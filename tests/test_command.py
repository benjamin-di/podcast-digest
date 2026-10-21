"""Command summarizer backend (``PODCAST_DIGEST_SUMMARIZER=command``).

``subprocess.run`` is stubbed: these tests pin the argv (local or over SSH),
the stdin request for both protocols, result parsing, failure classification
and the retry policy.
"""
from __future__ import annotations

import json
import re
import subprocess

import pytest

from podcast_digest import command
from podcast_digest.command import CommandCompleter, CommandConfig
from podcast_digest.llm import SummarizerUnavailable
from podcast_digest.summarize import SummarizationInput, summarize
from podcast_digest.transcribe import Segment


def _input() -> SummarizationInput:
    return SummarizationInput(
        podcast_name="The Test Podcast",
        title="Episode 1",
        published_at="2026-06-16T00:00:00+00:00",
        duration_seconds=3600,
        description="notes",
        transcript_text="hello world",
        segments=[Segment(start=0.0, end=2.0, text="hello world")],
    )


def _json_cfg(**over) -> CommandConfig:
    kw = dict(
        command="llm-runner", protocol="json", tier="value",
        ssh_host="runner.example", ssh_port=2222, ssh_user="dev", ssh_key="/k/id",
    )
    kw.update(over)
    return CommandConfig(**kw)


def _summarize(cfg: CommandConfig | None = None) -> str:
    return summarize(_input(), completer=CommandCompleter(cfg or _json_cfg()))


def _ndjson(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _result(text: str, *, is_error: bool = False, provider: str = "claude") -> dict:
    return {"version": 1, "type": "result", "provider": provider,
            "is_error": is_error, "text": text}


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Make the retry backoff instant so failure-shape tests are fast."""
    monkeypatch.setattr("podcast_digest.llm.time.sleep", lambda *_: None)


# ---- JSON protocol over SSH -------------------------------------------------


def test_json_happy_path_extracts_result_text(monkeypatch):
    captured = {}

    def fake_run(argv, *, input, capture_output, text, timeout):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_started", "run_id": "x"},
            _result("<h2>TL;DR</h2><p>Hi.</p>"),
        ))

    monkeypatch.setattr(subprocess, "run", fake_run)
    html = _summarize()

    assert html == "<h2>TL;DR</h2><p>Hi.</p>"
    # argv carries the ssh target, key, and remote command.
    assert captured["argv"][0] == "ssh"
    assert "dev@runner.example" in captured["argv"]
    assert captured["argv"][-1] == "llm-runner"
    assert "/k/id" in captured["argv"]
    assert "2222" in captured["argv"]
    # The request names the tier (which owns models + fallback chain) and the
    # system prompt; no model unless one is configured.
    cfg = json.loads(captured["input"])
    assert cfg["tier"] == "value"
    assert "model" not in cfg
    assert "TRANSCRIPT" in cfg["prompt"]
    assert cfg["system_prompt"]
    assert cfg["source"] == "podcast-digest"
    # Text-transform runs must ask the runner to disable any tool surface.
    assert cfg["tools"] == ""


def test_json_request_carries_model_and_omits_empty_tier(monkeypatch):
    captured = {}

    def fake_run(argv, *, input, **_kw):
        captured["input"] = input
        return _FakeProc(stdout=_ndjson(_result("<h2>TL;DR</h2><p>ok</p>")))

    monkeypatch.setattr(subprocess, "run", fake_run)
    _summarize(_json_cfg(tier="", model="some-model"))
    cfg = json.loads(captured["input"])
    assert cfg["model"] == "some-model"
    assert "tier" not in cfg


def test_strips_code_fence(monkeypatch):
    fenced = "```html\n<h2>TL;DR</h2><p>x</p>\n```"
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: _FakeProc(stdout=_ndjson(_result(fenced))),
    )
    assert _summarize() == "<h2>TL;DR</h2><p>x</p>"


def test_strips_brief_preamble_before_valid_html(monkeypatch):
    reply = (
        "I'll condense this transcript into the requested HTML email format. "
        "<h2>TL;DR</h2><p>x</p>"
    )
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stdout=_ndjson(_result(reply)))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _summarize() == "<h2>TL;DR</h2><p>x</p>"
    assert calls["n"] == 1


def test_file_pointer_result_retries_then_raises(monkeypatch):
    """An agent that writes the HTML to a file and replies with a pointer must
    never ship that pointer as the email body — it burns the retry budget and
    then raises (episode stays `transcribed`, retried next tick)."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stdout=_ndjson(
            _result("The HTML fragment is ready in `/tmp/summary.html`."),
        ))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid result"):
        _summarize()
    assert calls["n"] == 3  # first try + 2 retries


def test_invalid_result_then_valid_recovers(monkeypatch):
    good = "<h2>TL;DR</h2><p>ok</p>"
    outputs = ["Wrote the summary to /tmp/summary.html", good]
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(_result(outputs.pop(0)))),
    )
    assert _summarize() == good


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: _FakeProc(stderr="boom", returncode=255),
    )
    with pytest.raises(SummarizerUnavailable, match="failed rc=255"):
        _summarize()


def test_result_is_error_raises(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(_result("auth bad", is_error=True))),
    )
    with pytest.raises(RuntimeError, match="is_error"):
        _summarize()


def test_early_is_error_result_rescued_by_later_provider(monkeypatch):
    """Provider fallback emits a failed `result` with is_error=true and THEN a
    second provider's `result` that rescues the run (exit 0, no run_failed). The
    parser must return the good text, not raise on the first error result."""
    good = "<h2>TL;DR</h2><p>rescued</p>"
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_started", "run_id": "x"},
            _result("first provider failed", is_error=True),
            {"version": 1, "type": "provider_failed", "provider": "claude"},
            _result(good, provider="codex"),
        )),
    )
    assert _summarize() == good


def test_all_error_results_still_raise(monkeypatch):
    """When EVERY result is an error and none is rescued, still surface it."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(
            _result("err one", is_error=True),
            _result("err two", is_error=True, provider="codex"),
        )),
    )
    with pytest.raises(RuntimeError, match="is_error"):
        _summarize()


def test_validation_failure_with_terminal_marker_in_output_still_retries(monkeypatch):
    """A summary whose *content* happens to contain a terminal marker like
    "unauthorized" must NOT be misread as a terminal transport error — the
    validation failure is always retryable."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        # Not valid HTML (fails validation) AND contains "unauthorized".
        return _FakeProc(stdout=_ndjson(
            _result("The Fed said banks were unauthorized to lend."),
        ))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid result"):
        _summarize()
    assert calls["n"] == 3  # retried the full budget, not failed-fast as terminal


def test_all_error_with_terminal_marker_in_text_still_retries(monkeypatch):
    """An all-error stream whose provider text happens to contain a terminal
    marker ("unauthorized") must NOT be misclassified as terminal and skip the
    retry budget — the error carries model/provider text, not transport text."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stdout=_ndjson(
            _result("The guest said banks were unauthorized", is_error=True),
        ))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="is_error"):
        _summarize()
    assert calls["n"] == 3  # full retry budget, not failed-fast as terminal


def test_whitespace_result_does_not_clobber_good_earlier_result(monkeypatch):
    """A later whitespace-only result must not overwrite a real one."""
    good = "<h2>TL;DR</h2><p>real</p>"
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(
            _result(good), _result("   ", provider="codex"),
        )),
    )
    assert _summarize() == good


def test_run_failed_raises(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_failed", "reason": "all providers down",
             "auth_failure": False},
        )),
    )
    with pytest.raises(SummarizerUnavailable, match="run_failed"):
        _summarize()


def test_nonzero_exit_classifies_auth_failure(monkeypatch):
    blob = _ndjson(
        {"version": 1, "type": "run_failed", "reason": "exit_nonzero",
         "auth_failure": False},
        # later envelope flips to auth_failure — sticky flag must win.
        {"version": 1, "type": "run_failed", "reason": "auth_failure",
         "auth_failure": True},
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout="\x1b[31m" + blob + "\x1b[0m", returncode=1),
    )
    with pytest.raises(SummarizerUnavailable) as ei:
        _summarize()
    exc = ei.value
    assert exc.reason == "auth_failure"
    assert exc.auth_failure is True
    assert exc.rc == 1
    assert "\x1b" not in exc.raw  # ANSI stripped from the stored blob


@pytest.mark.parametrize("reason,auth", [("auth_failure", True), ("geo_blocked", False)])
def test_terminal_runner_reasons_are_not_retried(monkeypatch, reason, auth):
    """An auth failure, or a regional egress block (retrying from the same
    egress hits the same wall), must fail fast — not burn the retry budget."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_failed", "reason": reason, "auth_failure": auth},
        ), returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable):
        _summarize()
    assert calls["n"] == 1


def test_transient_nonzero_exit_retries(monkeypatch):
    """A non-auth nonzero exit is transient: retried up to the budget."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_failed", "reason": "exit_nonzero",
             "auth_failure": False},
        ), returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable):
        _summarize()
    assert calls["n"] == 3  # 1 try + 2 retries


def test_ssh_permission_denied_is_terminal(monkeypatch):
    """SSH-level auth markers live in stderr (stored in .raw, not the message)
    — they must still be detected as terminal and fail fast."""
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stderr="Permission denied (publickey).", returncode=255)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable, match="failed rc=255"):
        _summarize()
    assert calls["n"] == 1  # terminal: not retried


def test_failure_emits_elapsed_timing(monkeypatch, caplog):
    """A failed summarize logs summarize.failed with elapsed_s, so a
    slow-then-failing call is distinguishable from a fast failure."""
    import logging

    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: _FakeProc(stderr="boom", returncode=255),
    )
    with caplog.at_level(logging.WARNING, logger="podcast_digest.summarize"), \
            pytest.raises(RuntimeError):
        _summarize()
    rec = next(r for r in caplog.records if r.getMessage() == "summarize.failed")
    assert "elapsed_s" in rec.extras
    assert rec.extras["err_reason"] == "exit_nonzero"


def test_no_result_event_raises(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_kw: _FakeProc(stdout=_ndjson(
            {"version": 1, "type": "run_started", "run_id": "x"},
        )),
    )
    with pytest.raises(SummarizerUnavailable, match="no result"):
        _summarize()


def test_timeout_raises(monkeypatch):
    def fake_run(argv, **_kw):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=360)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable, match="timed out"):
        _summarize()


def test_missing_command_raises():
    with pytest.raises(RuntimeError, match="PODCAST_DIGEST_COMMAND is not set"):
        _summarize(_json_cfg(command=""))
    assert CommandCompleter(_json_cfg(command=" ")).config_problem()


def test_known_hosts_pins_strict(monkeypatch):
    captured = {}

    def fake_run(argv, **_kw):
        captured["argv"] = argv
        return _FakeProc(stdout=_ndjson(_result("<h2>TL;DR</h2><p>ok</p>")))

    monkeypatch.setattr(subprocess, "run", fake_run)
    _summarize(_json_cfg(ssh_known_hosts="/kh"))
    assert "UserKnownHostsFile=/kh" in captured["argv"]
    assert "StrictHostKeyChecking=yes" in captured["argv"]


# ---- text protocol, local ---------------------------------------------------


def test_text_protocol_runs_locally_with_prompt_on_stdin(monkeypatch):
    captured = {}

    def fake_run(argv, *, input, **_kw):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeProc(stdout="<h2>TL;DR</h2><p>local</p>\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    html = _summarize(CommandConfig(command="my-llm --model 'big one' -q"))
    assert html == "<h2>TL;DR</h2><p>local</p>"
    # Split shell-style, run locally (no ssh).
    assert captured["argv"] == ["my-llm", "--model", "big one", "-q"]
    # stdin carries the system prompt and the episode prompt as one text.
    assert captured["input"].startswith("You are an editorial assistant")
    assert "--- TRANSCRIPT" in captured["input"]


def test_text_protocol_nonzero_exit_is_transient(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        return _FakeProc(stderr="rate limited", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable) as ei:
        _summarize(CommandConfig(command="my-llm"))
    assert ei.value.reason == "exit_nonzero"
    assert calls["n"] == 3


def test_missing_executable_is_terminal_config_error(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, **_kw):
        calls["n"] += 1
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SummarizerUnavailable) as ei:
        _summarize(CommandConfig(command="not-installed"))
    assert ei.value.reason == "config_error"
    assert calls["n"] == 1


# ---- shared retry at the run boundary ---------------------------------------


def _complete(cfg: CommandConfig | None = None) -> str:
    return CommandCompleter(cfg or _json_cfg()).complete(
        system="s", prompt="p", purpose="summarize",
    )


def test_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_once(cfg, **_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("API Error: 500 internal server error")
        return "<p>ok</p>"

    monkeypatch.setattr(command, "_run_once", fake_once)
    assert _complete() == "<p>ok</p>"
    assert calls["n"] == 3  # 1 try + 2 retries, succeeds on the 3rd


def test_gives_up_after_retries(monkeypatch):
    calls = {"n": 0}

    def fake_once(cfg, **_kw):
        calls["n"] += 1
        raise RuntimeError("rate limit exceeded, try again")

    monkeypatch.setattr(command, "_run_once", fake_once)
    with pytest.raises(RuntimeError, match="rate limit"):
        _complete()
    assert calls["n"] == 3  # exhausts 1 try + 2 retries


def test_terminal_error_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_once(cfg, **_kw):
        calls["n"] += 1
        raise RuntimeError("PODCAST_DIGEST_COMMAND is not set")

    monkeypatch.setattr(command, "_run_once", fake_once)
    with pytest.raises(RuntimeError, match="is not set"):
        _complete()
    assert calls["n"] == 1  # terminal denylist: fail fast, no retry


def test_clean_purpose_uses_its_own_source_and_tail_limit(monkeypatch):
    captured = {}

    def fake_run(argv, *, input, **_kw):
        captured["cfg"] = json.loads(input)
        return _FakeProc(stdout=_ndjson(_result("cleaned")))

    monkeypatch.setattr(subprocess, "run", fake_run)
    CommandCompleter(_json_cfg()).complete(system="s", prompt="p", purpose="clean")
    assert captured["cfg"]["source"] == "podcast-digest-clean"
    assert captured["cfg"]["stdout_tail_limit"] > 64_000
    assert re.fullmatch(r"\d+", str(captured["cfg"]["timeout_seconds"]))
