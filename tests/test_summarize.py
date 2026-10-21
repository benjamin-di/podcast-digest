"""Backend-neutral LLM tasks (summary + cleanup) and the API backend.

The summary/cleanup tests drive a fake Completer; the ApiCompleter tests stub
the SDK clients, so no network or API key is involved.
"""
from __future__ import annotations

import re
import sys
import types

import pytest

from podcast_digest.llm import (
    ApiCompleter,
    SummarizerUnavailable,
    UnusableReply,
    provider_for_model,
)
from podcast_digest.summarize import (
    SummarizationInput,
    clean_transcript,
    summarize,
    summary_system_prompt,
)
from podcast_digest.transcribe import Segment


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    monkeypatch.setattr("podcast_digest.llm.time.sleep", lambda *_: None)


class FakeCompleter:
    """Records calls; replies via ``reply(system, prompt, purpose)``."""

    label = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def config_problem(self):
        return None

    def complete(self, *, system, prompt, purpose, guid=None, validate=None):
        self.calls.append({"system": system, "prompt": prompt, "purpose": purpose})
        text = self.reply(system, prompt, purpose)
        if validate is not None and validate(text):
            raise UnusableReply(f"invalid result: {text}")
        return text


def _input(**over) -> SummarizationInput:
    kw = dict(
        podcast_name="P", title="T", published_at="2026", duration_seconds=None,
        description=None, transcript_text="hello", segments=[],
    )
    kw.update(over)
    return SummarizationInput(**kw)


def _multi_chunk_input() -> SummarizationInput:
    return _input(
        transcript_text="x" * 100,
        segments=[Segment(start=float(i), end=float(i) + 1, text="word " * 40)
                  for i in range(20)],
    )


# ---- summary prompt ---------------------------------------------------------


def test_general_style_is_the_default_and_names_no_finance_terms():
    prompt = summary_system_prompt()
    assert "busy, curious reader" in prompt
    assert "SOFR" not in prompt and "tickers" not in prompt
    assert "<h2>TL;DR</h2>" in prompt


def test_finance_style_writes_for_a_finance_reader():
    prompt = summary_system_prompt("finance")
    assert "finance-literate reader" in prompt
    assert "SOFR" in prompt


def test_prompt_file_replaces_the_system_prompt_and_relaxes_validation(tmp_path):
    custom = tmp_path / "prompt.txt"
    custom.write_text("Summarize in three bullet points as an HTML <ul>.\n")
    fake = FakeCompleter(lambda *_: "<ul><li>a</li><li>b</li></ul>")
    html = summarize(_input(), completer=fake, prompt_file=str(custom))
    assert html == "<ul><li>a</li><li>b</li></ul>"  # no <h2> needed
    assert fake.calls[0]["system"] == "Summarize in three bullet points as an HTML <ul>."


def test_summary_with_builtin_prompt_requires_h2_sections():
    fake = FakeCompleter(lambda *_: "<p>no headings</p>")
    with pytest.raises(UnusableReply, match="invalid result"):
        summarize(_input(), completer=fake)


def test_style_reaches_the_language_guidance():
    fake = FakeCompleter(lambda *_: "<h2>TL;DR</h2>")
    summarize(_input(source_language="zh"), completer=fake, style="finance")
    assert "PBOC" in fake.calls[0]["prompt"]
    summarize(_input(source_language="zh"), completer=fake, style="general")
    assert "PBOC" not in fake.calls[1]["prompt"]


# ---- transcript cleanup -----------------------------------------------------


def _chunk_no(prompt: str) -> str:
    return re.search(r"chunk (\d+) of", prompt).group(1)


def test_clean_transcript_chunks_parallel_and_ordered():
    fake = FakeCompleter(lambda _s, prompt, _p: f"PART{_chunk_no(prompt)}")
    md = clean_transcript(_multi_chunk_input(), completer=fake, max_chunk_chars=400)

    assert len(fake.calls) >= 2                   # chunked
    assert all(c["purpose"] == "clean" for c in fake.calls)
    assert md.index("PART1") < md.index("PART2")  # order preserved despite parallelism
    assert md.startswith("# T")


def _dead_first_chunk(_s, prompt, _p):
    i = _chunk_no(prompt)
    if i == "1":
        raise SummarizerUnavailable("clean timed out after 360s", reason="timeout")
    return f"PART{i}"


def test_clean_transcript_all_or_nothing_by_default():
    """Default (allow_partial=False): a single failing chunk aborts the whole
    clean, so the pipeline can defer and retry the entire clean next tick."""
    with pytest.raises(SummarizerUnavailable, match="timed out"):
        clean_transcript(
            _multi_chunk_input(), completer=FakeCompleter(_dead_first_chunk),
            max_chunk_chars=400, max_parallel=2,
        )


def test_clean_transcript_partial_tolerates_one_dead_chunk():
    """allow_partial=True (final attempt): a dead chunk degrades to a marker
    while the survivors are kept."""
    md = clean_transcript(
        _multi_chunk_input(), completer=FakeCompleter(_dead_first_chunk),
        allow_partial=True, max_chunk_chars=400, max_parallel=2,
    )
    assert "PART2" in md                          # surviving chunks still attached
    assert "unavailable for this section" in md   # gap marker in place of chunk 1


def test_clean_transcript_partial_raises_when_all_fail():
    def dead(*_):
        raise SummarizerUnavailable("timed out", reason="timeout")

    with pytest.raises(RuntimeError, match="all .* chunk"):
        clean_transcript(
            _multi_chunk_input(), completer=FakeCompleter(dead),
            allow_partial=True, max_chunk_chars=400, max_parallel=2,
        )


# ---- API backend ------------------------------------------------------------


def test_provider_for_model_dispatch():
    assert provider_for_model("claude-sonnet-5") == "anthropic"
    assert provider_for_model("gpt-5") == "openai"
    assert provider_for_model("o3-mini") == "openai"
    # Anything else is an OpenAI-compatible model (Ollama, LM Studio, vLLM, …).
    assert provider_for_model("llama3.1:8b") == "openai"


def test_explicit_provider_overrides_the_name():
    assert ApiCompleter(model="my-claude-finetune", provider="anthropic").provider == "anthropic"


def test_missing_key_is_a_config_problem_and_unavailable_on_use():
    api = ApiCompleter(model="claude-sonnet-5")
    assert "ANTHROPIC_API_KEY" in api.config_problem()
    with pytest.raises(SummarizerUnavailable) as ei:
        api.complete(system="s", prompt="p", purpose="summarize")
    assert ei.value.reason == "config_error"
    # An OpenAI-compatible server needs no key.
    assert ApiCompleter(model="llama3", openai_base_url="http://localhost:11434/v1") \
        .config_problem() is None


class _FakeChat:
    def __init__(self, record, reply):
        self.record, self.reply = record, reply

    def create(self, **kw):
        self.record.update(kw)
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def _install_fake_openai(monkeypatch, record, reply="<h2>TL;DR</h2><p>ok</p>"):
    class FakeOpenAI:
        def __init__(self, **kw):
            record["client"] = kw
            self.chat = types.SimpleNamespace(completions=_FakeChat(record, reply))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


def test_openai_compatible_server_uses_chat_completions(monkeypatch):
    record: dict = {}
    _install_fake_openai(monkeypatch, record)
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:11434/v1")
    html = summarize(_input(), completer=api)
    assert html == "<h2>TL;DR</h2><p>ok</p>"
    assert record["client"]["base_url"] == "http://localhost:11434/v1"
    assert record["client"]["api_key"]  # the SDK needs a non-empty placeholder
    assert record["model"] == "llama3"
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["content"].startswith("Podcast: P")


def test_api_retries_an_unusable_reply_once(monkeypatch):
    replies = ["Wrote it to /tmp/x.html", "<h2>TL;DR</h2><p>second</p>"]
    record: dict = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_FakeChat(record, replies.pop(0)))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:1/v1")
    assert summarize(_input(), completer=api) == "<h2>TL;DR</h2><p>second</p>"


def _raising_openai(monkeypatch, exc_class_name: str, status: int | None = None):
    exc_type = type(exc_class_name, (Exception,), {})
    calls = {"n": 0}

    class FakeOpenAI:
        def __init__(self, **kw):
            pass

        @property
        def chat(self):
            def create(**kw):
                calls["n"] += 1
                e = exc_type("boom")
                if status is not None:
                    e.status_code = status
                raise e
            return types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    return calls


@pytest.mark.parametrize("name,status,reason", [
    ("APIConnectionError", None, "api_unavailable"),
    ("RateLimitError", 429, "api_unavailable"),
    ("InternalServerError", 503, "api_unavailable"),
    ("AuthenticationError", 401, "auth_failure"),
])
def test_api_outages_are_summarizer_unavailable(monkeypatch, name, status, reason):
    """Transport/auth failures are the backend's, not the episode's: they go on
    the transport budget and mute the heartbeat. The SDK already retried them,
    so they are not retried again here."""
    calls = _raising_openai(monkeypatch, name, status)
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:1/v1")
    with pytest.raises(SummarizerUnavailable) as ei:
        api.complete(system="s", prompt="p", purpose="summarize")
    assert ei.value.reason == reason
    assert calls["n"] == 1


def test_too_long_transcript_stays_an_episode_error(monkeypatch):
    """A request rejected because of this episode's transcript is the
    episode's problem: its own attempt budget, not the transport budget."""
    calls = _raising_openai(monkeypatch, "BadRequestError", 400)
    import openai as stub

    original = stub.OpenAI

    class TooLong(original):  # type: ignore[misc, valid-type]
        @property
        def chat(self):
            def create(**kw):
                calls["n"] += 1
                e = type("BadRequestError", (Exception,), {})("prompt is too long: 250000 tokens")
                e.status_code = 400
                raise e
            return types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=TooLong))
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:1/v1")
    with pytest.raises(Exception) as ei:
        api.complete(system="s", prompt="p", purpose="summarize")
    assert not isinstance(ei.value, SummarizerUnavailable)


@pytest.mark.parametrize("name,status", [("BadRequestError", 400), ("NotFoundError", 404)])
def test_rejected_configuration_is_summarizer_unavailable(monkeypatch, name, status):
    """An unknown model or an option the model rejects fails every call: a
    configuration error (episodes wait, the heartbeat goes quiet)."""
    _raising_openai(monkeypatch, name, status)
    api = ApiCompleter(model="no-such-model", openai_base_url="http://localhost:1/v1")
    with pytest.raises(SummarizerUnavailable) as ei:
        api.complete(system="s", prompt="p", purpose="summarize")
    assert ei.value.reason == "config_error"


def test_anthropic_requests_send_no_thinking_option_and_handle_refusals(monkeypatch):
    """Current Claude models reject `thinking: disabled`; the request must not
    carry a thinking option at all. A refusal is an unusable reply."""
    sent: list[dict] = []
    replies = iter(["refusal", "end_turn"])

    class Messages:
        def create(self, **kw):
            sent.append(kw)
            stop = next(replies)
            block = types.SimpleNamespace(type="text", text="" if stop == "refusal" else "<h2>TL;DR</h2>")
            return types.SimpleNamespace(
                content=[block], stop_reason=stop,
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
            )

    class FakeAnthropic:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic))
    api = ApiCompleter(model="claude-opus-5-5", anthropic_api_key="k")
    assert summarize(_input(), completer=api) == "<h2>TL;DR</h2>"
    assert len(sent) == 2                       # the refusal was retried once
    assert all("thinking" not in kw for kw in sent)
    assert sent[0]["max_tokens"] == 16000



def test_unreadable_prompt_file_is_a_summarizer_config_error(tmp_path):
    """Every episode would fail the same way: never an episode failure."""
    fake = FakeCompleter(lambda *_: "<h2>TL;DR</h2>")
    with pytest.raises(SummarizerUnavailable) as ei:
        summarize(_input(), completer=fake, prompt_file=str(tmp_path / "missing.txt"))
    assert ei.value.reason == "config_error"
    assert fake.calls == []


def test_reasoning_trace_never_reaches_the_summary_or_cleanup():
    reply = "<think>\nLet me plan the sections…\n</think>\n<h2>TL;DR</h2><p>ok</p>"
    assert summarize(_input(), completer=FakeCompleter(lambda *_: reply)) == \
        "<h2>TL;DR</h2><p>ok</p>"
    md = clean_transcript(
        _input(), completer=FakeCompleter(lambda *_: "<think>hm</think>**Host:** hi"),
    )
    assert "think" not in md and "**Host:** hi" in md


def test_reply_cut_off_at_the_output_limit_is_rejected(monkeypatch):
    record: dict = {}

    class CutChat(_FakeChat):
        def create(self, **kw):
            resp = super().create(**kw)
            resp.choices[0].finish_reason = "length"
            return resp

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(
                completions=CutChat(record, "<h2>TL;DR</h2><p>half a sum"),
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:1/v1")
    with pytest.raises(UnusableReply, match="output limit"):
        api.complete(system="s", prompt="p", purpose="clean")
    assert record["max_tokens"] == 16000  # cleanup gets the larger budget


def test_billing_refusal_is_summarizer_unavailable(monkeypatch):
    """A 400 for an exhausted credit balance fails every call until someone
    tops up: the backend's problem, not the episode's."""
    class FakeOpenAI:
        def __init__(self, **kw):
            pass

        @property
        def chat(self):
            def create(**kw):
                e = type("BadRequestError", (Exception,), {})(
                    "Your credit balance is too low to access the API",
                )
                e.status_code = 400
                raise e
            return types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    api = ApiCompleter(model="llama3", openai_base_url="http://localhost:1/v1")
    with pytest.raises(SummarizerUnavailable) as ei:
        api.complete(system="s", prompt="p", purpose="summarize")
    assert ei.value.reason == "config_error"
