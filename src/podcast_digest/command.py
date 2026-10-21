"""Command backend: run a command for each LLM call, locally or over SSH.

Used when ``PODCAST_DIGEST_SUMMARIZER=command``. Instead of calling a model API
in-process, each call runs ``PODCAST_DIGEST_COMMAND`` with the prompt on stdin —
for example a coding-agent CLI on a subscription (``claude -p``), ``llm``, or
``ollama run <model>``. With ``PODCAST_DIGEST_SSH_HOST`` set the command runs on
that host over SSH, so transcription and the model can live on different
machines without either holding the other's credentials.

Two protocols (``PODCAST_DIGEST_COMMAND_PROTOCOL``):

  - ``text`` (default): stdin is the system prompt and the user prompt as one
    text; stdout is the answer. Works with any CLI that reads a prompt on stdin.
  - ``json``: stdin is one JSON request; stdout is NDJSON events, of which the
    last non-error ``result`` event carries the answer and a ``run_failed``
    event classifies a failure. Suits a runner with provider fallback and
    machine-readable errors. The contract is in docs/command-backend.md.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import Literal

from .llm import Purpose, SummarizerUnavailable, UnusableReply, Validator, with_retries
from .log import oneline, strip_ansi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandConfig:
    """How to run the summarizer command."""

    command: str
    protocol: Literal["text", "json"] = "text"
    timeout_seconds: int = 300
    # JSON protocol only: optional runner tier and an explicit model request.
    tier: str = ""
    model: str | None = None
    # Run the command on this host over SSH (empty = run locally).
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key: str = ""
    ssh_known_hosts: str = ""


def _argv(cfg: CommandConfig) -> list[str]:
    if not cfg.ssh_host:
        return shlex.split(cfg.command)
    argv = [
        "ssh",
        "-p", str(cfg.ssh_port),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        # A run can sit quietly for minutes while the model thinks; keep the
        # connection alive so a long-but-healthy run isn't reaped.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
    ]
    if cfg.ssh_key:
        # IdentitiesOnly stops ssh from offering other agent keys first (which
        # can trip MaxAuthTries before our key is tried).
        argv += ["-i", cfg.ssh_key, "-o", "IdentitiesOnly=yes"]
    if cfg.ssh_known_hosts:
        argv += [
            "-o", f"UserKnownHostsFile={cfg.ssh_known_hosts}",
            "-o", "StrictHostKeyChecking=yes",
        ]
    else:
        # TOFU: pin the host key on first connect. Acceptable for a fixed host;
        # set PODCAST_DIGEST_SSH_KNOWN_HOSTS for strict pinning.
        argv += ["-o", "StrictHostKeyChecking=accept-new"]
    target = f"{cfg.ssh_user}@{cfg.ssh_host}" if cfg.ssh_user else cfg.ssh_host
    # The remote shell parses the command string, exactly as typed.
    return [*argv, target, cfg.command]


def _classify_failure(stdout: str) -> tuple[str, bool]:
    """Scan JSON-protocol output for the ``run_failed`` envelope and return
    ``(reason, auth_failure)``.

    A runner may flip-flop its classification across retries (one attempt
    reports ``auth_failure=true``, the next ``exit_nonzero``) — so the auth flag
    is sticky (any ``true`` wins) and an auth failure pins the reason, giving a
    stable signal to alert on. Falls back to ``exit_nonzero`` when no envelope
    is present (e.g. the process died before emitting one)."""
    reason = "exit_nonzero"
    auth = False
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "run_failed":
            auth = auth or bool(evt.get("auth_failure", False))
            reason = str(evt.get("reason") or reason)
    if auth:
        reason = "auth_failure"
    return reason, auth


class _AllResultsFailed(UnusableReply):
    """Every ``result`` event in the stream was ``is_error`` and none was
    rescued by a fallback provider — a provider-level error carrying its text."""


def _parse_result(stdout: str) -> str:
    """Pull the final answer out of a JSON-protocol NDJSON event stream.

    Provider fallback emits more than one ``result`` event: a failed provider
    can emit ``result`` with ``is_error=true`` and *then* a fallback rescues the
    run with a good ``result`` — the process exits 0 with no ``run_failed``. So
    we must NOT raise on the first error result; we scan the whole stream, keep
    the last non-empty *non-error* result, and only surface the error if no good
    result exists anywhere. (A terminal chain failure still arrives as
    ``run_failed`` and raises below.)
    """
    result_text = ""
    error_text: str | None = None  # None = no is_error result seen at all
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            # stderr/diagnostics can leak onto stdout in odd cases; skip them.
            continue
        etype = evt.get("type")
        if etype == "run_failed":
            reason, auth = _classify_failure(stdout)
            raise SummarizerUnavailable(
                f"command run_failed reason={reason}",
                reason=reason, auth_failure=auth,
                raw=strip_ansi(stdout.strip())[-1500:],
            )
        if etype == "result":
            if evt.get("is_error"):
                error_text = str(evt.get("text", "")) or error_text or ""
                continue
            # Keep the last non-empty non-error result. `.strip()` so a
            # whitespace-only later result can't clobber a real earlier one.
            text = evt.get("text", "")
            if text.strip():
                result_text = text
    if result_text:
        return result_text
    if error_text is not None:
        raise _AllResultsFailed(f"command result is_error: {error_text[:300]}")
    return ""


# Retries (in addition to the first try) for transient failures. A policy
# constant, like MAX_ATTEMPTS in db.py — not a per-deploy env knob.
_MAX_RETRIES = 2

# Substrings marking a *terminal* failure: retrying cannot help, so fail fast.
# Everything else is treated as transient and retried. A run is idempotent, so
# retry-by-default is safe; the set of transient errors is open-ended
# (timeouts, 5xx, rate limits, network resets) while terminal config/auth
# errors are few and stable — a denylist is more robust than trying to
# enumerate every transient pattern.
_TERMINAL_MARKERS = (
    "is not set",            # missing command / config
    "permission denied",
    "authentication failed",
    "unauthorized",
    "no such identity",      # bad / missing ssh key
    "command not found",     # the remote shell cannot find the command
)


def _is_terminal(exc: Exception) -> bool:
    # An unusable reply is always retryable — and its message carries
    # model/provider output we must NOT scan for terminal markers. Decide first.
    if isinstance(exc, UnusableReply):
        return False
    # An authenticated-provider failure or an explicitly terminal runner reason
    # cannot be fixed by retrying — fail fast (and stop re-logging the blob).
    # geo_blocked is terminal too: a regional egress block hits the same wall on
    # every retry from the same egress — the fix is routing, not a re-run.
    if getattr(exc, "auth_failure", False):
        return True
    if getattr(exc, "reason", "") in ("auth_failure", "config_error", "geo_blocked"):
        return True
    # Scan only transport-level text — the terse message and `.raw` (stderr:
    # "permission denied" / "no such identity"). Never the model output.
    haystack = f"{exc} {getattr(exc, 'raw', '')}".lower()
    return any(m in haystack for m in _TERMINAL_MARKERS)


# How much NDJSON a JSON-protocol runner may return per purpose (its
# stdout_tail_limit): a cleaned chunk is far longer than a summary.
_STDOUT_TAIL_LIMIT = {"summarize": 64_000, "clean": 200_000}


def _request(cfg: CommandConfig, *, system: str, prompt: str, purpose: Purpose) -> str:
    """The stdin payload for one run."""
    if cfg.protocol == "text":
        return f"{system}\n\n---\n\n{prompt}"
    config: dict[str, object] = {
        "prompt": prompt,
        "system_prompt": system,
        # Pure text transform: ask the runner to disable any tool surface.
        # Without this an agent can write its answer to a file and reply with a
        # pointer, which would ship verbatim as the email body.
        "tools": "",
        # The working directory is irrelevant to the work but must exist where
        # the runner starts its CLI; "/tmp" always exists.
        "cwd": "/tmp",
        "source": "podcast-digest" if purpose == "summarize" else "podcast-digest-clean",
        "timeout_seconds": cfg.timeout_seconds,
        "stdout_tail_limit": _STDOUT_TAIL_LIMIT[purpose],
    }
    if cfg.tier:
        config["tier"] = cfg.tier
    if cfg.model:
        config["model"] = cfg.model
    return json.dumps(config, ensure_ascii=False)


def _run_once(cfg: CommandConfig, *, system: str, prompt: str, purpose: Purpose) -> str:
    """One run. Returns the answer text; raises on any failure."""
    if not cfg.command.strip():
        raise RuntimeError("PODCAST_DIGEST_COMMAND is not set")
    where = f"ssh {cfg.ssh_host}" if cfg.ssh_host else "local"
    try:
        proc = subprocess.run(  # noqa: S603 - argv comes from operator config, not input
            _argv(cfg),
            input=_request(cfg, system=system, prompt=prompt, purpose=purpose),
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds + 60,
        )
    except subprocess.TimeoutExpired as e:
        raise SummarizerUnavailable(
            f"command {purpose} ({where}) timed out after {cfg.timeout_seconds + 60}s",
            reason="timeout",
        ) from e
    except OSError as e:
        # The executable itself is missing or not runnable: configuration.
        raise SummarizerUnavailable(
            f"command {purpose} ({where}) could not start: {e}",
            reason="config_error",
        ) from e

    if proc.returncode != 0:
        # A JSON runner reports the provider error on stdout (NDJSON); stderr
        # alone is often just a benign known_hosts warning. Keep both ends,
        # bounded: SSH-level markers (permission denied, …) land at the start of
        # stderr, the runner's failure message at the end of stdout.
        if cfg.protocol == "json":
            reason, auth = _classify_failure(proc.stdout)
        else:
            reason, auth = "exit_nonzero", False
        raw = strip_ansi(
            f"rc={proc.returncode} stderr={proc.stderr.strip()[:500]} "
            f"stdout={proc.stdout.strip()[-1500:]}"
        )
        raise SummarizerUnavailable(
            f"command {purpose} ({where}) failed rc={proc.returncode} reason={reason}",
            reason=reason, auth_failure=auth, rc=proc.returncode, raw=raw,
        )

    text = _parse_result(proc.stdout) if cfg.protocol == "json" else proc.stdout.strip()
    if not text:
        raise SummarizerUnavailable(
            f"command {purpose} ({where}) returned no result; "
            f"stderr={oneline(strip_ansi(proc.stderr.strip()))}",
            reason="no_result",
        )
    return text


class CommandCompleter:
    """Completer that runs ``PODCAST_DIGEST_COMMAND`` per call (see module doc)."""

    def __init__(self, cfg: CommandConfig) -> None:
        self.cfg = cfg
        where = f"@{cfg.ssh_host}" if cfg.ssh_host else ""
        self.label = f"command{where}:{cfg.protocol}"

    def config_problem(self) -> str | None:
        if not self.cfg.command.strip():
            return "PODCAST_DIGEST_COMMAND is not set"
        return None

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        purpose: Purpose,
        guid: str | None = None,
        validate: Validator | None = None,
    ) -> str:
        return with_retries(
            lambda: _run_once(self.cfg, system=system, prompt=prompt, purpose=purpose),
            validate=validate, is_terminal=_is_terminal, max_retries=_MAX_RETRIES,
            label=f"{purpose}:{self.label}", guid=guid,
        )
