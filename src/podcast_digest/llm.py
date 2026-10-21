"""LLM backends behind one call: ``complete(system=…, prompt=…) -> text``.

The summary and the transcript cleanup only need "send a system prompt and a
user prompt, get text back", so every backend implements :class:`Completer`:

  - :class:`ApiCompleter` — Anthropic, OpenAI, or any OpenAI-compatible server
    (Ollama, LM Studio, vLLM, …), called in-process.
  - :class:`podcast_digest.command.CommandCompleter` — one command per call,
    locally or on another host over SSH.

Failure classes the pipeline relies on:

  - :class:`SummarizerUnavailable` — the backend itself is unreachable or
    unusable (network, auth, outage, timeout, missing configuration). It is a
    property of the backend, not of the episode: the pipeline counts it on a
    separate transport budget and the heartbeat goes quiet while it lasts.
  - :class:`UnusableReply` — the backend answered, but the answer cannot be used
    (fails validation, or every provider reported an error). Always retried.
  - anything else — an episode-level failure, retried on the next tick within
    the episode's attempt budget.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

from .log import err_fields, event, oneline

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

Purpose = Literal["summarize", "clean"]
# Returns a short problem description, or None when the reply is acceptable.
Validator = Callable[[str], str | None]


class SummarizerUnavailable(RuntimeError):
    """The summarizer pathway itself is unreachable/unusable — transport failed
    or timed out, the backend returned no result, auth failed, or every provider
    is down. Distinct from a per-episode content error: a *sustained* run of
    these means the summarizer is out, which the scheduler's dead-man's-switch
    keys on. Subclasses RuntimeError so generic handlers keep working.

    Carries a machine-readable failure classification so call sites log
    ``err_reason`` / ``auth_failure`` as queryable fields instead of as a
    substring of a giant blob. ``raw`` holds the full (ANSI-stripped) transport
    output for DEBUG-level inspection."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unavailable",
        auth_failure: bool = False,
        rc: int | None = None,
        raw: str = "",
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.auth_failure = auth_failure
        self.rc = rc
        self.raw = raw


class UnusableReply(RuntimeError):
    """The backend replied but the reply is unusable. Always retryable, and its
    message embeds *model/provider output*, which must never be scanned for
    transport-level terminal markers: a summary literally containing
    "unauthorized" would otherwise be misclassified as terminal."""


class Completer(Protocol):
    """Anything that turns (system prompt, user prompt) into reply text."""

    #: Short description for logs, e.g. "anthropic:claude-sonnet-5".
    label: str

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        purpose: Purpose,
        guid: str | None = None,
        validate: Validator | None = None,
    ) -> str:
        """Return the reply text. Raises SummarizerUnavailable, UnusableReply,
        or another exception (see the module docstring)."""
        ...

    def config_problem(self) -> str | None:
        """A configuration error that makes every call fail, else None."""
        ...


def with_retries(
    call: Callable[[], str],
    *,
    validate: Validator | None,
    is_terminal: Callable[[Exception], bool],
    max_retries: int,
    label: str,
    guid: str | None,
) -> str:
    """Run ``call`` with bounded retry, exponential backoff and jitter.

    A reply that fails ``validate`` raises :class:`UnusableReply` and goes
    through the same retry budget (a re-run usually complies). Retries are
    logged at DEBUG — an outage that fans out across many chunks would otherwise
    flood the logs — and the caller logs one WARNING once the budget is spent.
    """
    for attempt in range(max_retries + 1):
        try:
            text = call()
            if validate is not None:
                problem = validate(text)
                if problem:
                    raise UnusableReply(
                        f"{label} returned an invalid result ({problem}): "
                        f"{oneline(text, limit=300)}"
                    )
            return text
        except Exception as e:  # noqa: BLE001 - reclassified by is_terminal
            if attempt == max_retries or is_terminal(e):
                raise
            backoff = min(2 ** attempt, 30) + random.uniform(0, 1.5)
            event(
                logger, logging.DEBUG, "llm.retry",
                label=label, guid=guid, attempt=attempt + 1, max=max_retries,
                backoff_s=round(backoff, 1), **err_fields(e),
            )
            raw = getattr(e, "raw", "")
            if raw:
                event(logger, logging.DEBUG, "llm.error_detail",
                      label=label, guid=guid, raw=oneline(raw, limit=2000))
            time.sleep(backoff)
    raise AssertionError("unreachable")  # the loop always returns or raises


# ---------------------------------------------------------------------------
# API backend (Anthropic / OpenAI / OpenAI-compatible)
# ---------------------------------------------------------------------------

# SDK exception class names (anthropic and openai share them) that mean the API
# itself is unusable rather than the request being wrong.
_AUTH_ERRORS = {"AuthenticationError", "PermissionDeniedError"}
_UNAVAILABLE_ERRORS = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "ServiceUnavailableError", "OverloadedError",
}

# Output budgets per task (max_tokens). Current models think adaptively before
# answering and the thinking shares this budget, so it is generous: a summary is
# ~1.5k tokens, a cleanup chunk about as much text as it receives (15k
# characters; far more tokens for CJK scripts). 16000 keeps a non-streaming
# request within the SDK's HTTP timeouts. A reply cut off at the budget is
# rejected, never used.
MAX_OUTPUT_TOKENS: dict[str, int] = {"summarize": 16000, "clean": 16000}

# A rejected request that is about the episode (its transcript is too long for
# the model) rather than the configuration (unknown model, unsupported option).
_EPISODE_REQUEST_ERRORS = ("prompt is too long", "too many tokens", "maximum context length",
                           "context_length_exceeded", "context window")


def provider_for_model(model: str) -> Literal["anthropic", "openai"]:
    """claude-* → Anthropic; anything else → OpenAI or an OpenAI-compatible server."""
    return "anthropic" if model.strip().lower().startswith("claude-") else "openai"


def _classify_sdk_error(e: Exception) -> Exception:
    """Map an SDK exception to SummarizerUnavailable when it is the API that is
    unusable; return it unchanged when the request itself was rejected."""
    names = {cls.__name__ for cls in type(e).__mro__}
    status = getattr(e, "status_code", None)
    if names & _AUTH_ERRORS or status in (401, 403):
        return SummarizerUnavailable(
            f"model API rejected the credentials: {oneline(str(e), limit=300)}",
            reason="auth_failure", auth_failure=True,
        )
    if "credit balance" in str(e).lower() or status == 402:
        # Billing: every call fails until someone tops up — the backend's
        # problem, not the episode's.
        return SummarizerUnavailable(
            f"model API refused for billing: {oneline(str(e), limit=300)}",
            reason="config_error",
        )
    if names & _UNAVAILABLE_ERRORS or status == 429 or (
        isinstance(status, int) and status >= 500
    ):
        return SummarizerUnavailable(
            f"model API unavailable: {oneline(str(e), limit=300)}",
            reason="api_unavailable",
        )
    if status in (400, 404, 422) or names & {"BadRequestError", "NotFoundError"}:
        if any(marker in str(e).lower() for marker in _EPISODE_REQUEST_ERRORS):
            return e  # this transcript is too long: the episode's problem
        # An unknown model id or an option the model rejects fails every call
        # the same way: a configuration error, never an episode failure.
        return SummarizerUnavailable(
            f"model API rejected the request (check PODCAST_DIGEST_MODEL): "
            f"{oneline(str(e), limit=300)}",
            reason="config_error",
        )
    return e


class ApiCompleter:
    """Call Anthropic, OpenAI, or an OpenAI-compatible server in-process."""

    def __init__(
        self,
        *,
        model: str,
        provider: Literal["auto", "anthropic", "openai"] = "auto",
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        openai_base_url: str = "",
    ) -> None:
        self.model = model
        self.provider = provider_for_model(model) if provider == "auto" else provider
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url.strip()
        self.label = f"{self.provider}:{model}"

    def config_problem(self) -> str | None:
        if self.provider == "anthropic" and not self.anthropic_api_key:
            return f"model {self.model!r} needs ANTHROPIC_API_KEY (not set)"
        if self.provider == "openai" and not (self.openai_api_key or self.openai_base_url):
            return (
                f"model {self.model!r} needs OPENAI_API_KEY (not set), or "
                "PODCAST_DIGEST_OPENAI_BASE_URL for an OpenAI-compatible server"
            )
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
        problem = self.config_problem()
        if problem:
            raise SummarizerUnavailable(problem, reason="config_error")

        def call() -> str:
            try:
                budget = MAX_OUTPUT_TOKENS[purpose]
                if self.provider == "anthropic":
                    text, truncated = self._anthropic(system, prompt, budget)
                else:
                    text, truncated = self._openai(system, prompt, budget)
            except Exception as e:  # noqa: BLE001 - reclassified below
                mapped = _classify_sdk_error(e)
                if mapped is e:
                    raise
                raise mapped from e
            if truncated:
                raise UnusableReply(
                    f"{self.label} hit the {MAX_OUTPUT_TOKENS[purpose]}-token output limit"
                )
            if not text.strip():
                raise UnusableReply(f"{self.label} returned empty content")
            return text

        # The SDKs already retry transport errors; retry here only for an
        # unusable reply (one extra attempt), never for a transport failure.
        return with_retries(
            call, validate=validate,
            is_terminal=lambda e: not isinstance(e, UnusableReply),
            max_retries=1, label=f"{purpose}:{self.label}", guid=guid,
        )

    def _anthropic(self, system: str, prompt: str, budget: int) -> tuple[str, bool]:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.anthropic_api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=budget,
            # No `thinking` option: current models reject disabling it, and the
            # default (adaptive) is right for every model. Its tokens share
            # max_tokens, which is sized for that.
            # The system prompt is byte-identical across episodes: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            # The model declined (a safety classifier, rarely triggered by a
            # podcast). Retried once; persistent refusals fail the episode.
            raise UnusableReply(f"{self.label} declined the request (refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        logger.info(
            "llm.done provider=anthropic input_tokens=%d output_tokens=%d cache_read=%d",
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )
        return text, getattr(resp, "stop_reason", None) == "max_tokens"

    def _openai(self, system: str, prompt: str, budget: int) -> tuple[str, bool]:
        from openai import OpenAI

        if self.openai_base_url:
            # OpenAI-compatible servers implement Chat Completions; many don't
            # implement the Responses API. Local servers ignore the key, but the
            # SDK requires a non-empty one.
            client = OpenAI(
                api_key=self.openai_api_key or "not-needed",
                base_url=self.openai_base_url,
            )
            chat = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=budget,
            )
            choice = chat.choices[0] if chat.choices else None
            text = (choice.message.content or "").strip() if choice else ""
            truncated = getattr(choice, "finish_reason", None) == "length"
            usage = getattr(chat, "usage", None)
            logger.info(
                "llm.done provider=openai-compatible input_tokens=%s output_tokens=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )
            return text, truncated

        client = OpenAI(api_key=self.openai_api_key)
        # Responses API: unifies reasoning + chat models, single `instructions` +
        # `input` shape.
        resp = client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=budget,
        )
        text = (resp.output_text or "").strip()
        details = getattr(resp, "incomplete_details", None)
        truncated = getattr(resp, "status", None) == "incomplete" and (
            getattr(details, "reason", None) == "max_output_tokens"
        )
        usage = getattr(resp, "usage", None)
        logger.info(
            "llm.done provider=openai input_tokens=%s output_tokens=%s",
            getattr(usage, "input_tokens", "?"),
            getattr(usage, "output_tokens", "?"),
        )
        return text, truncated


def build_completer(settings: Settings) -> Completer:
    """The configured backend. Never raises for missing credentials: those
    surface as ``config_problem()`` (checked at startup) and as
    SummarizerUnavailable on use."""
    if settings.summarizer_backend == "command":
        from .command import CommandCompleter, CommandConfig

        return CommandCompleter(CommandConfig(
            command=settings.PODCAST_DIGEST_COMMAND,
            protocol=settings.command_protocol,
            timeout_seconds=settings.PODCAST_DIGEST_COMMAND_TIMEOUT,
            tier=settings.PODCAST_DIGEST_TIER,
            model=settings.command_model,
            ssh_host=settings.PODCAST_DIGEST_SSH_HOST,
            ssh_port=settings.PODCAST_DIGEST_SSH_PORT,
            ssh_user=settings.PODCAST_DIGEST_SSH_USER,
            ssh_key=settings.PODCAST_DIGEST_SSH_KEY,
            ssh_known_hosts=settings.PODCAST_DIGEST_SSH_KNOWN_HOSTS,
        ))
    return ApiCompleter(
        model=settings.PODCAST_DIGEST_MODEL,
        provider=settings.PODCAST_DIGEST_PROVIDER,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_base_url=settings.PODCAST_DIGEST_OPENAI_BASE_URL,
    )
