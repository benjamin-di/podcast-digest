"""The two LLM tasks: the episode summary and the transcript cleanup.

Both are backend-neutral: they build prompts and validate replies, and the
configured :class:`~podcast_digest.llm.Completer` (an API or a command) runs
the model.

  - :func:`summarize` — one call per episode. Input: the transcript with
    `[hh:mm:ss]` timestamps interleaved. Output: an HTML fragment for the email.
  - :func:`clean_transcript` — the raw ASR transcript, chunked and cleaned in
    parallel into readable Markdown (speaker labels, punctuation, ad-reads
    removed), attached to the email and preferred as the summary's input.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .llm import Completer, SummarizerUnavailable
from .log import err_fields, event
from .transcribe import Segment

logger = logging.getLogger(__name__)

Style = Literal["general", "finance"]

# The summary system prompt. The `{…}` slots differ per style; everything else
# is shared. The rendered prompt is byte-identical across episodes so a provider
# can cache it.
_SYSTEM_PROMPT_TEMPLATE = """You are an editorial assistant condensing a podcast episode for {reader} who wants to decide quickly whether to listen, and to absorb the substance without doing so.

Voice: terse, specific, no fluff. Preserve concrete details that matter — {details}. Strip filler, anecdotes about the hosts' weekends, and audience-greeting boilerplate.

Output: a single HTML fragment suitable for an email body. No `<html>`, `<head>`, or `<body>` tags — just the inner content starting with `<p>` or `<h2>`. Use only these tags: `<h2>`, `<h3>`, `<p>`, `<ul>`, `<ol>`, `<li>`, `<strong>`, `<em>`, `<br>`, `<a>`. No CSS, no inline styles, no `<div>`, no `<table>`.

Structure, in this exact order:

<h2>TL;DR</h2>
<p>One sentence. The single thing this episode is about.</p>

<h2>Speakers</h2>
<ul>
  <li><strong>Host name</strong> — one-line bio (firm, role)</li>
  <li><strong>Guest name</strong> — one-line bio</li>
</ul>
(If hosts/guests aren't clearly named in the transcript, write "Hosts not identified in transcript." — never invent.)

<h2>Key takeaways</h2>
<ul>
  <li>5-8 bullets. Each bullet is one sharp claim or framework, not a topic header. Lead with the conclusion, then a clause of supporting detail.</li>
</ul>

<h2>Notable claims &amp; predictions</h2>
<ul>
  <li>Specific calls with their hedges preserved. Quote {quantities} and time horizons when given.</li>
</ul>
(Omit this section entirely if the episode is purely retrospective / explanatory.)

<h2>Timestamped highlights</h2>
<ul>
  <li><strong>[hh:mm:ss]</strong> — short label of what's discussed at that moment</li>
</ul>
5-10 entries. Use real timestamps from the transcript. Format as `[hh:mm:ss]` even if `hh` is 00.

<h2>Worth listening in full?</h2>
<p>One short paragraph: yes / no / skim, and why. Be opinionated.</p>

<h2>Jargon</h2>
<ul>
  <li><strong>TERM</strong> — one-line, plain-English definition, framed by how this episode used it</li>
</ul>
{jargon} 3-8 entries, fewer is fine. Define only terms that actually appear above, and keep the body itself terse — the glossary is the decoder, so do NOT expand acronyms inline. Omit this whole section — heading included — if nothing qualifies; never emit an empty list or a "none" placeholder.

Rules:
- Never invent facts, numbers, or attributions. If something is unclear in the transcript, omit it rather than guess.
- The transcript will contain ASR errors ({asr_examples}). Use context to repair only where you're confident; otherwise pass through.
- Do not include a closing sign-off or "let me know if..." preamble. The HTML fragment is the entire output.
- Return the HTML fragment directly as your reply text. NEVER write it to a file, use any tool, or reply with a file path or a status note ("the fragment is ready in...") — your reply text is inserted into the email body verbatim."""

_STYLE_SLOTS: dict[str, dict[str, str]] = {
    "general": {
        "reader": "a busy, curious reader",
        "details": "names, numbers, dates, sources, examples, time horizons",
        "quantities": "figures",
        "jargon": (
            "Decode only the specialized terms a well-read generalist wouldn't "
            "instantly recognize — field-specific acronyms, named programs or "
            "institutions, technical terms and metrics. Skip terms this reader "
            "already knows (GDP, CEO, AI, the EU)."
        ),
        "asr_examples": "mistranscribed names, terms and numbers",
    },
    "finance": {
        "reader": "a finance-literate reader",
        "details": "tickers, percentages, dates, named institutions, levels, time horizons",
        "quantities": "levels",
        "jargon": (
            "Decode only the specialized terms a finance-literate generalist "
            "wouldn't instantly recognize — niche acronyms, named "
            "facilities/programs, instrument types, sector-specific metrics "
            "(e.g. SOFR, BTFP, RRP, DSCR, NIM, CLO, TIC). Skip terms this reader "
            "already knows (CPI, GDP, FOMC, EPS, P/E, ETF, IPO, the Fed / ECB)."
        ),
        "asr_examples": "mistranscribed tickers, names",
    },
}


def summary_system_prompt(style: Style = "general", prompt_file: str = "") -> str:
    """The system prompt for the summary: a custom file, else the built-in
    prompt for ``style``."""
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return _SYSTEM_PROMPT_TEMPLATE.format(**_STYLE_SLOTS[style])


@dataclass
class SummarizationInput:
    """Everything the model needs about an episode."""

    podcast_name: str
    title: str
    published_at: str
    duration_seconds: int | None
    description: str | None
    transcript_text: str
    segments: list[Segment]
    # Non-English transcript language (ISO 639-1, e.g. "zh"); None = English.
    # When set, per-language guidance is appended to the *user* message so the
    # cached system prompt stays byte-identical across feeds.
    source_language: str | None = None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(
    inp: SummarizationInput,
    *,
    completer: Completer,
    style: Style = "general",
    prompt_file: str = "",
    guid: str | None = None,
) -> str:
    """Return the summary as an HTML fragment.

    Raises on any failure (backend error, invalid reply after retries) so the
    pipeline leaves the episode at its prior stage and retries on the next tick.
    """
    event(
        logger, logging.INFO, "summarize.start",
        backend=completer.label, guid=guid,
        podcast=inp.podcast_name, title=inp.title, segments=len(inp.segments),
        transcript_chars=len(inp.transcript_text),
    )
    started = time.perf_counter()
    try:
        try:
            system = summary_system_prompt(style, prompt_file)
        except OSError as e:
            # Every episode would fail the same way: a configuration error of
            # the summarizer, not an episode failure.
            raise SummarizerUnavailable(
                f"cannot read PODCAST_DIGEST_SUMMARY_PROMPT_FILE: {e}", reason="config_error",
            ) from e
        html = completer.complete(
            system=system,
            prompt=_build_user_message(inp, style=style),
            purpose="summarize",
            guid=guid,
            # A custom prompt chooses its own sections; only the built-in one
            # guarantees <h2> headings.
            validate=_validate_custom_html if prompt_file else _validate_summary_html,
        )
    except Exception as e:
        # Emit timing on failure too, so a slow-then-failing summarizer is
        # distinguishable from a fast failure (the dead-man's-switch cares).
        event(logger, logging.WARNING, "summarize.failed", guid=guid,
              elapsed_s=round(time.perf_counter() - started, 1), **err_fields(e))
        raise
    event(
        logger, logging.INFO, "summarize.done",
        guid=guid, chars=len(html), elapsed_s=round(time.perf_counter() - started, 1),
    )
    return _normalize_summary_html(html)


def _normalize_summary_html(text: str) -> str:
    """Return the usable HTML fragment from an otherwise valid reply.

    Models occasionally prepend a short acknowledgement despite the system
    prompt. Accept that harmless shape only when the required ``<h2>`` marker
    is present, and discard everything before it. File pointers and other
    prose-only replies still fail validation.
    """
    body = _strip_code_fence(strip_reasoning(text)).strip()
    if not body.startswith("<"):
        first_heading = body.find("<h2>")
        if first_heading >= 0:
            body = body[first_heading:]
    return body


def _validate_custom_html(text: str) -> str | None:
    """A reply to a custom prompt must at least be markup, not a chatty
    substitute such as a file pointer."""
    if not _normalize_summary_html(text).startswith("<"):
        return "does not start with an HTML tag"
    return None


def _validate_summary_html(text: str) -> str | None:
    """Check that a reply is the digest HTML fragment, not a chatty substitute.
    An agent CLI with tools occasionally writes the HTML to a file and answers
    with a pointer ("The HTML fragment is ready in /tmp/summary.html") — which,
    unchecked, ships verbatim as the email body. Returns a short problem
    description, or None if the text looks right."""
    problem = _validate_custom_html(text)
    if problem:
        return problem
    if "<h2>" not in _normalize_summary_html(text):
        return "missing required <h2> sections"
    return None


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------


def _language_guidance(lang: str, style: Style = "general") -> str:
    """Instructions for summarizing a non-English transcript into English.

    The base system prompt is written for English; this adds the transcript's
    language plus, for Mandarin, romanization and the 万/亿 numeric-unit rules
    (LLMs frequently fumble those, producing order-of-magnitude errors in the
    takeaways) and, for the finance style, China-specific market jargon.
    """
    if lang == "zh":
        jargon = (
            "In the Jargon section decode China-specific terms (PBOC, LPR, RRR, "
            "MLF, LGFV, A-/H-shares, SOE, Two Sessions). "
            if style == "finance"
            else "In the Jargon section decode China-specific terms. "
        )
        return (
            "The transcript is Mandarin Chinese. Write the ENTIRE summary in English. "
            "Romanize Chinese names/places in pinyin with the characters in parentheses "
            "on first mention (e.g. Pan Gongsheng (潘功胜)). "
            + jargon
            + "Convert Chinese numeric units correctly: 万 = 10^4, 亿 = 10^8."
        )
    names = "names/tickers" if style == "finance" else "names"
    return (
        f"The transcript language is {lang}. Write the ENTIRE summary in English; "
        f"keep {names} as spoken."
    )


def _build_user_message(inp: SummarizationInput, *, style: Style = "general") -> str:
    header_lines = [
        f"Podcast: {inp.podcast_name}",
        f"Episode title: {inp.title}",
    ]
    if inp.published_at:
        header_lines.append(f"Published: {inp.published_at}")
    if inp.duration_seconds:
        header_lines.append(f"Duration: {_fmt_dur(inp.duration_seconds)}")
    if inp.description:
        # Trim show notes; we just want a hint for the model, not the world.
        desc = re.sub(r"<[^>]+>", " ", inp.description)
        desc = re.sub(r"\s+", " ", desc).strip()[:1500]
        header_lines.append(f"Show notes: {desc}")
    if inp.source_language:
        header_lines.append(_language_guidance(inp.source_language, style))

    body = _format_transcript(inp.segments, inp.transcript_text)
    return (
        "\n".join(header_lines)
        + "\n\n--- TRANSCRIPT (timestamps are speech start times) ---\n\n"
        + body
    )


def _format_transcript(segments: list[Segment], fallback: str) -> str:
    """Interleave `[hh:mm:ss]` markers every ~30s of speech, then the text."""
    if not segments:
        return fallback

    out: list[str] = []
    last_stamp = -1e9
    STAMP_EVERY = 30.0
    for seg in segments:
        if seg.start - last_stamp >= STAMP_EVERY:
            out.append(f"\n[{_fmt_hms(seg.start)}]")
            last_stamp = seg.start
        out.append(seg.text.strip())
    return " ".join(out).strip()


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_dur(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


# Reasoning models served through OpenAI-compatible APIs often prefix the answer
# with their trace in <think>…</think>; it must never reach the email.
_REASONING_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(s: str) -> str:
    return _REASONING_RE.sub("", s, count=1)


_CODE_FENCE_RE = re.compile(r"^```(?:html|markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fence(s: str) -> str:
    """If the model wrapped the output in ```html ... ```, peel it off."""
    m = _CODE_FENCE_RE.match(s.strip())
    return m.group(1).strip() if m else s


# ---------------------------------------------------------------------------
# Transcript cleanup → Markdown (attached to the digest)
# ---------------------------------------------------------------------------

CLEANUP_SYSTEM_PROMPT = """You clean a raw, machine-generated (ASR) podcast transcript into readable Markdown.

You receive ONE CHUNK of a longer transcript, with `[hh:mm:ss]` timestamps. Clean only this chunk:
- Add speaker labels in bold at each turn (the person's real name if it's clear from context, else **Host:** / **Guest:** / **Speaker:**).
- Fix obvious ASR errors (misheard words, homophones, mangled names/tickers/numbers) using context — never invent content.
- Add proper punctuation, capitalization, and paragraph breaks.
- Remove filler ("um", "uh", false starts) and host ad-reads / sponsor breaks.
- Preserve ALL substantive content. Keep a `[hh:mm:ss]` timestamp at the start of each major turn or topic shift.

Output ONLY the cleaned Markdown for this chunk — no preamble, no document title, no "here is…", no code fences. Return it directly as your reply text; never write it to a file or reply with a file path or status note. It will be concatenated with the other chunks."""


def _chunk_transcript(
    segments: list[Segment], fallback_text: str, *, max_chars: int,
) -> list[str]:
    """Split the timestamped transcript into ~max_chars chunks at segment
    boundaries so each cleanup call fits a single model response."""
    if not segments:
        text = fallback_text.strip()
        if not text:
            return []
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    last_stamp = -1e9
    STAMP_EVERY = 30.0
    for seg in segments:
        piece = ""
        if seg.start - last_stamp >= STAMP_EVERY:
            piece += f"\n[{_fmt_hms(seg.start)}] "
            last_stamp = seg.start
        piece += seg.text.strip() + " "
        if cur_len + len(piece) > max_chars and cur:
            chunks.append("".join(cur).strip())
            cur, cur_len, last_stamp = [], 0, -1e9
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur).strip())
    return chunks


# Inserted in place of a chunk that fails every retry, so one stubborn chunk
# (persistent timeout / transient API error) degrades the attachment to a small
# gap instead of discarding the entire .md.
_CHUNK_FAILED_MARKER = "_[transcript cleanup unavailable for this section]_"


def clean_transcript(
    inp: SummarizationInput,
    *,
    completer: Completer,
    guid: str | None = None,
    allow_partial: bool = False,
    max_chunk_chars: int = 15_000,
    max_parallel: int = 3,
) -> str:
    """Clean the raw transcript into a readable Markdown document. Chunked so
    arbitrarily long episodes fit single model responses, and chunks are
    cleaned concurrently so wall time is ~the slowest chunk, not the sum. Each
    call already retries transient failures (see ``llm.with_retries``).

    Completeness policy via ``allow_partial``:
      - ``False`` (default): all-or-nothing. Any chunk that still fails after its
        retries aborts the whole clean (raises). The pipeline defers and retries
        the *entire* clean next tick, aiming for a complete transcript.
      - ``True``: best-effort. A chunk that fails is replaced by a short marker
        and the surviving chunks are kept; raises only if *every* chunk fails.
        Used on the final attempt so one stubborn chunk doesn't cost the whole
        attachment (matters most for long, many-chunk episodes).
    """
    chunks = _chunk_transcript(
        inp.segments, inp.transcript_text, max_chars=max_chunk_chars,
    )
    if not chunks:
        raise RuntimeError("clean_transcript: empty transcript")

    n = len(chunks)
    event(
        logger, logging.INFO, "clean.start",
        backend=completer.label, guid=guid, title=inp.title, chunks=n,
        parallel=min(max_parallel, n), allow_partial=allow_partial,
        transcript_chars=len(inp.transcript_text),
    )
    started = time.perf_counter()

    def _clean_chunk(idx_chunk: tuple[int, str]) -> str | None:
        """Clean one chunk. Returns None only when ``allow_partial`` and the
        chunk failed (caller substitutes a marker); otherwise re-raises."""
        i, chunk = idx_chunk
        header = (
            f"Podcast: {inp.podcast_name}\nEpisode: {inp.title}\n"
            f"(transcript chunk {i} of {n})\n\n--- CHUNK ---\n\n"
        )
        try:
            part = completer.complete(
                system=CLEANUP_SYSTEM_PROMPT, prompt=header + chunk,
                purpose="clean", guid=guid,
            )
            return _strip_code_fence(strip_reasoning(part)).strip()
        except Exception as e:  # noqa: BLE001
            if not allow_partial:
                raise
            event(logger, logging.WARNING, "clean.chunk_failed",
                  guid=guid, chunk=i, chunks=n, **err_fields(e))
            return None

    try:
        if n == 1:
            cleaned_parts = [_clean_chunk((1, chunks[0]))]
        else:
            with ThreadPoolExecutor(max_workers=min(max_parallel, n)) as ex:
                # ex.map preserves input order, so chunks reassemble correctly.
                cleaned_parts = list(ex.map(_clean_chunk, enumerate(chunks, start=1)))

        ok = sum(1 for p in cleaned_parts if p)
        if ok == 0:
            raise RuntimeError(
                f"clean_transcript: all {n} chunk(s) failed; nothing to attach"
            )
    except Exception as e:
        # Emit timing on failure too (mirrors summarize.failed): a slow clean
        # that ultimately fails should be visible, not just a fast one.
        event(logger, logging.WARNING, "clean.failed", guid=guid,
              elapsed_s=round(time.perf_counter() - started, 1), **err_fields(e))
        raise

    if ok < n:
        event(logger, logging.WARNING, "clean.partial",
              guid=guid, title=inp.title, ok=ok, failed=n - ok)

    body = "\n\n".join(p if p else _CHUNK_FAILED_MARKER for p in cleaned_parts)
    md = f"# {inp.title}\n\n_{inp.podcast_name}_\n\n{body}\n"
    event(logger, logging.INFO, "clean.done",
          guid=guid, chars=len(md), ok_chunks=ok, chunks=n,
          elapsed_s=round(time.perf_counter() - started, 1))
    return md
