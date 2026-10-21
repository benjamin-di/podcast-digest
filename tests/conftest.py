"""Shared pytest fixtures for the integration tests.

The integration tests exercise the full `run_tick` pipeline with everything
external mocked:

  - iTunes Search + RSS fetch + audio download → respx intercepts httpx
  - whisper.cpp subprocess → `mock_transcribe` patches the function
  - Anthropic / OpenAI summarizer → `mock_llm` patches the function
  - mail → `FakeMailer` drop-in

This catches state-machine regressions (dry-run handling, idempotency,
failure-resume) without needing live network, an LLM key, or a 1.5 GB
whisper model. Unit tests for the individual modules live in their own
files; this conftest is shared by the integration tests only.
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from podcast_digest.config import Feed, FeedsFile, Settings
from podcast_digest.db import Database
from podcast_digest.transcribe import Segment, Transcript

FIXTURES = Path(__file__).parent / "fixtures"
RSS_BYTES = (FIXTURES / "sample_feed.xml").read_bytes()
ITUNES_BODY = (FIXTURES / "itunes_search.json").read_bytes()

# URLs the sample feed and iTunes fixture reference. Kept here so the tests
# don't have to re-derive them; if you swap the fixtures, swap these too.
FEED_URL = "https://example.com/feed.xml"
EPISODE_URLS = {
    "ep-3-guid": "https://cdn.example.com/ep3.mp3",
    "ep-2-guid": "https://cdn.example.com/ep2.mp3",
    "ep-1-guid": "https://cdn.example.com/ep1.mp3",
}
FAKE_MP3 = b"FAKE-MP3-PAYLOAD-FOR-TESTS"


@pytest.fixture(autouse=True)
def isolated_settings_env(monkeypatch):
    """Tests never read the developer's configuration: no `.env` file (the
    quickstart puts one in the repository root) and no PODCAST_DIGEST_* or
    provider variables from the shell. Tests that need a setting set it."""
    import os

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith(("PODCAST_DIGEST_", "PODCAST_MCP_")) or name in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        ):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def mock_public_dns(monkeypatch):
    """Keep outbound-policy tests deterministic without touching real DNS."""
    public = ipaddress.ip_address("93.184.216.34")
    monkeypatch.setattr(
        "podcast_digest.net._resolve_host_addresses",
        lambda _host, _port: (public,),
    )


@pytest.fixture(autouse=True)
def mock_ffprobe_ok(monkeypatch):
    """Tests download FAKE_MP3 bytes, which a real ffprobe would reject — and
    download() now verifies media before publishing. Make the probe
    deterministic: decodable, and long enough (2 h) to clear every
    `min_minutes` length gate in the fixtures. Tests that steer the gate patch
    `pipeline.probe_duration` as before; test_audio.py restores the real
    probe where it exercises ffprobe itself."""
    async def fake_ffprobe(_path):
        return "ok", 7200.0, ""

    monkeypatch.setattr("podcast_digest.audio._ffprobe", fake_ffprobe)


# ---------------------------------------------------------------------------
# Mailer
# ---------------------------------------------------------------------------


class FakeMailer:
    """Drop-in for a Mailer. Captures sends; can be configured to fail.

    The pipeline calls `.send(to=, subject=, body_html=)` via `asyncio.to_thread`,
    so this needs to be sync — same signature as the real one.
    """

    def __init__(self, fail: bool = False, exc: Exception | None = None):
        self.sent: list[dict[str, Any]] = []
        self.fail = fail
        self.exc = exc or RuntimeError("synthetic mail failure")

    def send(self, *, to, subject, body_html, attachments=None, **_extra):
        if self.fail:
            raise self.exc
        self.sent.append({
            "to": to, "subject": subject, "body_html": body_html,
            "attachments": attachments or [],
        })


# ---------------------------------------------------------------------------
# Config + state
# ---------------------------------------------------------------------------


def _base_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PODCAST_DIGEST_MAIL_FROM", "bot@example.com")
    monkeypatch.setenv("PODCAST_DIGEST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """A Settings instance (API summarizer) pointed at an isolated tmp dir.

    Everything sensitive uses placeholder values: the tests mock the SDKs and
    the mailer that would consume them.
    """
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("PODCAST_DIGEST_MODEL", "claude-sonnet-5")
    s = Settings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s


@pytest.fixture
def command_settings(monkeypatch, tmp_path):
    """Settings in `command` summarizer mode (JSON protocol over SSH), with no
    LLM key set: summarization is delegated to the command (stubbed by
    `mock_llm` / the tests' own patches)."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("PODCAST_DIGEST_SUMMARIZER", "command")
    monkeypatch.setenv("PODCAST_DIGEST_COMMAND", "llm-runner")
    monkeypatch.setenv("PODCAST_DIGEST_COMMAND_PROTOCOL", "json")
    monkeypatch.setenv("PODCAST_DIGEST_SSH_HOST", "runner.example")
    s = Settings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s


@pytest.fixture
def db(settings):
    d = Database(settings.db_path)
    yield d
    d.close()


@pytest.fixture
def feeds_file():
    """Single-feed config matching our iTunes fixture (collectionName).

    `min_minutes=60` is chosen so only the newest episode in the sample RSS
    (90 min) qualifies — the 45-min ep-2 and 3-min teaser ep-1 are filtered
    out. That keeps the state-machine tests (idempotency, dry-run, resume,
    MAX_ATTEMPTS) focused on a single episode without the second-tick
    catchup behavior muddying expectations.

    Tests that exercise the catchup / min-minutes filter pass their own
    FeedsFile in (see test_min_minutes_filters_short_episodes_at_parse_time).
    """
    return FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="The Test Podcast", search="The Test Podcast", min_minutes=60)],
    )


# ---------------------------------------------------------------------------
# Module-level mocks (LLM, transcription)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the summarizer with a deterministic HTML producer.

    The fake echoes the episode title so tests can assert the right episode
    flowed through, and records every call (incl. the configured backend) so
    tests can assert call counts and routing.
    """
    calls: list[dict[str, Any]] = []

    def fake_summarize(inp, *, completer, **_kw):
        calls.append({
            "title": inp.title, "backend": completer.label,
            "source_language": inp.source_language,
            "transcript_text": inp.transcript_text,
        })
        return (
            f"<h2>TL;DR</h2><p>Test summary of {inp.title!r} on {inp.podcast_name!r}.</p>"
            f"<h2>Key takeaways</h2><ul><li>Hello.</li></ul>"
        )

    monkeypatch.setattr("podcast_digest.pipeline.summarize", fake_summarize)
    fake_summarize.calls = calls  # type: ignore[attr-defined]
    return fake_summarize


@pytest.fixture
def mock_transcribe(monkeypatch):
    """Replace `transcribe()` with a stub that fakes both whisper outputs.

    Crucially this writes both the `.txt` and `.json` sidecars to the
    persistent transcripts dir so the resume code path (load_whisper_json
    against an on-disk file) works on re-entry, exactly as it does in prod.
    """
    transcript = Transcript(
        text="hello world this is a test transcript",
        segments=[
            Segment(start=0.0, end=2.0, text="hello world"),
            Segment(start=2.0, end=4.0, text="this is a test transcript"),
        ],
    )
    calls: list[dict[str, Any]] = []

    async def fake_transcribe(
        audio_path, *, out_dir, basename, model_path=None, language=None, **_kw,
    ):
        calls.append({
            "audio_path": str(audio_path), "basename": basename,
            "model_path": str(model_path), "language": language,
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{basename}.txt"
        json_path = out_dir / f"{basename}.json"
        txt_path.write_text(transcript.text, encoding="utf-8")
        json_path.write_text(json.dumps({
            "transcription": [
                {"offsets": {"from": int(s.start * 1000), "to": int(s.end * 1000)},
                 "text": s.text}
                for s in transcript.segments
            ],
        }), encoding="utf-8")
        return transcript, txt_path, json_path

    monkeypatch.setattr("podcast_digest.pipeline.transcribe", fake_transcribe)
    fake_transcribe.calls = calls  # type: ignore[attr-defined]
    return fake_transcribe


# ---------------------------------------------------------------------------
# HTTP (respx)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http():
    """Mock iTunes Search + RSS + every episode-audio URL.

    Yields the respx router so individual tests can override routes
    (e.g. simulate a 500 from a CDN).
    """
    with respx.mock(assert_all_called=False) as router:
        router.get("https://itunes.apple.com/search").mock(
            return_value=httpx.Response(200, content=ITUNES_BODY,
                                        headers={"content-type": "application/json"})
        )
        router.get(FEED_URL).mock(
            return_value=httpx.Response(200, content=RSS_BYTES,
                                        headers={"content-type": "application/xml"})
        )
        for url in EPISODE_URLS.values():
            router.get(url).mock(
                return_value=httpx.Response(
                    200,
                    content=FAKE_MP3,
                    headers={
                        "content-type": "audio/mpeg",
                        "content-length": str(len(FAKE_MP3)),
                    },
                )
            )
        yield router
