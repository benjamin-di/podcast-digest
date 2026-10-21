"""End-to-end pipeline tests.

Each test wires `run_tick` against mocked HTTP, transcribe, summarize, and
mailer (see conftest.py) and asserts both the user-visible effect (mail
sent, file on disk) and the persisted DB state.

What's covered:
  - Happy path: discover -> download -> transcribe -> summarize -> email
  - First-run-only-newest guard (no back-catalog flood)
  - Idempotency: re-running the tick after success is a no-op
  - --dry-run: no email, no status advance, audio preserved for next real run
  - Resume after a failed mail send: status preserved at 'summarized',
    next tick completes without re-running transcribe/summarize
  - MAX_ATTEMPTS cap: a persistently failing episode stops being retried
  - min_minutes filter: short episodes are skipped at parse time
"""
from __future__ import annotations

from pathlib import Path

import pytest

from podcast_digest.db import MAX_ATTEMPTS
from podcast_digest.emitted import EmittedLog
from podcast_digest.llm import SummarizerUnavailable
from podcast_digest.pipeline import run_tick

from .conftest import FEED_URL, RSS_BYTES, FakeMailer

NEWEST_GUID = "ep-3-guid"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_sends_one_email_and_cleans_audio(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_file,
        db=db, mailer=mailer, dry_run=False,
    )

    # First-run on a brand-new feed only takes the latest episode, so even
    # though the sample RSS has 3 items, we email exactly one.
    assert stats == {"feeds": 1, "new_episodes": 1, "emailed": 1, "failed": 0,
                     "summarize_failed": 0, "summarize_ok": 1}

    # Mailer received exactly one structured send.
    assert len(mailer.sent) == 1
    sent = mailer.sent[0]
    assert sent["to"] == "reader@example.com"
    assert sent["subject"].startswith("[Podcast] The Test Podcast — Episode 3:")
    assert "Test summary of 'Episode 3: Long episode'" in sent["body_html"]

    # Stubbed LLM was called once, through the configured API backend.
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["backend"] == "anthropic:claude-sonnet-5"

    # DB: episode is at 'emailed' and therefore drops out of the pending queue.
    assert db.episodes_in_progress(FEED_URL) == []
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "emailed"
    assert ep.emailed_at is not None
    assert ep.terminal_reason == "sent"
    assert ep.summary_html is not None

    # Audio was cleaned up; transcript was kept.
    assert ep.audio_path is None
    assert ep.transcript_path is not None
    assert Path(ep.transcript_path).exists()


# ---------------------------------------------------------------------------
# Command summarizer backend (transcribe here, summarize via a command over SSH)
# ---------------------------------------------------------------------------


async def test_command_backend_summarizes_and_emails(
    command_settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """In `command` mode (no LLM key) the pipeline routes summarization to the
    command backend, then emails locally and advances to `emailed` — same end
    state as the API backend."""
    mailer = FakeMailer()

    stats = await run_tick(
        settings=command_settings, feeds_file=feeds_file,
        db=db, mailer=mailer, dry_run=False,
    )

    assert stats == {"feeds": 1, "new_episodes": 1, "emailed": 1, "failed": 0,
                     "summarize_failed": 0, "summarize_ok": 1}

    # The command backend over SSH was used (not an API SDK).
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["backend"] == "command@runner.example:json"

    # Mail still sent from this host; body is the produced summary.
    assert len(mailer.sent) == 1
    assert "Test summary of 'Episode 3: Long episode'" in mailer.sent[0]["body_html"]

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "emailed"
    assert ep.audio_path is None  # cleaned after send


async def test_cleaned_transcript_attached_and_idempotent(
    command_settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
    monkeypatch,
):
    """ATTACH_TRANSCRIPT=cleaned attaches an agent-cleaned .md, and a send
    failure does NOT re-run the (expensive) cleanup — the .clean.md persists."""
    command_settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT = "cleaned"
    clean_calls: list[str] = []

    def fake_clean(inp, *, completer, **_kw):
        clean_calls.append(inp.title)
        return f"# {inp.title}\n\ncleaned body"

    monkeypatch.setattr(
        "podcast_digest.pipeline.clean_transcript", fake_clean
    )

    # Run 1: mail fails — cleanup ran once, .clean.md persisted, nothing sent.
    failing = FakeMailer(fail=True)
    await run_tick(settings=command_settings, feeds_file=feeds_file,
                   db=db, mailer=failing, dry_run=False)
    assert len(clean_calls) == 1
    assert failing.sent == []

    # Run 2: good mailer — NOT re-cleaned (file reused), attachment present.
    good = FakeMailer()
    await run_tick(settings=command_settings, feeds_file=feeds_file,
                   db=db, mailer=good, dry_run=False)
    assert len(clean_calls) == 1            # idempotent: not re-cleaned
    assert len(good.sent) == 1
    atts = good.sent[0]["attachments"]
    assert len(atts) == 1
    assert atts[0].name.endswith(".md")
    assert b"cleaned body" in atts[0].content


async def test_cleaned_clean_failure_defers_email_until_clean_succeeds(
    command_settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
    monkeypatch,
):
    """A transient cleanup failure must NOT ship a degraded digest: the episode
    parks at `transcribed` (no summary, no email) and the whole clean is retried
    next tick, so the summary is always built from the cleaned transcript."""
    command_settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT = "cleaned"
    calls = {"n": 0}

    def fake_clean(inp, *, completer, allow_partial=False, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("clean timed out after 360s")
        return f"# {inp.title}\n\ncleaned body"

    monkeypatch.setattr(
        "podcast_digest.pipeline.clean_transcript", fake_clean
    )

    # Tick 1: clean fails → nothing sent, episode deferred, no summary built.
    m1 = FakeMailer()
    s1 = await run_tick(settings=command_settings, feeds_file=feeds_file,
                        db=db, mailer=m1, dry_run=False)
    assert m1.sent == []
    assert s1["emailed"] == 0
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "transcribed"   # parked, not advanced to summarized
    assert ep.summary_html is None      # summary NOT generated from raw ASR
    assert ep.clean_attempts == 1
    assert ep.attempts == 0             # defer doesn't burn the main retry budget

    # Tick 2: clean succeeds → single digest, summary from cleaned text, .md attached.
    m2 = FakeMailer()
    await run_tick(settings=command_settings, feeds_file=feeds_file,
                   db=db, mailer=m2, dry_run=False)
    assert len(m2.sent) == 1
    atts = m2.sent[0]["attachments"]
    assert len(atts) == 1 and atts[0].name.endswith(".md")
    assert b"cleaned body" in atts[0].content
    ep2 = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep2 is not None and ep2.status == "emailed"


async def test_clean_transport_outage_suppresses_healthy_heartbeat(
    command_settings, db, feeds_file, mock_transcribe, mock_http, monkeypatch,
):
    """A swallowed cleanup transport failure still reaches tick telemetry."""
    command_settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT = "cleaned"

    def transport_down(inp, *, completer, **_kw):
        raise SummarizerUnavailable("cleanup failed: connection refused")

    monkeypatch.setattr(
        "podcast_digest.pipeline.clean_transcript", transport_down
    )
    stats = await run_tick(
        settings=command_settings,
        feeds_file=feeds_file,
        db=db,
        mailer=FakeMailer(),
        dry_run=False,
    )

    assert stats["summarize_failed"] == 1
    assert stats["summarize_ok"] == 0
    assert stats["failed"] == 0  # deferred cleanup does not burn episode budget
    episode = db.get_episode(FEED_URL, NEWEST_GUID)
    assert episode is not None and episode.status == "transcribed"
    assert episode.clean_attempts == 1


async def test_cleaned_gives_up_after_budget_and_ships_raw(
    command_settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
    monkeypatch,
):
    """If cleanup never recovers, the digest must still ship after CLEAN_MAX_TICKS
    — degraded to a raw-transcript summary with no attachment (delivery is never
    blocked forever)."""
    from podcast_digest.pipeline import CLEAN_MAX_TICKS

    command_settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT = "cleaned"

    def always_fail(inp, *, completer, allow_partial=False, **_kw):
        raise RuntimeError("clean timed out after 360s")

    monkeypatch.setattr(
        "podcast_digest.pipeline.clean_transcript", always_fail
    )

    # The first CLEAN_MAX_TICKS-1 ticks defer (no email).
    for i in range(CLEAN_MAX_TICKS - 1):
        m = FakeMailer()
        await run_tick(settings=command_settings, feeds_file=feeds_file,
                       db=db, mailer=m, dry_run=False)
        assert m.sent == []
        ep = db.get_episode(FEED_URL, NEWEST_GUID)
        assert ep is not None and ep.status == "transcribed"
        assert ep.clean_attempts == i + 1

    # Final tick: give up → ship the digest from raw, no attachment.
    m_final = FakeMailer()
    await run_tick(settings=command_settings, feeds_file=feeds_file,
                   db=db, mailer=m_final, dry_run=False)
    assert len(m_final.sent) == 1
    assert m_final.sent[0]["attachments"] == []
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.status == "emailed"
    assert ep.clean_attempts == CLEAN_MAX_TICKS


async def test_summarizer_failure_stays_at_transcribed_and_retries(
    command_settings, db, feeds_file, mock_transcribe, mock_http, monkeypatch,
):
    """If the summarizer fails, the episode must stay at `transcribed`
    (not drift), so the next tick retries summarize+email without
    re-downloading or re-transcribing."""
    def boom(inp, *, completer, **_kw):
        raise SummarizerUnavailable("summarize failed rc=255: connection refused")

    monkeypatch.setattr("podcast_digest.pipeline.summarize", boom)
    mailer = FakeMailer()

    stats = await run_tick(
        settings=command_settings, feeds_file=feeds_file,
        db=db, mailer=mailer, dry_run=False,
    )
    assert stats["failed"] == 1
    # A SummarizerUnavailable failure is counted as a summarizer-transport failure, which
    # is what the dead-man's-switch keys on (suppresses the heartbeat).
    assert stats["summarize_failed"] == 1
    assert stats["emailed"] == 0
    assert mailer.sent == []

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "transcribed"   # high-water mark held; no drift
    assert ep.summary_html is None
    transcribe_calls = len(mock_transcribe.calls)

    # Next tick with a working summarizer: completes without re-transcribing.
    def ok(inp, *, completer, **_kw):
        return "<p>recovered</p>"

    monkeypatch.setattr("podcast_digest.pipeline.summarize", ok)
    stats2 = await run_tick(
        settings=command_settings, feeds_file=feeds_file,
        db=db, mailer=mailer, dry_run=False,
    )
    assert stats2["emailed"] == 1
    assert len(mailer.sent) == 1
    assert len(mock_transcribe.calls) == transcribe_calls  # no re-transcribe
    ep2 = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep2 is not None and ep2.status == "emailed"


async def test_multiday_summarizer_outage_does_not_drop_episode_then_recovers(
    command_settings, db, feeds_file, mock_transcribe, mock_http, monkeypatch,
):
    """F2/R1: a summarizer outage spanning MORE ticks than MAX_ATTEMPTS must NOT
    silently drop the episode (the original audit bug). The transport failures go
    on the separate generous budget; the episode stays pending across the whole
    outage and completes the moment the summarizer returns."""
    from podcast_digest.db import MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_MAX_ATTEMPTS

    def boom(inp, *, completer, **_kw):
        raise SummarizerUnavailable("summarize failed rc=255: connection refused")

    monkeypatch.setattr("podcast_digest.pipeline.summarize", boom)
    mailer = FakeMailer()

    # Outage for well beyond MAX_ATTEMPTS ticks (but within the transport budget).
    outage_ticks = MAX_ATTEMPTS + 3
    assert outage_ticks < SUMMARIZE_TRANSPORT_MAX_ATTEMPTS
    for _ in range(outage_ticks):
        await run_tick(settings=command_settings, feeds_file=feeds_file,
                       db=db, mailer=mailer, dry_run=False)

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "transcribed"          # still pending, NOT dropped
    assert ep.attempts == 0                     # general budget untouched
    assert ep.summarize_attempts == outage_ticks
    assert mailer.sent == []

    # Summarizer returns → the episode summarizes + emails, and its counters reset.
    monkeypatch.setattr("podcast_digest.pipeline.summarize",
                        lambda inp, *, completer, **_kw: "<p>recovered</p>")
    stats = await run_tick(settings=command_settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1
    assert len(mailer.sent) == 1
    ep2 = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep2.status == "emailed" and ep2.summarize_attempts == 0


# ---------------------------------------------------------------------------
# Concurrency guard: a second tick skips while one holds the lock
# ---------------------------------------------------------------------------


async def test_concurrent_tick_is_skipped_no_duplicate_send(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """If another process (serve loop or manual run) already holds the tick
    lock, run_tick must skip entirely — no feeds processed, no emails. This is
    the guard against the duplicate-email race."""
    from podcast_digest.pipeline import _acquire_tick_lock

    held = _acquire_tick_lock(settings)
    assert held is not None  # we now hold the lock, simulating a running tick
    try:
        mailer = FakeMailer()
        stats = await run_tick(
            settings=settings, feeds_file=feeds_file,
            db=db, mailer=mailer, dry_run=False,
        )
        assert stats.get("skipped") is True
        assert stats["emailed"] == 0
        assert mailer.sent == []
        assert len(mock_llm.calls) == 0
    finally:
        held.close()

    # Lock released → a subsequent tick runs normally and emails.
    mailer2 = FakeMailer()
    stats2 = await run_tick(
        settings=settings, feeds_file=feeds_file,
        db=db, mailer=mailer2, dry_run=False,
    )
    assert stats2.get("skipped") is None
    assert stats2["emailed"] == 1
    assert len(mailer2.sent) == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_second_tick_is_a_noop(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    mailer = FakeMailer()

    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=False)
    # Second tick — nothing new on the (mocked) feed
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["new_episodes"] == 0
    assert stats["emailed"] == 0
    assert len(mailer.sent) == 1
    # And the LLM was not called a second time.
    assert len(mock_llm.calls) == 1


# ---------------------------------------------------------------------------
# Dry-run does not poison the DB
# ---------------------------------------------------------------------------


async def test_dry_run_does_not_advance_status_or_send(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=True)

    # Nothing emailed; `emailed` counter only increments on a real send.
    assert stats["emailed"] == 0
    assert mailer.sent == []

    # Status stops at 'summarized'. Critically, the next real run MUST be
    # able to pick up and finish — verifying that here:
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "summarized"
    assert ep.summary_html is not None  # work wasn't wasted
    # Audio kept on disk so the next real run doesn't re-download.
    assert ep.audio_path is not None
    assert Path(ep.audio_path).exists()

    # Real run now: should email, no re-transcribe, no re-summarize.
    transcribe_calls_before = len(mock_transcribe.calls)
    llm_calls_before = len(mock_llm.calls)
    stats2 = await run_tick(settings=settings, feeds_file=feeds_file,
                            db=db, mailer=mailer, dry_run=False)
    assert stats2["emailed"] == 1
    assert len(mailer.sent) == 1
    # No new transcribe / summarize calls — picked up exactly at the email stage.
    assert len(mock_transcribe.calls) == transcribe_calls_before
    assert len(mock_llm.calls) == llm_calls_before


# ---------------------------------------------------------------------------
# Resume after a failed send (regression test for the H3 bug we fixed)
# ---------------------------------------------------------------------------


async def test_failed_send_preserves_high_water_mark_and_resumes(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    # Run #1: mailer raises. Pipeline should leave status at 'summarized',
    # bump attempts, and NOT re-do transcribe/summarize on the next tick.
    failing_mailer = FakeMailer(fail=True)

    stats1 = await run_tick(settings=settings, feeds_file=feeds_file,
                            db=db, mailer=failing_mailer, dry_run=False)
    assert stats1["failed"] == 1
    assert stats1["emailed"] == 0

    ep_after_fail = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep_after_fail is not None
    # H3 regression check: status stays at the last *successful* stage.
    assert ep_after_fail.status == "summarized"
    assert ep_after_fail.attempts == 1
    assert ep_after_fail.last_error is not None
    assert "synthetic mail failure" in ep_after_fail.last_error
    assert ep_after_fail.failed_at is not None

    transcribe_calls = len(mock_transcribe.calls)
    llm_calls = len(mock_llm.calls)

    # Run #2: working mailer. Email should send, no extra transcribe/summarize.
    good_mailer = FakeMailer()
    stats2 = await run_tick(settings=settings, feeds_file=feeds_file,
                            db=db, mailer=good_mailer, dry_run=False)
    assert stats2["emailed"] == 1
    assert len(good_mailer.sent) == 1
    assert len(mock_transcribe.calls) == transcribe_calls
    assert len(mock_llm.calls) == llm_calls

    # last_error cleared on the successful stage transition.
    ep_after_success = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep_after_success is not None
    assert ep_after_success.status == "emailed"
    assert ep_after_success.last_error is None


# ---------------------------------------------------------------------------
# MAX_ATTEMPTS cap
# ---------------------------------------------------------------------------


async def test_max_attempts_eventually_drops_episode_from_pending(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    failing_mailer = FakeMailer(fail=True)

    # Burn through the retry budget.
    for _ in range(MAX_ATTEMPTS):
        await run_tick(settings=settings, feeds_file=feeds_file,
                       db=db, mailer=failing_mailer, dry_run=False)

    # The row is still there (with the error preserved) but no longer pending —
    # so it stops eating bandwidth + LLM tokens every tick.
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.attempts == MAX_ATTEMPTS
    assert ep.status == "summarized"
    assert db.episodes_in_progress(FEED_URL) == []

    # The final drop GC'd every download artifact for the episode (final audio
    # AND any resume fragments) — dropped episodes must not hold disk forever.
    assert not settings.audio_dir.exists() or not list(settings.audio_dir.rglob("*.*"))
    assert db.get_episode(FEED_URL, NEWEST_GUID).audio_path is None

    # One more tick — even with a working mailer, the capped episode is skipped.
    good_mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=good_mailer, dry_run=False)
    assert stats["emailed"] == 0
    assert good_mailer.sent == []


# ---------------------------------------------------------------------------
# min_minutes filter
# ---------------------------------------------------------------------------


async def test_min_minutes_filters_short_episodes_at_parse_time(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """Verifies short episodes are filtered out AND that first sight of a feed
    suppresses the back-catalog (no flood) — only the latest is emailed.

    sample_feed.xml: ep-3 (90 min), ep-2 (45 min), ep-1 (3 min teaser).
    With min_minutes=15, ep-1 is filtered. First tick emails ONLY ep-3 (newest)
    and records ep-2 as already-seen (status 'emailed', never sent). A later
    tick must NOT back-fill ep-2.
    """
    from podcast_digest.config import Feed, FeedsFile

    feeds_short_filter = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="The Test Podcast", search="The Test Podcast", min_minutes=15)],
    )
    mailer = FakeMailer()

    s1 = await run_tick(settings=settings, feeds_file=feeds_short_filter,
                        db=db, mailer=mailer, dry_run=False)
    assert s1["new_episodes"] == 1  # only the latest is processed
    assert s1["emailed"] == 1       # the 90-min one (ep-3)

    s2 = await run_tick(settings=settings, feeds_file=feeds_short_filter,
                        db=db, mailer=mailer, dry_run=False)
    assert s2["new_episodes"] == 0  # archive suppressed on first run; no back-fill
    assert s2["emailed"] == 0
    assert len(mailer.sent) == 1    # ONLY ep-3 ever emailed

    # ep-1 (3 min) filtered out entirely (never inserted).
    assert db.get_episode(FEED_URL, "ep-1-guid") is None
    # ep-2 recorded as already-seen (suppressed), but never actually emailed.
    ep2 = db.get_episode(FEED_URL, "ep-2-guid")
    assert ep2 is not None and ep2.status == "emailed"
    assert ep2.terminal_reason == "backlog_suppressed"
    assert ep2.emailed_at is None
    assert all(s["subject"].find("Episode 2") == -1 for s in mailer.sent)
    # ep-3 (newest) actually emailed.
    assert db.get_episode(FEED_URL, NEWEST_GUID).status == "emailed"


async def test_ffprobe_backstop_skips_short_real_duration(
    settings, db, mock_llm, mock_transcribe, mock_http, monkeypatch,
):
    """A clip whose *declared* duration clears the floor but whose *real* audio
    is below it must be skipped before transcription.

    ep-3 declares 90 min (passes the parse-time filter), but we make ffprobe
    report 40 s. The backstop must terminal-skip it: no transcribe, no email,
    status 'emailed' (seen, never sent), audio cleaned up.
    """
    from podcast_digest.config import Feed, FeedsFile

    async def fake_probe(_path):
        return 40.0  # 40 seconds — a conference soundbite

    monkeypatch.setattr("podcast_digest.pipeline.probe_duration", fake_probe)

    feeds = FeedsFile(
        recipient="reader@example.com",
        feeds=[Feed(name="The Test Podcast", rss_url=FEED_URL, min_minutes=15)],
    )
    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["emailed"] == 0
    assert stats["failed"] == 0
    assert len(mailer.sent) == 0
    # Transcription was never spent on the clip.
    assert len(mock_transcribe.calls) == 0
    # ep-3 is terminal (deduped, never retried) but was never sent.
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.status == "emailed"
    assert ep.terminal_reason == "below_min_duration"
    assert ep.emailed_at is None
    assert ep.audio_path is None  # disposable teaser cleaned up


async def test_global_min_minutes_default_applies_to_unset_feed(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """A feed with no `min_minutes` inherits the file-level floor at parse time.

    With the file default at 15 and the feed unset, the 3-min teaser (ep-1) is
    filtered out exactly as if the feed had set min_minutes=15 itself.
    """
    from podcast_digest.config import Feed, FeedsFile

    feeds = FeedsFile(
        recipient="reader@example.com",
        min_minutes=15,
        feeds=[Feed(name="The Test Podcast", rss_url=FEED_URL)],  # min_minutes unset
    )
    mailer = FakeMailer()
    await run_tick(settings=settings, feeds_file=feeds,
                   db=db, mailer=mailer, dry_run=False)

    # ep-1 (3 min) filtered at parse time → never inserted.
    assert db.get_episode(FEED_URL, "ep-1-guid") is None


# ---------------------------------------------------------------------------
# CDN 500 → episode failed, status not advanced past 'discovered'
# ---------------------------------------------------------------------------


async def test_audio_cdn_failure_marks_episode_failed_without_status_drift(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """When the CDN returns 500, the episode is marked failed at the download
    stage — status stays at the insert-time default ('discovered') so the
    next tick retries the download (not a non-existent transcribe stage)."""
    import httpx as _httpx
    mock_http.get("https://cdn.example.com/ep3.mp3").mock(
        return_value=_httpx.Response(500, content=b"oh no"),
    )

    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["failed"] == 1
    assert stats["emailed"] == 0

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "discovered"
    assert ep.attempts == 1
    assert ep.audio_path is None
    # transcribe was never called.
    assert len(mock_transcribe.calls) == 0


# ---------------------------------------------------------------------------
# Pinned RSS URL skips iTunes lookup
# ---------------------------------------------------------------------------


async def test_pinned_rss_url_bypasses_itunes_search(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """If `rss_url` is set on the feed, we never hit iTunes Search."""
    from podcast_digest.config import Feed, FeedsFile

    feeds_pinned = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="The Test Podcast", rss_url=FEED_URL, min_minutes=15)],
    )
    mailer = FakeMailer()

    await run_tick(settings=settings, feeds_file=feeds_pinned,
                   db=db, mailer=mailer, dry_run=False)

    # iTunes route was registered by the conftest fixture but should never
    # have been hit. respx tracks per-route call counts on the router.
    itunes_calls = sum(
        route.call_count for route in mock_http.routes
        if "itunes.apple.com" in str(route.pattern)
    )
    assert itunes_calls == 0
    assert len(mailer.sent) == 1


async def test_search_feed_resolves_itunes_once_then_uses_cache(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """F5: a `search` feed must hit iTunes only on the FIRST tick; thereafter it
    reuses the cached identity URL. Re-resolving every tick lets a feedUrl change
    re-key the feed under a new PK (first_run re-fires → re-email; in-progress
    episodes orphaned)."""
    from podcast_digest.config import Feed, FeedsFile

    feeds = FeedsFile(
        recipient="reader@example.com",
        feeds=[Feed(name="The Test Podcast", search="The Test Podcast", min_minutes=15)],
    )
    mailer = FakeMailer()

    def itunes_calls():
        return sum(
            route.call_count for route in mock_http.routes
            if "itunes.apple.com" in str(route.pattern)
        )

    await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False)
    assert itunes_calls() == 1  # resolved on first sight

    await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False)
    await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False)
    assert itunes_calls() == 1  # cached identity URL reused, no re-resolution
    assert len(mailer.sent) == 1  # and no spurious re-email


async def test_first_run_insert_and_suppress_are_atomic(
    settings, db, mock_llm, mock_transcribe, mock_http, monkeypatch,
):
    """F4: if the process dies between inserting the back-catalog and suppressing
    it, the archive must NOT persist at 'discovered' (which the uncapped pending
    loop would later download+email). The insert+suppress runs in one tx, so a
    crash mid-way rolls back BOTH — nothing is inserted."""
    from podcast_digest.config import Feed, FeedsFile

    feeds = FeedsFile(
        recipient="reader@example.com",
        # min_minutes=0 so all three sample episodes are unseen (2 suppressed).
        feeds=[Feed(name="The Test Podcast", rss_url=FEED_URL, min_minutes=0)],
    )

    # Make set_status blow up on the FIRST suppress write, simulating a crash
    # between the insert loop and the suppress completing.
    real_set_status = db.set_status
    boom = {"armed": True}

    def exploding_set_status(*a, **k):
        if boom["armed"] and k.get("status") == "emailed":
            boom["armed"] = False
            raise RuntimeError("simulated crash mid-suppress")
        return real_set_status(*a, **k)

    monkeypatch.setattr(db, "set_status", exploding_set_status)

    mailer = FakeMailer()
    # The feed-level handler swallows the exception (feed.failed); the tick returns.
    await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False)

    # The transaction rolled back: NONE of the episodes were persisted, so the
    # next tick re-derives them under the cap (no orphaned 'discovered' archive).
    for guid in ("ep-1-guid", "ep-2-guid", NEWEST_GUID):
        assert db.get_episode(FEED_URL, guid) is None

    # Recovery: a clean tick now processes only the newest (cap), suppresses rest.
    monkeypatch.setattr(db, "set_status", real_set_status)
    await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False)
    assert len(mailer.sent) == 1  # exactly one email — no back-catalog flood
    assert db.get_episode(FEED_URL, NEWEST_GUID).status == "emailed"


async def test_duplicate_guid_in_one_poll_is_processed_once(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """A repeated GUID cannot suppress the selected newest copy before send."""
    import httpx

    from podcast_digest.config import Feed, FeedsFile

    duplicate_feed = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>The Test Podcast</title>
      <item><title>Newest canonical copy</title><guid>ep-3-guid</guid>
        <pubDate>Wed, 22 Jul 2026 10:00:00 GMT</pubDate>
        <enclosure url="https://cdn.example.com/ep3.mp3"
                   type="audio/mpeg" length="1"/></item>
      <item><title>Older duplicate copy</title><guid>ep-3-guid</guid>
        <pubDate>Tue, 21 Jul 2026 10:00:00 GMT</pubDate>
        <enclosure url="https://cdn.example.com/ep3.mp3"
                   type="audio/mpeg" length="1"/></item>
    </channel></rss>"""
    mock_http.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=duplicate_feed)
    )
    feeds = FeedsFile(
        recipient="reader@example.com",
        feeds=[Feed(name="The Test Podcast", rss_url=FEED_URL)],
    )
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["new_episodes"] == 1
    assert stats["emailed"] == 1
    assert len(mailer.sent) == 1
    assert "Newest canonical copy" in mailer.sent[0]["subject"]
    episode = db.get_episode(FEED_URL, NEWEST_GUID)
    assert episode is not None and episode.status == "emailed"


# ---------------------------------------------------------------------------
# YouTube single-video source
# ---------------------------------------------------------------------------


async def test_youtube_video_full_pipeline_and_idempotent(
    settings, db, mock_llm, mock_transcribe, monkeypatch,
):
    """A `youtube_video` feed: yt-dlp metadata + audio are stubbed; the episode
    flows through transcribe → summarize → email exactly once, and a second tick
    is a no-op that does NOT re-hit yt-dlp metadata (short-circuited once seen)."""
    from podcast_digest import youtube
    from podcast_digest.config import Feed, FeedsFile
    from podcast_digest.feeds import EpisodeMeta

    watch = "https://www.youtube.com/watch?v=FIRESIDE123"
    identity = "https://www.youtube.com/watch?v=FIRESIDE123"

    meta_calls: list[str] = []
    dl_calls: list[str] = []

    async def fake_meta(url, *, cookies=None):
        meta_calls.append(url)
        return EpisodeMeta(
            guid="yt:video:FIRESIDE123",
            title="A fireside chat on market structure",
            published_at="2026-06-10T00:00:00+00:00",
            duration_seconds=3600,
            audio_url=identity,
            audio_type="video/youtube",
            description="Conference fireside",
        )

    async def fake_dl(url, *, dest_dir, basename, fmt="bestaudio/best", cookies=None):
        dl_calls.append(url)
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / f"{basename}.mp3"
        p.write_bytes(b"FAKE-YT-AUDIO")
        return p

    monkeypatch.setattr(youtube, "single_video_episode", fake_meta)
    monkeypatch.setattr(youtube, "download_audio", fake_dl)

    feeds = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="Conference Talks", youtube_video=watch)],
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)
    assert stats == {"feeds": 1, "new_episodes": 1, "emailed": 1, "failed": 0,
                     "summarize_failed": 0, "summarize_ok": 1}
    assert len(mailer.sent) == 1
    assert mailer.sent[0]["subject"].startswith(
        "[Podcast] Conference Talks — A fireside chat on market structure"
    )
    # The watch URL is the "listen" link in the email body.
    assert watch in mailer.sent[0]["body_html"]
    assert meta_calls == [watch]
    assert dl_calls == [watch]

    ep = db.get_episode(identity, "yt:video:FIRESIDE123")
    assert ep is not None and ep.status == "emailed"
    assert ep.audio_path is None  # cleaned after send

    # Second tick: already seen → no metadata fetch, no new email.
    stats2 = await run_tick(settings=settings, feeds_file=feeds,
                            db=db, mailer=mailer, dry_run=False)
    assert stats2["emailed"] == 0
    assert len(mailer.sent) == 1
    assert meta_calls == [watch]  # NOT called again


# ---------------------------------------------------------------------------
# Direct-audio single-episode source (one-shot)
# ---------------------------------------------------------------------------


async def test_audio_url_full_pipeline_and_idempotent(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """An `audio_url` feed: a direct .mp3 flows through download → transcribe →
    summarize → email exactly once; a second tick is deduped (no new email)."""
    import hashlib

    from podcast_digest.config import Feed, FeedsFile

    audio = "https://cdn.example.com/clip.mp3"
    mock_http.get(audio).respond(
        200,
        headers={"content-type": "audio/mpeg", "content-length": "9"},
        content=b"FAKEAUDIO",
    )

    feeds = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="Random Clip", audio_url=audio)],
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)
    assert stats == {"feeds": 1, "new_episodes": 1, "emailed": 1, "failed": 0,
                     "summarize_failed": 0, "summarize_ok": 1}
    assert len(mailer.sent) == 1
    assert mailer.sent[0]["subject"].startswith("[Podcast] Random Clip — Random Clip")
    assert audio in mailer.sent[0]["body_html"]

    guid = f"audio:{hashlib.sha1(audio.encode()).hexdigest()[:16]}"
    ep = db.get_episode(audio, guid)
    assert ep is not None and ep.status == "emailed"
    # A bare audio URL has no metadata, so we must NOT fabricate a release date
    # (the old bug stamped the email with the processing time).
    assert ep.published_at == ""
    assert "Released" not in mailer.sent[0]["body_html"]

    # Second tick: already seen → deduped, no new email.
    stats2 = await run_tick(settings=settings, feeds_file=feeds,
                            db=db, mailer=mailer, dry_run=False)
    assert stats2["emailed"] == 0
    assert len(mailer.sent) == 1


async def test_audio_url_uses_captured_episode_metadata(
    settings, db, mock_llm, mock_transcribe, mock_http,
):
    """An `audio_url` one-shot ingested from an Apple episode carries the real
    release date + runtime (persisted by the MCP at ingest); the pipeline uses
    them instead of stamping the processing time / omitting the duration."""
    import hashlib

    from podcast_digest.config import Feed, FeedsFile

    audio = "https://cdn.example.com/roundtable.mp3"
    mock_http.get(audio).respond(
        200,
        headers={"content-type": "audio/mpeg", "content-length": "9"},
        content=b"FAKEAUDIO",
    )

    feeds = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(
            name="Guest Name: The Minerals Supply Chain",
            audio_url=audio,
            published_at="2026-06-10T07:00:00Z",
            duration_seconds=1440,
            show="Weekly Roundtable",
        )],
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1
    # Subject leads with the real show name (not the episode title) and carries
    # the real runtime; body carries the real release date.
    subject = mailer.sent[0]["subject"]
    assert subject.startswith("[Podcast] Weekly Roundtable — Guest Name")
    assert "(24m)" in subject
    assert "Released" in mailer.sent[0]["body_html"]

    guid = f"audio:{hashlib.sha1(audio.encode()).hexdigest()[:16]}"
    ep = db.get_episode(audio, guid)
    assert ep is not None
    assert ep.published_at == "2026-06-10T07:00:00Z"
    assert ep.duration_seconds == 1440


async def test_oneoff_dedups_against_show_title_already_emitted(
    settings, db, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """A one-shot of an episode already emitted (e.g. via its podcast feed, or a
    backfilled mailbox entry) is suppressed on the (show, title) key — because
    the one-shot now keys on the real show name, not its own episode-title
    `name`. The guid differs, so only the secondary key can catch it; this is the
    one-shot/RSS double-send the show-name change closes."""
    import hashlib

    from podcast_digest.config import Feed, FeedsFile

    show = "Weekly Roundtable"
    title = "Guest Name: The Minerals Supply Chain"
    audio = "https://cdn.example.com/roundtable.mp3"

    # Title-only emitted entry (empty feed_url+guid), as a backfill-from-mail or
    # another feed's prior send records it.
    log_path = tmp_path / "emitted.tsv"
    log_path.write_text(
        f"\t\t2026-06-28T00:00:00+00:00\t{show}\t{title}\n", encoding="utf-8",
    )
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)

    feeds = FeedsFile(
        recipient="reader@example.com", poll_seconds=21600,
        feeds=[Feed(name=title, audio_url=audio, show=show)],
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)

    # Suppressed at discovery: no send, no local record.
    assert stats["emailed"] == 0
    assert mailer.sent == []
    guid = f"audio:{hashlib.sha1(audio.encode()).hexdigest()[:16]}"
    assert db.get_episode(audio, guid) is None


async def test_youtube_channel_takes_newest_video(
    settings, db, mock_llm, mock_transcribe, monkeypatch,
):
    """A `youtube` channel feed enumerates uploads via yt-dlp, enriches the newest
    unseen video's metadata, and (on first run) emails only that newest video."""
    from podcast_digest import youtube
    from podcast_digest.config import Feed, FeedsFile
    from podcast_digest.feeds import EpisodeMeta

    feed_url = "https://www.youtube.com/feeds/videos.xml?channel_id=UC" + "z" * 22
    new_watch = "https://www.youtube.com/watch?v=NEWVIDEO0001"

    async def fake_resolve(spec, *, client):
        return feed_url

    async def fake_list(spec, *, limit=15, cookies=None):
        # Flat enumeration: newest-first, provisional title, no date.
        return "Conference Talks", [
            EpisodeMeta(
                guid="yt:video:NEWVIDEO0001", title="(provisional teaser title)",
                published_at="", duration_seconds=1800, audio_url=new_watch,
                audio_type="video/youtube", description=None,
            ),
            EpisodeMeta(
                guid="yt:video:OLDVIDEO0002", title="Older", published_at="",
                duration_seconds=900,
                audio_url="https://www.youtube.com/watch?v=OLDVIDEO0002",
                audio_type="video/youtube", description=None,
            ),
        ]

    async def fake_single(url, *, cookies=None):
        # Only the newest unseen episode is enriched to authoritative metadata.
        assert url == new_watch
        return EpisodeMeta(
            guid="yt:video:NEWVIDEO0001", title="Newest",
            published_at="2026-06-15T00:00:00+00:00", duration_seconds=1800,
            audio_url=new_watch, audio_type="video/youtube", description="desc",
        )

    async def fake_dl(url, *, dest_dir, basename, fmt="bestaudio/best", cookies=None):
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / f"{basename}.mp3"
        p.write_bytes(b"FAKE-YT-AUDIO")
        return p

    monkeypatch.setattr(youtube, "resolve_channel_feed", fake_resolve)
    monkeypatch.setattr(youtube, "list_channel_uploads", fake_list)
    monkeypatch.setattr(youtube, "single_video_episode", fake_single)
    monkeypatch.setattr(youtube, "download_audio", fake_dl)

    feeds = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="Conference Talks", youtube="@conferencetalks")],
    )
    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds,
                           db=db, mailer=mailer, dry_run=False)

    assert stats == {"feeds": 1, "new_episodes": 1, "emailed": 1, "failed": 0,
                     "summarize_failed": 0, "summarize_ok": 1}
    assert len(mailer.sent) == 1
    # The email uses the enriched (authoritative) title, not the flat teaser.
    assert "Newest" in mailer.sent[0]["subject"]
    # Newest emailed; older suppressed as seen (kept on cheap flat metadata).
    assert db.get_episode(feed_url, "yt:video:NEWVIDEO0001").status == "emailed"
    assert db.get_episode(feed_url, "yt:video:OLDVIDEO0002").status == "emailed"


# ---------------------------------------------------------------------------
# Shared delivery log dedup (another instance already sent it)
# ---------------------------------------------------------------------------


async def test_emitted_log_skips_episode_already_sent_by_other_node(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """If the shared delivery log already lists the newest episode (the
    primary sent it before going down), the standby must NOT re-send it — the
    no-duplicate-emails guarantee. The episode is skipped entirely: no mail, no
    new_episode, and a fresh local DB."""
    log_path = tmp_path / "emitted.tsv"
    log_path.write_text(
        f"{FEED_URL}\t{NEWEST_GUID}\t2026-06-28T00:00:00+00:00\n", encoding="utf-8",
    )
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["emailed"] == 0
    assert stats["new_episodes"] == 0
    assert mailer.sent == []
    # Skipped, not recorded locally either — exactly as if already seen.
    assert db.get_episode(FEED_URL, NEWEST_GUID) is None


async def test_emitted_log_skips_strong_guid_from_differently_named_feed(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """One Acast episode ID appears in an
    umbrella feed and a differently named show-specific feed."""
    import httpx

    acast_guid = "5f0c1e2d3b4a596877665544"
    rss = RSS_BYTES.replace(NEWEST_GUID.encode(), acast_guid.encode())
    mock_http.get(FEED_URL).mock(return_value=httpx.Response(200, content=rss))

    log_path = tmp_path / "emitted.tsv"
    EmittedLog.load(log_path).record(
        "https://access.acast.com/rss/umbrella",
        acast_guid,
        "2026-07-29T00:00:00+00:00",
        "World News Daily",
        "A Brave New Week",
    )
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["new_episodes"] == 0
    assert stats["emailed"] == 0
    assert mailer.sent == []
    assert db.get_episode(FEED_URL, acast_guid) is None


async def test_emitted_log_records_guid_after_send(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """A real send appends the episode to the (enabled-but-empty) emitted log,
    so another instance would then skip it — the write half of the contract."""
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["emailed"] == 1
    assert len(mailer.sent) == 1
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parts = lines[0].split("\t")
    assert (parts[0], parts[1]) == (FEED_URL, NEWEST_GUID)


async def test_recut_full_program_is_suppressed_before_second_summary(
    settings, db, feeds_file, mock_llm, mock_http, tmp_path, monkeypatch,
):
    """Different feed, GUID, title, and a lightly edited transcript still
    produce one digest; the shorter excerpt policy is covered in test_dedupe."""
    import json

    import httpx

    from podcast_digest.config import Feed, FeedsFile
    from podcast_digest.transcribe import Transcript

    original_tokens = [f"token{i}" for i in range(1000)]
    edited_tokens = original_tokens.copy()
    edited_tokens[400:450] = [f"replacement{i}" for i in range(50)]
    transcripts = [" ".join(original_tokens), " ".join(edited_tokens)]
    transcribe_calls = 0

    async def transcribe_variant(_audio_path, *, out_dir, basename, **_kwargs):
        nonlocal transcribe_calls
        text = transcripts[transcribe_calls]
        transcribe_calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{basename}.txt"
        json_path = out_dir / f"{basename}.json"
        txt_path.write_text(text, encoding="utf-8")
        json_path.write_text(json.dumps({"transcription": []}), encoding="utf-8")
        return Transcript(text=text, segments=[]), txt_path, json_path

    monkeypatch.setattr("podcast_digest.pipeline.transcribe", transcribe_variant)
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    first = await run_tick(
        settings=settings,
        feeds_file=feeds_file,
        db=db,
        mailer=mailer,
        dry_run=False,
    )
    assert first["emailed"] == 1
    assert len(mock_llm.calls) == 1

    second_feed_url = "https://example.com/renamed.xml"
    second_guid = "different-platform-guid"
    renamed_rss = RSS_BYTES.replace(
        NEWEST_GUID.encode(), second_guid.encode()
    ).replace(
        b"Episode 3: Long episode", b"Renamed full interview"
    )
    mock_http.get(second_feed_url).mock(
        return_value=httpx.Response(200, content=renamed_rss)
    )
    second_feeds = FeedsFile(
        recipient="reader@example.com",
        feeds=[
            Feed(
                name="Different Video Channel",
                rss_url=second_feed_url,
                min_minutes=60,
            )
        ],
    )

    second = await run_tick(
        settings=settings,
        feeds_file=second_feeds,
        db=db,
        mailer=mailer,
        dry_run=False,
    )

    assert second["new_episodes"] == 1
    assert second["emailed"] == 0
    assert len(mailer.sent) == 1
    assert len(mock_llm.calls) == 1
    duplicate = db.get_episode(second_feed_url, second_guid)
    assert duplicate is not None
    assert duplicate.status == "emailed"
    assert duplicate.terminal_reason == "duplicate_transcript_near"
    assert duplicate.emailed_at is None
    assert duplicate.summary_html is None
    assert duplicate.content_sha256 is not None
    assert duplicate.audio_path is None


async def test_emitted_log_dry_run_does_not_record(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """A dry-run never sends, so it must never record — otherwise a dry-run would
    make another instance skip an episode that was never actually emailed."""
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    await run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=True,
    )

    assert mailer.sent == []
    assert not log_path.exists()


async def test_emitted_log_suppresses_stale_inprogress_resume(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """A stale in-progress row (summarized-but-not-emailed) for an episode the
    OTHER instance already sent must be suppressed on resume, not re-emailed —
    the standby-activation hardening."""
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    # Seed a stale in-progress row for the newest episode...
    db.insert_episode(feed_url=FEED_URL, guid=NEWEST_GUID, title="Episode 3: Long episode",
                      published_at="2026-06-01T00:00:00+00:00", duration_seconds=5400,
                      audio_url="https://cdn.example.com/ep3.mp3")
    db.set_status(feed_url=FEED_URL, guid=NEWEST_GUID, status="summarized",
                  summary_html="<p>old</p>")
    # ...and record it as already emailed elsewhere (by show+title).
    EmittedLog.load(log_path).record_title("The Test Podcast", "Episode 3: Long episode")
    mailer = FakeMailer()

    await run_tick(settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=False)

    assert mailer.sent == []  # NOT re-sent
    suppressed = db.get_episode(FEED_URL, NEWEST_GUID)
    assert suppressed.status == "emailed"
    assert suppressed.terminal_reason == "duplicate_identity"
    assert suppressed.emailed_at is None


# ---------------------------------------------------------------------------
# Post-audit hardening: re-key escape hatch, probably-delivered, sanitize-at-
# production, enrichment live/durationless gate
# ---------------------------------------------------------------------------


async def test_search_feed_rekeys_when_cached_url_dies(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """F5 escape hatch: a host migration kills the cached feed URL. The feed
    must re-resolve once, adopt the NEW URL (re-key), and MIGRATE its episode
    rows + polled watermark to the new PK — so even on a single-instance deploy
    (no emitted log; this test deliberately leaves it off) nothing is
    re-emailed, no first_run re-fires, and in-flight state survives."""
    import json as _json

    mailer = FakeMailer()

    s1 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    assert s1["emailed"] == 1
    assert db.get_resolved_url("The Test Podcast", "The Test Podcast") == FEED_URL

    # Host migration: the old URL starts failing; iTunes now points elsewhere.
    new_url = "https://example.com/feed-v2.xml"
    itunes_v2 = _json.loads((Path(__file__).parent / "fixtures" / "itunes_search.json").read_text())
    itunes_v2["results"][0]["feedUrl"] = new_url
    import httpx as _httpx
    mock_http.get(FEED_URL).mock(return_value=_httpx.Response(500))
    mock_http.get("https://itunes.apple.com/search").mock(
        return_value=_httpx.Response(200, content=_json.dumps(itunes_v2).encode(),
                                     headers={"content-type": "application/json"}))
    from .conftest import RSS_BYTES
    mock_http.get(new_url).mock(
        return_value=_httpx.Response(200, content=RSS_BYTES,
                                     headers={"content-type": "application/xml"}))

    s2 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    # Re-keyed to the new URL…
    assert db.get_resolved_url("The Test Podcast", "The Test Podcast") == new_url
    # …with the episode history migrated to the new PK: the already-sent
    # episode is a known row under the new URL (no emitted log needed), so no
    # duplicate email and no back-catalog re-suppression churn.
    assert s2["emailed"] == 0 and s2["new_episodes"] == 0
    assert len(mailer.sent) == 1
    migrated = db.get_episode(new_url, NEWEST_GUID)
    assert migrated is not None and migrated.status == "emailed"
    assert db.get_episode(FEED_URL, NEWEST_GUID) is None  # old PK vacated


async def test_search_feed_transient_outage_does_not_rekey(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """A dead cached URL with an UNCHANGED resolution must fail the tick (feed
    error), not re-key: only a genuinely different resolver answer re-keys, so
    a transient CDN outage can't orphan the feed's history."""
    import httpx as _httpx

    mailer = FakeMailer()
    s1 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    assert s1["emailed"] == 1

    mock_http.get(FEED_URL).mock(return_value=_httpx.Response(500))
    s2 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    # iTunes still answers FEED_URL → no re-key, the feed just fails this tick.
    assert db.get_resolved_url("The Test Podcast", "The Test Podcast") == FEED_URL
    assert s2["emailed"] == 0 and len(mailer.sent) == 1


async def test_feed_outage_does_not_strand_pending_delivery(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """Persisted work resumes even when the current RSS poll is unavailable."""
    import httpx as _httpx

    # First tick reaches summarized, then the synthetic mail failure leaves the
    # row pending for delivery.
    first = FakeMailer(fail=True)
    s1 = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db,
        mailer=first, dry_run=False,
    )
    assert s1["failed"] == 1
    pending = db.get_episode(FEED_URL, NEWEST_GUID)
    assert pending is not None and pending.status == "summarized"

    # Discovery now fails at the cached RSS URL and resolves to the same URL.
    # The already-produced digest must still be delivered from durable state.
    mock_http.get(FEED_URL).mock(return_value=_httpx.Response(500))
    recovered = FakeMailer()
    s2 = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db,
        mailer=recovered, dry_run=False,
    )

    assert s2["new_episodes"] == 0
    assert s2["emailed"] == 1
    assert s2["failed"] == 0
    assert len(recovered.sent) == 1
    assert db.get_episode(FEED_URL, NEWEST_GUID).status == "emailed"


async def test_delayed_feed_redirect_to_internal_address_is_blocked(
    settings, db, mock_http,
):
    """A stored public URL is re-validated if its redirect target later changes."""
    import httpx as _httpx

    from podcast_digest.config import Feed, FeedsFile

    public_url = "https://attacker.example/feed"
    internal_url = "http://127.0.0.1:8080/private"
    public = mock_http.get(public_url).mock(
        return_value=_httpx.Response(302, headers={"location": internal_url})
    )
    internal = mock_http.get(internal_url).mock(
        return_value=_httpx.Response(200, content=b"<rss/>")
    )
    feeds = FeedsFile(
        recipient="reader@example.com",
        feeds=[Feed(name="Untrusted feed", rss_url=public_url)],
    )

    stats = await run_tick(
        settings=settings, feeds_file=feeds, db=db,
        mailer=FakeMailer(), dry_run=False,
    )

    assert stats["new_episodes"] == 0
    assert public.called
    assert not internal.called


async def test_mail_probably_delivered_marks_sent_without_duplicate(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """F9: a response-phase timeout AFTER the request was fully transmitted to
    Graph means the send almost certainly happened — the episode must be marked
    emailed (+ recorded in the emitted log) instead of re-sending up to
    MAX_ATTEMPTS duplicate digests."""
    from podcast_digest.mail import MailProbablyDelivered

    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer(fail=True, exc=MailProbablyDelivered("response timed out"))

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1 and stats["failed"] == 0
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep.status == "emailed"
    assert ep.terminal_reason == "sent"
    assert ep.emailed_at is not None
    assert NEWEST_GUID in log_path.read_text()  # shared delivery record still written

    # Next tick with a healthy mailer: nothing to re-send.
    mailer2 = FakeMailer()
    s2 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer2, dry_run=False)
    assert s2["emailed"] == 0 and mailer2.sent == []


async def test_summary_sanitized_at_production_choke_point(
    settings, db, feeds_file, mock_transcribe, mock_http, monkeypatch,
):
    """F8 at the right altitude: the sanitizer runs where the summary is
    PRODUCED, so the DB row itself is clean — every consumer (email, wiki,
    future readers) inherits it instead of each needing its own defense."""

    def evil_summarize(inp, *, completer, **_kw):
        return (
            "<h2>TL;DR</h2><p>fine</p>"
            "<script>alert(1)</script>"
            '<p><a href="javascript:steal()">click</a></p>'
        )

    monkeypatch.setattr("podcast_digest.pipeline.summarize", evil_summarize)
    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1

    stored = db.get_episode(FEED_URL, NEWEST_GUID).summary_html
    assert "<script" not in stored and "javascript:" not in stored
    assert "<p>fine</p>" in stored
    assert "<script" not in mailer.sent[0]["body_html"]


async def test_enrich_drops_durationless_entry_when_metadata_fetch_fails(
    db, monkeypatch,
):
    """A flat-mode entry with no duration whose full-metadata fetch also fails
    is overwhelmingly a live/premiere — it must be dropped for THIS tick (it
    would otherwise take the cap-1 slot and pre-suppress the real newest
    upload), while a durationed entry degrades to flat metadata as before."""
    from podcast_digest import youtube
    from podcast_digest.feeds import EpisodeMeta
    from podcast_digest.pipeline import _enrich_new_youtube

    def _meta(vid, dur):
        return EpisodeMeta(
            guid=f"yt:video:{vid}", title="?", published_at="",
            duration_seconds=dur,
            audio_url=f"https://www.youtube.com/watch?v={vid}",
            audio_type="video/youtube", description=None,
        )

    calls: list[str] = []

    async def authoritative_or_live(url, *, cookies=None):
        calls.append(url)
        if "AAAAAAAAAAA" in url:
            raise youtube.FeedError("Premieres in 2 hours")
        return EpisodeMeta(
            guid="yt:video:BBBBBBBBBBB",
            title="Authoritative newest VOD",
            published_at="2026-07-22T10:00:00+00:00",
            duration_seconds=901,
            audio_url=url,
            audio_type="video/youtube",
            description=None,
        )

    monkeypatch.setattr(
        youtube, "single_video_episode", authoritative_or_live
    )
    # With a one-slot budget, the dropped live entry must not prevent the next
    # VOD from receiving authoritative validation.
    out = await _enrich_new_youtube(
        [_meta("AAAAAAAAAAA", None), _meta("BBBBBBBBBBB", 900)],
        feed_url="https://f/x", db=db, cookies=None, limit=1,
    )
    assert [m.guid for m in out] == ["yt:video:BBBBBBBBBBB"]
    assert out[0].title == "Authoritative newest VOD"
    assert out[0].duration_seconds == 901
    assert len(calls) == 2


async def test_rekey_not_committed_until_fresh_url_fetches(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """The re-key must be adopted only AFTER the fresh URL proves fetchable:
    committing first would let one transient outage + one flaky resolver
    answer permanently swap the known-good cached URL for a dead one."""
    import json as _json

    import httpx as _httpx

    mailer = FakeMailer()
    s1 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    assert s1["emailed"] == 1

    # Cached URL 500s AND the resolver flakes to a different URL that is ALSO
    # dead: the tick fails, but the cache must still hold the original.
    dead_url = "https://example.com/feed-flaky.xml"
    itunes_v2 = _json.loads(
        (Path(__file__).parent / "fixtures" / "itunes_search.json").read_text())
    itunes_v2["results"][0]["feedUrl"] = dead_url
    mock_http.get(FEED_URL).mock(return_value=_httpx.Response(500))
    mock_http.get(dead_url).mock(return_value=_httpx.Response(500))
    mock_http.get("https://itunes.apple.com/search").mock(
        return_value=_httpx.Response(200, content=_json.dumps(itunes_v2).encode(),
                                     headers={"content-type": "application/json"}))

    s2 = await run_tick(settings=settings, feeds_file=feeds_file,
                        db=db, mailer=mailer, dry_run=False)
    assert s2["emailed"] == 0
    # Unverified identity NOT adopted; the recoverable original is retried
    # next tick once its outage passes.
    assert db.get_resolved_url("The Test Podcast", "The Test Podcast") == FEED_URL
    assert db.get_episode(FEED_URL, NEWEST_GUID) is not None  # history intact


async def test_rekey_recovers_if_cache_write_crashes_after_state_migration(
    db, monkeypatch,
):
    """A crash between migration and cache update is safe on the next tick."""
    from podcast_digest.feeds import FeedError
    from podcast_digest.pipeline import _fetch_via_cached_identity

    old_url = "https://old.example/rss"
    new_url = "https://new.example/rss"
    db.upsert_feed(url=old_url, name="Crash-safe")
    db.mark_feed_polled(old_url)
    db.insert_episode(
        feed_url=old_url,
        guid="guid-1",
        title="Episode",
        published_at="2026-07-22T10:00:00+00:00",
        duration_seconds=60,
        audio_url="https://cdn.example/episode.mp3",
    )
    db.set_resolved_url("Crash-safe", "search term", old_url)

    async def resolve():
        return new_url, None

    async def fetch(url):
        if url == old_url:
            raise FeedError("old host unavailable")
        return b"fresh feed"

    real_set_resolved = db.set_resolved_url

    def crash_on_cache(name, spec, url):
        if url == new_url:
            raise RuntimeError("simulated crash before cache commit")
        real_set_resolved(name, spec, url)

    monkeypatch.setattr(db, "set_resolved_url", crash_on_cache)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _fetch_via_cached_identity(
            db=db,
            feed_name="Crash-safe",
            spec="search term",
            resolve=resolve,
            fetch=fetch,
        )

    # State moved first, while the cache still points at the dead old URL.
    assert db.get_episode(old_url, "guid-1") is None
    assert db.get_episode(new_url, "guid-1") is not None
    assert db.get_resolved_url("Crash-safe", "search term") == old_url

    # Retrying repeats the no-op migration and then commits the cache.
    monkeypatch.setattr(db, "set_resolved_url", real_set_resolved)
    url, _extra, result = await _fetch_via_cached_identity(
        db=db,
        feed_name="Crash-safe",
        spec="search term",
        resolve=resolve,
        fetch=fetch,
    )
    assert (url, result) == (new_url, b"fresh feed")
    assert db.get_resolved_url("Crash-safe", "search term") == new_url
    assert db.get_episode(new_url, "guid-1") is not None


# ---------------------------------------------------------------------------
# Non-English feeds (feeds.toml `language`) — Mandarin-first support
# ---------------------------------------------------------------------------


async def test_mandarin_feed_multilingual_model_english_summary(
    settings, db, mock_llm, mock_transcribe, mock_http, tmp_path,
):
    """A `language = "zh"` feed transcribes with the multilingual whisper model
    in Chinese, the summarizer is told the transcript is Mandarin (and emits
    English), and the email carries a "Translated from Mandarin" note — while
    the episode title (the delivery-log dedup key) stays untouched."""
    from podcast_digest.config import Feed, FeedsFile

    feeds_zh = FeedsFile(
        recipient="reader@example.com",
        poll_seconds=21600,
        feeds=[Feed(name="The Test Podcast", search="The Test Podcast",
                    min_minutes=60, language="zh")],
    )
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_zh, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["emailed"] == 1
    # Transcription selected the multilingual model with `-l zh`.
    assert len(mock_transcribe.calls) == 1
    call = mock_transcribe.calls[0]
    assert call["model_path"] == str(settings.PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL)
    assert call["language"] == "zh"
    # The summarizer got the source language (its output stays English).
    assert mock_llm.calls[0]["source_language"] == "zh"
    # One English email with the translation note; subject/title untouched.
    assert len(mailer.sent) == 1
    sent = mailer.sent[0]
    assert "Translated from Mandarin" in sent["body_html"]
    assert sent["subject"].startswith("[Podcast] The Test Podcast — Episode 3:")
    # Delivery-log dedup key unchanged: the emitted log records the ORIGINAL
    # (show, title) — a translated title would re-send from the standby.
    cols = log_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert (cols[0], cols[1]) == (FEED_URL, NEWEST_GUID)
    assert cols[3] == "The Test Podcast"
    assert cols[4] == "Episode 3: Long episode"


async def test_english_feed_regression_keeps_en_model_and_no_note(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """No `language` in feeds.toml → the English path is byte-for-byte what it
    was: .en model, `-l en`, no source_language, no translation note."""
    mailer = FakeMailer()

    stats = await run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer, dry_run=False,
    )

    assert stats["emailed"] == 1
    call = mock_transcribe.calls[0]
    assert call["model_path"] == str(settings.PODCAST_DIGEST_WHISPER_MODEL)
    assert call["language"] == "en"
    assert mock_llm.calls[0]["source_language"] is None
    assert "Translated from" not in mailer.sent[0]["body_html"]
