"""Focused regression tests for enclosure refresh, Cantonese metadata, and CJK chunking."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from podcast_digest.config import Feed, FeedsFile, Settings
from podcast_digest.db import Database
from podcast_digest.feeds import EpisodeMeta
from podcast_digest.pipeline import _wrap_email, run_tick
from podcast_digest.summarize import _chunk_transcript, _language_guidance
from podcast_digest.transcribe import Segment


def test_refresh_pending_episode_media_updates_only_pre_transcribe(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        feed_url = "https://feeds.example.com/show.xml"
        db.upsert_feed(url=feed_url, name="Example Show")
        db.insert_episode(
            feed_url=feed_url,
            guid="ep-1",
            title="Episode 1",
            published_at="2026-11-19T00:00:00+00:00",
            duration_seconds=None,
            audio_url="https://cdn.example.com/ep1.mp3?sig=expired",
        )

        changed = db.refresh_pending_episode_media(
            feed_url=feed_url,
            guid="ep-1",
            audio_url="https://cdn.example.com/ep1.mp3?sig=fresh",
            duration_seconds=1800,
        )
        assert changed is True
        ep = db.get_episode(feed_url, "ep-1")
        assert ep is not None
        assert ep.audio_url == "https://cdn.example.com/ep1.mp3?sig=fresh"
        assert ep.duration_seconds == 1800

        # Idempotent when nothing changed
        assert db.refresh_pending_episode_media(
            feed_url=feed_url,
            guid="ep-1",
            audio_url="https://cdn.example.com/ep1.mp3?sig=fresh",
            duration_seconds=1800,
        ) is False

        # Once transcribed, audio_url is frozen
        db.set_status(feed_url=feed_url, guid="ep-1", status="transcribed")
        assert db.refresh_pending_episode_media(
            feed_url=feed_url,
            guid="ep-1",
            audio_url="https://cdn.example.com/ep1.mp3?sig=newer",
            duration_seconds=1900,
        ) is False
        ep = db.get_episode(feed_url, "ep-1")
        assert ep is not None
        assert ep.audio_url == "https://cdn.example.com/ep1.mp3?sig=fresh"
    finally:
        db.close()


async def test_pipeline_refreshes_expired_enclosure_url_on_retry(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        feed_url = "https://feeds.example.com/show.xml"
        db.upsert_feed(url=feed_url, name="Example Show")
        db.mark_feed_polled(feed_url)
        db.insert_episode(
            feed_url=feed_url,
            guid="ep-1",
            title="Episode 1",
            published_at="2026-11-19T00:00:00+00:00",
            duration_seconds=1200,
            audio_url="https://cdn.example.com/ep1.mp3?sig=expired",
        )

        fresh_meta = EpisodeMeta(
            guid="ep-1",
            title="Episode 1",
            published_at="2026-11-19T00:00:00+00:00",
            duration_seconds=1200,
            audio_url="https://cdn.example.com/ep1.mp3?sig=fresh",
            audio_type="audio/mpeg",
            description=None,
        )
        seen_urls: list[str] = []

        async def fake_download(url: str, **kwargs):
            seen_urls.append(url)
            raise RuntimeError("stop after recording refreshed url")

        settings = Settings(PODCAST_DIGEST_DATA_DIR=tmp_path)
        feeds = FeedsFile(
            recipient="reader@example.com",
            feeds=[Feed(name="Example Show", rss_url=feed_url)],
        )
        with (
            patch("podcast_digest.pipeline._discover_episodes", new=AsyncMock(return_value=(feed_url, None, [fresh_meta]))),
            patch("podcast_digest.pipeline.audio_download", new=AsyncMock(side_effect=fake_download)),
        ):
            await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=None, dry_run=True)

        assert seen_urls == ["https://cdn.example.com/ep1.mp3?sig=fresh"]
        ep = db.get_episode(feed_url, "ep-1")
        assert ep is not None
        assert ep.audio_url == "https://cdn.example.com/ep1.mp3?sig=fresh"
    finally:
        db.close()


def test_cantonese_email_header_and_summary_guidance():
    html = _wrap_email(
        "<h2>TL;DR</h2><p>Summary.</p>",
        "Regional Markets",
        "Episode 1",
        "https://cdn.example.com/ep1.mp3",
        original_language="yue",
    )
    assert "Translated from Cantonese" in html

    guidance = _language_guidance("yue", style="finance")
    assert "Cantonese Chinese" in guidance
    assert "万 = 10^4" in guidance
    assert "亿 = 10^8" in guidance


def test_cjk_chunking_splits_densely_while_preserving_ascii():
    ascii_text = "a" * 100
    assert len(_chunk_transcript([], ascii_text, max_chars=100)) == 1

    # 100 CJK chars weigh 300 (> 100), so they split into 3-4 chunks
    cjk_text = "中" * 100
    cjk_chunks = _chunk_transcript([], cjk_text, max_chars=100)
    assert len(cjk_chunks) == 4
    assert "".join(cjk_chunks) == cjk_text

    # Segmented CJK path also respects weighted length
    segs = [Segment(start=float(i * 30), end=float((i + 1) * 30), text="中" * 30) for i in range(4)]
    seg_chunks = _chunk_transcript(segs, "", max_chars=150)
    assert len(seg_chunks) == 4
