"""Tests for email UI redesign, read-only MCP archive tools, and backend hygiene."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from podcast_digest.config import Feed, FeedsFile, Settings
from podcast_digest.db import Database
from podcast_digest.delivery import Envelope, send_digest
from podcast_digest.feeds import EpisodeMeta
from podcast_digest.mail import MailAttachment, _html_to_text
from podcast_digest.pipeline import _wrap_email, run_tick
from podcast_digest_mcp import archive

from .conftest import FEED_URL, FakeMailer


def test_wrap_email_links_youtube_timestamps_and_strips_style_in_plaintext():
    summary = (
        "<h2>TL;DR</h2><p>Summary.</p>"
        "<h2>Timestamped highlights</h2>"
        "<ul><li><strong>[00:02:15]</strong> — Opening point</li>"
        "<li><strong>[01:05:00]</strong> — Closing call</li></ul>"
    )
    html = _wrap_email(
        summary,
        "Lecture Hall",
        "Macro Outlook",
        "https://www.youtube.com/watch?v=abc12345678",
        published_at="2026-11-20T00:00:00+00:00",
        duration_seconds=4200,
    )
    assert 'href="https://www.youtube.com/watch?v=abc12345678&amp;t=135s"' in html
    assert 'href="https://www.youtube.com/watch?v=abc12345678&amp;t=3900s"' in html
    plain = _html_to_text(html)
    assert ".pd-wrap" not in plain
    assert "prefers-color-scheme" not in plain
    assert "Macro Outlook" in plain
    assert "[00:02:15] — Opening point" in plain


def test_secret_file_loading_and_validation(tmp_path: Path):
    key_file = tmp_path / "caller.key"
    key_file.write_text("  sk-test-from-file\n", encoding="utf-8")

    s = Settings(ANTHROPIC_API_KEY_FILE=key_file)
    assert s.ANTHROPIC_API_KEY == "sk-test-from-file"

    # Direct env value takes precedence over file
    s2 = Settings(ANTHROPIC_API_KEY="sk-direct", ANTHROPIC_API_KEY_FILE=key_file)
    assert s2.ANTHROPIC_API_KEY == "sk-direct"

    # Missing file fails closed
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY_FILE"):
        Settings(ANTHROPIC_API_KEY_FILE=tmp_path / "missing.key")

    # Empty file fails closed
    empty_file = tmp_path / "empty.key"
    empty_file.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY_FILE"):
        Settings(ANTHROPIC_API_KEY_FILE=empty_file)


async def test_finalize_delivery_prunes_base64_attachments_and_reopen_migrates(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url=FEED_URL, name="Show")
        db.insert_episode(
            feed_url=FEED_URL, guid="ep-1", title="Episode 1",
            published_at="2026-11-20T00:00:00+00:00", duration_seconds=600,
            audio_url="https://cdn.example.com/ep1.mp3",
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-1", status="summarized",
            summary_html="<h2>TL;DR</h2><p>Hi</p>",
        )
        payload = b"# Cleaned Transcript\n" + (b"word " * 500)
        env = Envelope(
            to="reader@example.com",
            subject="[Podcast] Show — Episode 1",
            body_html="<p>Hi</p>",
            attachments=(MailAttachment("Episode-1.md", payload, "text/markdown; charset=utf-8"),),
        )
        await send_digest(db, FakeMailer(), FEED_URL, "ep-1", env, show="Show")
        row_before = db.get_delivery(FEED_URL, "ep-1")
        assert row_before is not None
        assert '"content":' in row_before.envelope

        db.finalize_delivery(row_before.operation_id)
        row_after = db.get_delivery(FEED_URL, "ep-1")
        assert row_after is not None and row_after.finalized_at is not None
        parsed_json = json.loads(row_after.envelope)
        assert parsed_json["attachments"] == [{
            "name": "Episode-1.md",
            "content_type": "text/markdown; charset=utf-8",
            "size_bytes": len(payload),
            "pruned": True,
        }]
        # Envelope.parse still tolerates pruned attachments
        parsed_env = Envelope.parse(row_after.envelope)
        assert parsed_env.attachments[0].name == "Episode-1.md"
        assert parsed_env.attachments[0].content == b""

        # Simulate a legacy finalized row that still had base64 content
        db._conn.execute(
            "UPDATE digest_deliveries SET envelope=? WHERE operation_id=?",
            (row_before.envelope, row_before.operation_id),
        )
    finally:
        db.close()

    reopened = Database(tmp_path / "state.db")
    try:
        migrated = reopened.get_delivery(FEED_URL, "ep-1")
        assert migrated is not None
        assert '"content":' not in migrated.envelope
        assert '"pruned": true' in migrated.envelope
    finally:
        reopened.close()


async def test_configurable_max_new_per_tick(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url=FEED_URL, name="Daily Show")
        db.mark_feed_polled(FEED_URL)  # steady state (not first_run)
        metas = [
            EpisodeMeta(
                guid=f"ep-{i}", title=f"Ep {i}",
                published_at=f"2026-11-{20 - i:02d}T00:00:00+00:00",
                duration_seconds=900,
                audio_url=f"https://cdn.example.com/ep{i}.mp3",
                audio_type="audio/mpeg", description=None,
            )
            for i in range(3)
        ]
        settings = Settings(PODCAST_DIGEST_DATA_DIR=tmp_path)
        feeds = FeedsFile(
            recipient="reader@example.com",
            max_new_per_tick=1,
            feeds=[Feed(name="Daily Show", rss_url=FEED_URL, max_new_per_tick=2)],
        )
        with (
            patch("podcast_digest.pipeline._discover_episodes", new=AsyncMock(return_value=(FEED_URL, None, metas))),
            patch("podcast_digest.pipeline._advance_episode", new=AsyncMock(return_value=False)),
        ):
            stats = await run_tick(settings=settings, feeds_file=feeds, db=db, mailer=None, dry_run=True)

        assert stats["new_episodes"] == 2
        assert db.get_episode(FEED_URL, "ep-0").status == "discovered"
        assert db.get_episode(FEED_URL, "ep-1").status == "discovered"
        assert db.get_episode(FEED_URL, "ep-2").terminal_reason == "backlog_suppressed"
    finally:
        db.close()


def test_mcp_archive_status_search_get_and_readonly_wal(tmp_path: Path):
    data_dir = tmp_path / "data"
    transcripts = data_dir / "transcripts" / "show--abc"
    transcripts.mkdir(parents=True)
    raw_txt = transcripts / "ep1--123.txt"
    clean_md = transcripts / "ep1--123.clean.md"
    raw_txt.write_text("raw transcript about central bank liquidity", encoding="utf-8")
    clean_md.write_text(
        "# Macro Deep Dive\n\n**Host:** Let's discuss quantitative tightening and SOFR spreads.\n",
        encoding="utf-8",
    )

    db = Database(data_dir / "state.db")
    try:
        db.upsert_feed(url=FEED_URL, name="Macro Pod")
        db.insert_episode(
            feed_url=FEED_URL, guid="ep-macro-1", title="Macro Deep Dive",
            published_at="2026-11-20T00:00:00+00:00", duration_seconds=2400,
            audio_url="https://cdn.example.com/macro1.mp3",
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-macro-1", status="transcribed",
            transcript_path=str(raw_txt), segments_path=str(transcripts / "ep1--123.json"),
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-macro-1", status="summarized",
            summary_html="<h2>TL;DR</h2><p>Liquidity and SOFR overview.</p>",
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-macro-1", status="emailed",
            emailed_at="2026-11-20T02:00:00+00:00", terminal_reason="sent",
        )
        # Also test read-only directory fallback on WAL database
        status = archive.digest_status(data_dir, limit=5)
        assert status["available"] is True
        assert status["counts"]["emailed"] == 1
        assert status["recent_delivered"][0]["has_transcript"] is True

        search = archive.search_transcripts(data_dir, query="quantitative tightening", feed="macro")
        assert search["available"] is True
        assert search["total_matches"] == 1
        assert search["results"][0]["guid"] == "ep-macro-1"
        assert search["results"][0]["transcript_kind"] == "cleaned"

        ep = archive.get_episode(data_dir, query_or_guid="ep-macro-1", max_chars=1000)
        assert ep["found"] is True
        assert ep["transcript_kind"] == "cleaned"
        assert "SOFR spreads" in ep["transcript"]
        assert ep["transcript_truncated"] is False

        # Make data_dir read-only to verify WAL snapshot fallback works without write access
        os.chmod(data_dir, 0o555)
        try:
            ro_status = archive.digest_status(data_dir, limit=5)
            assert ro_status["available"] is True
            assert ro_status["counts"]["emailed"] == 1
        finally:
            os.chmod(data_dir, 0o755)
    finally:
        db.close()


def test_clean_transcript_passes_show_notes_to_every_chunk():
    from podcast_digest.summarize import SummarizationInput, clean_transcript

    seen_prompts: list[str] = []

    class FakeCompleter:
        label = "fake"

        def complete(self, *, system: str, prompt: str, purpose: str, guid: str | None = None, validate=None) -> str:
            seen_prompts.append(prompt)
            return "**Jane Doe:** Cleaned chunk."

        def config_problem(self) -> str | None:
            return None

    inp = SummarizationInput(
        podcast_name="Macro Pod",
        title="Liquidity Outlook",
        published_at="2026-11-21T00:00:00+00:00",
        duration_seconds=3600,
        description="<p>Host <b>Alex</b> interviews guest <em>Jane Doe</em> on SOFR.</p>",
        transcript_text=("alpha " * 300) + ("beta " * 300),
        segments=[],
    )
    md = clean_transcript(inp, completer=FakeCompleter(), max_chunk_chars=1800)
    assert len(seen_prompts) == 2
    assert all("Show notes: Host Alex interviews guest Jane Doe on SOFR." in p for p in seen_prompts)
    assert "**Jane Doe:** Cleaned chunk." in md


def test_fts5_search_index_backfill_and_bm25_query(tmp_path: Path):
    from podcast_digest.pipeline import _backfill_search_index

    data_dir = tmp_path / "data"
    transcripts = data_dir / "transcripts" / "show--xyz"
    transcripts.mkdir(parents=True)
    raw_txt = transcripts / "ep1--abc.txt"
    clean_md = transcripts / "ep1--abc.clean.md"
    raw_txt.write_text("raw text", encoding="utf-8")
    clean_md.write_text(
        "# Repo Plumbing\n\n**Jane Doe:** Reverse-repo balances dropped below $50B.\n",
        encoding="utf-8",
    )

    db = Database(data_dir / "state.db")
    try:
        db.upsert_feed(url=FEED_URL, name="Plumbing Weekly")
        db.insert_episode(
            feed_url=FEED_URL, guid="ep-fts-1", title="Repo Plumbing",
            published_at="2026-11-21T00:00:00+00:00", duration_seconds=1800,
            audio_url="https://cdn.example.com/repo.mp3",
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-fts-1", status="transcribed",
            transcript_path=str(raw_txt), segments_path=str(transcripts / "ep1--abc.json"),
        )
        db.set_status(
            feed_url=FEED_URL, guid="ep-fts-1", status="summarized",
            summary_html="<h2>TL;DR</h2><p>Front-end repo funding tightens.</p>",
        )
        _backfill_search_index(db)

        # Punctuation and quotes in user query must never trigger FTS5 syntax errors
        res = archive.search_transcripts(data_dir, query='reverse-repo "balances"', feed="plumbing")
        assert res["available"] is True
        assert res["engine"] == "fts5"
        assert res["total_matches"] == 1
        assert res["results"][0]["guid"] == "ep-fts-1"
        assert res["results"][0]["transcript_kind"] == "cleaned"
    finally:
        db.close()
