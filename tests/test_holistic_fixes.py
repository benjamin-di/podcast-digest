"""Regression tests for holistic bug, UI/UX, MCP, and efficiency fixes."""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from local_transcriber.whisper import _atomic_publish, _validate_artifact_id
from pydantic import ValidationError

from podcast_digest import __main__ as cli_main
from podcast_digest import feeds, pipeline, publisher, summarize, webhook
from podcast_digest.command import _classify_failure, _parse_result
from podcast_digest.config import Feed, FeedsFile
from podcast_digest.db import Database
from podcast_digest.emitted import EmittedLog
from podcast_digest.feeds import EpisodeMeta
from podcast_digest.mail import SmtpMailer, _html_to_text
from podcast_digest.sanitize import sanitize_summary_html
from podcast_digest.transcribe import Segment, Transcript
from podcast_digest.webhook import DeliveredEpisode, WebhookClient, WebhookError
from podcast_digest_mcp import archive, feeds_io, routing


def test_atomic_writes_respect_process_umask_for_mcp(tmp_path: Path):
    probe = tmp_path / "umask-probe"
    probe.write_text("x", encoding="utf-8")
    expected_mode = stat.S_IMODE(probe.stat().st_mode)

    target = tmp_path / "transcript.clean.md"
    pipeline._atomic_write_text(target, "# Episode\n\nHello.\n")
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode

    src = tmp_path / "staged.txt"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "published.txt"
    real_replace = os.replace
    calls = 0

    def _fake_replace(a, b):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(18, "EXDEV")
        return real_replace(a, b)

    with patch("local_transcriber.whisper.os.replace", side_effect=_fake_replace):
        _atomic_publish(src, dst)
    assert stat.S_IMODE(dst.stat().st_mode) == expected_mode


def test_archive_warns_on_unreadable_transcript(tmp_path: Path, caplog):
    tdir = tmp_path / "transcripts" / "show--abc"
    tdir.mkdir(parents=True)
    raw = tdir / "ep1.txt"
    raw.write_text("hello world", encoding="utf-8")

    with (
        patch("podcast_digest_mcp.archive.os.access", return_value=False),
        patch.object(Path, "read_text", side_effect=PermissionError("denied")),
        caplog.at_level(logging.WARNING),
    ):
        assert archive._has_transcript(tmp_path, str(raw)) is False
        assert archive._read_transcript(tmp_path, str(raw)) == (None, None)
    assert any("archive.transcript_unreadable" in r.message for r in caplog.records)


def test_webhook_rejects_redirects_and_ignores_non_utf8_without_ack(monkeypatch):
    import http.server
    import threading

    requests_seen: list[str] = []

    class _RedirectServer(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests_seen.append(f"POST {self.path}")
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/secret")
            self.end_headers()

        def do_GET(self):
            requests_seen.append(f"GET {self.path}")
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _RedirectServer)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.handle_request, daemon=True)
    thread.start()
    ep = DeliveredEpisode(
        show="Synthetic", title="Ep 1", published_at="2026-01-01",
        guid="g1", listen_url="https://cdn.example/1.mp3",
        summary="<p>Hi</p>", transcript="Hello",
    )
    try:
        client = WebhookClient(url=f"http://127.0.0.1:{port}/ingest", token="tok-secret", ack_field="")
        with pytest.raises(WebhookError, match="HTTP 302"):
            client.push(ep)
    finally:
        thread.join(timeout=2)
        srv.server_close()
    assert requests_seen == ["POST /ingest"]

    class _Binary200:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n=-1):
            return b"\xff\xfe\xfd"

    monkeypatch.setattr(webhook, "_open", lambda req, timeout=None: _Binary200())
    client = WebhookClient(url="https://hooks.example.com/ingest", ack_field="")
    assert client.push(ep) == ""


async def test_fetch_page_enforces_size_limit(monkeypatch):
    monkeypatch.setattr(feeds, "MAX_PAGE_BYTES", 100)
    monkeypatch.setattr(feeds, "validate_outbound_url_async", AsyncMock())

    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 200))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(feeds.FeedError, match="exceeded"):
            await feeds.fetch_page("https://feeds.example.com/big.xml", client=client)


def test_multiline_subject_and_emitted_log_sanitization(tmp_path: Path):
    subj = pipeline._build_subject("Daily\nShow", "Episode\r\nTitle\tOne", 3600)
    assert "\n" not in subj and "\r" not in subj
    assert subj == "[Podcast] Daily Show — Episode Title One (1h00m)"

    # Duplicate show == title omits redundant prefix.
    dedup_subj = pipeline._build_subject("Standalone Video", "Standalone Video", 600)
    assert dedup_subj == "[Podcast] Standalone Video (10m)"

    # Long subjects truncate cleanly while preserving duration suffix.
    long_subj = pipeline._build_subject("Show", "A" * 300, 1800)
    assert len(long_subj) <= 240
    assert long_subj.endswith("… (30m)")

    # EmittedLog sanitizes CR and Unicode line separators so reload never splits a row.
    log_path = tmp_path / "emitted.tsv"
    log = EmittedLog.load(log_path)
    assert log.record(
        "https://feeds.example.com/rss", "guid-1", "2026-11-23T00:00:00+00:00",
        "Show\rName", "Line1\u2028Line2\r\nLine3",
    )
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_smtp_invalid_header_is_classified_as_mail_error():
    mailer = SmtpMailer(host="127.0.0.1", port=1, sender="bot\nbad@example.com")
    with pytest.raises(Exception) as exc:
        mailer.send(to="user@example.com", subject="Hi", body_html="<p>Hi</p>")
    from podcast_digest.mail import MailError
    assert isinstance(exc.value, MailError)


async def test_youtube_channel_teaser_does_not_suppress_qualifying_episode(
    settings, db, mock_llm, mock_transcribe, mock_http, monkeypatch,
):
    from .conftest import FakeMailer

    teaser_known = EpisodeMeta(
        guid="yt:video:teaser00001",
        title="30s Teaser",
        published_at="2026-11-23T12:00:00+00:00",
        duration_seconds=30,
        audio_url="https://www.youtube.com/watch?v=teaser00001",
        audio_type="audio/mp4",
        description=None,
    )
    teaser_flat_unknown = EpisodeMeta(
        guid="yt:video:teaser00002",
        title="Durationless Flat Teaser",
        published_at="2026-11-23T11:00:00+00:00",
        duration_seconds=None,
        audio_url="https://www.youtube.com/watch?v=teaser00002",
        audio_type="audio/mp4",
        description=None,
    )
    teaser_enriched_short = EpisodeMeta(
        guid="yt:video:teaser00002",
        title="Enriched 45s Short",
        published_at="2026-11-23T11:00:00+00:00",
        duration_seconds=45,
        audio_url="https://www.youtube.com/watch?v=teaser00002",
        audio_type="audio/mp4",
        description=None,
    )
    full_ep = EpisodeMeta(
        guid="yt:video:fullep00001",
        title="Full 30m Interview",
        published_at="2026-11-22T12:00:00+00:00",
        duration_seconds=1800,
        audio_url="https://www.youtube.com/watch?v=fullep00001",
        audio_type="audio/mp4",
        description=None,
    )

    by_url = {
        teaser_known.audio_url: teaser_known,
        teaser_flat_unknown.audio_url: teaser_enriched_short,
        full_ep.audio_url: full_ep,
    }
    channel_feed = "https://www.youtube.com/feeds/videos.xml?channel_id=UCtest123456"
    monkeypatch.setattr(
        pipeline.youtube, "resolve_channel_feed",
        AsyncMock(return_value=channel_feed),
    )
    monkeypatch.setattr(
        pipeline.youtube, "list_channel_uploads",
        AsyncMock(return_value=("Synthetic Channel", [teaser_known, teaser_flat_unknown, full_ep])),
    )
    monkeypatch.setattr(
        pipeline.youtube, "single_video_episode",
        AsyncMock(side_effect=lambda url, **kw: by_url[url]),
    )
    monkeypatch.setattr(pipeline, "audio_download", AsyncMock(return_value=settings.audio_dir / "ep.m4a"))
    (settings.audio_dir / "ep.m4a").write_bytes(b"fake")
    monkeypatch.setattr(pipeline, "probe_duration", AsyncMock(return_value=1800.0))

    feeds_file = FeedsFile(
        recipient="reader@example.com",
        min_minutes=10,
        feeds=[Feed(name="Synthetic Channel", youtube="https://www.youtube.com/@synthetic")],
    )
    mailer = FakeMailer()
    stats = await pipeline.run_tick(
        settings=settings, feeds_file=feeds_file, db=db, mailer=mailer,
    )
    assert stats["emailed"] == 1
    assert len(mailer.sent) == 1
    assert "Full 30m Interview" in mailer.sent[0]["subject"]
    delivered = db.get_episode(channel_feed, full_ep.guid)
    assert delivered is not None and delivered.status == "emailed" and delivered.terminal_reason == "sent"


def test_attachment_filename_preserves_unicode_and_legacy_guid_updates_fts(tmp_path: Path):
    assert pipeline._attachment_filename("央行降准 — Macro Update", ".md") == "央行降准-Macro-Update.md"

    db = Database(tmp_path / "state.db")
    try:
        feed_url = "https://feeds.example/rss"
        old_guid = "https://cdn.example/ep.mp3?sig=1"
        new_guid = "synthetic:v1:" + "b" * 64
        db.insert_episode(
            feed_url=feed_url, guid=old_guid, title="Legacy Ep",
            published_at="2026-11-01T00:00:00+00:00", duration_seconds=600,
            audio_url=old_guid,
        )
        db.set_status(
            feed_url=feed_url, guid=old_guid, status="emailed",
            summary_html="<p>Legacy summary</p>", emailed_at="2026-11-01T01:00:00+00:00",
            terminal_reason="sent",
        )
        db.upsert_search_index(
            feed_url=feed_url, guid=old_guid, show="Legacy Show", title="Legacy Ep",
            summary_text="Legacy summary", transcript_text="Legacy transcript",
        )
        assert db.migrate_legacy_synthetic_guid(
            feed_url=feed_url, guid=new_guid, title="Legacy Ep",
            published_at="2026-11-01T00:00:00+00:00", duration_seconds=600,
            audio_url="https://cdn.example/ep.mp3?sig=2",
        )
    finally:
        db.close()

    hits = archive.search_transcripts(tmp_path, query="Legacy", limit=5)
    assert hits["total_matches"] == 1
    assert hits["results"][0]["guid"] == new_guid


def test_email_ui_ux_full_document_preheader_and_timestamp_links():
    summary = (
        "<h2>TL;DR</h2><p>Central banks held rates steady while signaling patience.</p>"
        "<h2>Timestamped highlights</h2>"
        "<ul>"
        "<li><strong>[00:02:15]</strong> — Opening point</li>"
        "<li>[1:05:00] — Deep dive</li>"
        '<li><a href="https://example.com/existing">[00:10:00]</a> — Already linked</li>'
        "</ul>"
    )
    html = pipeline._wrap_email(
        summary,
        "Same Title",
        "Same Title",
        "https://www.youtube.com/watch?v=abc12345678",
        published_at="2026-11-20T00:00:00+00:00",
        duration_seconds=4200,
    )
    assert html.startswith("<!DOCTYPE html><html><head>")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "pd-preheader" in html
    assert "Central banks held rates steady" in html
    assert "<!--[if mso]>" in html
    # Redundant eyebrow omitted when podcast == title.
    assert 'class="pd-eyebrow"' not in html
    # YouTube CTA label.
    assert "Watch on YouTube" in html
    # Deep links for both [00:02:15] and [1:05:00], no nested <a> on existing link.
    assert 'href="https://www.youtube.com/watch?v=abc12345678&amp;t=135s"' in html
    assert 'href="https://www.youtube.com/watch?v=abc12345678&amp;t=3900s"' in html
    assert "<a href=\"https://example.com/existing\"" in html
    assert "<a href=\"https://www.youtube.com/watch?v=abc12345678&amp;t=600s\"" not in html

    plain = _html_to_text(html)
    assert "Watch on YouTube (https://www.youtube.com/watch?v=abc12345678)" in plain
    assert "---" in plain


def test_mcp_search_fts5_total_matches_cjk_quotes_and_get_episode_feed_disambiguation(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://a.example/rss", name="Show Alpha")
        db.upsert_feed(url="https://b.example/rss", name="Show Beta")
        for i in range(4):
            db.insert_episode(
                feed_url="https://a.example/rss", guid=f"ep-{i}",
                title=f"Rates Episode {i}", published_at=f"2026-11-0{i+1}T00:00:00+00:00",
                duration_seconds=1800, audio_url=f"https://cdn.example/{i}.mp3",
            )
            db.set_status(
                feed_url="https://a.example/rss", guid=f"ep-{i}",
                status="emailed", summary_html="<p>Liquidity and bond yields discussion.</p>",
                emailed_at=f"2026-11-0{i+1}T01:00:00+00:00", terminal_reason="sent",
            )
            db.upsert_search_index(
                feed_url="https://a.example/rss", guid=f"ep-{i}",
                show="Show Alpha", title=f"Rates Episode {i}",
                summary_text="Liquidity and bond yields discussion.",
                transcript_text="Central bank liquidity and treasury bond yields.",
            )
        # Same guid "ep-0" on Show Beta + Chinese episode.
        db.insert_episode(
            feed_url="https://b.example/rss", guid="ep-0",
            title="央行政策讨论", published_at="2026-11-10T00:00:00+00:00",
            duration_seconds=1800, audio_url="https://cdn.example/zh.mp3",
        )
        db.set_status(
            feed_url="https://b.example/rss", guid="ep-0",
            status="emailed", summary_html="<p>央行降准释放流动性</p>",
            emailed_at="2026-11-10T01:00:00+00:00", terminal_reason="sent",
        )
        db.upsert_search_index(
            feed_url="https://b.example/rss", guid="ep-0",
            show="Show Beta", title="央行政策讨论",
            summary_text="央行降准释放流动性", transcript_text="央行降准释放流动性",
        )
        # All episodes are indexed; unindexed_searchable_episodes must be empty.
        assert db.unindexed_searchable_episodes() == []
    finally:
        db.close()

    # True total_matches > limit in FTS5 mode.
    res = archive.search_transcripts(tmp_path, query="liquidity", limit=2)
    assert res["engine"] == "fts5"
    assert res["total_matches"] == 4
    assert len(res["results"]) == 2
    assert res["results"][0]["feed_url"] == "https://a.example/rss"

    # Quoted phrase works cleanly.
    res_q = archive.search_transcripts(tmp_path, query='"bond yields"', limit=5)
    assert res_q["total_matches"] == 4

    # CJK search falls back from unicode61 and returns the Chinese episode.
    res_zh = archive.search_transcripts(tmp_path, query="降准", limit=5)
    assert res_zh["total_matches"] == 1
    assert res_zh["results"][0]["show"] == "Show Beta"

    # get_episode disambiguates colliding GUID "ep-0" via feed parameter.
    ep_a = archive.get_episode(tmp_path, query_or_guid="ep-0", feed="Show Alpha")
    ep_b = archive.get_episode(tmp_path, query_or_guid="ep-0", feed="Show Beta")
    assert ep_a["show"] == "Show Alpha"
    assert ep_b["show"] == "Show Beta"


async def test_feeds_io_min_minutes_zero_and_validation(tmp_path: Path):
    f = tmp_path / "feeds.toml"
    f.write_text('recipient = "reader@example.com"\nmin_minutes = 5\n', encoding="utf-8")
    status, _ = await feeds_io.add_feed(
        f, name="Short News", rss_url="https://news.example/rss", min_minutes=0,
    )
    assert status == "added"
    doc = feeds_io.read(f)
    assert doc.min_minutes == 5
    assert doc.feeds[0].min_minutes == 0

    # Blank name must be rejected by FeedsFile validation before writing.
    with pytest.raises(ValidationError):
        await feeds_io.add_feed(f, name="   ", rss_url="https://bad.example/rss")


def test_routing_apple_and_youtube_canonicalization():
    client = httpx.Client()
    try:
        with patch("podcast_digest_mcp.routing.validate_outbound_url"):
            c1 = routing.classify("https://podcasts.apple.com/us/podcast/id123456/", http=client)
            assert c1.kind == "itunes_podcast" and c1.canonical == "123456"

            c2 = routing.classify("https://youtu.be/dQw4w9WgXcQ?t=42", http=client)
            assert c2.kind == "youtube_video"
            assert c2.canonical == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

            with pytest.raises(ValueError, match="unrecognised YouTube video URL"):
                routing.classify("https://www.youtube.com/shorts", http=client)
    finally:
        client.close()


def test_artifact_key_leading_underscore_is_valid_transcriber_id():
    key = pipeline._artifact_key("_123")
    assert key[0].isalnum()
    assert _validate_artifact_id(key) == key


def test_publisher_rejects_non_finite_json_and_accepts_cjk():
    raw_json = json.dumps({
        "segments": [
            {"startTime": float("nan"), "endTime": 1.0, "body": "bad"},
            {"startTime": 0.0, "endTime": 10.0, "body": "good text"},
        ]
    })
    segs = publisher._parse_json(raw_json)
    assert len(segs) == 1 and segs[0].text == "good text"

    cjk_text = "这是一段很长的中文播客逐字稿测试内容" * 40  # ~680 CJK chars -> ~340 word-equivalents
    t = Transcript(text=cjk_text, segments=[Segment(start=0.0, end=120.0, text=cjk_text)])
    assert publisher.implausible(t, declared_seconds=120) is None


def test_summarize_preamble_fence_and_sanitize_title_and_strict_clean():
    raw = "Sure, here is the summary:\n```HTML\n<H2>TL;DR</H2><p>Ok.</p>\n```"
    assert summarize._validate_summary_html(raw) is None
    normalized = summarize._normalize_summary_html(raw)
    assert normalized == "<H2>TL;DR</H2><p>Ok.</p>"

    cleaned = sanitize_summary_html("<html><head><title>Leak</title></head><body><h2>TL;DR</h2><p>Hi</p></body></html>")
    assert "Leak" not in cleaned
    assert "<h2>TL;DR</h2><p>Hi</p>" in cleaned

    class _EmptyChunkCompleter:
        label = "fake"

        def __init__(self):
            self.calls = 0

        def complete(self, **kw):
            self.calls += 1
            return "<think>only reasoning</think>" if self.calls == 1 else "Cleaned chunk 2"

        def config_problem(self):
            return None

    inp = summarize.SummarizationInput(
        podcast_name="Show", title="Ep", published_at="",
        duration_seconds=None, description=None,
        transcript_text="a" * 30, segments=[],
    )
    with pytest.raises(RuntimeError, match="empty content"):
        summarize.clean_transcript(
            inp, completer=_EmptyChunkCompleter(), allow_partial=False, max_chunk_chars=15,
        )


def test_command_parser_ignores_scalars_and_null_text():
    stdout = '42\n"note"\n{"type":"result","text":null}\n{"type":"result","text":"ok"}\n'
    assert _parse_result(stdout) == "ok"
    assert _classify_failure('123\n{"type":"run_failed","reason":"timeout"}\n') == ("timeout", False)


def test_cli_unknown_feed_exits_two(tmp_path: Path, monkeypatch, capsys):
    f = tmp_path / "feeds.toml"
    f.write_text(
        'recipient = "reader@example.com"\n[[feeds]]\nname = "Real Show"\nrss_url = "https://a.example/rss"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PODCAST_DIGEST_CONFIG", str(f))
    monkeypatch.setenv("PODCAST_DIGEST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    rc = cli_main.main(["run", "--dry-run", "--feed", "Typo Show"])
    assert rc == 2
    assert "unknown --feed" in capsys.readouterr().err
