"""Tests for the MCP audio one-shot path: Apple episode lookup + feeds_io."""
from __future__ import annotations

import pytest

pytest.importorskip("tomlkit")  # feeds_io needs tomlkit (the `mcp` extra)

import httpx
import respx

from podcast_digest_mcp import apple, feeds_io

# -- apple.lookup_episode ----------------------------------------------------


@respx.mock
async def test_lookup_episode_matches_trackid():
    respx.get("https://itunes.apple.com/lookup").respond(
        200,
        json={"resultCount": 2, "results": [
            {"wrapperType": "track", "kind": "podcast", "collectionName": "Show"},
            {"wrapperType": "podcastEpisode", "trackId": 999,
             "trackName": "Ep Title", "episodeUrl": "https://cdn.example/ep.mp3",
             "trackTimeMillis": 1874000, "releaseDate": "2026-06-20T08:00:00Z",
             "collectionName": "Show"},
        ]},
    )
    async with httpx.AsyncClient() as http:
        ep = await apple.lookup_episode("111", "999", http=http)
    assert ep.audio_url == "https://cdn.example/ep.mp3"
    assert ep.title == "Ep Title"
    assert ep.duration_seconds == 1874  # 1_874_000 ms // 1000
    assert ep.collection_name == "Show"


@respx.mock
async def test_lookup_episode_not_found_raises():
    respx.get("https://itunes.apple.com/lookup").respond(
        200, json={"resultCount": 1, "results": [
            {"wrapperType": "podcastEpisode", "trackId": 1,
             "episodeUrl": "https://x/y.mp3"}]},
    )
    async with httpx.AsyncClient() as http:
        with pytest.raises(ValueError, match="could not resolve"):
            await apple.lookup_episode("111", "999", http=http)


# -- feeds_io audio_url ------------------------------------------------------

SEED = 'recipient = "me@example.com"\npoll_seconds = 3600\n'


async def test_add_feed_audio_url(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    status, _ = await feeds_io.add_feed(p, name="My Clip", audio_url="https://cdn/x.mp3")
    assert status == "added"
    doc = feeds_io.read(p)
    assert len(doc.feeds) == 1
    assert doc.feeds[0].audio_url == "https://cdn/x.mp3"
    assert doc.feeds[0].name == "My Clip"


async def test_add_feed_audio_url_dedup(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    await feeds_io.add_feed(p, name="A", audio_url="https://cdn/x.mp3")
    status, existing = await feeds_io.add_feed(p, name="B", audio_url="https://cdn/x.mp3")
    assert status == "already_subscribed"
    assert existing == "A"


async def test_add_feed_rejects_two_sources(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        await feeds_io.add_feed(
            p, name="X", audio_url="https://cdn/x.mp3", rss_url="https://f/r.xml",
        )


async def test_add_feed_writes_and_reads_language(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    await feeds_io.add_feed(
        p, name="Example Mandarin Show", youtube="https://youtube.com/@x", language="zh",
    )
    # Round-trips through read()...
    assert feeds_io.read(p).feeds[0].language == "zh"
    # ...and lands as a real key the digest's config.Feed will pick up.
    assert 'language = "zh"' in p.read_text(encoding="utf-8")


async def test_add_feed_omits_language_when_none(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    await feeds_io.add_feed(p, name="EN Show", rss_url="https://f/r.xml", language=None)
    assert feeds_io.read(p).feeds[0].language is None
    assert "language" not in p.read_text(encoding="utf-8")


# -- subscription_list type labeling -----------------------------------------


def test_feed_type_labels_subscriptions_vs_oneoffs():
    pytest.importorskip("fastmcp")  # tools.py imports fastmcp
    from podcast_digest_mcp.tools import _feed_type

    assert _feed_type("rss_url") == "subscription"
    assert _feed_type("search") == "subscription"
    assert _feed_type("youtube") == "subscription"
    assert _feed_type("youtube_video") == "oneoff"
    assert _feed_type("audio_url") == "oneoff"
    assert _feed_type(None) == "subscription"


async def test_append_signals_processing(tmp_path):
    # A successful add bumps the wake sentinel so the digest runs a tick soon.
    pytest.importorskip("fastmcp")
    from podcast_digest.wake import read_wake
    from podcast_digest_mcp import tools

    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    res = await tools._append(p, kind="oneoff_audio", name="Clip", audio_url="https://c/x.mp3")
    assert res["status"] == "added"
    assert res["processing_triggered"] is True
    assert read_wake(p) > 0


async def test_append_strips_tracking_and_dedups_on_clean_value(tmp_path):
    # A `?si=` link is stored clean, and a different `?si=` token for the same
    # video dedups on the stripped value (not just via the name backstop).
    pytest.importorskip("fastmcp")
    from podcast_digest_mcp import tools

    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    r1 = await tools._append(
        p, kind="oneoff_youtube_video", name="Vid A",
        youtube_video="https://youtu.be/exampleVid2?si=AAA",
    )
    assert r1["status"] == "added"
    assert feeds_io.read(p).feeds[0].youtube_video == "https://youtu.be/exampleVid2"
    r2 = await tools._append(
        p, kind="oneoff_youtube_video", name="Vid B",  # different name on purpose
        youtube_video="https://youtu.be/exampleVid2?si=BBB",
    )
    assert r2["status"] == "already_subscribed"  # matched on the stripped value


async def test_append_dedup_does_not_signal(tmp_path):
    # Re-adding an identical entry is a no-op and must NOT bump the sentinel.
    pytest.importorskip("fastmcp")
    from podcast_digest.wake import read_wake
    from podcast_digest_mcp import tools

    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    await tools._append(p, kind="oneoff_audio", name="Clip", audio_url="https://c/x.mp3")
    token_after_first = read_wake(p)
    res = await tools._append(p, kind="oneoff_audio", name="Clip", audio_url="https://c/x.mp3")
    assert res["status"] == "already_subscribed"
    assert read_wake(p) == token_after_first  # unchanged


async def test_append_persists_episode_metadata(tmp_path):
    # An Apple-episode ingest carries the resolved release date + runtime; they
    # must round-trip into the feed entry so the pipeline shows the real date
    # instead of fabricating the processing time.
    pytest.importorskip("fastmcp")
    from podcast_digest_mcp import tools

    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    res = await tools._append(
        p, kind="oneoff_audio", name="Guest Name on Critical Minerals",
        audio_url="https://c/roundtable.mp3",
        published_at="2026-06-10T07:00:00Z", duration_seconds=1440,
        show="Weekly Roundtable",
    )
    assert res["status"] == "added"
    entry = feeds_io.read(p).feeds[0]
    assert entry.published_at == "2026-06-10T07:00:00Z"
    assert entry.duration_seconds == 1440
    assert entry.show == "Weekly Roundtable"


async def test_append_omits_metadata_when_absent(tmp_path):
    # A bare audio URL has no metadata to capture — the keys must stay absent so
    # the pipeline falls back to "" (omit the release line), never a fake date.
    pytest.importorskip("fastmcp")
    from podcast_digest_mcp import tools

    p = tmp_path / "feeds.toml"
    p.write_text(SEED, encoding="utf-8")
    await tools._append(p, kind="oneoff_audio", name="Clip", audio_url="https://c/x.mp3")
    entry = feeds_io.read(p).feeds[0]
    assert entry.published_at is None
    assert entry.duration_seconds is None
