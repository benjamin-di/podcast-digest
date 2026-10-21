"""YouTube source: URL parsing, channel resolution, Atom parsing, and the
audio.download routing hook."""
from __future__ import annotations

import pytest

from podcast_digest import audio, youtube
from podcast_digest.feeds import FeedError

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(url):
    assert youtube.extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_rejects_garbage():
    with pytest.raises(FeedError):
        youtube.extract_video_id("https://example.com/not-a-video")


def test_canonical_watch_url():
    assert (
        youtube.canonical_watch_url("abc12345678")
        == "https://www.youtube.com/watch?v=abc12345678"
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=x", True),
        ("https://youtu.be/x", True),
        ("http://youtube-nocookie.com/embed/x", True),
        ("https://cdn.example.com/ep.mp3", False),
        ("", False),
        # F3 hardening: only http(s) hosts that ARE a YouTube domain route to
        # yt-dlp. A feed-controlled enclosure that merely *mentions* youtube.com
        # (in the path/query) or uses a non-http scheme must NOT be misrouted —
        # the substring match used to send these to yt-dlp's argv.
        ("https://evil.test/x?ref=youtube.com", False),
        ("https://youtube.com.evil.test/x", False),
        ("--exec=touch /pwned#youtube.com", False),
        ("javascript:alert(1)//youtube.com", False),
        ("file:///etc/passwd?youtube.com", False),
    ],
)
def test_is_youtube_url(url, expected):
    assert youtube.is_youtube_url(url) is expected


# ---------------------------------------------------------------------------
# Channel feed resolution
# ---------------------------------------------------------------------------


async def test_resolve_channel_feed_from_raw_id():
    cid = "UC" + "a" * 22
    out = await youtube.resolve_channel_feed(cid, client=None)
    assert out == f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"


async def test_resolve_channel_feed_from_channel_url():
    cid = "UC" + "b" * 22
    out = await youtube.resolve_channel_feed(
        f"https://www.youtube.com/channel/{cid}", client=None
    )
    assert out.endswith(cid)


async def test_resolve_channel_feed_passthrough_videos_xml():
    url = "https://www.youtube.com/feeds/videos.xml?channel_id=UCxxxxxxxxxxxxxxxxxxxxxx"
    assert await youtube.resolve_channel_feed(url, client=None) == url


async def test_resolve_channel_feed_from_handle(monkeypatch):
    cid = "UC" + "c" * 22

    async def fake_fetch(url, *, client):
        assert "@examplechannel" in url
        return f'... "channelId":"{cid}" ...'.encode()

    monkeypatch.setattr(youtube, "fetch_page", fake_fetch)
    out = await youtube.resolve_channel_feed("@examplechannel", client=None)
    assert out.endswith(cid)


async def test_resolve_channel_feed_prefers_canonical_over_featured(monkeypatch):
    """A channel page embeds featured/related channels' "channelId" entries
    *before* its own canonical id. The resolver must pick the RSS alternate
    link or "externalId", not the first "channelId" in the HTML (this once
    resolved @examplechannel to a featured second channel)."""
    featured = "UC" + "f" * 22
    canonical = "UC" + "m" * 22

    async def fake_fetch(url, *, client):
        return (
            f'"channelId":"{featured}" "channelId":"{featured}" '
            f'"externalId":"{canonical}"'
        ).encode()

    monkeypatch.setattr(youtube, "fetch_page", fake_fetch)
    out = await youtube.resolve_channel_feed("@examplechannel", client=None)
    assert out.endswith(canonical)


async def test_resolve_channel_feed_prefers_rss_link_first(monkeypatch):
    featured = "UC" + "f" * 22
    canonical = "UC" + "m" * 22

    async def fake_fetch(url, *, client):
        return (
            f'"channelId":"{featured}" '
            f'href="https://www.youtube.com/feeds/videos.xml?channel_id={canonical}"'
        ).encode()

    monkeypatch.setattr(youtube, "fetch_page", fake_fetch)
    out = await youtube.resolve_channel_feed("@examplechannel", client=None)
    assert out.endswith(canonical)


# ---------------------------------------------------------------------------
# Channel upload enumeration (yt-dlp --flat-playlist)
# ---------------------------------------------------------------------------

# Two JSON lines as yt-dlp --flat-playlist --dump-json emits them, newest-first.
_FLAT_JSON = (
    b'{"id": "VIDEOID1234", "title": "Newest clip", "duration": 1800.0,'
    b' "playlist_uploader": "Conference Talks"}\n'
    b'{"id": "VIDEOID5678", "title": "Older clip", "duration": 900,'
    b' "playlist_uploader": "Conference Talks"}\n'
)


async def test_list_channel_uploads(monkeypatch):
    captured = {}

    async def fake_stdout(cmd, *, label, **_kw):
        captured["cmd"] = cmd
        return _FLAT_JSON

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    cid = "UC" + "a" * 22
    name, eps = await youtube.list_channel_uploads(
        f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}", limit=5
    )
    # The bare videos.xml PK is turned into a /videos channel URL for yt-dlp,
    # requesting exactly `limit` entries.
    assert captured["cmd"][-1] == f"https://www.youtube.com/channel/{cid}/videos"
    assert captured["cmd"][captured["cmd"].index("--playlist-end") + 1] == "5"
    assert name == "Conference Talks"
    assert [e.guid for e in eps] == ["yt:video:VIDEOID1234", "yt:video:VIDEOID5678"]
    first = eps[0]
    assert first.title == "Newest clip"
    assert first.audio_url == "https://www.youtube.com/watch?v=VIDEOID1234"
    assert first.duration_seconds == 1800  # flat gives duration (the Atom feed didn't)
    assert first.published_at == ""        # absent in flat mode; enriched per-episode
    assert first.audio_type == "video/youtube"


async def test_list_channel_uploads_no_channel_id_raises(monkeypatch):
    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    with pytest.raises(FeedError):
        await youtube.list_channel_uploads("https://www.youtube.com/playlist?list=PLxyz")


async def test_list_channel_uploads_empty_channel_is_not_an_error(monkeypatch):
    # A channel with no uploads → yt-dlp exits 0 with empty stdout → zero
    # episodes, NOT a feed error (matches the old empty-Atom-feed behavior).
    async def fake_stdout(cmd, *, label, **_kw):
        return b""

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    name, eps = await youtube.list_channel_uploads("UC" + "a" * 22)
    assert name is None
    assert eps == []


async def test_list_channel_uploads_skips_entries_without_id(monkeypatch):
    async def fake_stdout(cmd, *, label, **_kw):
        return b'{"title": "no id", "duration": 100}\n{"id": "GOODID00001", "duration": 200}\n'

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    _name, eps = await youtube.list_channel_uploads("UC" + "a" * 22)
    assert [e.guid for e in eps] == ["yt:video:GOODID00001"]


async def test_list_channel_uploads_filters_non_downloadable(monkeypatch):
    """F10: a live stream / scheduled premiere / members-only video must not
    occupy the newest slot and pre-suppress the real newest upload behind it."""
    lines = [
        b'{"id": "LIVE00000001", "title": "LIVE now", "live_status": "is_live", "duration": null}',
        b'{"id": "PREMIERE0001", "title": "Premiere", "live_status": "is_upcoming", "duration": null}',
        b'{"id": "NODUR0000001", "title": "durationless", "duration": null}',
        b'{"id": "MEMBERS00001", "title": "members only", "availability": "subscriber_only", "duration": 900}',
        b'{"id": "REALVIDEO001", "title": "Real upload", "duration": 1800}',
    ]

    async def fake_stdout(cmd, *, label, **_kw):
        return b"\n".join(lines) + b"\n"

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    _name, eps = await youtube.list_channel_uploads("UC" + "a" * 22)
    assert [e.guid for e in eps] == ["yt:video:REALVIDEO001"]

    # Members-only stays skipped even WITH cookies: memberships are per-channel
    # purchases, so a global cookies file can't prove this channel's
    # subscriber_only video is downloadable — admitting it would recreate the
    # exact slot poisoning (fail every tick, pre-suppress the real newest).
    _name, eps = await youtube.list_channel_uploads("UC" + "a" * 22, cookies="/c.txt")
    assert [e.guid for e in eps] == ["yt:video:REALVIDEO001"]


async def test_list_channel_uploads_all_durationless_falls_back(monkeypatch):
    """R4/F10: if flat mode returns NO durations for any entry (a yt-dlp quirk,
    not a channel of live streams), don't blank the feed — keep the entries."""
    lines = [
        b'{"id": "VIDA00000001", "title": "A"}',
        b'{"id": "VIDB00000002", "title": "B"}',
    ]

    async def fake_stdout(cmd, *, label, **_kw):
        return b"\n".join(lines) + b"\n"

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    _name, eps = await youtube.list_channel_uploads("UC" + "a" * 22)
    # Kept (newest-first), not dropped.
    assert [e.guid for e in eps] == ["yt:video:VIDA00000001", "yt:video:VIDB00000002"]
    # But a positively-live entry is STILL dropped even in the fallback set.
    lines2 = [
        b'{"id": "LIVE00000001", "title": "live", "live_status": "is_live"}',
        b'{"id": "VIDC00000003", "title": "C"}',
    ]

    async def fake_stdout2(cmd, *, label, **_kw):
        return b"\n".join(lines2) + b"\n"

    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout2)
    _name, eps = await youtube.list_channel_uploads("UC" + "a" * 22)
    assert [e.guid for e in eps] == ["yt:video:VIDC00000003"]


async def test_list_channel_uploads_passes_end_of_options_separator(monkeypatch):
    """F3: the channel URL is passed after `--` so a `-`-leading value can never
    be parsed as a yt-dlp option."""
    captured = {}

    async def fake_stdout(cmd, *, label, **_kw):
        captured["cmd"] = cmd
        return b""

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_stdout", fake_stdout)
    await youtube.list_channel_uploads("UC" + "a" * 22)
    cmd = captured["cmd"]
    assert cmd[-2] == "--"
    assert cmd[-1].startswith("https://www.youtube.com/channel/")


# ---------------------------------------------------------------------------
# audio.download routing
# ---------------------------------------------------------------------------


async def test_audio_download_routes_youtube(monkeypatch, tmp_path):
    captured = {}

    async def fake_dl(url, *, dest_dir, basename, fmt, cookies):
        captured.update(url=url, fmt=fmt, cookies=cookies)
        p = dest_dir / f"{basename}.mp3"
        p.write_bytes(b"yt-audio")
        return p

    monkeypatch.setattr(youtube, "download_audio", fake_dl)
    out = await audio.download(
        "https://youtu.be/VIDEOID1234",
        dest_dir=tmp_path,
        basename="ep",
        ytdlp_format="bestaudio",
        ytdlp_cookies="/tmp/c.txt",
    )
    assert out.read_bytes() == b"yt-audio"
    assert captured == {
        "url": "https://youtu.be/VIDEOID1234",
        "fmt": "bestaudio",
        "cookies": "/tmp/c.txt",
    }


async def test_download_audio_atomic_on_success(monkeypatch, tmp_path):
    """F11: yt-dlp writes to a `.dl` temp basename and is atomically renamed to
    the final name only on a clean exit."""
    async def fake_run(cmd, *, timeout, label, **_kw):
        # yt-dlp's -o points at the temp basename; simulate it writing the mp3.
        o = cmd[cmd.index("-o") + 1]
        (tmp_path / o.replace("%(ext)s", "mp3").split("/")[-1]).write_bytes(b"full-audio")
        return 0, b"", b""

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "run_capture", fake_run)
    out = await youtube.download_audio(
        "https://youtu.be/x", dest_dir=tmp_path, basename="ep",
    )
    assert out == tmp_path / "ep.mp3"
    assert out.read_bytes() == b"full-audio"
    # No temp leftovers.
    assert list(tmp_path.glob("ep.dl*")) == []


async def test_download_audio_no_truncated_cache_on_failure(monkeypatch, tmp_path):
    """F11: a non-zero exit that left a partial mp3 must NOT leave a `basename.mp3`
    that the next tick would accept as cached (half an episode transcribed)."""
    async def fake_run(cmd, *, timeout, label, **_kw):
        # yt-dlp partially writes then dies: a truncated temp mp3 remains.
        o = cmd[cmd.index("-o") + 1]
        (tmp_path / o.replace("%(ext)s", "mp3").split("/")[-1]).write_bytes(b"HALF")
        return 1, b"", b"boom"

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "run_capture", fake_run)
    with pytest.raises(FeedError):
        await youtube.download_audio(
            "https://youtu.be/x", dest_dir=tmp_path, basename="ep",
        )
    # The final path was never created, and the truncated temp is cleaned up.
    assert not (tmp_path / "ep.mp3").exists()
    assert list(tmp_path.glob("ep.dl*")) == []


async def test_audio_download_non_youtube_uses_http(monkeypatch, tmp_path):
    async def fake_stream(client, audio_url, dest_dir, basename):
        p = dest_dir / f"{basename}.mp3"
        p.write_bytes(b"http-audio")
        return p

    monkeypatch.setattr(audio, "_download_stream", fake_stream)
    out = await audio.download(
        "https://cdn.example.com/ep.mp3", dest_dir=tmp_path, basename="ep",
    )
    assert out.read_bytes() == b"http-audio"


async def test_download_audio_timeout_keeps_partial_for_resume(monkeypatch, tmp_path):
    """The download timeout is a per-tick time slice, not a death sentence: a
    throttled pull killed at the bound keeps its `.part` bytes so yt-dlp's
    default --continue resumes next tick instead of restarting from zero five
    times and then dropping the episode."""
    from podcast_digest.proc import SubprocessTimeout

    partial = tmp_path / "ep.dl.webm.part"
    tmp_final = tmp_path / "ep.dl.mp3"

    async def fake_run_capture(cmd, *, timeout, label, **_kw):
        partial.write_bytes(b"x" * 1000)   # bytes downloaded before the kill
        tmp_final.write_bytes(b"trunc")    # half-written ffmpeg output
        raise SubprocessTimeout(f"{label} timed out after {timeout}s")

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "run_capture", fake_run_capture)
    with pytest.raises(FeedError, match="resumes next tick"):
        await youtube.download_audio(
            "https://www.youtube.com/watch?v=VIDEOID1234",
            dest_dir=tmp_path, basename="ep",
        )
    assert partial.exists()          # NOT cleaned up — resume material
    # The postprocess output MUST die: yt-dlp skips the transcode when its
    # output already exists, so a kept truncated mp3 would be promoted to the
    # final file by the next tick's rc==0 check (the F11 truncated-cache bug).
    assert not tmp_final.exists()
    # A real yt-dlp failure (rc != 0) still cleans: that state may be poisoned.

    async def fake_run_capture_rc1(cmd, *, timeout, label, **_kw):
        partial.write_bytes(b"x")
        return 1, b"", b"some extractor error"

    monkeypatch.setattr(youtube, "run_capture", fake_run_capture_rc1)
    with pytest.raises(FeedError, match="rc=1"):
        await youtube.download_audio(
            "https://www.youtube.com/watch?v=VIDEOID1234",
            dest_dir=tmp_path, basename="ep",
        )
    assert not partial.exists()


async def test_single_video_episode_rejects_live_and_upcoming(monkeypatch):
    """Full (non-flat) metadata carries a reliable live_status — the
    authoritative gate that catches an untagged live/premiere readmitted by the
    flat-mode all-durationless quirk fallback before it can take the cap-1 slot."""
    import json as _json

    payload = {"id": "LIVEVIDEO001", "title": "Premiere!",
               "live_status": "is_upcoming", "duration": None}

    async def fake_run_capture(cmd, *, timeout, label, **_kw):
        return 0, _json.dumps(payload).encode(), b""

    monkeypatch.setattr(youtube, "_require_yt_dlp", lambda: None)
    monkeypatch.setattr(youtube, "run_capture", fake_run_capture)
    with pytest.raises(FeedError, match="live stream/premiere"):
        await youtube.single_video_episode("https://www.youtube.com/watch?v=LIVEVIDEO001")
    # Once the premiere has aired it looks like a normal VOD → admitted.
    payload.update(live_status="was_live", duration=3600, upload_date="20260701")
    ep = await youtube.single_video_episode("https://www.youtube.com/watch?v=LIVEVIDEO001")
    assert ep.duration_seconds == 3600


async def test_resolve_channel_feed_reads_a_real_html_channel_page():
    """Regression: the channel page is HTML, not a feed. Resolution must not go
    through the RSS fetcher (which rejects anything that isn't RSS/Atom)."""
    import httpx
    import respx

    cid = "UC" + "d" * 22
    page = (
        "<!DOCTYPE html><html><head>"
        f'<link rel="alternate" type="application/rss+xml" '
        f'href="https://www.youtube.com/feeds/videos.xml?channel_id={cid}">'
        "</head><body>channel</body></html>"
    )
    with respx.mock() as router:
        router.get("https://www.youtube.com/@examplechannel").mock(
            return_value=httpx.Response(200, text=page, headers={"content-type": "text/html"}),
        )
        async with httpx.AsyncClient() as client:
            out = await youtube.resolve_channel_feed(
                "https://www.youtube.com/@examplechannel", client=client,
            )
    assert out.endswith(cid)
