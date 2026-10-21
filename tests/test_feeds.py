from pathlib import Path

import httpx
import pytest

from podcast_digest.feeds import (
    SYNTHETIC_GUID_PREFIX,
    _entry_duration,
    _entry_published,
    parse_episodes,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_episodes_returns_newest_first():
    body = (FIXTURES / "sample_feed.xml").read_bytes()
    eps = parse_episodes(body)
    assert [e.guid for e in eps] == ["ep-3-guid", "ep-2-guid", "ep-1-guid"]
    # Duration parsed in three different formats
    assert eps[0].duration_seconds == 5400      # 01:30:00
    assert eps[1].duration_seconds == 45 * 60   # 45:00
    assert eps[2].duration_seconds == 180       # integer seconds
    # Description from CDATA survived
    assert eps[0].description and "episode three" in eps[0].description.lower()


def test_entry_published_always_iso_or_empty():
    """F6: _entry_published never returns a raw RFC-822 string (which would sort
    lexicographically above every ISO date) nor a fabricated now() — only an ISO
    UTC timestamp or "" (which sorts oldest)."""
    # feedparser-structured date → ISO.
    import time
    struct = time.struct_time((2025, 6, 10, 8, 0, 0, 0, 0, 0))
    assert _entry_published({"published_parsed": struct}) == "2025-06-10T08:00:00+00:00"
    # Raw RFC-822 string feedparser couldn't structure → parsed to ISO UTC.
    assert _entry_published(
        {"published": "Tue, 10 Jun 2025 08:00:00 GMT"}
    ) == "2025-06-10T08:00:00+00:00"
    # Unparseable → "" (NOT the raw string, NOT now()).
    assert _entry_published({"published": "sometime last week"}) == ""
    # Missing entirely → "" (NOT now()).
    assert _entry_published({}) == ""


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # feedparser date fallback
def test_bad_dated_backcatalog_does_not_steal_newest_slot():
    """F6 regression: an entry with an unparseable/missing date must sort OLDEST,
    not newest — otherwise it steals the cap-1 slot and pre-suppresses the real
    newest episode. Build a feed where a bad-dated item precedes ISO ones."""
    feed = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>T</title>
      <item><title>Undated teaser</title><guid>bad</guid>
        <pubDate>garbage date</pubDate>
        <enclosure url="https://cdn/bad.mp3" type="audio/mpeg" length="1"/></item>
      <item><title>Genuinely newest</title><guid>new</guid>
        <pubDate>Wed, 02 Jul 2026 10:00:00 GMT</pubDate>
        <enclosure url="https://cdn/new.mp3" type="audio/mpeg" length="1"/></item>
      <item><title>Older</title><guid>old</guid>
        <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
        <enclosure url="https://cdn/old.mp3" type="audio/mpeg" length="1"/></item>
    </channel></rss>"""
    eps = parse_episodes(feed)
    # Real newest first; the bad-dated entry sinks to the bottom (oldest).
    assert [e.guid for e in eps] == ["new", "old", "bad"]
    assert eps[-1].published_at == ""


def test_parse_episodes_filters_by_min_minutes():
    body = (FIXTURES / "sample_feed.xml").read_bytes()
    eps = parse_episodes(body, min_minutes=15)
    assert [e.guid for e in eps] == ["ep-3-guid", "ep-2-guid"]


def test_parse_episodes_audio_url_and_type():
    body = (FIXTURES / "sample_feed.xml").read_bytes()
    eps = parse_episodes(body)
    assert eps[0].audio_url == "https://cdn.example.com/ep3.mp3"
    assert eps[0].audio_type == "audio/mpeg"


def test_missing_guid_ignores_rotating_enclosure_query():
    def body(signature: str) -> bytes:
        return f"""<?xml version="1.0"?>
        <rss version="2.0"><channel><title>T</title><item>
          <title>Unsigned episode</title>
          <pubDate>Wed, 22 Jul 2026 10:00:00 GMT</pubDate>
          <enclosure url="https://cdn.example/audio.mp3?signature={signature}"
                     type="audio/mpeg" length="1"/>
        </item></channel></rss>""".encode()

    first = parse_episodes(body("first"))[0]
    second = parse_episodes(body("rotated"))[0]
    assert first.guid.startswith(SYNTHETIC_GUID_PREFIX)
    assert first.guid == second.guid
    assert first.audio_url != second.audio_url


def test_explicit_guid_is_preserved_verbatim():
    body = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>T</title><item>
      <title>Explicit</title><guid>publisher-guid-123</guid>
      <enclosure url="https://cdn.example/audio.mp3?signature=x"
                 type="audio/mpeg" length="1"/>
    </item></channel></rss>"""
    assert parse_episodes(body)[0].guid == "publisher-guid-123"


def test_entry_duration_formats():
    # _entry_duration accepts a Mapping-like with .get; FeedParserDict satisfies this.
    # The function is private but pivotal — exercise the format conversions directly.
    assert _entry_duration({"itunes_duration": "01:30:00"}) == 5400
    assert _entry_duration({"itunes_duration": "45:00"}) == 2700
    assert _entry_duration({"itunes_duration": "30"}) == 30
    assert _entry_duration({"itunes_duration": "garbage"}) is None
    assert _entry_duration({}) is None


# ---------------------------------------------------------------------------
# fetch_rss: curl fallback redirect-following + non-feed body rejection
# ---------------------------------------------------------------------------

RSS_MIN = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
STUB = b"Temporary Redirect. Redirecting to https://cdn.example/feed.rss"


def _scripted_curl(monkeypatch, hops):
    """Scripted curl for curlfetch: each hop is (status, redirect_url, body)."""
    calls = []

    async def fake_run_capture(cmd, *, timeout, label, env=None):
        status, redirect_url, body = hops[len(calls)]
        calls.append(cmd[-1])
        Path(cmd[cmd.index("-o") + 1]).write_bytes(body)
        return 0, f"{status}\t{redirect_url}".encode(), b""

    monkeypatch.setattr("podcast_digest.curlfetch.run_capture", fake_run_capture)
    return calls


async def test_fetch_rss_curl_follows_redirect(monkeypatch):
    """A 403-blocked host whose URL is itself a redirect (the Substack shape):
    the curl fallback must follow the validated hop and return the real feed."""
    from podcast_digest import feeds

    calls = _scripted_curl(monkeypatch, [
        (307, "https://cdn.example/feed.rss", STUB),
        (200, "", RSS_MIN),
    ])
    transport = httpx.MockTransport(lambda req: httpx.Response(403))
    async with httpx.AsyncClient(transport=transport) as client:
        body = await feeds.fetch_rss("https://api.substack.example/feed.rss", client=client)
    assert body == RSS_MIN
    assert calls == [
        "https://api.substack.example/feed.rss",
        "https://cdn.example/feed.rss",
    ]


async def test_fetch_rss_rejects_non_feed_body(monkeypatch):
    """A success status carrying a non-feed body (stub, challenge page) must
    raise loudly — feedparser would silently yield zero entries forever."""
    from podcast_digest import feeds

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"<html>Just a moment...</html>")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(feeds.FeedError, match="does not look like"):
            await feeds.fetch_rss("https://example.com/feed.xml", client=client)


async def test_fetch_rss_curl_stub_rejected(monkeypatch):
    """End-to-end: httpx 403 → curl fallback serves a text stub with 200 →
    the feed-body guard converts silent zero-entries into a loud FeedError."""
    from podcast_digest import feeds

    _scripted_curl(monkeypatch, [(200, "", STUB)])
    transport = httpx.MockTransport(lambda req: httpx.Response(403))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(feeds.FeedError, match="does not look like"):
            await feeds.fetch_rss("https://api.substack.example/feed.rss", client=client)
