"""Unit tests for the MCP URL classifier."""
from __future__ import annotations

import httpx
import pytest
import respx

from podcast_digest import net
from podcast_digest_mcp.routing import classify, strip_tracking_params


@pytest.mark.parametrize("url,expected", [
    # share/analytics tokens dropped
    ("https://youtu.be/exampleVid2?si=AbCdEfGhIjKlMnOp", "https://youtu.be/exampleVid2"),
    ("https://cdn.example/ep.mp3?utm_source=x&utm_medium=y", "https://cdn.example/ep.mp3"),
    ("https://x.test/a?fbclid=abc&gclid=def", "https://x.test/a"),
    # real params kept; only the tracking one removed
    ("https://feeds.example/show?format=mp3&si=abc", "https://feeds.example/show?format=mp3"),
    # no query / non-URL / nothing to strip → unchanged
    ("https://feeds.example/show", "https://feeds.example/show"),
    ("@exampleshow", "@exampleshow"),
    ("https://feeds.example/show?token=keepme", "https://feeds.example/show?token=keepme"),
])
def test_strip_tracking_params(url, expected):
    assert strip_tracking_params(url) == expected


def test_strip_tracking_params_passthrough_none():
    assert strip_tracking_params(None) is None


@pytest.mark.parametrize("url,kind", [
    ("https://www.youtube.com/watch?v=abc12345DEF", "youtube_video"),
    ("https://youtu.be/xyz98765GHI", "youtube_video"),
    ("https://www.youtube.com/shorts/qrs44455JKL", "youtube_video"),
    ("https://music.youtube.com/watch?v=mus12345DEF", "youtube_video"),
    ("https://www.youtube.com/@lecturehall", "youtube_channel"),
    ("https://www.youtube.com/channel/UCexampleChannelId000000", "youtube_channel"),
    ("https://www.youtube.com/c/SomeChannel", "youtube_channel"),
])
def test_youtube_classification(url, kind):
    with httpx.Client() as http:
        c = classify(url, http=http)
    assert c.kind == kind


def test_classify_apple_podcasts_subscription():
    url = "https://podcasts.apple.com/us/podcast/show-name/id123456"
    with httpx.Client() as http:
        c = classify(url, http=http)
    assert c.kind == "itunes_podcast"
    assert c.canonical == "123456"


def test_classify_apple_episode_url():
    url = "https://podcasts.apple.com/us/podcast/show-name/id123456?i=987654321"
    with httpx.Client() as http:
        c = classify(url, http=http)
    assert c.kind == "itunes_episode"
    assert c.canonical == "123456"  # collection id (from /id<N>)
    assert c.extra["episode_track_id"] == "987654321"  # episode id (from ?i=)


@respx.mock
def test_classify_rss_by_content_type():
    respx.head("https://feed.example/show").respond(
        200, headers={"content-type": "application/rss+xml"},
    )
    with httpx.Client() as http:
        c = classify("https://feed.example/show", http=http)
    assert c.kind == "rss"


@respx.mock
def test_classify_rss_by_body_sniff():
    respx.head("https://feed.example/show").respond(
        200, headers={"content-type": "text/html"},
    )
    respx.get("https://feed.example/show").respond(
        200,
        headers={"content-type": "text/html"},
        content=b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>',
    )
    with httpx.Client() as http:
        c = classify("https://feed.example/show", http=http)
    assert c.kind == "rss"


@respx.mock
def test_classify_audio_url_by_content_type():
    respx.head("https://cdn.example/ep.mp3").respond(
        200, headers={"content-type": "audio/mpeg"},
    )
    with httpx.Client() as http:
        c = classify("https://cdn.example/ep.mp3", http=http)
    assert c.kind == "audio"
    assert c.canonical == "https://cdn.example/ep.mp3"
    assert c.extra["content_type"] == "audio/mpeg"


@respx.mock
def test_classify_audio_url_by_get_when_head_ambiguous():
    # HEAD says octet-stream (some CDNs), GET reveals the audio content-type.
    respx.head("https://cdn.example/ep").respond(
        200, headers={"content-type": "application/octet-stream"},
    )
    respx.get("https://cdn.example/ep").respond(
        200, headers={"content-type": "audio/mp4"}, content=b"\x00\x00",
    )
    with httpx.Client() as http:
        c = classify("https://cdn.example/ep", http=http)
    assert c.kind == "audio"


def test_classify_rejects_non_http():
    with httpx.Client() as http, pytest.raises(ValueError):
        classify("file:///etc/passwd", http=http)


@pytest.mark.parametrize("url", [
    "http://localhost/feed.xml",
    "http://127.0.0.1:8000/feed",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/rss",
    "http://192.168.1.1/feed.xml",
    "http://podcast-digest-mcp.local/feed",
])
def test_classify_rejects_internal_hosts(url):
    # SSRF guard: the arbitrary-URL branch must refuse internal/private targets
    # before issuing any HTTP probe.
    with httpx.Client() as http, pytest.raises(ValueError, match="internal"):
        classify(url, http=http)


@respx.mock
def test_classify_rejects_redirect_to_internal_address():
    public = respx.head("https://feed.example/show").respond(
        302, headers={"location": "http://169.254.169.254/latest/meta-data/"},
    )
    internal = respx.head("http://169.254.169.254/latest/meta-data/").respond(
        200, headers={"content-type": "application/rss+xml"},
    )

    with httpx.Client() as http, pytest.raises(ValueError, match="internal"):
        classify("https://feed.example/show", http=http)

    assert public.called
    assert not internal.called


def test_classify_rejects_hostname_resolving_to_private_address(monkeypatch):
    monkeypatch.setattr(
        net,
        "_resolve_host_addresses",
        lambda _host, _port: (net.ipaddress.ip_address("10.0.0.8"),),
    )
    with httpx.Client() as http, pytest.raises(ValueError, match="internal"):
        classify("https://feed.example/show", http=http)


@respx.mock
def test_classify_body_sniff_stops_after_bounded_prefix():
    class CountingStream(httpx.SyncByteStream):
        def __init__(self):
            self.chunks_read = 0

        def __iter__(self):
            for index in range(100):
                self.chunks_read += 1
                prefix = b"<rss>" if index == 0 else b""
                yield prefix + b"x" * (1024 - len(prefix))

    stream = CountingStream()
    respx.head("https://feed.example/large").respond(
        200, headers={"content-type": "application/octet-stream"},
    )
    respx.get("https://feed.example/large").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            stream=stream,
        )
    )

    with httpx.Client() as http:
        result = classify("https://feed.example/large", http=http)

    assert result.kind == "rss"
    assert stream.chunks_read <= 4
