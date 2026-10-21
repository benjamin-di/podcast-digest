"""Classify an inbound URL so `ingest` knows which `[[feeds]]` shape to write.

The MCP maps each inbound URL flavour onto a digest `Feed` source:

  - rss             → rss_url=        (subscription)
  - itunes_podcast  → iTunes Lookup   → rss_url=        (subscription)
  - youtube_channel → youtube=        (subscription)
  - youtube_video   → youtube_video=  (one-shot)
  - audio           → audio_url=      (one-shot, a direct .mp3/.m4a URL)
  - itunes_episode  → iTunes Lookup   → audio_url=      (one-shot)

A direct audio URL is stored verbatim; an Apple Podcasts *episode* deep-link
(`/id<coll>?i=<track>`) is resolved to its direct enclosure URL by `apple.py`
before being written, so the digest never has to know about Apple.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import (
    parse_qs,
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

import httpx

from podcast_digest.net import validate_outbound_url

# Share / analytics query params that never affect what gets fetched. Stripped
# from a URL before it's stored so feeds.toml stays clean (and so the per-share
# `?si=` token isn't persisted) and an identical link with a different token
# dedups on the value. Anything matching `utm_*` is also dropped.
_TRACKING_PARAMS = frozenset({"si", "fbclid", "gclid", "mc_cid", "mc_eid"})


def strip_tracking_params(value: str | None) -> str | None:
    """Drop share/analytics query params from a URL. No-op for non-URLs / no query.

    Conservative: only an allowlist of pure-tracking params (+ `utm_*`) is
    removed — never arbitrary query, which could be a signed/required parameter.
    """
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.query:
        return value
    kept = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    return urlunparse(parsed._replace(query=urlencode(kept)))

YOUTUBE_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "youtube-nocookie.com",
})

APPLE_PODCAST_PATH = re.compile(r"/podcast/[^/]+/id(\d+)$")


@dataclass(frozen=True)
class Classification:
    # kind ∈ {rss, itunes_podcast, youtube_channel, youtube_video, audio,
    #         itunes_episode}
    kind: str
    canonical: str          # the URL/value to store (or the iTunes collection id)
    extra: dict             # kind-specific metadata (e.g. episode_track_id)


_WATCH_RE = re.compile(r"^/(?:watch|shorts|live|embed)(?:/[\w-]{11})?")

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_SNIFF_BYTES = 4096


@dataclass(frozen=True)
class _Probe:
    headers: httpx.Headers
    content: bytes


def _probe(
    http: httpx.Client, method: str, url: str, *, body_limit: int = 0,
) -> _Probe:
    """Issue a bounded request while validating every redirect destination."""
    current = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        validate_outbound_url(current)
        with http.stream(
            method, current, follow_redirects=False, timeout=10.0,
        ) as response:
            location = response.headers.get("location")
            if response.status_code in _REDIRECT_STATUSES and location:
                if redirect_count >= _MAX_REDIRECTS:
                    raise ValueError(f"too many redirects while classifying {url}")
                current = urljoin(current, location)
                continue
            content = bytearray()
            if body_limit:
                for chunk in response.iter_bytes(chunk_size=min(body_limit, 4096)):
                    remaining = body_limit - len(content)
                    content.extend(chunk[:remaining])
                    if len(content) >= body_limit:
                        break
            return _Probe(headers=httpx.Headers(response.headers), content=bytes(content))
    raise AssertionError("redirect loop exhausted without returning")


def classify(url: str, *, http: httpx.Client) -> Classification:
    """Sniff a URL and decide which `[[feeds]]` source field it maps to.

    Raises ValueError for unsupported schemes, unrecognised URLs, or kinds
    the upstream pipeline can't handle today (Apple episode `?i=`, raw audio).
    """
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {p.scheme!r}")
    host = (p.hostname or "").lower()
    validate_outbound_url(url)

    # YouTube — distinguish a single watch URL from a channel URL.
    if host in YOUTUBE_HOSTS:
        # /watch?v=, /shorts/, /live/, /embed/, youtu.be/<id> → single video
        if host == "youtu.be" or _WATCH_RE.match(p.path):
            return Classification(kind="youtube_video", canonical=url, extra={})
        # /@handle, /c/<name>, /channel/<UC...>, /user/<name> → channel
        return Classification(kind="youtube_channel", canonical=url, extra={})

    # Apple Podcasts. The page URL carries both the collection id (in /id<N>)
    # and, for an episode deep-link, the episode track id (in ?i=<N>).
    if host.endswith("podcasts.apple.com"):
        m = APPLE_PODCAST_PATH.search(p.path)
        if not m:
            raise ValueError(f"unrecognised Apple Podcasts URL: {url}")
        collection_id = m.group(1)
        episode = parse_qs(p.query).get("i")
        if episode:
            # Single episode → resolved to a direct audio enclosure downstream.
            return Classification(
                kind="itunes_episode", canonical=collection_id,
                extra={"episode_track_id": episode[0]},
            )
        return Classification(kind="itunes_podcast", canonical=collection_id, extra={})

    # Heuristic: anything else over HTTP(S) is either an RSS feed or unsupported.
    # Redirect destinations are re-validated by `_probe`.

    # HEAD probe by content-type, then a short GET body sniff if HEAD fails or
    # the content-type is ambiguous (HTML wrapping pages, CDN error stubs, etc.).
    ct = ""
    try:
        response = _probe(http, "HEAD", url)
        ct = (
            response.headers.get("content-type") or ""
        ).split(";", 1)[0].strip().lower()
    except httpx.HTTPError:
        pass

    if ct in {"application/rss+xml", "application/atom+xml", "application/xml", "text/xml"}:
        return Classification(kind="rss", canonical=url, extra={})
    if ct.startswith("audio/"):
        return Classification(kind="audio", canonical=url, extra={"content_type": ct})

    try:
        response = _probe(http, "GET", url, body_limit=_SNIFF_BYTES)
        head = response.content.decode("utf-8", "replace")
        get_ct = (
            response.headers.get("content-type") or ""
        ).split(";", 1)[0].strip().lower()
    except httpx.HTTPError as e:
        raise ValueError(f"could not classify {url}: GET failed: {e}") from e
    if get_ct.startswith("audio/"):
        return Classification(kind="audio", canonical=url, extra={"content_type": get_ct})
    if "<rss" in head.lower() or "<feed" in head.lower():
        return Classification(kind="rss", canonical=url, extra={})

    raise ValueError(f"unrecognised URL (not RSS, YouTube, or Apple Podcasts): {url}")
