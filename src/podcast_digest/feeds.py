"""RSS feed discovery + parsing.

Two responsibilities:
  1. Resolve a feed name -> RSS URL via the iTunes Search API (when the user
     gives a `search` string instead of a `rss_url`).
  2. Parse an RSS feed and return new episodes that aren't in the DB yet.

iTunes Search is a public, key-free endpoint that podcast clients have used
since 2005. It returns a `feedUrl` field on every podcast hit.
"""
from __future__ import annotations

import calendar
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import feedparser
import httpx

from . import USER_AGENT
from .config import Feed
from .curlfetch import CurlFetchError, fetch_to_file
from .net import validate_outbound_url_async
from .proc import SubprocessTimeout
from .publisher import TranscriptLink, transcript_links

logger = logging.getLogger(__name__)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
SYNTHETIC_GUID_PREFIX = "synthetic:v1:"


@dataclass(frozen=True)
class EpisodeMeta:
    """A single episode parsed off the RSS feed."""

    guid: str
    title: str
    published_at: str            # ISO8601 UTC
    duration_seconds: int | None
    audio_url: str
    audio_type: str | None       # e.g. "audio/mpeg"
    description: str | None      # show notes; HTML possible
    # Publisher transcripts linked from the item (Podcasting 2.0).
    transcripts: tuple[TranscriptLink, ...] = ()


class FeedError(RuntimeError):
    """Wraps any feed-level error (network, parse, no audio)."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def resolve_feed_url(feed: Feed, *, client: httpx.AsyncClient) -> tuple[str, int | None]:
    """Resolve a Feed (which may have `search` only) to a concrete RSS URL.

    Returns (rss_url, itunes_collection_id). itunes_collection_id is None when
    the caller already pinned `rss_url` directly.
    """
    if feed.rss_url:
        return feed.rss_url, None
    if not feed.search:
        raise FeedError(f"feed {feed.name!r} has neither rss_url nor search")

    params: dict[str, str | int] = {
        "term": feed.search, "media": "podcast", "limit": 5, "entity": "podcast",
    }
    r = await client.get(ITUNES_SEARCH_URL, params=params)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        raise FeedError(f"iTunes Search returned no results for {feed.search!r}")
    target = feed.name.lower()
    exact = next(
        (
            r
            for r in results
            if (r.get("collectionName") or "").lower() == target
            or (r.get("trackName") or "").lower() == target
        ),
        None,
    )
    if exact is None:
        # No exact name match — be loud about the guess so misroutes are
        # diagnosable from logs. We still pick the top hit (matches typical
        # podcast-client behavior) but cache it permanently in feeds.url, so
        # log enough context to catch a wrong pick on the first run.
        chosen = results[0]
        candidates = [
            (r.get("collectionName") or r.get("trackName") or "?", r.get("feedUrl"))
            for r in results[:5]
        ]
        logger.warning(
            "feeds.itunes_no_exact_match search=%r name=%r "
            "chosen=%r feedUrl=%r candidates=%s",
            feed.search, feed.name,
            chosen.get("collectionName") or chosen.get("trackName"),
            chosen.get("feedUrl"),
            candidates,
        )
    else:
        chosen = exact
        logger.info(
            "feeds.itunes_resolved name=%r feedUrl=%r collectionId=%s",
            chosen.get("collectionName") or chosen.get("trackName"),
            chosen.get("feedUrl"), chosen.get("collectionId"),
        )

    rss = chosen.get("feedUrl")
    if not rss:
        raise FeedError(f"iTunes result for {feed.search!r} has no feedUrl")
    return rss, chosen.get("collectionId")


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------


async def fetch_page(url: str, *, client: httpx.AsyncClient) -> bytes:
    """Download a page body over the guarded client.

    Some hosts (e.g. Substack behind Cloudflare) fingerprint-block httpx's TLS
    stack with a 403/429 while serving `curl` fine. For those we fall back to a
    curl subprocess (curl ships in the image) rather than giving up.
    """
    await validate_outbound_url_async(url)
    try:
        r = await client.get(url, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.content
    except httpx.HTTPStatusError as e:
        if e.response.status_code not in (403, 429):
            raise
        logger.warning(
            "feeds.fetch_blocked status=%d url=%s — retrying via curl",
            e.response.status_code, url,
        )
        final_url = str(e.response.request.url)
        await validate_outbound_url_async(final_url)
        return await _fetch_rss_curl(final_url)


async def fetch_rss(rss_url: str, *, client: httpx.AsyncClient) -> bytes:
    """Download the raw RSS body. feedparser handles parsing in a separate step."""
    body = await fetch_page(rss_url, client=client)
    # Reject non-feed bodies served with a success status (redirect stub,
    # challenge page): feedparser would quietly parse them into ZERO entries
    # and the feed would silently stop discovering episodes forever. A loud
    # feed.failed beats an invisible one.
    if not _looks_like_feed(body):
        raise FeedError(
            f"body from {rss_url} does not look like an RSS/Atom feed "
            f"(first bytes: {body[:80]!r})"
        )
    return body


# Every feedparser-parseable podcast feed is RSS2 (`<rss`), Atom (`<feed`) or
# legacy RDF (`<rdf`); one of these appears in the prologue. Deliberately
# generous — a false rejection would break a working feed.
_FEED_MARKERS = (b"<rss", b"<feed", b"<rdf")


def _looks_like_feed(body: bytes) -> bool:
    prologue = body[:65536].lower()
    return any(marker in prologue for marker in _FEED_MARKERS)


async def _fetch_rss_curl(rss_url: str) -> bytes:
    """Fetch via a curl subprocess — bypasses TLS-fingerprint blocks that hit
    httpx. The block can land on httpx's first request, so `rss_url` may be an
    unresolved redirect — curlfetch follows the chain one validated hop at a
    time (a bare `-L` would skip outbound-URL validation entirely)."""
    fd, tmp_name = tempfile.mkstemp(prefix="rss-curl-", suffix=".xml")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        await fetch_to_file(
            rss_url, dest=tmp, user_agent=USER_AGENT,
            max_time="30", timeout=45.0, label="curl rss",
        )
        out = tmp.read_bytes()
    except (CurlFetchError, SubprocessTimeout) as e:
        raise FeedError(str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)
    if not out:
        raise FeedError(f"curl fallback returned an empty body for {rss_url}")
    return out


def parse_episodes(rss_body: bytes, *, min_minutes: int = 0) -> list[EpisodeMeta]:
    """Parse RSS bytes into EpisodeMeta entries. Sorted newest-first.

    feedparser is forgiving: it will parse Atom and RSS, with or without iTunes
    namespace. We pull GUID with `id`, audio from `enclosures[0]`, and duration
    from the iTunes namespace where present.
    """
    parsed = feedparser.parse(rss_body)
    if parsed.bozo and not parsed.entries:
        raise FeedError(f"feed unparseable: {parsed.bozo_exception}")

    # feedparser keeps only one <podcast:transcript> per item, and items often
    # list several formats: read the tags straight from the XML. Matched by
    # GUID, else by position (feedparser keeps document order).
    links = transcript_links(rss_body)
    episodes: list[EpisodeMeta] = []
    for index, entry in enumerate(parsed.entries):
        try:
            ep = _entry_to_episode(entry)
        except FeedError as e:
            logger.warning("skipping entry: %s", e)
            continue
        found = links.get(str(entry.get("id") or "")) or links.get(f"#{index}")
        if found:
            ep = replace(ep, transcripts=found)
        if min_minutes and ep.duration_seconds is not None and ep.duration_seconds < min_minutes * 60:
            continue
        episodes.append(ep)

    episodes.sort(key=lambda e: e.published_at, reverse=True)
    return episodes


def _entry_to_episode(entry) -> EpisodeMeta:
    enclosures = entry.get("enclosures") or []
    audio = next((e for e in enclosures if str(e.get("type", "")).startswith("audio")), None)
    if not audio:
        # Some feeds put the only enclosure without a type — fall back to first.
        audio = enclosures[0] if enclosures else None
    if not audio or not audio.get("href"):
        raise FeedError(f"entry has no audio enclosure: {entry.get('title', '?')}")

    title = entry.get("title", "Untitled")
    published_at = _entry_published(entry)
    guid = entry.get("id") or entry.get("guid")
    if not guid:
        guid = _synthetic_guid(
            title=str(title),
            published_at=published_at,
            entry_url=str(entry.get("link") or ""),
            audio_url=str(audio["href"]),
        )
    duration = _entry_duration(entry)
    description = entry.get("summary") or entry.get("description")

    return EpisodeMeta(
        guid=str(guid),
        title=str(title),
        published_at=published_at,
        duration_seconds=duration,
        audio_url=str(audio["href"]),
        audio_type=audio.get("type"),
        description=description,
    )


def _synthetic_guid(
    *, title: str, published_at: str, entry_url: str, audio_url: str,
) -> str:
    """Build a stable identity for an entry whose publisher omitted its GUID.

    Signed podcast-CDN enclosure URLs routinely rotate their query parameters.
    Hash only durable metadata and query-free URL components so a signature
    refresh does not turn the same episode into a new DB row and email.
    """
    locator = entry_url or audio_url
    parts = urlsplit(locator)
    stable_locator = urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, "", "")
    )
    material = "\0".join((title.strip(), published_at, stable_locator))
    return SYNTHETIC_GUID_PREFIX + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _entry_published(entry) -> str:
    """Return an ISO-8601 UTC timestamp, or "" when the date is unknown.

    ALWAYS ISO or empty — never a raw RFC-822 string and never ``now()``. Callers
    sort episodes newest-first by this value (and the DB does ``ORDER BY
    published_at DESC``); a raw string like ``"Tue, 10 Jun 2025…"`` sorts
    lexicographically *above* every ISO ``"2026-…"`` (``'T' > '2'``), and a
    ``now()`` fallback sorts as the newest of all — either way a single
    bad-/undated back-catalog entry would steal the cap-1 slot and pre-suppress
    the genuinely newest episode (sent → never; unrecoverable). Empty sorts
    oldest (correct for an undated entry) and makes the email omit the date line
    rather than fabricating one.
    """
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct:
        # feedparser hands us a UTC `time.struct_time`. Convert via
        # `calendar.timegm` (UTC-aware); `time.mktime` would interpret it as
        # local time and shift the timestamp by the container's TZ offset.
        dt = datetime.fromtimestamp(calendar.timegm(struct), tz=UTC)
        return dt.isoformat(timespec="seconds")
    # feedparser couldn't structure it — try to parse the raw string ourselves
    # (RFC 822 is the RSS norm) before giving up. Normalize to UTC ISO.
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            parsed_dt = parsedate_to_datetime(str(raw))
        except (TypeError, ValueError):
            parsed_dt = None
        if parsed_dt is not None:
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=UTC)
            return parsed_dt.astimezone(UTC).isoformat(timespec="seconds")
    return ""


def _entry_duration(entry) -> int | None:
    # iTunes spec: itunes:duration is "HH:MM:SS", "MM:SS", or seconds as string.
    raw = entry.get("itunes_duration") or entry.get("duration")
    if not raw:
        return None
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, sec = nums[:3]
    return h * 3600 + m * 60 + sec
