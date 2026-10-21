"""The MCP tools mobile clients call: ingest, preview, subscription_list,
subscription_remove, process_now.

Every tool reads or writes the shared config dir — the digest container does
everything downstream. No DB writes, no background tasks: a feed appended to
`feeds.toml` is processed by the digest (the bind-mounted config dir means the
running digest sees the new file without a restart). `ingest` additionally bumps
a wake sentinel (`process_now` does so on demand) so the digest runs a tick
within seconds instead of waiting out the poll interval — see podcast_digest.wake.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import httpx
from fastmcp import FastMCP

from podcast_digest.feeds import fetch_rss, parse_episodes
from podcast_digest.net import validate_outbound_url_async
from podcast_digest.wake import signal_wake
from podcast_digest.youtube import (
    list_channel_uploads,
    resolve_channel_feed,
    single_video_episode,
)

from . import apple, feeds_io, lang, routing

logger = logging.getLogger(__name__)


# Annotations tell clients which tools only read and which change state, so
# they can confirm destructive calls (MCP 2025-03-26; honoured by claude.ai,
# Claude Code, ChatGPT and Codex).
def register(
    mcp: FastMCP,
    *,
    feeds_path: Path,
    http: httpx.AsyncClient,
    http_sync: httpx.Client,
) -> None:
    """Wire the four tools onto the FastMCP app."""

    # -- ingest -----------------------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def ingest(
        url: str,
        name: str | None = None,
        min_minutes: int = 0,
        language: str | None = None,
    ) -> dict:
        """Add a URL to feeds.toml.

        Smart routing:
        * RSS feed URL → subscription, source = `rss_url`
        * Apple Podcasts podcast page (`id<N>` without `?i=`) → resolved to
          RSS via iTunes Lookup, source = `rss_url`
        * YouTube channel URL (`/@handle`, `/channel/UC...`, `/c/...`,
          `/user/...`) → subscription, source = `youtube`
        * YouTube watch URL (`/watch?v=`, `/shorts/`, `youtu.be/...`) →
          one-shot, source = `youtube_video` (processed exactly once)
        * Direct audio URL (`.mp3`/`.m4a`, `audio/*`) → one-shot,
          source = `audio_url`
        * Apple Podcasts *episode* URL (`/id<N>?i=<trackId>`) → resolved to
          its direct enclosure via iTunes Lookup → one-shot, source = `audio_url`

        Language: leave `language` unset and the spoken language is auto-detected
        (RSS `<language>`, else the script of the latest titles) — a non-English
        feed transcribes with the multilingual model and still emails an English
        summary. Pass an ISO-639-1 code (e.g. `"zh"`, `"ja"`) to force it, or
        `"en"` to force the English path. `preview` reports what would be
        detected, so call it first if unsure.

        Idempotent: if an identical entry (same source value or same name)
        already exists, returns `already_subscribed` instead of appending.
        """
        try:
            c = await asyncio.to_thread(routing.classify, url, http=http_sync)
        except ValueError as e:
            return {"error": str(e)}

        if c.kind == "rss":
            title, declared, titles = await _rss_meta(c.canonical, http=http)
            final_name = name or title or c.canonical
            lang_code = lang.resolve_language(language, declared, [final_name, *titles])
            return await _append(
                feeds_path, kind="subscription",
                name=final_name, rss_url=c.canonical, min_minutes=min_minutes,
                language=lang_code,
            )

        if c.kind == "itunes_podcast":
            p = await apple.lookup_podcast(c.canonical, http=http)
            if error := await _publisher_url_error(p.rss_url):
                return {"error": error}
            _title, declared, titles = await _rss_meta(p.rss_url, http=http)
            final_name = name or p.name
            lang_code = lang.resolve_language(language, declared, [final_name, *titles])
            return await _append(
                feeds_path, kind="subscription",
                name=final_name, rss_url=p.rss_url, min_minutes=min_minutes,
                language=lang_code,
            )

        if c.kind == "youtube_channel":
            channel_feed = await resolve_channel_feed(c.canonical, client=http)
            ch_name, titles = await _channel_meta(channel_feed)
            channel_name = name or ch_name or c.canonical
            lang_code = lang.resolve_language(language, None, [channel_name, *titles])
            return await _append(
                feeds_path, kind="subscription",
                name=channel_name, youtube=c.canonical, min_minutes=min_minutes,
                language=lang_code,
            )

        if c.kind == "youtube_video":
            video_name = name or await _youtube_video_title(c.canonical) or c.canonical
            lang_code = lang.resolve_language(language, None, [video_name])
            return await _append(
                feeds_path, kind="oneoff_youtube_video",
                name=video_name, youtube_video=c.canonical, min_minutes=min_minutes,
                language=lang_code,
            )

        if c.kind == "audio":
            audio_name = name or await _audio_title(c.canonical, http=http) or c.canonical
            lang_code = lang.resolve_language(language, None, [audio_name])
            return await _append(
                feeds_path, kind="oneoff_audio",
                name=audio_name, audio_url=c.canonical, min_minutes=min_minutes,
                language=lang_code,
            )

        if c.kind == "itunes_episode":
            try:
                ep = await apple.lookup_episode(
                    c.canonical, c.extra["episode_track_id"], http=http,
                )
            except ValueError as e:
                return {"error": str(e)}
            if error := await _publisher_url_error(ep.audio_url):
                return {"error": error}
            final_name = name or ep.title
            lang_code = lang.resolve_language(
                language, None, [final_name, ep.collection_name or ""],
            )
            return await _append(
                feeds_path, kind="oneoff_audio",
                name=final_name, audio_url=ep.audio_url, min_minutes=min_minutes,
                language=lang_code,
                # Persist the resolved metadata — a bare enclosure can't be
                # re-resolved at processing time, so the pipeline would otherwise
                # stamp the email with the processing date and no runtime. `show`
                # is the podcast name (the one-shot's `name` is the episode title).
                published_at=ep.published_at, duration_seconds=ep.duration_seconds,
                show=ep.collection_name,
            )

        return {"error": f"unhandled classification: {c.kind}"}

    # -- preview ----------------------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    async def preview(url: str) -> dict:
        """Dry-run of `ingest`: resolve the URL and describe what would be added.

        No file writes. For an RSS feed or Apple Podcasts podcast page, returns
        the feed title + the last 3 episode titles, and whether those episodes
        link a publisher transcript (used instead of local transcription). For a YouTube channel, the
        channel name + last 3 video titles. For a YouTube video, title +
        duration via yt-dlp. For a direct audio URL, the derived name +
        content-type + size. For an Apple Podcasts episode, the resolved title,
        duration, and enclosure URL.
        """
        try:
            c = await asyncio.to_thread(routing.classify, url, http=http_sync)
        except ValueError as e:
            return {"error": str(e)}

        if c.kind == "rss":
            body = await fetch_rss(c.canonical, client=http)
            episodes = parse_episodes(body)
            name = _rss_title_from_body(body) or "(unknown)"
            titles = [e.title for e in episodes[:3]]
            return {
                "kind": "subscription",
                "source": "rss_url",
                "name": name,
                "rss_url": c.canonical,
                "latest_episodes": titles,
                # Whether recent episodes link a publisher transcript, which the
                # digest then uses instead of transcribing the audio.
                "publisher_transcripts": any(e.transcripts for e in episodes[:3]),
                "language": lang.resolve_language(
                    None, _rss_language_from_body(body), [name, *titles],
                ),
            }

        if c.kind == "itunes_podcast":
            p = await apple.lookup_podcast(c.canonical, http=http)
            if error := await _publisher_url_error(p.rss_url):
                return {"error": error}
            body = await fetch_rss(p.rss_url, client=http)
            episodes = parse_episodes(body)
            titles = [e.title for e in episodes[:3]]
            return {
                "kind": "subscription",
                "source": "rss_url",
                "name": p.name,
                "rss_url": p.rss_url,
                "latest_episodes": titles,
                # Whether recent episodes link a publisher transcript, which the
                # digest then uses instead of transcribing the audio.
                "publisher_transcripts": any(e.transcripts for e in episodes[:3]),
                "language": lang.resolve_language(
                    None, _rss_language_from_body(body), [p.name, *titles],
                ),
            }

        if c.kind == "youtube_channel":
            channel_feed = await resolve_channel_feed(c.canonical, client=http)
            channel_name, episodes = await list_channel_uploads(channel_feed, limit=5)
            titles = [e.title for e in episodes]
            return {
                "kind": "subscription",
                "source": "youtube",
                "name": channel_name or c.canonical,
                "youtube": c.canonical,
                "latest_episodes": titles[:3],
                "language": lang.resolve_language(None, None, [channel_name or "", *titles]),
            }

        if c.kind == "youtube_video":
            video_ep = await single_video_episode(c.canonical)
            return {
                "kind": "oneoff_youtube_video",
                "source": "youtube_video",
                "name": video_ep.title,
                "youtube_video": c.canonical,
                "duration_seconds": video_ep.duration_seconds,
                "language": lang.resolve_language(None, None, [video_ep.title]),
            }

        if c.kind == "audio":
            head = await _audio_head(c.canonical, http=http)
            audio_name = _audio_name(c.canonical, head) or c.canonical
            return {
                "kind": "oneoff_audio",
                "source": "audio_url",
                "name": audio_name,
                "audio_url": c.canonical,
                "content_type": head.get("content_type") or c.extra.get("content_type"),
                "size_bytes": head.get("size_bytes"),
                "language": lang.resolve_language(None, None, [audio_name]),
            }

        if c.kind == "itunes_episode":
            try:
                apple_ep = await apple.lookup_episode(
                    c.canonical, c.extra["episode_track_id"], http=http,
                )
            except ValueError as e:
                return {"error": str(e)}
            if error := await _publisher_url_error(apple_ep.audio_url):
                return {"error": error}
            return {
                "kind": "oneoff_audio",
                "source": "audio_url",
                "name": apple_ep.title,
                "collection": apple_ep.collection_name,
                "audio_url": apple_ep.audio_url,
                "duration_seconds": apple_ep.duration_seconds,
                "published_at": apple_ep.published_at,
                "language": lang.resolve_language(
                    None, None, [apple_ep.title, apple_ep.collection_name or ""],
                ),
            }

        return {"error": f"unhandled classification: {c.kind}"}

    # -- subscription_list ------------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    async def subscription_list() -> dict:
        """Read feeds.toml and return every entry with its source kind and type.

        feeds.toml holds two kinds of entry, distinguished by `type`:
        * `subscription` — recurring (`rss_url` / `search` / `youtube`); the
          newest episode is picked up every poll tick, indefinitely.
        * `oneoff` — a single episode (`youtube_video` / `audio_url`); processed
          exactly once, then deduped. Its entry lingers as a record of what was
          ingested; use `subscription_remove(name)` to drop it.
        """
        doc = await asyncio.to_thread(feeds_io.read, feeds_path)
        items = []
        for f in doc.feeds:
            source = _source_of(f)
            items.append({
                "name": f.name,
                "type": _feed_type(source),
                "source": source,
                "value": getattr(f, source) if source else None,
                "min_minutes": f.min_minutes,
                "language": f.language,
            })
        counts = {
            "subscriptions": sum(1 for it in items if it["type"] == "subscription"),
            "oneoffs": sum(1 for it in items if it["type"] == "oneoff"),
        }
        return {
            "recipient": doc.recipient,
            "poll_seconds": doc.poll_seconds,
            "counts": counts,
            "feeds": items,
        }

    # -- subscription_remove ---------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False})
    async def subscription_remove(name: str) -> dict:
        """Remove every feeds.toml entry whose `name` matches exactly."""
        removed = await feeds_io.remove_feed(feeds_path, name=name)
        return {"removed": removed, "name": name}

    # -- process_now ------------------------------------------------------

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def process_now() -> dict:
        """Trigger the digest to run a processing tick now, instead of waiting
        for the next scheduled poll.

        `ingest` already does this automatically after adding a feed/episode;
        use this to force a run manually (e.g. after editing feeds.toml directly,
        or to retry a feed that failed). The digest picks up the signal within a
        few seconds and downloads/transcribes/summarizes/emails any pending
        episodes. Safe to call repeatedly: ticks are single-flight (a run already
        in progress is never duplicated) and a no-op when nothing is pending.
        """
        return {"triggered": _signal_processing(feeds_path)}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _publisher_url_error(url: str) -> str | None:
    """Validate a URL learned from third-party metadata before persisting it."""
    try:
        await validate_outbound_url_async(url)
    except ValueError as exc:
        return f"refusing unsafe publisher URL: {exc}"
    return None


async def _append(
    feeds_path: Path,
    *,
    kind: str,
    name: str,
    rss_url: str | None = None,
    youtube: str | None = None,
    youtube_video: str | None = None,
    audio_url: str | None = None,
    min_minutes: int = 0,
    language: str | None = None,
    published_at: str | None = None,
    duration_seconds: int | None = None,
    show: str | None = None,
) -> dict:
    # Strip share/analytics params before storing so feeds.toml stays clean and
    # the same link with a different `?si=`/`utm_*` token dedups on the value.
    rss_url = routing.strip_tracking_params(rss_url)
    youtube = routing.strip_tracking_params(youtube)
    youtube_video = routing.strip_tracking_params(youtube_video)
    audio_url = routing.strip_tracking_params(audio_url)

    # Dedup + append happen atomically under feeds_io's write lock, so two
    # concurrent ingest()s of the same URL can't both append (TOCTOU-free).
    status, existing = await feeds_io.add_feed(
        feeds_path,
        name=name,
        rss_url=rss_url,
        youtube=youtube,
        youtube_video=youtube_video,
        audio_url=audio_url,
        min_minutes=min_minutes,
        language=language,
        published_at=published_at,
        duration_seconds=duration_seconds,
        show=show,
    )
    if status == "already_subscribed":
        return {"kind": kind, "status": "already_subscribed", "name": existing}
    # Kick the digest now (write feeds.toml first — done above — then signal, so
    # the feed is visible when the triggered tick loads it). Best-effort: a feed
    # that's added but not signalled still gets picked up on the next poll tick.
    triggered = _signal_processing(feeds_path)
    return {
        "kind": kind,
        "status": "added",
        "name": name,
        "rss_url": rss_url,
        "youtube": youtube,
        "youtube_video": youtube_video,
        "audio_url": audio_url,
        "language": language,
        "processing_triggered": triggered,
    }


def _signal_processing(feeds_path: Path) -> bool:
    """Bump the wake sentinel so the digest runs a tick within seconds. Returns
    False (and logs) if signalling fails — never fails the caller."""
    try:
        signal_wake(feeds_path)
        return True
    except OSError as e:
        logger.warning("mcp.signal_wake_failed err=%s", e)
        return False


def _source_of(f: feeds_io.FeedEntry) -> str | None:
    for k in ("rss_url", "search", "youtube", "youtube_video", "audio_url"):
        if getattr(f, k):
            return k
    return None


# One-shot sources are processed exactly once; everything else recurs each tick.
_ONESHOT_SOURCES = frozenset({"youtube_video", "audio_url"})


def _feed_type(source: str | None) -> str:
    return "oneoff" if source in _ONESHOT_SOURCES else "subscription"


# -- audio one-shot helpers -------------------------------------------------

_AUDIO_EXTS = (".mp3", ".m4a", ".aac", ".ogg", ".wav", ".flac")
_DISPOSITION_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)


async def _audio_head(url: str, *, http: httpx.AsyncClient) -> dict:
    """HEAD a direct-audio URL for content-type, size, and a filename hint.

    Best-effort: any error yields an empty dict so preview/ingest still work
    off the URL alone.
    """
    info: dict = {}
    try:
        r = await http.head(url)
        ct = (r.headers.get("content-type") or "").split(";", 1)[0].strip()
        info["content_type"] = ct or None
        cl = r.headers.get("content-length")
        info["size_bytes"] = int(cl) if cl and cl.isdigit() else None
        m = _DISPOSITION_FILENAME.search(r.headers.get("content-disposition", ""))
        info["filename"] = m.group(1).strip() if m else None
    except Exception:
        pass
    return info


def _audio_name(url: str, head: dict) -> str | None:
    """Derive a human title: Content-Disposition filename, else the URL basename,
    with any audio extension stripped."""
    from urllib.parse import unquote, urlparse

    candidate = head.get("filename") or unquote(urlparse(url).path.rsplit("/", 1)[-1])
    candidate = (candidate or "").strip()
    for ext in _AUDIO_EXTS:
        if candidate.lower().endswith(ext):
            candidate = candidate[: -len(ext)]
            break
    return candidate.strip() or None


async def _audio_title(url: str, *, http: httpx.AsyncClient) -> str | None:
    return _audio_name(url, await _audio_head(url, http=http))


def _rss_title_from_body(body: bytes) -> str | None:
    import feedparser
    parsed = feedparser.parse(body)
    return (parsed.feed.get("title") if hasattr(parsed, "feed") else None) or None


def _rss_language_from_body(body: bytes) -> str | None:
    """The channel-level `<language>` declared by an RSS/Atom feed, if any."""
    import feedparser
    parsed = feedparser.parse(body)
    return (parsed.feed.get("language") if hasattr(parsed, "feed") else None) or None


async def _rss_meta(
    rss_url: str, *, http: httpx.AsyncClient
) -> tuple[str | None, str | None, list[str]]:
    """`(title, declared_language, latest_titles)` for a feed. All best-effort:
    a fetch/parse failure yields ``(None, None, [])`` so ingest still proceeds
    off the URL alone."""
    try:
        body = await fetch_rss(rss_url, client=http)
        titles = [e.title for e in parse_episodes(body)[:5]]
        return _rss_title_from_body(body), _rss_language_from_body(body), titles
    except Exception:
        return None, None, []


async def _channel_meta(channel_feed: str) -> tuple[str | None, list[str]]:
    """`(channel_name, latest_titles)` for a YouTube channel; ``(None, [])`` on
    error. Titles feed the script-based language sniff (YouTube declares none)."""
    try:
        name, eps = await list_channel_uploads(channel_feed, limit=5)
        return name, [e.title for e in eps]
    except Exception:
        return None, []


async def _youtube_video_title(url: str) -> str | None:
    try:
        ep = await single_video_episode(url)
        return ep.title
    except Exception:
        return None
