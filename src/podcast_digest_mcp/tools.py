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
from typing import Annotated

import httpx
from fastmcp import FastMCP
from pydantic import Field

from podcast_digest.feeds import FeedError, fetch_rss, parse_episodes
from podcast_digest.net import validate_outbound_url_async
from podcast_digest.wake import signal_wake
from podcast_digest.youtube import (
    list_channel_uploads,
    resolve_channel_feed,
    single_video_episode,
)

from . import apple, archive, feeds_io, lang, routing

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
    data_dir: Path | None = None,
) -> None:
    """Wire the MCP tools onto the FastMCP app."""

    # -- ingest -----------------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Add podcast or episode",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        }
    )
    async def ingest(
        url: Annotated[
            str,
            Field(
                description=(
                    "URL to add: RSS feed, Apple Podcasts show or episode, YouTube channel or video, "
                    "or direct audio file (.mp3/.m4a)."
                )
            ),
        ],
        name: Annotated[
            str | None,
            Field(description="Optional display name override; defaults to the resolved show, channel, or episode title."),
        ] = None,
        min_minutes: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional minimum episode duration in minutes "
                    "(omit to inherit file default; 0 disables filtering)."
                ),
            ),
        ] = None,
        language: Annotated[
            str | None,
            Field(
                description=(
                    "Optional ISO-639-1 spoken language override (e.g. 'en', 'zh', 'ja'); "
                    "omitted auto-detects from feed metadata and titles."
                )
            ),
        ] = None,
    ) -> dict:
        """Subscribe to a podcast/YouTube channel or queue a one-shot episode/audio URL for digest delivery.

        Classifies the URL automatically: RSS feeds, Apple Podcasts shows, and YouTube channels become
        recurring subscriptions; individual YouTube videos, Apple Podcasts episodes, and direct audio links
        are queued once. Non-English audio transcribes with the multilingual model and still delivers an
        English summary. Idempotent: duplicate entries return `already_subscribed` without re-adding.
        """
        if name is not None:
            name = name.strip() or None
        try:
            c = await asyncio.to_thread(routing.classify, url, http=http_sync)

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
                ep = await apple.lookup_episode(
                    c.canonical, c.extra["episode_track_id"], http=http,
                )
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
        except (ValueError, FeedError, httpx.HTTPError) as e:
            return {"error": str(e)}

    # -- preview ----------------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Preview podcast URL",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        }
    )
    async def preview(
        url: Annotated[
            str,
            Field(description="Podcast, RSS, Apple Podcasts, YouTube, or direct audio URL to classify without saving."),
        ],
    ) -> dict:
        """Dry-run of ingest: resolve a URL and report its classification, detected language, and recent titles.

        Makes no writes. Returns feed/channel metadata and latest 3 titles (plus publisher transcript detection
        for RSS/Apple Podcasts), or episode title, duration, and enclosure/media metadata for one-shot URLs.
        """
        try:
            c = await asyncio.to_thread(routing.classify, url, http=http_sync)

            if c.kind == "rss":
                body = await fetch_rss(c.canonical, client=http)
                rss_title, declared_lang, episodes = await asyncio.to_thread(
                    _parse_rss_once, body,
                )
                name = rss_title or "(unknown)"
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
                        None, declared_lang, [name, *titles],
                    ),
                }

            if c.kind == "itunes_podcast":
                p = await apple.lookup_podcast(c.canonical, http=http)
                if error := await _publisher_url_error(p.rss_url):
                    return {"error": error}
                body = await fetch_rss(p.rss_url, client=http)
                _rss_title, declared_lang, episodes = await asyncio.to_thread(
                    _parse_rss_once, body,
                )
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
                        None, declared_lang, [p.name, *titles],
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
                apple_ep = await apple.lookup_episode(
                    c.canonical, c.extra["episode_track_id"], http=http,
                )
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
        except (ValueError, FeedError, httpx.HTTPError) as e:
            return {"error": str(e)}

    # -- subscription_list ------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "List podcast subscriptions",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def subscription_list(
        include_completed_oneoffs: Annotated[
            bool,
            Field(description="Include one-shot episodes that have already been processed (hidden by default)."),
        ] = False,
    ) -> dict:
        """List configured recurring subscriptions and queued one-shot episodes with their source kind and settings.

        Completed one-shot episodes are omitted by default; pass include_completed_oneoffs=true to include them.
        """
        doc = await asyncio.to_thread(feeds_io.read, feeds_path)
        has_oneoffs = any(_feed_type(_source_of(f)) == "oneoff" for f in doc.feeds)
        completed = (
            set() if (include_completed_oneoffs or not has_oneoffs)
            else await asyncio.to_thread(archive.completed_oneoff_keys, data_dir)
        )
        items = []
        hidden_completed = 0
        for f in doc.feeds:
            source = _source_of(f)
            value = getattr(f, source) if source else None
            feed_type = _feed_type(source)
            if (
                feed_type == "oneoff"
                and not include_completed_oneoffs
                and archive.oneoff_is_completed(source, value, completed)
            ):
                hidden_completed += 1
                continue
            items.append({
                "name": f.name,
                "type": feed_type,
                "source": source,
                "value": value,
                "min_minutes": f.min_minutes if f.min_minutes is not None else doc.min_minutes,
                "language": f.language,
            })
        counts = {
            "subscriptions": sum(1 for it in items if it["type"] == "subscription"),
            "oneoffs": sum(1 for it in items if it["type"] == "oneoff"),
        }
        if hidden_completed:
            counts["completed_oneoffs_hidden"] = hidden_completed
        return {
            "recipient": doc.recipient,
            "poll_seconds": doc.poll_seconds,
            "min_minutes": doc.min_minutes,
            "counts": counts,
            "feeds": items,
        }

    # -- digest_status ----------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Digest pipeline status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def digest_status(
        limit: Annotated[int, Field(ge=1, le=100, description="Maximum recent deliveries and errors to return.")] = 10,
    ) -> dict:
        """Read digest pipeline status: stage counts, recent deliveries, in-progress episodes, held deliveries, and feed errors."""
        return await asyncio.to_thread(archive.digest_status, data_dir, limit=limit)

    # -- search_transcripts -----------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Search podcast transcripts",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def search_transcripts(
        query: Annotated[str, Field(description="Keyword or phrase query over saved podcast summaries and transcripts.")],
        feed: Annotated[str | None, Field(description="Optional feed or show name filter.")] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum matching episodes to return.")] = 10,
    ) -> dict:
        """Search saved podcast summaries and transcripts by keyword(s).

        Returns matching episodes ranked by relevance with context snippets and GUIDs for get_episode.
        """
        return await asyncio.to_thread(
            archive.search_transcripts, data_dir, query=query, feed=feed, limit=limit,
        )

    # -- get_episode ------------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Get podcast episode",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def get_episode(
        query_or_guid: Annotated[str, Field(description="Episode GUID from search_transcripts or a title search string.")],
        feed: Annotated[str | None, Field(description="Optional feed URL or show name to disambiguate colliding GUIDs.")] = None,
        include_transcript: Annotated[bool, Field(description="Include the saved transcript text alongside the summary.")] = True,
        max_chars: Annotated[int, Field(ge=1000, le=200_000, description="Maximum transcript characters returned in this slice.")] = 40_000,
        offset: Annotated[int, Field(ge=0, description="Character offset for paginating long transcripts.")] = 0,
    ) -> dict:
        """Read one episode's metadata, summary, and saved transcript (cleaned Markdown preferred, raw text fallback).

        Use offset and max_chars to paginate long transcripts.
        """
        return await asyncio.to_thread(
            archive.get_episode,
            data_dir,
            query_or_guid=query_or_guid,
            feed=feed,
            include_transcript=include_transcript,
            max_chars=max_chars,
            offset=offset,
        )

    # -- subscription_remove ---------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Remove podcast subscription",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def subscription_remove(
        name: Annotated[str, Field(description="Exact subscription or one-shot entry name from subscription_list.")],
    ) -> dict:
        """Remove every configured subscription or one-shot entry whose name matches exactly."""
        removed = await feeds_io.remove_feed(feeds_path, name=name)
        return {"removed": removed, "name": name}

    # -- process_now ------------------------------------------------------

    @mcp.tool(
        annotations={
            "title": "Trigger digest processing",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        }
    )
    async def process_now() -> dict:
        """Signal the digest pipeline to run a processing tick immediately instead of waiting for the next poll.

        Safe to call repeatedly: ticks are single-flight and a no-op when no episodes are pending.
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
    min_minutes: int | None = None,
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


def _parse_rss_once(body: bytes) -> tuple[str | None, str | None, list]:
    """Parse an RSS/Atom body once and return (feed_title, declared_language, episodes)."""
    import feedparser
    parsed = feedparser.parse(body)
    feed_obj = parsed.feed if hasattr(parsed, "feed") else {}
    title = feed_obj.get("title") or None
    declared = feed_obj.get("language") or None
    episodes = parse_episodes(body)
    return title, declared, episodes


async def _rss_meta(
    rss_url: str, *, http: httpx.AsyncClient
) -> tuple[str | None, str | None, list[str]]:
    """`(title, declared_language, latest_titles)` for a feed. All best-effort:
    a fetch/parse failure yields ``(None, None, [])`` so ingest still proceeds
    off the URL alone."""
    try:
        body = await fetch_rss(rss_url, client=http)
        title, declared, episodes = await asyncio.to_thread(_parse_rss_once, body)
        return title, declared, [e.title for e in episodes[:5]]
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
