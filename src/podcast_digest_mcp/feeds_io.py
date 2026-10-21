"""Read/write feeds.toml while preserving comments and key order.

The digest re-parses feeds.toml on every poll tick, so an atomic in-place
rewrite is picked up live with no restart. The bind-mount on the digest side
is the directory (not the single file), so a tempfile+rename within the same
dir lands at the same logical path with a new inode — the container's next
open() sees the new contents.

Schema mirrors podcast_digest.config.Feed: every entry has `name` plus exactly
one source field (`rss_url`, `search`, `youtube`, `youtube_video`, or
`audio_url`).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import tomlkit


@dataclass(frozen=True)
class FeedEntry:
    name: str
    rss_url: str | None
    search: str | None
    youtube: str | None
    youtube_video: str | None
    audio_url: str | None
    min_minutes: int
    # Non-English spoken language as a bare ISO-639-1 code (e.g. "zh"); None/"en"
    # keeps the English transcription path. Mirrors podcast_digest.config.Feed.
    language: str | None = None
    # Episode metadata captured at ingest for audio one-shots (Apple episode →
    # enclosure). Mirrors podcast_digest.config.Feed; unset for every other source.
    published_at: str | None = None
    duration_seconds: int | None = None
    # Real podcast/show name for a one-shot (its `name` is the episode title).
    show: str | None = None


@dataclass(frozen=True)
class FeedsDoc:
    recipient: str
    poll_seconds: int
    feeds: list[FeedEntry]


# One lock per process — serialises concurrent ingest() writers.
_WRITE_LOCK = asyncio.Lock()


def read(path: Path) -> FeedsDoc:
    raw = tomlkit.parse(path.read_text(encoding="utf-8"))
    feeds: list[FeedEntry] = []
    for f in raw.get("feeds", []):
        feeds.append(FeedEntry(
            name=str(f["name"]),
            rss_url=_opt_str(f, "rss_url"),
            search=_opt_str(f, "search"),
            youtube=_opt_str(f, "youtube"),
            youtube_video=_opt_str(f, "youtube_video"),
            audio_url=_opt_str(f, "audio_url"),
            min_minutes=int(f.get("min_minutes", 0)),
            language=_opt_str(f, "language"),
            published_at=_opt_str(f, "published_at"),
            duration_seconds=_opt_int(f, "duration_seconds"),
            show=_opt_str(f, "show"),
        ))
    recipient = raw.get("recipient")
    if recipient is None:
        raise ValueError(f"{path} is missing the required 'recipient' key")
    return FeedsDoc(
        recipient=str(recipient),
        poll_seconds=int(raw.get("poll_seconds", 21600)),
        feeds=feeds,
    )


def _opt_str(table, key: str) -> str | None:
    v = table.get(key)
    return str(v) if v is not None else None


def _opt_int(table, key: str) -> int | None:
    v = table.get(key)
    return int(v) if v is not None else None


async def add_feed(
    path: Path,
    *,
    name: str,
    rss_url: str | None = None,
    search: str | None = None,
    youtube: str | None = None,
    youtube_video: str | None = None,
    audio_url: str | None = None,
    min_minutes: int = 0,
    language: str | None = None,
    published_at: str | None = None,
    duration_seconds: int | None = None,
    show: str | None = None,
) -> tuple[str, str]:
    """Dedup-check and append a [[feeds]] entry atomically.

    The whole read → dedup → write runs under `_WRITE_LOCK`, so two concurrent
    ingest()s of the same URL can't both pass the dedup check and double-append.

    Returns ``("added", name)`` on append, or ``("already_subscribed", existing)``
    if an entry with the same name or same source value already exists.
    """
    sources = [s for s in (rss_url, search, youtube, youtube_video, audio_url) if s]
    if len(sources) != 1:
        raise ValueError(
            "exactly one of rss_url / search / youtube / youtube_video / audio_url "
            "must be set"
        )
    async with _WRITE_LOCK:
        return await asyncio.to_thread(
            _add_feed_blocking,
            path, name, rss_url, search, youtube, youtube_video, audio_url, min_minutes,
            language, published_at, duration_seconds, show,
        )


def _add_feed_blocking(
    path: Path,
    name: str,
    rss_url: str | None,
    search: str | None,
    youtube: str | None,
    youtube_video: str | None,
    audio_url: str | None,
    min_minutes: int,
    language: str | None,
    published_at: str | None,
    duration_seconds: int | None,
    show: str | None,
) -> tuple[str, str]:
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    feeds = doc.get("feeds")
    if feeds is None:
        feeds = tomlkit.aot()
        doc["feeds"] = feeds

    for f in feeds:
        existing = str(f.get("name", ""))
        if existing == name:
            return ("already_subscribed", existing)
        if rss_url and _opt_str(f, "rss_url") == rss_url:
            return ("already_subscribed", existing)
        if search and _opt_str(f, "search") == search:
            return ("already_subscribed", existing)
        if youtube and _opt_str(f, "youtube") == youtube:
            return ("already_subscribed", existing)
        if youtube_video and _opt_str(f, "youtube_video") == youtube_video:
            return ("already_subscribed", existing)
        if audio_url and _opt_str(f, "audio_url") == audio_url:
            return ("already_subscribed", existing)

    entry = tomlkit.table()
    entry["name"] = name
    if rss_url:
        entry["rss_url"] = rss_url
    if search:
        entry["search"] = search
    if youtube:
        entry["youtube"] = youtube
    if youtube_video:
        entry["youtube_video"] = youtube_video
    if audio_url:
        entry["audio_url"] = audio_url
    # Non-English feeds route to the multilingual whisper model (`-l <language>`).
    if language:
        entry["language"] = language
    if min_minutes:
        entry["min_minutes"] = min_minutes
    # Episode metadata for audio one-shots — write-once at ingest so the email
    # shows the real release date / runtime (Apple episode lookup).
    if published_at:
        entry["published_at"] = published_at
    if duration_seconds:
        entry["duration_seconds"] = duration_seconds
    if show:
        entry["show"] = show
    feeds.append(entry)

    _atomic_write(path, tomlkit.dumps(doc))
    return ("added", name)


async def remove_feed(path: Path, *, name: str) -> bool:
    """Remove a feed by exact name match. Returns True if anything was removed."""
    async with _WRITE_LOCK:
        return await asyncio.to_thread(_remove_feed_blocking, path, name)


def _remove_feed_blocking(path: Path, name: str) -> bool:
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    feeds = doc.get("feeds")
    if not feeds:
        return False
    new_feeds = tomlkit.aot()
    removed = False
    for f in feeds:
        if str(f.get("name", "")) == name:
            removed = True
            continue
        new_feeds.append(f)
    if not removed:
        return False
    doc["feeds"] = new_feeds
    _atomic_write(path, tomlkit.dumps(doc))
    return True


def _atomic_write(path: Path, content: str) -> None:
    # tempfile in the SAME directory so os.replace is atomic across the rename
    # (same filesystem). fsync the file, then rename, then fsync the dir.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
