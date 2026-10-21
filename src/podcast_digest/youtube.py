"""YouTube source support.

YouTube isn't a podcast: there's no audio enclosure to download. Instead:

  - **Channels** are enumerated with ``yt-dlp --flat-playlist`` over the
    channel's ``/videos`` tab. (YouTube's old key-free Atom feed at
    ``feeds/videos.xml?channel_id=UC...`` now returns 404 — globally deprecated
    as of 2026-06.) Flat enumeration is reliable for the video id, duration, and
    reverse-chronological order; the title is provisional and the upload date is
    absent, so callers enrich the few episodes they emit via
    ``single_video_episode``. Each entry's "audio_url" is the watch URL.
  - **Single videos** have no feed, so we pull metadata once via
    ``yt-dlp --dump-single-json``.
  - **Audio** for either is extracted with ``yt-dlp`` (which needs ffmpeg, already
    in the image) — the pipeline's ffmpeg pass then re-encodes it to WAV like any
    other download.

The watch URL flows through the pipeline as ``EpisodeMeta.audio_url`` and is
detected by ``audio.download``, which routes YouTube URLs here instead of the
plain HTTP streamer.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .feeds import EpisodeMeta, FeedError, fetch_page
from .net import HTTP_SCHEMES
from .proc import SubprocessTimeout, run_capture

logger = logging.getLogger(__name__)

# Hosts we treat as YouTube watch URLs.
_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
# A channel id is "UC" + 22 url-safe base64 chars.
_CHANNEL_ID_RE = re.compile(r"(UC[\w-]{22})")
_VIDEOS_XML = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
# How many recent uploads to enumerate per channel. Mirrors the entry count the
# old Atom feed returned, so first-sight back-catalog suppression covers the
# recent archive; only the newest unseen episode is ever enriched/emitted.
_CHANNEL_FEED_LIMIT = 15

# Subprocess timeouts (seconds). yt-dlp is known to stall on throttled/hung
# connections; without a bound one stall wedges the whole serial tick until a
# manual container restart. Generous — a real audio pull of a long episode is
# minutes, not the ceiling here. See proc.run_capture.
_YTDLP_META_TIMEOUT = 180.0      # flat enumeration / single-video metadata
_YTDLP_DOWNLOAD_TIMEOUT = 1800.0  # full audio extract + mp3 transcode


def is_youtube_url(url: str) -> bool:
    """True if `url` is an http(s) URL on a YouTube host.

    Parses the URL and matches the *host* (exact or a dotted subdomain of a
    known YouTube domain) rather than substring-scanning the whole string — so a
    feed-controlled enclosure like ``http://evil.test/x?ref=youtube.com`` is not
    misrouted to yt-dlp, and a non-http scheme is rejected outright.
    """
    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme.lower() not in HTTP_SCHEMES:
        return False
    host = (parts.hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _YOUTUBE_HOSTS)


def extract_video_id(url: str) -> str:
    """Pull the 11-char video id from any common YouTube URL form.

    Handles ``watch?v=ID``, ``youtu.be/ID``, ``/embed/ID``, ``/shorts/ID`` and a
    bare id. Raises FeedError if none is found.
    """
    s = url.strip()
    if re.fullmatch(r"[\w-]{11}", s):
        return s
    patterns = (
        r"[?&]v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"/embed/([\w-]{11})",
        r"/shorts/([\w-]{11})",
        r"/live/([\w-]{11})",
    )
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    raise FeedError(f"could not extract a YouTube video id from {url!r}")


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------------------------
# Channel discovery
# ---------------------------------------------------------------------------


async def resolve_channel_feed(spec: str, *, client: httpx.AsyncClient) -> str:
    """Resolve a channel reference to its ``videos.xml`` Atom feed URL.

    Accepts a raw ``UC...`` id, a ``/channel/UC...`` URL, an already-formed
    ``videos.xml`` URL (returned as-is), or a handle/custom URL (``@handle``,
    ``/c/Name``, ``/user/Name``) — the latter requires one page fetch to read the
    channel id out of the HTML.
    """
    s = spec.strip()
    if "feeds/videos.xml" in s:
        return s
    # Raw id or /channel/UC... — no network needed.
    if re.fullmatch(r"UC[\w-]{22}", s):
        return _VIDEOS_XML.format(cid=s)
    m = re.search(r"/channel/(UC[\w-]{22})", s)
    if m:
        return _VIDEOS_XML.format(cid=m.group(1))

    # Handle / custom URL → fetch the page and read the channel id from it.
    # Match canonical signals first: the RSS alternate link and "externalId"
    # always refer to the page's own channel, whereas the first "channelId" in
    # the raw HTML is often a *featured/related* channel rendered in the page
    # data (a featured channel would otherwise win).
    url = s if s.startswith("http") else f"https://www.youtube.com/{s.lstrip('/')}"
    # An HTML page, not a feed: fetch_page (not fetch_rss, which rejects
    # anything that isn't RSS/Atom). Same guarded client and curl fallback.
    body = await fetch_page(url, client=client)
    html = body.decode("utf-8", errors="replace")
    for pat in (
        r'feeds/videos\.xml\?channel_id=(UC[\w-]{22})',
        r'"externalId":"(UC[\w-]{22})"',
        r'"channelId":"(UC[\w-]{22})"',
    ):
        m = re.search(pat, html)
        if m:
            return _VIDEOS_XML.format(cid=m.group(1))
    m = _CHANNEL_ID_RE.search(html)
    if not m:
        raise FeedError(f"could not resolve a channel id for {spec!r}")
    return _VIDEOS_XML.format(cid=m.group(1))


async def list_channel_uploads(
    spec: str, *, limit: int = _CHANNEL_FEED_LIMIT, cookies: str | None = None,
) -> tuple[str | None, list[EpisodeMeta]]:
    """Enumerate a channel's most recent uploads via ``yt-dlp --flat-playlist``.

    Replaces the dead ``videos.xml`` Atom feed. `spec` is any reference carrying
    a ``UC...`` channel id (a raw id, a ``/channel/UC...`` URL, or the legacy
    ``videos.xml`` URL we still use as the feed's stable primary key).

    Returns ``(channel_name, episodes)``, newest-first. Flat enumeration is
    reliable for the video id, duration, and order, but the ``title`` is
    provisional (yt-dlp can surface a teaser/cached title) and there is no upload
    date — so ``published_at`` is left empty for callers to enrich via
    :func:`single_video_episode` on the few episodes they actually emit.
    """
    _require_yt_dlp()
    m = _CHANNEL_ID_RE.search(spec)
    if not m:
        raise FeedError(f"could not find a channel id in {spec!r}")
    channel_url = f"https://www.youtube.com/channel/{m.group(1)}/videos"
    cmd = [
        "yt-dlp", "--flat-playlist", "--playlist-end", str(limit),
        "--dump-json", "--no-warnings",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd += ["--", channel_url]
    out = await _ytdlp_stdout(cmd, label=f"channel {m.group(1)}")

    name: str | None = None
    episodes: list[EpisodeMeta] = []
    skipped = 0
    # Entries skipped ONLY by the duration-None heuristic are held aside: if the
    # duration filter would remove *every* entry (a yt-dlp/API quirk where flat
    # mode returns no durations at all — vs a channel genuinely full of live
    # streams), we fall back to keeping them rather than blanking the feed. The
    # fallback CAN readmit an untagged live/premiere (in the quirk state the
    # duration heuristic — the only reliable flat-mode live signal — is
    # nullified); the enrichment step's full-metadata live check is the
    # backstop that catches those before they can take the newest slot (see
    # single_video_episode + pipeline._enrich_new_youtube).
    durationless: list[EpisodeMeta] = []
    for line in out.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        name = name or data.get("playlist_uploader") or data.get("channel")
        vid = data.get("id")
        if not vid:
            logger.debug("youtube.skip_entry no video id: %r", data.get("title"))
            skipped += 1
            continue
        verdict = _entry_verdict(data)
        if verdict == "skip":
            skipped += 1
            continue
        dur = data.get("duration")
        meta = EpisodeMeta(
            guid=f"yt:video:{vid}",
            title=str(data.get("title") or "Untitled"),
            published_at="",  # absent in flat mode; enriched per-episode
            duration_seconds=int(dur) if dur else None,
            audio_url=canonical_watch_url(vid),
            audio_type="video/youtube",
            description=None,
        )
        (durationless if verdict == "no_duration" else episodes).append(meta)

    quirk_fallback = False
    if not episodes and durationless:
        # No entry had a duration at all — that's a metadata quirk, not a channel
        # of live streams. Don't blank the feed; keep them (newest-first order
        # preserved). Enrichment re-checks liveness with full metadata.
        quirk_fallback = True
        episodes = durationless
        durationless = []
    # One summary line per poll (the per-entry detail is DEBUG): the same
    # entries re-classify identically every tick, so per-entry INFO would
    # repeat forever on a 6h cadence.
    if skipped or durationless or quirk_fallback:
        logger.info(
            "youtube.enumerated kept=%d skipped=%d durationless_dropped=%d "
            "no_durations_quirk=%s",
            len(episodes), skipped, len(durationless), quirk_fallback,
        )
    return name, episodes


# live_status / availability values that mean "can't download this now". These
# are populated by yt-dlp when known; in flat-playlist mode they're often None
# for a normal public video, so `duration is None` is the reliable catch for a
# live stream / scheduled premiere (neither has a duration yet).
_UNDOWNLOADABLE_LIVE = frozenset({"is_live", "is_upcoming", "post_live"})
_MEMBERS_ONLY = frozenset({"subscriber_only", "premium_only", "needs_auth"})


def _entry_verdict(data: dict) -> str:
    """Classify a flat-enumeration entry: ``"ok"``, ``"skip"``, or ``"no_duration"``.

    A live stream, scheduled premiere, or members-only video can appear newest on
    ``/videos`` yet cannot be pulled — with cap=1 it would occupy the single
    processing slot and pre-suppress the real newest upload behind it (never
    sent, unrecoverable), while itself failing every tick. Those return
    ``"skip"``. Members-only is skipped UNCONDITIONALLY: memberships are bought
    per channel, so a global cookies file proves the account exists — not that
    it can download THIS channel's subscriber_only video; admitting one on the
    strength of cookies for some other channel recreates the exact slot
    poisoning this filter prevents. (A membered channel the operator actually
    wants needs a per-feed opt-in, not a global inference.)

    A missing duration is a *soft* signal (``"no_duration"``): almost always a
    live/premiere (no duration yet), but occasionally a flat-mode metadata quirk
    for a real VOD — so the caller drops it when real VODs are also present, but
    keeps it if it's the only kind of entry, rather than risk blanking the feed.

    Per-entry logs are DEBUG: flat enumeration re-sees the same ~15 entries every
    tick, so INFO here repeats identical lines forever (the caller emits one
    per-poll summary instead).
    """
    title = data.get("title")
    live_status = data.get("live_status")
    if live_status in _UNDOWNLOADABLE_LIVE:
        logger.debug("youtube.skip_entry live_status=%s title=%r", live_status, title)
        return "skip"
    availability = data.get("availability")
    if availability in _MEMBERS_ONLY:
        logger.debug("youtube.skip_entry availability=%s title=%r", availability, title)
        return "skip"
    if data.get("duration") is None:
        logger.debug("youtube.skip_entry no duration (live/premiere?) title=%r", title)
        return "no_duration"
    return "ok"


# ---------------------------------------------------------------------------
# Single-video metadata
# ---------------------------------------------------------------------------


async def single_video_episode(
    url: str, *, cookies: str | None = None
) -> EpisodeMeta:
    """Build an EpisodeMeta for one video via ``yt-dlp --dump-single-json``.

    Called at most once per video (the pipeline short-circuits once the episode
    is known), so the yt-dlp round-trip is paid once, not every tick.
    """
    _require_yt_dlp()
    cmd = [
        "yt-dlp", "--dump-single-json", "--skip-download", "--no-playlist",
        "--no-warnings",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd += ["--", url]
    try:
        rc, out, err = await run_capture(
            cmd, timeout=_YTDLP_META_TIMEOUT, label="yt-dlp metadata",
        )
    except SubprocessTimeout as e:
        raise FeedError(str(e)) from e
    if rc != 0 or not out:
        raise FeedError(
            f"yt-dlp metadata failed (rc={rc}) for {url}: "
            f"{err.decode(errors='replace')[-300:]}"
        )
    data = json.loads(out)
    # Full (non-flat) metadata carries a reliable live_status — unlike flat
    # enumeration, where it's usually None. This is the authoritative gate that
    # catches a live stream / scheduled premiere no flat-mode heuristic could
    # (e.g. one readmitted by the all-durationless quirk fallback) before it
    # can occupy the newest processing slot and pre-suppress the real upload
    # behind it. A premiere self-heals: once it airs it gains a duration and a
    # normal VOD status, and the next tick admits it.
    live_status = data.get("live_status")
    if data.get("is_live") or live_status in _UNDOWNLOADABLE_LIVE:
        raise FeedError(
            f"video is a live stream/premiere (live_status={live_status}) — "
            f"not downloadable yet: {url}"
        )
    vid = data.get("id") or extract_video_id(url)
    return EpisodeMeta(
        guid=f"yt:video:{vid}",
        title=str(data.get("title") or "Untitled"),
        published_at=_upload_date_to_iso(data.get("upload_date")),
        duration_seconds=int(data["duration"]) if data.get("duration") else None,
        audio_url=canonical_watch_url(vid),
        audio_type="video/youtube",
        description=data.get("description"),
    )


def _upload_date_to_iso(yyyymmdd: str | None) -> str:
    """``YYYYMMDD`` → ISO-8601 UTC, or "" when absent.

    Empty (not ``now()``) on a missing date, matching feeds._entry_published: a
    fabricated ``now()`` would sort as the newest episode and could steal the
    cap-1 slot from a genuinely newer upload. "" sorts oldest and omits the date.
    """
    if yyyymmdd and len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return (
            datetime(
                int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), tzinfo=UTC
            ).isoformat(timespec="seconds")
        )
    return ""


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------


async def download_audio(
    url: str,
    *,
    dest_dir: Path,
    basename: str,
    fmt: str = "bestaudio/best",
    cookies: str | None = None,
) -> Path:
    """Extract audio from a YouTube URL to ``dest_dir/basename.mp3`` with yt-dlp.

    yt-dlp uses ffmpeg to extract to MP3; the transcribe step re-encodes to WAV
    anyway, so MP3 (vs the native webm/m4a) just buys a deterministic path.
    Idempotent: returns immediately if the final file already exists.
    """
    _require_yt_dlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{basename}.mp3"
    if final.exists() and final.stat().st_size > 0:
        logger.info("youtube.cached path=%s bytes=%d", final, final.stat().st_size)
        return final

    # Download + transcode to a `.dl`-suffixed temp basename, then atomically
    # rename to the final name ONLY on a clean exit. Without this, a process kill
    # mid-postprocess (or an extractor that fails after partially writing the
    # mp3) leaves a non-empty *truncated* `basename.mp3` that the cache check
    # above would accept next tick — half an episode transcribed + emailed with
    # no error. Mirrors the `.partial` + os.replace discipline in audio.py.
    tmp_base = f"{basename}.dl"
    tmp_final = dest_dir / f"{tmp_base}.mp3"
    outtmpl = str(dest_dir / f"{tmp_base}.%(ext)s")

    def _cleanup_tmp() -> None:
        for frag in dest_dir.glob(f"{tmp_base}.*"):
            frag.unlink(missing_ok=True)

    # Resume semantics: yt-dlp resumes `.part` files by default (--continue),
    # so the download timeout is a per-tick TIME SLICE, not a death sentence —
    # a throttled multi-hour pull keeps its bytes across ticks and converges
    # instead of restarting from zero five times and then being dropped.
    # The ONE artifact that must never survive a kill is the postprocess
    # output (tmp_final): ffmpeg writes it non-atomically, and yt-dlp SKIPS
    # the transcode when its output already exists — a truncated .dl.mp3 kept
    # for "resume" would be promoted verbatim by the rc==0 check below (the
    # F11 truncated-cache bug). So: delete tmp_final before every run (stale
    # copy from a killed/crashed attempt) and on timeout; keep the .part
    # download fragments; full cleanup only on rc != 0 (state may be
    # poisoned) and after success (stray intermediates).
    tmp_final.unlink(missing_ok=True)
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "-x", "--audio-format", "mp3",
        "--no-playlist", "--no-progress", "--no-warnings",
        "-o", outtmpl,
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd += ["--", url]
    logger.info("youtube.download url=%s dest=%s", url, final)
    try:
        rc, _out, err = await run_capture(
            cmd, timeout=_YTDLP_DOWNLOAD_TIMEOUT, label="yt-dlp download",
        )
    except SubprocessTimeout as e:
        tmp_final.unlink(missing_ok=True)  # never let a truncated mp3 survive
        raise FeedError(f"{e} (partial download kept; resumes next tick)") from e
    if rc != 0 or not (tmp_final.exists() and tmp_final.stat().st_size > 0):
        _cleanup_tmp()
        raise FeedError(
            f"yt-dlp download failed (rc={rc}) for {url}: "
            f"{err.decode(errors='replace')[-400:]}"
        )
    os.replace(tmp_final, final)
    _cleanup_tmp()  # any non-mp3 intermediates yt-dlp may have left
    logger.info("youtube.downloaded path=%s bytes=%d", final, final.stat().st_size)
    return final


def _require_yt_dlp() -> None:
    if not shutil.which("yt-dlp"):
        raise FeedError("yt-dlp not on PATH (needed for YouTube feeds)")


async def _ytdlp_stdout(
    cmd: list[str], *, label: str, timeout: float = _YTDLP_META_TIMEOUT
) -> bytes:
    """Run a yt-dlp command and return its stdout bytes.

    Raises FeedError on a non-zero exit (surfacing the stderr tail) so a genuinely
    failed enumeration becomes a visible feed error — same as the old fetch_rss
    404 path. A zero exit with empty stdout is NOT an error: it's a channel with
    no uploads, which the caller renders as zero episodes (matching the old empty-
    Atom-feed behavior). Bounded by `timeout` so a hung yt-dlp can't wedge the tick.
    """
    try:
        rc, out, err = await run_capture(cmd, timeout=timeout, label=f"yt-dlp {label}")
    except SubprocessTimeout as e:
        raise FeedError(str(e)) from e
    if rc != 0:
        raise FeedError(
            f"yt-dlp {label} failed (rc={rc}): "
            f"{err.decode(errors='replace')[-300:]}"
        )
    return out
