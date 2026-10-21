"""Streaming audio download with redirect + atomic rename.

Podcasts are large (90-min MP3 ≈ 80MB). We stream to a `.partial` file and
atomically rename on success so a crash mid-download never leaves a corrupt
final file that we'd later try to transcribe.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from . import youtube
from .curlfetch import fetch_to_file
from .feeds import USER_AGENT
from .net import validate_async_request, validate_outbound_url_async
from .proc import SubprocessTimeout, run_capture

logger = logging.getLogger(__name__)

CHUNK = 256 * 1024
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=15.0, read=120.0)
# Some hosts (Substack behind Cloudflare) fingerprint-block httpx's TLS with a
# 403/429 but serve `curl` fine — same as the RSS fetch in feeds.py.
CURL_MAX_TIME = "900"
# Outer wall-clock bound on the curl subprocess: its own --max-time should trip
# first; this is the kill-if-curl-itself-hangs backstop (a hair over --max-time).
_CURL_TIMEOUT = 960.0
_FFPROBE_TIMEOUT = 120.0


class InvalidMediaError(RuntimeError):
    """ffprobe read the downloaded bytes and they are not decodable media."""


async def download(
    audio_url: str,
    *,
    dest_dir: Path,
    basename: str,
    client: httpx.AsyncClient | None = None,
    ytdlp_format: str = "bestaudio/best",
    ytdlp_cookies: str | None = None,
) -> Path:
    """Download `audio_url` to `dest_dir/basename + ext` atomically.

    Returns the final path. Idempotent: if the final file already exists and is
    non-empty, returns immediately. Verifies Content-Length when the server
    sends it; raises on truncation so we don't transcribe partial MP3s.

    YouTube watch URLs have no audio enclosure, so they're routed to yt-dlp
    (`youtube.download_audio`) instead of the plain HTTP streamer.
    """
    await validate_outbound_url_async(audio_url)
    if youtube.is_youtube_url(audio_url):
        return await youtube.download_audio(
            audio_url, dest_dir=dest_dir, basename=basename,
            fmt=ytdlp_format, cookies=ytdlp_cookies,
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Streaming requires its own client w/ follow_redirects because podcast
    # CDNs (libsyn, megaphone, acast) chain 2-3 redirects to land on the MP3.
    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
        event_hooks={"request": [validate_async_request]},
    )

    try:
        try:
            return await _download_stream(active_client, audio_url, dest_dir, basename)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                logger.warning(
                    "audio.fetch_blocked status=%d url=%s — retrying via curl",
                    e.response.status_code, audio_url,
                )
                final_url = str(e.response.request.url)
                await validate_outbound_url_async(final_url)
                return await _download_curl(
                    final_url, dest_dir=dest_dir, basename=basename,
                )
            raise
    finally:
        if owns_client:
            await active_client.aclose()


async def _download_stream(
    client: httpx.AsyncClient, audio_url: str, dest_dir: Path, basename: str,
) -> Path:
    # Always pass our generous DOWNLOAD_TIMEOUT explicitly: the caller's client
    # typically has a tight default (~30s read) that would trip on an 80MB MP3.
    async with client.stream(
        "GET", audio_url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        ext = _ext_from_response(resp)
        final = dest_dir / f"{basename}{ext}"
        if await _reusable_cached(final):
            return final

        partial = dest_dir / f"{basename}{ext}.partial"
        total = int(resp.headers.get("content-length") or 0)
        written = 0
        try:
            with partial.open("wb") as f:
                async for chunk in resp.aiter_bytes(CHUNK):
                    f.write(chunk)
                    written += len(chunk)
            # Truncation detection: a TCP close mid-stream is silent in httpx.
            # We treat any positive content-length mismatch as a hard error
            # so we don't summarize half an episode.
            if total > 0 and written < total:
                raise RuntimeError(
                    f"truncated download: got {written} of {total} bytes "
                    f"from {audio_url}"
                )
            # Never publish bytes that don't decode as media: a 200 can still
            # be an interstitial/error page, and a poisoned final file would
            # be picked up by the cached-file check and re-fed to
            # transcription on every retry until the episode drops.
            await verify_media(partial, source=audio_url)
            os.replace(partial, final)
        except BaseException:
            # On any failure (truncation, cancellation, IO error) drop the
            # partial so the next retry starts clean.
            partial.unlink(missing_ok=True)
            raise
        logger.info(
            "audio.downloaded path=%s bytes=%d expected=%d", final, written, total
        )
        return final


async def _download_curl(
    audio_url: str, *, dest_dir: Path, basename: str,
) -> Path:
    """Download via a curl subprocess — bypasses TLS-fingerprint 403/429 blocks
    (Substack/Cloudflare) that hit httpx. The block can land on httpx's FIRST
    request — before any redirect — so `audio_url` may itself be an unresolved
    redirect (Substack 307s api.substack.com audio to a signed substackcdn.com
    URL). curlfetch follows the chain one validated hop at a time; truncation
    against a declared Content-Length is caught by curl itself (exit 18)."""
    ext = _ext_from_url(audio_url)
    final = dest_dir / f"{basename}{ext}"
    if await _reusable_cached(final):
        return final
    partial = dest_dir / f"{basename}{ext}.partial"
    try:
        await fetch_to_file(
            audio_url, dest=partial, user_agent=USER_AGENT,
            max_time=CURL_MAX_TIME, timeout=_CURL_TIMEOUT, label="curl audio",
        )
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError(
                f"curl audio download produced no data for {audio_url}"
            )
        # Same contract as the httpx path: never publish non-media bytes —
        # before this check a redirect stub saved as `.mp3` was cached forever
        # and fed to transcription on every retry until the episode dropped.
        await verify_media(partial, source=audio_url)
        size = partial.stat().st_size
        os.replace(partial, final)
    except BaseException:
        # Timeout (child already killed), cancellation, spawn error, or a bad
        # download — never leave a stale .partial behind. After os.replace the
        # partial is gone, so on the success path this is unreachable.
        partial.unlink(missing_ok=True)
        raise
    logger.info("audio.downloaded_curl path=%s bytes=%d", final, size)
    return final


async def _reusable_cached(final: Path) -> bool:
    """True if `final` exists and verifies as media. A cached non-media
    artifact (e.g. a redirect stub published as `.mp3` by a pre-0.1.47 run) is
    deleted so the caller re-downloads instead of re-serving the poison."""
    if not (final.exists() and final.stat().st_size > 0):
        return False
    try:
        await verify_media(final, source=str(final))
    except InvalidMediaError as exc:
        logger.warning("audio.cached_invalid path=%s err=%s — refetching", final, exc)
        final.unlink(missing_ok=True)
        return False
    logger.info("audio.cached path=%s bytes=%d", final, final.stat().st_size)
    return True


async def _ffprobe(path: Path) -> tuple[str, float | None, str]:
    """Run ffprobe on `path`. Returns `(outcome, duration, detail)`:

      "ok"          — ffprobe decoded the file (duration may still be None if
                      it printed something unparsable)
      "invalid"     — ffprobe ran and REJECTED the file: the bytes are not
                      decodable media (redirect stub, HTML error page, junk)
      "unavailable" — ffprobe missing or hung: nothing is known about the file

    Splitting "invalid" from "unavailable" is load-bearing: verify_media
    rejects only on "invalid", so a missing ffprobe binary can degrade
    validation without silently dropping every episode.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "--",  # end of options: a path beginning with '-' can't become a flag
        str(path),
    ]
    try:
        rc, stdout, stderr = await run_capture(
            cmd, timeout=_FFPROBE_TIMEOUT, label="ffprobe",
        )
    except SubprocessTimeout as exc:
        return "unavailable", None, f"timeout: {exc}"
    except OSError as exc:
        # ffprobe binary missing/unspawnable (e.g. not installed on the host).
        return "unavailable", None, str(exc)
    if rc != 0:
        return "invalid", None, f"rc={rc}: {stderr.decode(errors='replace')[:200]}"
    raw = stdout.decode(errors="replace").strip()
    try:
        return "ok", float(raw), ""
    except ValueError:
        return "ok", None, f"bad duration {raw!r}"


async def probe_duration(path: Path) -> float | None:
    """Real media duration in seconds via ffprobe, or None if it can't be read.

    The pre-transcribe length gate relies on this: declared feed durations are
    often missing or wrong (and absent entirely for YouTube / direct-audio
    sources), so we measure the downloaded file itself. ffprobe ships with the
    ffmpeg already on PATH in the image. Returning None on any failure means an
    unreadable probe never blocks an episode — we fall through to transcription.
    """
    outcome, duration, detail = await _ffprobe(path)
    if outcome != "ok":
        logger.warning("ffprobe.%s path=%s err=%s", outcome, path, detail)
        return None
    if duration is None:
        logger.warning("ffprobe.bad_duration path=%s %s", path, detail)
    return duration


async def verify_media(path: Path, *, source: str) -> None:
    """Raise :class:`InvalidMediaError` if ffprobe positively rejects `path`.

    Called before a downloaded file is atomically published (and on cached
    files before reuse) so non-media bytes served with a success status — a
    redirect stub, a challenge page — fail the DOWNLOAD, leaving the episode
    cleanly retryable, instead of poisoning transcription downstream. An
    unavailable/hung ffprobe never blocks (same contract as probe_duration):
    we only refuse when the tool affirmatively read the bytes and they are
    not media.
    """
    outcome, _duration, detail = await _ffprobe(path)
    if outcome == "invalid":
        raise InvalidMediaError(
            f"downloaded file is not decodable media ({detail}) "
            f"path={path.name} source={source}"
        )
    if outcome == "unavailable":
        logger.warning("verify_media.skipped path=%s err=%s", path, detail)


def cleanup(path: Path | str | None) -> None:
    """Delete `path` if it exists. Swallow errors — cleanup is best-effort."""
    if not path:
        return
    p = Path(path)
    try:
        p.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("audio.cleanup_failed path=%s err=%s", p, e)


def _ext_from_response(resp: httpx.Response) -> str:
    """Best-effort extension. Prefer Content-Type, fall back to URL path."""
    ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    if ct in mapping:
        return mapping[ct]
    # URL path fallback (strip query)
    return _ext_from_url(str(resp.url))


def _ext_from_url(url: str) -> str:
    """Extension from a URL path (no response headers, for the curl path)."""
    url_path = url.split("?", 1)[0]
    for ext in (".mp3", ".m4a", ".aac", ".ogg", ".wav"):
        if url_path.lower().endswith(ext):
            return ext
    return ".mp3"  # podcasts are MP3 ~99% of the time
