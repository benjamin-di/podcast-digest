"""Orchestration: one tick = poll every feed, advance every episode."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import logging
import os
import tempfile
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from . import USER_AGENT, publisher, youtube
from .audio import cleanup as audio_cleanup
from .audio import download as audio_download
from .audio import probe_duration
from .config import Feed, FeedsFile, Settings
from .db import (
    SUMMARIZE_TRANSPORT_MAX_ATTEMPTS,
    WIKI_PUSH_MAX_ATTEMPTS,
    Database,
    EpisodeRow,
    stage_lt,
)
from .dedupe import (
    CONTENT_WINDOW_DAYS,
    TranscriptFingerprint,
    find_content_match,
    fingerprint_transcript,
)
from .emitted import EmittedLog
from .feeds import (
    SYNTHETIC_GUID_PREFIX,
    EpisodeMeta,
    FeedError,
    fetch_rss,
    parse_episodes,
    resolve_feed_url,
)
from .llm import SummarizerUnavailable, build_completer
from .log import err_fields, event
from .mail import MailAttachment, Mailer, MailProbablyDelivered
from .net import validate_async_request
from .sanitize import safe_listen_url, sanitize_summary_html
from .summarize import SummarizationInput, clean_transcript, summarize
from .transcribe import Transcript, load_whisper_json, transcribe
from .webhook import DeliveredEpisode, WebhookClient, WebhookError

logger = logging.getLogger(__name__)

# How many of the newest unseen episodes a feed processes per tick; the rest
# (older, newest-first) are recorded but pre-marked seen (never sent). At 1 this
# means "only the latest new episode per show per tick" — and it doubles as a
# hard back-catalog-flood guard regardless of first_run/history/restarts. Bump
# higher only if a show regularly drops multiple episodes between polls.
MAX_NEW_PER_TICK = 1

# How many ticks to keep retrying transcript cleanup before giving up and
# shipping the digest from the raw transcript (no/partial attachment). Cleanup
# gates the summary (the summary is built from the cleaned text), so a transient
# summarizer failure parks the episode at `transcribed` and retries the whole clean
# next tick rather than degrading immediately. A policy constant, like
# MAX_ATTEMPTS in db.py. The final attempt accepts a partial clean.
CLEAN_MAX_TICKS = 3


def _build_webhook(settings: Settings) -> WebhookClient | None:
    """A webhook client, or None when the feature is unconfigured (no URL).

    The empty URL is the master off-switch: with it unset, no push is ever
    attempted and the tick stats carry no webhook keys.
    """
    if not settings.webhook_url:
        return None
    return WebhookClient(
        url=settings.webhook_url,
        token=settings.PODCAST_DIGEST_WEBHOOK_TOKEN,
        timeout=settings.PODCAST_DIGEST_WEBHOOK_TIMEOUT,
        fields=settings.webhook_fields,
        ack_field=settings.webhook_ack_field,
    )


def _acquire_tick_lock(settings: Settings):
    """Take an exclusive, non-blocking flock so two ticks never run at once.

    The scheduler (`serve`) and a manual `run` are separate processes sharing
    the same SQLite + state dir. Without a guard they race the same episodes
    and email each one multiple times (the status check and the status write
    aren't atomic across processes). Returns the held file object on success,
    or None if another tick already holds the lock.
    """
    lock_path = settings.PODCAST_DIGEST_DATA_DIR / "tick.lock"
    f = lock_path.open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


async def run_tick(
    *,
    settings: Settings,
    feeds_file: FeedsFile,
    db: Database,
    mailer: Mailer | None,
    dry_run: bool = False,
    only_feed: str | None = None,
) -> dict:
    """Process every feed once. Returns a small stats dict for logging."""
    settings.ensure_dirs()

    # `summarize_failed` counts episodes that failed *at the summarizer transport*
    # (SummarizerUnavailable) — the signal the dead-man's-switch keys on, separate from
    # the generic `failed` (content/email hiccups) and from feed-resolution
    # errors. `summarize_ok` counts successful summarizations: one success
    # proves the transport is healthy, which lets the heartbeat distinguish a
    # real outage (failures, zero successes) from a single stuck episode
    # (its canary fails while everything else summarizes fine).
    stats = {"feeds": 0, "new_episodes": 0, "emailed": 0, "failed": 0,
             "summarize_failed": 0, "summarize_ok": 0}

    # Single-flight: skip if another tick (serve loop or a manual run) holds
    # the lock. Prevents duplicate emails from concurrent ticks.
    lock = _acquire_tick_lock(settings)
    if lock is None:
        logger.warning("tick.skip another tick is already in progress")
        stats["skipped"] = True
        return stats

    try:
        return await _run_tick_locked(
            settings=settings, feeds_file=feeds_file, db=db,
            mailer=mailer, dry_run=dry_run, only_feed=only_feed, stats=stats,
        )
    finally:
        lock.close()  # releases the flock


async def _run_tick_locked(
    *,
    settings: Settings,
    feeds_file: FeedsFile,
    db: Database,
    mailer: Mailer | None,
    dry_run: bool,
    only_feed: str | None,
    stats: dict,
) -> dict:
    webhook = _build_webhook(settings)
    webhook_since = db.webhook_enabled_since(webhook is not None)
    # Shared delivery log: episodes another instance already emailed are treated
    # as seen, so this one never re-sends them. Disabled (no-op) unless
    # PODCAST_DIGEST_EMITTED_LOG is set. See emitted.py.
    emitted = EmittedLog.load(settings.PODCAST_DIGEST_EMITTED_LOG)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        event_hooks={"request": [validate_async_request]},
    ) as client:
        for feed in feeds_file.feeds:
            if only_feed and feed.name != only_feed:
                continue
            stats["feeds"] += 1
            try:
                new, emailed, failed = await _process_feed(
                    feed=feed,
                    recipient=feeds_file.recipient,
                    min_minutes=feeds_file.min_minutes_for(feed),
                    settings=settings,
                    db=db,
                    mailer=mailer,
                    client=client,
                    dry_run=dry_run,
                    webhook=webhook,
                    webhook_since=webhook_since,
                    emitted=emitted,
                    stats=stats,
                )
                stats["new_episodes"] += new
                stats["emailed"] += emailed
                stats["failed"] += failed
            except Exception as e:  # noqa: BLE001
                event(logger, logging.ERROR, "feed.failed",
                      feed=feed.name, exc_info=True, **err_fields(e))
                # Best-effort: feed_url may be unknown if resolution failed.
                with contextlib.suppress(Exception):
                    db.mark_feed_polled(
                        url=db.get_feed_rss_url(feed.name) or feed.rss_url or feed.name,
                        error=str(e)[:500],
                    )
    if "webhook_pushed" in stats:
        logger.info(
            "tick.webhook pushed=%d failed=%d",
            stats["webhook_pushed"], stats["webhook_failed"],
        )
    return stats


async def _process_feed(
    *,
    feed: Feed,
    recipient: str,
    min_minutes: int,
    settings: Settings,
    db: Database,
    mailer: Mailer | None,
    client: httpx.AsyncClient,
    dry_run: bool,
    webhook: WebhookClient | None = None,
    webhook_since: str | None = None,
    emitted: EmittedLog | None = None,
    stats: dict | None = None,
) -> tuple[int, int, int]:
    emitted = emitted or EmittedLog(None)
    try:
        identity_url, itunes_id, episodes = await _discover_episodes(
            feed=feed, min_minutes=min_minutes, settings=settings, db=db, client=client,
        )
    except Exception as discovery_error:  # noqa: BLE001
        known_identity = _known_feed_identity(feed, db)
        if known_identity is None:
            raise
        identity_url = known_identity
        event(
            logger, logging.ERROR, "feed.failed",
            feed=feed.name, source=identity_url, exc_info=True,
            **err_fields(discovery_error),
        )
        emailed_count, failed_count = await _resume_pending_episodes(
            episodes=[],
            rss_url=identity_url,
            feed=feed,
            recipient=recipient,
            min_minutes=min_minutes,
            settings=settings,
            db=db,
            mailer=mailer,
            client=client,
            dry_run=dry_run,
            emitted=emitted,
            stats=stats,
        )
        if webhook is not None and feed.webhook and not dry_run and stats is not None:
            await _sweep_webhook_pushes(
                feed=feed, rss_url=identity_url, db=db,
                webhook=webhook, since=webhook_since, stats=stats,
            )
        db.mark_feed_polled(identity_url, error=str(discovery_error)[:500])
        return 0, emailed_count, failed_count
    first_run = db.upsert_feed(
        url=identity_url, name=feed.name, itunes_collection_id=itunes_id,
    )
    rss_url = identity_url  # PK in the feeds/episodes tables for this feed
    episodes = _dedupe_discovered_episodes(episodes, feed_name=feed.name)
    for meta in episodes:
        if meta.guid.startswith(SYNTHETIC_GUID_PREFIX):
            db.migrate_legacy_synthetic_guid(
                feed_url=rss_url,
                guid=meta.guid,
                title=meta.title,
                published_at=meta.published_at,
                duration_seconds=meta.duration_seconds,
                audio_url=meta.audio_url,
            )
    event(
        logger, logging.INFO, "feed.polled",
        feed=feed.name, source=identity_url, parsed=len(episodes),
        first_run=first_run,
    )

    # Unseen = not in this instance's local DB AND not already emailed by another
    # instance (the shared delivery log). The latter is what lets a second or a
    # replacement instance skip everything already sent and pick up only the
    # episodes that dropped meanwhile — no duplicates.
    new_episodes: list[EpisodeMeta] = []
    for episode in episodes:
        if db.has_episode(rss_url, episode.guid):
            continue
        if emitted.contains(
            rss_url, episode.guid, feed.display_show, episode.title,
        ):
            event(
                logger,
                logging.INFO,
                "episode.identity_duplicate",
                feed=feed.name,
                guid=episode.guid,
                title=episode.title,
            )
            continue
        new_episodes.append(episode)

    # Record every unseen episode, but only ever *process* the newest few; the
    # rest are pre-marked 'emailed' (seen, never sent). `new_episodes` is
    # newest-first. The cap is the hard guard against back-catalog floods:
    #   - first sight of a feed → process only the latest (cap 1), suppress the
    #     whole archive;
    #   - steady state → allow a small catch-up;
    #   - and even if `first_run` is wrong (old data / version skew / a restart
    #     mid-first-run left the archive unseen with last_polled_at set), the cap
    #     still bounds it — no unbounded flood.
    cap = 1 if first_run else MAX_NEW_PER_TICK
    to_process = new_episodes[:cap]
    to_suppress = new_episodes[cap:]

    # Insert every unseen episode AND pre-suppress the beyond-cap back-catalog in
    # a SINGLE transaction. Otherwise a crash (SIGKILL/OOM/power-loss) between the two autocommit loops leaves the whole
    # archive persisted at `discovered`: next tick `has_episode` hides them from
    # the cap logic while `episodes_in_progress` returns every one (the pending
    # loop is uncapped) → the entire back catalogue is downloaded, transcribed,
    # summarized, and EMAILED. The tx makes "inserted" and "suppressed" all-or-
    # nothing, so the cap is never bypassed.
    if new_episodes:  # skip the no-op BEGIN/COMMIT on the steady-state tick
        with db.tx():
            for meta in new_episodes:
                db.insert_episode(
                    feed_url=rss_url, guid=meta.guid, title=meta.title,
                    published_at=meta.published_at,
                    duration_seconds=meta.duration_seconds, audio_url=meta.audio_url,
                )
            for meta in to_suppress:
                db.set_status(
                    feed_url=rss_url, guid=meta.guid,
                    status="emailed", terminal_reason="backlog_suppressed",
                )
    if to_suppress:
        logger.info(
            "feed.backlog_suppressed name=%r first_run=%s processing=%d suppressed=%d",
            feed.name, first_run, len(to_process), len(to_suppress),
        )
    new_episodes = to_process

    # Resume any episode that didn't reach emailed (includes the ones we just
    # inserted, plus any leftovers from a previous crash).
    emailed_count, failed_count = await _resume_pending_episodes(
        episodes=episodes,
        rss_url=rss_url,
        feed=feed,
        recipient=recipient,
        min_minutes=min_minutes,
        settings=settings,
        db=db,
        mailer=mailer,
        client=client,
        dry_run=dry_run,
        emitted=emitted,
        stats=stats,
    )

    # Non-fatal post-email step: push emailed episodes to the delivery webhook.
    # Mirrors the transcript-cleanup resilience model — a receiver/network
    # failure is logged and retried on a later tick (bounded by
    # WIKI_PUSH_MAX_ATTEMPTS), never blocking or delaying the digest. Skipped in
    # dry-run, when the feature is unconfigured (webhook is None), or when this
    # feed opts out (webhook = false).
    if webhook is not None and feed.webhook and not dry_run and stats is not None:
        await _sweep_webhook_pushes(
            feed=feed, rss_url=rss_url, db=db, webhook=webhook,
            since=webhook_since, stats=stats,
        )

    db.mark_feed_polled(rss_url)
    return len(new_episodes), emailed_count, failed_count


async def _resume_pending_episodes(
    *,
    episodes: list[EpisodeMeta],
    rss_url: str,
    feed: Feed,
    recipient: str,
    min_minutes: int,
    settings: Settings,
    db: Database,
    mailer: Mailer | None,
    client: httpx.AsyncClient,
    dry_run: bool,
    emitted: EmittedLog,
    stats: dict | None,
) -> tuple[int, int]:
    """Advance persisted work independently of the current discovery result."""
    emailed_count = 0
    failed_count = 0
    for ep in db.episodes_in_progress(rss_url):
        # If another instance already emailed this (per the shared delivery log),
        # suppress it — mark emailed without sending — so a stale in-progress row
        # from this instance's own DB (e.g. an old checkpoint) can't duplicate.
        if emitted.contains(rss_url, ep.guid, feed.display_show, ep.title):
            db.set_status(
                feed_url=rss_url, guid=ep.guid, status="emailed",
                terminal_reason="duplicate_identity",
            )
            continue
        # One transport canary per TICK, not per feed: once a transport failure
        # has been seen this tick the outage is established — running every
        # feed's canary too would just serialize N more multi-minute summarizer
        # timeouts into the tick for no new information. Recovery still
        # converges in one tick: with no failure yet, every feed's canary runs.
        if (
            ep.summarize_attempts >= SUMMARIZE_TRANSPORT_MAX_ATTEMPTS
            and stats is not None
            and stats.get("summarize_failed", 0) > 0
        ):
            continue
        pending_meta = next((m for m in episodes if m.guid == ep.guid), None)
        try:
            advanced = await _advance_episode(
                ep=ep,
                meta=pending_meta,
                feed=feed,
                recipient=recipient,
                min_minutes=min_minutes,
                settings=settings,
                db=db,
                mailer=mailer,
                client=client,
                dry_run=dry_run,
                emitted=emitted,
                stats=stats,
            )
            if advanced:
                emailed_count += 1
        except Exception as e:  # noqa: BLE001
            failed_count += 1
            # A summarizer-transport failure (vs a content/email error) is the
            # signal the dead-man's-switch watches — track it separately.
            is_transport = isinstance(e, SummarizerUnavailable)
            if is_transport and stats is not None:
                stats["summarize_failed"] += 1
            event(
                logger, logging.ERROR, "episode.failed",
                feed=feed.name, guid=ep.guid, title=ep.title,
                exc_info=True, **err_fields(e),
            )
            # A summarizer-transport outage goes on the separate, far more
            # generous `summarize_attempts` budget (not the small `attempts` one),
            # so a multi-day summarizer outage can't silently exhaust MAX_ATTEMPTS and
            # dark-hole every pending episode right as the heartbeat clears — yet
            # it's still bounded, so an episode that deterministically fails
            # transport every tick eventually drops. See db.mark_failed.
            # mark_failed reports whether THIS failure exhausted the episode's
            # budget (computed from the post-bump DB value — the tick-start
            # snapshot on `ep` is stale whenever an earlier stage succeeded
            # this tick and reset the counters). If so, the episode leaves the
            # pending queue for good: GC its download artifacts (final audio
            # AND resume fragments like yt-dlp .part files) so dropped
            # episodes can't accumulate hundreds of MB each on the data volume.
            exhausted = db.mark_failed(
                rss_url, ep.guid, f"{type(e).__name__}: {e}", transport=is_transport,
            )
            if exhausted:
                _cleanup_dropped_episode_audio(settings, feed, ep, db)
    return emailed_count, failed_count


def _known_feed_identity(feed: Feed, db: Database) -> str | None:
    """Return a persisted or source-intrinsic identity without network I/O."""
    if feed.audio_url:
        return feed.audio_url
    if feed.youtube_video:
        try:
            return youtube.canonical_watch_url(
                youtube.extract_video_id(feed.youtube_video)
            )
        except youtube.FeedError:
            return None
    if feed.rss_url:
        return feed.rss_url
    if feed.search:
        return (
            db.get_resolved_url(feed.name, feed.search)
            or db.get_unique_feed_url(feed.name)
        )
    if feed.youtube:
        return (
            db.get_resolved_url(feed.name, feed.youtube)
            or db.get_unique_feed_url(feed.name)
        )
    return db.get_unique_feed_url(feed.name)


def _dedupe_discovered_episodes(
    episodes: list[EpisodeMeta], *, feed_name: str,
) -> list[EpisodeMeta]:
    """Keep the newest occurrence of each GUID before applying backlog caps."""
    seen: set[str] = set()
    unique: list[EpisodeMeta] = []
    duplicate_count = 0
    for meta in episodes:
        if meta.guid in seen:
            duplicate_count += 1
            continue
        seen.add(meta.guid)
        unique.append(meta)
    if duplicate_count:
        logger.warning(
            "feed.duplicate_guids name=%r duplicates=%d",
            feed_name, duplicate_count,
        )
    return unique


async def _enrich_new_youtube(
    episodes: list[EpisodeMeta],
    *,
    feed_url: str,
    db: Database,
    cookies: str | None,
    limit: int = MAX_NEW_PER_TICK,
) -> list[EpisodeMeta]:
    """Replace flat-enumeration metadata with authoritative metadata for the
    newest *unseen* uploads.

    Flat channel enumeration is reliable for the video id, duration, and order,
    but carries a provisional title and no upload date. Fetch the real title +
    date via `single_video_episode` until `limit` viable unseen candidates are
    selected — which is all the pipeline will emit this tick — and leave the
    remaining back-catalog on cheap flat metadata (it's suppressed, never
    shown). Failed durationless/live candidates are dropped without consuming
    that selection budget; other per-video failures degrade to flat metadata.
    """
    selected = 0
    out: list[EpisodeMeta] = []
    for meta in episodes:  # newest-first
        # Spend the budget on candidates that can actually occupy a processing
        # slot. A durationless entry whose authoritative fetch fails is dropped
        # for this tick, so it must not consume the only slot and leave the real
        # newest VOD behind it unvalidated.
        if selected < limit and not db.has_episode(feed_url, meta.guid):
            try:
                out.append(
                    await youtube.single_video_episode(meta.audio_url, cookies=cookies)
                )
                selected += 1
                continue
            except youtube.FeedError as e:
                if meta.duration_seconds is None:
                    # No duration in flat mode AND the full-metadata fetch
                    # failed (or said live/premiere): overwhelmingly a
                    # not-yet-downloadable entry. Admitting it on flat metadata
                    # would let it take the cap-1 slot and pre-suppress the
                    # real newest upload behind it. Drop it from THIS tick
                    # only — nothing is persisted, so the next tick re-sees it;
                    # a premiere self-heals once it airs and gains a duration.
                    logger.warning(
                        "youtube.enrich_dropped_durationless guid=%s err=%s "
                        "(benign for a premiere — it self-heals once aired; "
                        "the SAME guid repeating across many ticks means a "
                        "real durationless VOD is being starved by metadata "
                        "fetch failures)",
                        meta.guid, e,
                    )
                    continue
                logger.warning(
                    "youtube.enrich_failed guid=%s err=%s — using flat metadata",
                    meta.guid, e,
                )
                selected += 1
        out.append(meta)
    return out


async def _discover_episodes(
    *,
    feed: Feed,
    min_minutes: int,
    settings: Settings,
    db: Database,
    client: httpx.AsyncClient,
) -> tuple[str, int | None, list[EpisodeMeta]]:
    """Resolve a feed to (identity_url, itunes_collection_id, episodes).

    The identity URL is the stable per-feed key in the DB. Discovery differs by
    source: podcast RSS (iTunes-resolved or pinned), a YouTube channel feed, or a
    single YouTube video.
    """
    # Single direct-audio episode: a one-shot keyed by the audio URL. Like the
    # YouTube-video one-shot, once we've seen it we skip rediscovery entirely —
    # the episode is driven from the DB by `episodes_in_progress` thereafter.
    # (The MCP resolves Apple Podcasts episode URLs to a direct enclosure and
    # writes it here, so the pipeline stays provider-agnostic.)
    if feed.audio_url:
        identity_url = feed.audio_url
        guid = f"audio:{hashlib.sha1(feed.audio_url.encode()).hexdigest()[:16]}"
        if db.has_episode(identity_url, guid):
            return identity_url, None, []
        ep = EpisodeMeta(
            guid=guid,
            title=feed.name,
            # Real release date/runtime if the MCP captured them at ingest (Apple
            # episode); otherwise "" / None so the email omits those lines rather
            # than fabricating the processing time. A bare audio URL carries no
            # metadata, so this is the honest floor.
            published_at=feed.published_at or "",
            duration_seconds=feed.duration_seconds,
            audio_url=feed.audio_url,
            audio_type=None,
            description=None,
        )
        return identity_url, None, [ep]

    # Single YouTube video: a one-episode feed keyed by its canonical watch URL.
    # Once we've seen it, skip the yt-dlp metadata round-trip entirely — the
    # episode is driven from the DB by `episodes_in_progress` thereafter.
    if feed.youtube_video:
        video_id = youtube.extract_video_id(feed.youtube_video)
        identity_url = youtube.canonical_watch_url(video_id)
        if db.has_episode(identity_url, f"yt:video:{video_id}"):
            return identity_url, None, []
        ep = await youtube.single_video_episode(
            feed.youtube_video,
            cookies=settings.PODCAST_DIGEST_YTDLP_COOKIES or None,
        )
        return identity_url, None, [ep]

    # YouTube channel: enumerate recent uploads via yt-dlp (YouTube's old
    # videos.xml Atom endpoint now 404s globally). `feed_url` — the legacy
    # videos.xml URL `resolve_channel_feed` derives — stays the feed's stable
    # primary key so existing episode history isn't orphaned and re-emailed.
    if feed.youtube:
        youtube_spec = feed.youtube
        # Cached identity, keyed by the exact configured spec. Re-resolving a
        # `@handle`/custom URL each tick re-scrapes the channel page, which can
        # pick a *different* channel id (a featured/related channel) → a new PK
        # → first_run re-fires and the newest already-sent video is re-emailed,
        # orphaning the old history. Keying on the spec means editing
        # `youtube:` in feeds.toml naturally re-resolves.
        async def _resolve_yt() -> tuple[str, None]:
            return await youtube.resolve_channel_feed(youtube_spec, client=client), None

        cookies = settings.PODCAST_DIGEST_YTDLP_COOKIES or None
        feed_url, _extra, (_name, episodes) = await _fetch_via_cached_identity(
            db=db, feed_name=feed.name, spec=youtube_spec,
            resolve=_resolve_yt,
            fetch=lambda url: youtube.list_channel_uploads(url, cookies=cookies),
        )
        episodes = await _enrich_new_youtube(
            episodes, feed_url=feed_url, db=db, cookies=cookies,
        )
        return feed_url, None, episodes

    # Podcast RSS (the original path). For a `search` feed, the cached resolved
    # URL prevents per-tick iTunes re-resolution from re-keying the feed under a
    # new PK on a feedUrl change (host migration, http→https, iTunes reordering
    # a no-exact-match result) — first_run would re-fire and re-email. A pinned
    # `rss_url` is already stable (resolve_feed_url returns it as-is), so it
    # skips the cache entirely.
    if feed.search and not feed.rss_url:
        rss_url, itunes_id, body = await _fetch_via_cached_identity(
            db=db, feed_name=feed.name, spec=feed.search,
            resolve=lambda: resolve_feed_url(feed, client=client),
            fetch=lambda url: fetch_rss(url, client=client),
        )
    else:
        rss_url, itunes_id = await resolve_feed_url(feed, client=client)
        body = await fetch_rss(rss_url, client=client)
    return rss_url, itunes_id, parse_episodes(body, min_minutes=min_minutes)


async def _fetch_via_cached_identity(
    *,
    db: Database,
    feed_name: str,
    spec: str,
    resolve,
    fetch,
):
    """Fetch a feed's content via its cached identity URL, with a bounded
    re-key escape hatch. Returns ``(url, resolve_extra, fetch_result)``.

    ``resolve`` is an async no-arg callable returning ``(url, extra)`` (extra:
    e.g. the iTunes collection id, None on a cache hit); ``fetch`` is an async
    ``url -> result`` callable that raises FeedError/httpx.HTTPError on a dead
    or unusable URL.

    Policy, shared by every source type so it can't drift per-branch:
      - cache hit → fetch without re-resolving (identity is stable-by-default);
      - fetch failure on a CACHED url → re-resolve once, and re-key ONLY if
        resolution answers differently — a transient outage with a stable
        resolution re-raises rather than re-keying the feed;
      - a fresh resolution (miss or re-key) is recorded via set_resolved_url,
        which REPLACES the (name, spec) row, so the newest answer always wins.
    """
    extra = None
    cached = db.get_resolved_url(feed_name, spec)
    url = cached
    if url is None:
        url, extra = await resolve()
        db.set_resolved_url(feed_name, spec, url)
    try:
        result = await fetch(url)
    except (FeedError, httpx.HTTPError):
        if cached is None:
            raise
        fresh_url, fresh_extra = await resolve()
        if fresh_url == url:
            raise
        # Adopt the new identity only AFTER it proves fetchable: committing
        # first would let one transient outage + one flaky resolver answer
        # permanently swap a recoverable known-good URL for a dead one.
        result = await fetch(fresh_url)
        event(logger, logging.WARNING, "feed.rekeyed",
              feed=feed_name, old=url, new=fresh_url)
        # Carry the feed's state to the new PK: keeps first_run from
        # re-firing (no duplicate email on single-instance deploys, where the
        # emitted log is off) and keeps in-flight episodes alive instead of
        # re-inserting them under the new URL and pre-suppressing them. Migrate
        # BEFORE updating the cache: if the process dies between the writes,
        # the old cached URL fails again next tick and this idempotent migration
        # repeats. The opposite order could strand state under the old PK.
        migrated = db.rekey_feed(url, fresh_url)
        if migrated:
            logger.info("feed.rekeyed_migrated old=%s new=%s episodes=%d",
                        url, fresh_url, migrated)
        db.set_resolved_url(feed_name, spec, fresh_url)
        url, extra = fresh_url, fresh_extra
    return url, extra, result


async def _advance_episode(
    *,
    ep: EpisodeRow,
    meta: EpisodeMeta | None,
    feed: Feed,
    recipient: str,
    min_minutes: int,
    settings: Settings,
    db: Database,
    mailer: Mailer | None,
    client: httpx.AsyncClient,
    dry_run: bool,
    emitted: EmittedLog | None = None,
    stats: dict | None = None,
) -> bool:
    """Advance one episode through the pipeline. Returns True if we emailed."""
    emitted = emitted or EmittedLog(None)
    feed_slug = _feed_storage_key(feed, ep.feed_url)
    safe_guid = _artifact_key(ep.guid)

    transcript_path = Path(ep.transcript_path) if ep.transcript_path else None
    segments_path = Path(ep.segments_path) if ep.segments_path else None

    # We only need audio if we still need to transcribe. After transcription
    # the audio file is disposable — skip the (re-)download entirely for
    # episodes resuming at `transcribed` or later. This matters when we resume
    # after the email step failed: we'd otherwise re-fetch 80MB for nothing.
    need_transcribe = stage_lt(ep.status, "transcribed") or not (
        transcript_path and transcript_path.exists()
        and segments_path and segments_path.exists()
    )

    # ---- 0. Publisher transcript. When the feed links a usable transcript
    # (Podcasting 2.0), it replaces download + local transcription. Anything
    # unusable — a fetch error, an unknown format, a transcript too short for
    # the episode — falls through to whisper in this same pass, so it never
    # costs a tick. Written in whisper's artifact shape, so every later stage
    # and every resume treats it like a whisper transcript.
    transcript: Transcript | None = None
    is_one_shot = bool(feed.youtube_video or feed.audio_url)
    if (
        need_transcribe
        and settings.PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS
        and feed.publisher_transcripts
        and meta is not None
        and meta.transcripts
    ):
        pub = await publisher.fetch(
            meta.transcripts, client=client, language=feed.source_language,
            declared_seconds=ep.duration_seconds, guid=ep.guid,
        )
        if pub is not None:
            # The length floor normally checks the downloaded audio; with no
            # download, use the transcript's own timing when the feed declared
            # no duration (a declared one was already checked at parse time).
            covered = pub.end_seconds
            if (
                min_minutes and not is_one_shot and ep.duration_seconds is None
                and covered is not None and covered < min_minutes * 60
            ):
                logger.info(
                    "episode.too_short feed=%r guid=%r title=%r seconds=%.0f floor=%dm "
                    "skipping (publisher transcript)",
                    feed.name, ep.guid, ep.title, covered, min_minutes,
                )
                db.set_status(
                    feed_url=ep.feed_url, guid=ep.guid, status="emailed",
                    terminal_reason="below_min_duration",
                )
                return False
            transcript_dir = settings.transcripts_dir / feed_slug
            transcript_dir.mkdir(parents=True, exist_ok=True)
            txt_path = transcript_dir / f"{safe_guid}.txt"
            json_path = transcript_dir / f"{safe_guid}.json"
            _atomic_write_text(txt_path, pub.transcript.text)
            _atomic_write_text(json_path, publisher.to_whisper_json(pub))
            if settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT == "cleaned":
                # Already punctuated and usually speaker-labelled: attach it
                # as readable Markdown instead of spending an LLM cleanup.
                _atomic_write_text(
                    txt_path.with_suffix(".clean.md"),
                    publisher.to_markdown(pub, title=ep.title, show=feed.display_show),
                )
            db.set_status(
                feed_url=ep.feed_url, guid=ep.guid,
                status="transcribed",
                transcript_path=str(txt_path), segments_path=str(json_path),
            )
            transcript = pub.transcript
            transcript_path, segments_path = txt_path, json_path
            need_transcribe = False

    # ---- 1. Download
    audio_path = Path(ep.audio_path) if ep.audio_path else None
    if need_transcribe and (
        stage_lt(ep.status, "downloaded") or not (audio_path and audio_path.exists())
    ):
        audio_dir = settings.audio_dir / feed_slug
        audio_path = await audio_download(
            ep.audio_url,
            dest_dir=audio_dir,
            basename=safe_guid,
            client=client,
            ytdlp_format=settings.PODCAST_DIGEST_YTDLP_FORMAT,
            ytdlp_cookies=settings.PODCAST_DIGEST_YTDLP_COOKIES or None,
        )
        db.set_status(
            feed_url=ep.feed_url, guid=ep.guid,
            status="downloaded", audio_path=str(audio_path),
        )

    # ---- 1b. Length backstop. The declared feed duration is often missing or
    # wrong (and absent for YouTube channels / direct-audio streams), so before
    # spending a transcription we probe the *real* audio and skip anything under
    # the floor — conference soundbites, teasers, promos. Terminal-skip = mark
    # emailed (seen, never sent), like the back-catalog suppressor, so it's
    # deduped and never retried; the teaser audio is disposable. A None probe
    # (unreadable) falls through to transcription rather than silently dropping.
    #
    # Scoped to recurring streams only. Single-pick one-shots (`youtube_video`,
    # `audio_url`) are an explicit user choice — exempt, matching how the
    # parse-time `min_minutes` filter already treats them.
    if need_transcribe and min_minutes and not is_one_shot and audio_path and audio_path.exists():
        real_seconds = await probe_duration(audio_path)
        if real_seconds is not None and real_seconds < min_minutes * 60:
            logger.info(
                "episode.too_short feed=%r guid=%r title=%r seconds=%.0f floor=%dm skipping",
                feed.name, ep.guid, ep.title, real_seconds, min_minutes,
            )
            audio_cleanup(audio_path)
            db.clear_audio_path(ep.feed_url, ep.guid)
            db.set_status(
                feed_url=ep.feed_url, guid=ep.guid, status="emailed",
                terminal_reason="below_min_duration",
            )
            return False

    # ---- 2. Transcribe (only compute if needed; defer load until summarize stage)
    if need_transcribe:
        if audio_path is None or not audio_path.exists():
            raise RuntimeError(
                f"transcribe needs audio but path is missing: {audio_path!r}"
            )
        transcript_dir = settings.transcripts_dir / feed_slug
        # Non-English feeds (feeds.toml `language`) transcribe with the
        # multilingual model in the original language; the summarizer then
        # reads that transcript and emits English. English feeds keep the .en
        # model + configured language, byte-for-byte as before.
        lang = feed.source_language
        whisper_model = (
            settings.PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL if lang
            else settings.PODCAST_DIGEST_WHISPER_MODEL
        )
        transcript, txt_path, json_path = await transcribe(
            audio_path,
            out_dir=transcript_dir,
            basename=safe_guid,
            whisper_bin=settings.PODCAST_DIGEST_WHISPER_BIN,
            model_path=whisper_model,
            language=lang or settings.PODCAST_DIGEST_WHISPER_LANGUAGE,
            threads=settings.PODCAST_DIGEST_WHISPER_THREADS,
        )
        transcript_path = txt_path
        segments_path = json_path
        db.set_status(
            feed_url=ep.feed_url, guid=ep.guid,
            status="transcribed",
            transcript_path=str(txt_path), segments_path=str(json_path),
        )

    # ---- 2b. Full-program content identity. Different publishers/platforms
    # can give one program unrelated GUIDs, titles, and enclosure URLs. Compare
    # the raw transcript before cleanup/summarization so a renamed or lightly
    # re-cut copy does not consume more LLM work or emit another digest.
    content_fingerprint = _load_or_create_content_fingerprint(
        ep=ep,
        transcript=transcript,
        transcript_path=transcript_path,
        db=db,
    )
    cutoff = datetime.now(UTC) - timedelta(days=CONTENT_WINDOW_DAYS)
    candidates = db.delivered_content_since(cutoff.isoformat(timespec="seconds"))
    candidates.extend(emitted.content_candidates(cutoff))
    content_match = find_content_match(content_fingerprint, candidates)
    if content_match is not None:
        event(
            logger,
            logging.INFO,
            "episode.content_duplicate",
            feed=feed.name,
            guid=ep.guid,
            title=ep.title,
            kind=content_match.kind,
            score=round(content_match.score, 4),
            matched_feed=content_match.candidate.feed_url,
            matched_guid=content_match.candidate.guid,
            matched_title=content_match.candidate.title,
            dry_run=dry_run,
        )
        if not dry_run:
            if audio_path and audio_path.exists():
                audio_cleanup(audio_path)
                db.clear_audio_path(ep.feed_url, ep.guid)
            db.set_status(
                feed_url=ep.feed_url,
                guid=ep.guid,
                status="emailed",
                terminal_reason=f"duplicate_transcript_{content_match.kind}",
            )
            return False

    completer = build_completer(settings)

    # ---- 3a. Clean transcript → Markdown (when attaching cleaned). Done BEFORE
    # summarizing so the summary is built from the *cleaned* text (correct
    # names/tickers, speaker-aware, ad-reads removed) rather than raw ASR.
    #
    # Cleanup is a real prerequisite, so it uses the same resilience model as the
    # other stages: on failure the episode does NOT advance — it stays at
    # `transcribed` and the whole clean is retried next tick, bounded by
    # CLEAN_MAX_TICKS via `clean_attempts` (a counter separate from `attempts`,
    # so cleanup retries never trip MAX_ATTEMPTS and drop the episode). Only
    # after exhausting that budget do we degrade: the final attempt accepts a
    # partial clean, and if even that yields nothing we summarize from raw and
    # ship without an attachment. The .clean.md persists so a later
    # mail/summarize failure never re-runs the (expensive) cleanup.
    attachments: list[MailAttachment] = []
    cleaned_md: str | None = None
    attach_mode = settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT
    can_attach = (
        not dry_run
        and stage_lt(ep.status, "emailed")
        and transcript_path is not None
        and Path(transcript_path).exists()
    )

    if attach_mode == "raw" and can_attach:
        assert transcript_path is not None
        tpath = Path(transcript_path)
        try:
            attachments.append(MailAttachment(
                name=f"{_safe_basename(ep.title)}.txt",
                content=tpath.read_bytes(),
                content_type="text/plain; charset=utf-8",
            ))
        except OSError as e:
            logger.warning(
                "transcript.attach_failed mode=raw guid=%r err=%s", ep.guid, e,
            )

    elif attach_mode == "cleaned" and can_attach:
        assert transcript_path is not None
        tpath = Path(transcript_path)
        clean_path = tpath.with_suffix(".clean.md")
        # Ensure the cleaned transcript exists — but only while we can still feed
        # it to the summary (before `summarized`). Defer (don't advance) on a
        # transient failure; degrade once the per-episode budget is spent.
        if not _is_nonempty_file(clean_path) and stage_lt(ep.status, "summarized"):
            is_final = ep.clean_attempts + 1 >= CLEAN_MAX_TICKS
            try:
                if transcript is None:
                    assert segments_path is not None
                    transcript = load_whisper_json(
                        segments_path, fallback_text_path=tpath,
                    )
                md = await asyncio.to_thread(
                    clean_transcript,
                    SummarizationInput(
                        podcast_name=feed.display_show, title=ep.title,
                        published_at=ep.published_at,
                        duration_seconds=ep.duration_seconds,
                        description=meta.description if meta else None,
                        transcript_text=transcript.text,
                        segments=transcript.segments,
                    ),
                    completer=completer,
                    guid=ep.guid,
                    allow_partial=is_final,
                )
                if not md.strip():
                    raise RuntimeError("transcript cleanup returned empty Markdown")
                _atomic_write_text(clean_path, md)
            except Exception as e:  # noqa: BLE001
                db.bump_clean_attempt(ep.feed_url, ep.guid)
                if isinstance(e, SummarizerUnavailable) and stats is not None:
                    # Cleanup uses the same backend as summarization but
                    # intentionally defers in-place instead of raising. Surface
                    # the outage explicitly so the scheduler cannot emit a
                    # false healthy heartbeat while every clean is parked.
                    stats["summarize_failed"] += 1
                if not is_final:
                    event(
                        logger, logging.INFO, "episode.clean_deferred",
                        guid=ep.guid, attempt=ep.clean_attempts + 1,
                        max=CLEAN_MAX_TICKS, **err_fields(e),
                    )
                    return False  # stay at `transcribed`; retry the clean next tick
                event(
                    logger, logging.WARNING, "episode.clean_gave_up",
                    guid=ep.guid, after_ticks=ep.clean_attempts + 1,
                    note="summarizing from raw, no attachment", **err_fields(e),
                )
        # Attach the cleaned transcript if we have one (this tick or a prior one).
        if _is_nonempty_file(clean_path):
            cleaned_md = clean_path.read_text(encoding="utf-8")
            attachments.append(MailAttachment(
                name=f"{_safe_basename(ep.title)}.md",
                content=clean_path.read_bytes(),
                content_type="text/markdown; charset=utf-8",
            ))

    # ---- 3b. Summarize (prefer the cleaned transcript when available)
    summary_html = ep.summary_html
    if stage_lt(ep.status, "summarized") or not summary_html:
        if cleaned_md is not None:
            # The cleaned Markdown already carries speaker labels + [hh:mm:ss]
            # timestamps; feed it directly as the transcript text.
            summary_input = SummarizationInput(
                podcast_name=feed.display_show, title=ep.title,
                published_at=ep.published_at,
                duration_seconds=ep.duration_seconds,
                description=meta.description if meta else None,
                transcript_text=cleaned_md, segments=[],
                source_language=feed.source_language,
            )
        else:
            if transcript is None:
                if transcript_path is None or segments_path is None:
                    raise RuntimeError(
                        "summarize needs a transcript but no path is recorded"
                    )
                transcript = load_whisper_json(
                    segments_path, fallback_text_path=transcript_path,
                )
            summary_input = SummarizationInput(
                podcast_name=feed.display_show, title=ep.title,
                published_at=ep.published_at,
                duration_seconds=ep.duration_seconds,
                description=meta.description if meta else None,
                transcript_text=transcript.text, segments=transcript.segments,
                source_language=feed.source_language,
            )
        # Run the blocking call off the event loop (an in-process SDK call or a
        # command). A failure raises, so the episode stays at its prior stage
        # and the next tick retries.
        summary_html = await asyncio.to_thread(
            summarize, summary_input,
            completer=completer,
            style=settings.PODCAST_DIGEST_SUMMARY_STYLE,
            prompt_file=settings.PODCAST_DIGEST_SUMMARY_PROMPT_FILE,
            guid=ep.guid,
        )
        # Sanitize at the choke point where the summary is PRODUCED, so every
        # consumer — the DB row, the email, the webhook, anything added later
        # — sees only the allowed tag set. (The email wrapper sanitizes again
        # for rows stored before this existed; that second pass is a no-op on
        # anything written here.) Empty-after-sanitize means the model emitted
        # nothing but disallowed markup — treat like an empty response: fail
        # and retry rather than persist a blank digest.
        summary_html = sanitize_summary_html(summary_html)
        if not summary_html.strip():
            raise RuntimeError("summary empty after sanitization")
        db.set_status(
            feed_url=ep.feed_url, guid=ep.guid,
            status="summarized", summary_html=summary_html,
        )
        if stats is not None:
            stats["summarize_ok"] = stats.get("summarize_ok", 0) + 1
        # Deliberately NO bulk reset of other episodes' transport counters
        # here: this episode's own counters were just cleared by set_status,
        # and a post-outage throttled backlog drains itself one canary per
        # tick (each success clears its own counters the same way). A bulk
        # reset would also zero a deterministically-failing episode's hard-cap
        # progress every time its feed publishes normally — making the drop
        # cap unreachable on any active feed and resurrecting already-dropped
        # episodes.

    # ---- 4. Email
    sent_now = False
    if stage_lt(ep.status, "emailed"):
        subject = _build_subject(feed.display_show, ep.title, ep.duration_seconds)
        if dry_run:
            # In dry-run mode we intentionally do NOT advance status to
            # `emailed` — otherwise a single dry-run would silently make the
            # next real run skip the episode.
            logger.info(
                "mail.dry_run subject=%r bytes=%d (status NOT advanced)",
                subject, len(summary_html),
            )
        else:
            if mailer is None:
                raise RuntimeError("no mailer configured for a real (non-dry-run) tick")
            try:
                await asyncio.to_thread(
                    mailer.send,
                    to=recipient,
                    subject=subject,
                    body_html=_wrap_email(
                        summary_html, feed.display_show, ep.title, ep.audio_url,
                        published_at=ep.published_at,
                        duration_seconds=ep.duration_seconds,
                        original_language=feed.source_language,
                        transcript_source=(
                            publisher.source_of(segments_path) if segments_path else "whisper"
                        ),
                    ),
                    attachments=attachments,
                )
            except MailProbablyDelivered as e:
                # The message was fully transmitted; only the server's verdict
                # never arrived. Sending isn't idempotent, so retrying would
                # re-send the digest up to MAX_ATTEMPTS times — the duplicate
                # storm this branch exists to prevent. Treat as delivered:
                # advance + record, and say so loudly. Worst case (the server
                # really did drop it) is one missing digest, visible here and
                # recoverable; the alternative was N duplicates.
                event(logger, logging.WARNING, "mail.probably_delivered",
                      guid=ep.guid, title=ep.title, **err_fields(e))
            now = datetime.now(UTC).isoformat(timespec="seconds")
            db.set_status(
                feed_url=ep.feed_url, guid=ep.guid,
                status="emailed", emailed_at=now, terminal_reason="sent",
            )
            # Record in the shared delivery log so another instance won't
            # re-send this episode. Best-effort: a failed append never fails the send.
            emitted.record(
                ep.feed_url,
                ep.guid,
                now,
                feed.display_show,
                ep.title,
                fingerprint=content_fingerprint,
            )
            sent_now = True

    # ---- 5. Cleanup audio (only after a real send; dry-run keeps audio so the
    # next real tick doesn't have to re-download).
    if sent_now and audio_path and audio_path.exists():
        audio_cleanup(audio_path)
        db.clear_audio_path(ep.feed_url, ep.guid)

    return sent_now


def _load_or_create_content_fingerprint(
    *,
    ep: EpisodeRow,
    transcript: Transcript | None,
    transcript_path: Path | None,
    db: Database,
) -> TranscriptFingerprint:
    if (
        ep.content_sha256 is not None
        and ep.content_minhash is not None
        and ep.content_token_count is not None
    ):
        try:
            return TranscriptFingerprint.from_storage(
                ep.content_sha256,
                ep.content_minhash,
                ep.content_token_count,
            )
        except ValueError:
            logger.warning(
                "episode.invalid_content_fingerprint feed=%s guid=%s recomputing",
                ep.feed_url,
                ep.guid,
            )

    if transcript is not None:
        transcript_text = transcript.text
    elif transcript_path is not None:
        transcript_text = transcript_path.read_text(encoding="utf-8")
    else:
        raise RuntimeError("content fingerprint needs a persisted transcript")

    fingerprint = fingerprint_transcript(transcript_text)
    db.set_content_fingerprint(ep.feed_url, ep.guid, fingerprint)
    return fingerprint


# ---------------------------------------------------------------------------
# Delivery webhook (non-fatal, post-email)
# ---------------------------------------------------------------------------
# Delivery state lives in the episodes table's wiki_* columns (named after the
# first receiver this fed; kept for schema compatibility).


async def _sweep_webhook_pushes(
    *,
    feed: Feed,
    rss_url: str,
    db: Database,
    webhook: WebhookClient,
    stats: dict,
    since: str | None = None,
) -> None:
    """Push every emailed-but-unpushed episode of this feed to the webhook.

    Uniform path for both freshly-emailed episodes (eligible the same tick) and
    earlier ones whose push failed (retried until the budget is spent). Every
    failure is swallowed — this step must never break the digest. The blocking
    POST runs in a thread (like mailer.send): its payload carries a full
    transcript, and a wedged receiver must not freeze the event loop — the
    scheduler's wake polling and every concurrent subprocess timeout live there.
    """
    stats.setdefault("webhook_pushed", 0)
    stats.setdefault("webhook_failed", 0)
    for ep in db.episodes_pending_wiki(rss_url, emailed_since=since):
        transcript_text = _read_transcript_for_webhook(ep)
        if transcript_text is None:
            # Emailed but no transcript on disk (degraded episode) — nothing
            # useful to deliver. Burn an attempt so it doesn't retry forever.
            db.bump_wiki_attempt(ep.feed_url, ep.guid)
            logger.warning("webhook.skip_no_transcript guid=%r title=%r", ep.guid, ep.title)
            continue
        try:
            # summary_html is sanitized at production (and legacy rows by the
            # one-time DB migration), but the webhook is an outbound surface
            # like the email, so it gets the same cheap belt-and-braces pass —
            # the at-rest invariant has a real hole: a version ROLLBACK after
            # the migration stamp writes raw rows the re-upgrade never
            # re-cleans (user_version is already 1). The feed-controlled
            # enclosure URL gets its scheme gate: http(s) or nothing.
            episode = DeliveredEpisode(
                show=feed.display_show,
                title=ep.title,
                published_at=ep.published_at,
                guid=ep.guid,
                listen_url=safe_listen_url(ep.audio_url) or "",
                summary=sanitize_summary_html(ep.summary_html or ""),
                transcript=transcript_text,
            )
            ack = await asyncio.to_thread(webhook.push, episode)
            # This is the persistence boundary: never advance the watermark
            # without the receiver's acknowledgement when one is required, even
            # if a replacement/test client violates WebhookClient's contract.
            if webhook.ack_field and not (ack or "").strip():
                raise WebhookError(f"webhook response missing {webhook.ack_field!r}")
            db.set_wiki_pushed(ep.feed_url, ep.guid)
            stats["webhook_pushed"] += 1
        except Exception as e:  # noqa: BLE001
            db.bump_wiki_attempt(ep.feed_url, ep.guid)
            stats["webhook_failed"] += 1
            logger.warning(
                "webhook.push_failed guid=%r attempt=%d/%d err=%s",
                ep.guid, ep.wiki_push_attempts + 1, WIKI_PUSH_MAX_ATTEMPTS, e,
            )


def _read_transcript_for_webhook(ep: EpisodeRow) -> str | None:
    """The push payload transcript: prefer the cleaned `.clean.md` (speaker
    labels + timestamps), fall back to the raw `.txt`. None if neither is on
    disk."""
    if not ep.transcript_path:
        return None
    tpath = Path(ep.transcript_path)
    for p in (tpath.with_suffix(".clean.md"), tpath):
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup_dropped_episode_audio(
    settings: Settings, feed: Feed, ep: EpisodeRow, db: Database,
) -> None:
    """Best-effort removal of every download artifact of a dropped episode.

    Covers not just the final audio file (audio.cleanup's job on the happy
    path) but the resume fragments a timed-out yt-dlp keeps between ticks
    (`<basename>.dl.*`, `.part`, `.partial`) — once the episode leaves the
    pending queue nothing would ever touch them again. `_artifact_key` output
    is ASCII and glob-safe, and every
    artifact is `<basename>.<something>`, so the `.` anchor makes the glob
    exact — a bare `{base}*` would also eat a sibling episode whose basename
    merely starts with this one's (numeric guids: '?p=123' vs '?p=1234').
    """
    bases = {_artifact_key(ep.guid), _safe_basename(ep.guid)}
    removed = 0
    directories = {
        settings.audio_dir / _feed_storage_key(feed, ep.feed_url),
        settings.audio_dir / feed.identity(),
    }
    if ep.audio_path:
        directories.add(Path(ep.audio_path).parent)
    for directory in directories:
        for base in bases:
            for p in directory.glob(f"{base}.*"):
                with contextlib.suppress(OSError):
                    p.unlink()
                    removed += 1
    if removed:
        db.clear_audio_path(ep.feed_url, ep.guid)
        logger.info(
            "episode.dropped_gc guid=%r files_removed=%d", ep.guid, removed,
        )


def _safe_basename(guid: str) -> str:
    """Filesystem-safe basename derived from a (possibly URL-shaped) GUID."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in guid)
    cleaned = cleaned.strip("-")
    if len(cleaned) > 100:
        # Tail is more discriminating for URL-shaped GUIDs.
        cleaned = cleaned[-100:]
    return cleaned or "episode"


def _artifact_key(guid: str) -> str:
    """Collision-resistant ASCII artifact id accepted by local-transcriber."""
    ascii_guid = unicodedata.normalize("NFKD", guid).encode(
        "ascii", "ignore",
    ).decode("ascii")
    readable = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in ascii_guid
    ).strip("-")
    readable = (readable or "episode")[:64].rstrip("-") or "episode"
    digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()[:32]
    return f"{readable}--{digest}"


def _feed_storage_key(feed: Feed, feed_url: str) -> str:
    """ASCII feed slug plus source hash, preventing same-name collisions."""
    ascii_name = unicodedata.normalize("NFKD", feed.name).encode(
        "ascii", "ignore",
    ).decode("ascii").lower()
    slug = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in ascii_name
    ).strip("-")
    slug = (slug or "feed")[:64].rstrip("-") or "feed"
    digest = hashlib.sha256(feed_url.encode("utf-8")).hexdigest()[:12]
    return f"{slug}--{digest}"


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    """Durably publish UTF-8 text without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _fmt_duration(seconds: int | None) -> str:
    if not seconds:
        return ""
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _fmt_release_date(published_at: str | None) -> str:
    """ISO8601 -> 'D Mon YYYY' (e.g. '10 Jun 2025'); falls back to the raw string."""
    if not published_at:
        return ""
    try:
        dt = datetime.fromisoformat(published_at)
    except ValueError:
        return _escape(published_at)
    return dt.strftime("%-d %b %Y")


def _build_subject(podcast: str, title: str, duration: int | None) -> str:
    dur = _fmt_duration(duration)
    return f"[Podcast] {podcast} — {title}{f' ({dur})' if dur else ''}"


# Display names for the language codes we expect to see in feeds.toml; any
# other code falls back to itself in the email note.
_LANGUAGE_NAMES = {
    "zh": "Mandarin",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}


def _wrap_email(
    summary_html: str,
    podcast: str,
    title: str,
    audio_url: str,
    published_at: str | None = None,
    duration_seconds: int | None = None,
    original_language: str | None = None,
    transcript_source: str = "whisper",
) -> str:
    """Add a title + episode-metadata header and a footer around the model output.

    The metadata line carries the podcast name, release date, and runtime so the
    digest is useful on its own without opening the episode. The model summary is
    sanitized at production and re-sanitized here (see sanitize.py) — the
    email is the outward surface, so it never trusts storage, cheap as the
    second pass is.
    """
    meta_bits = [_escape(podcast)]
    released = _fmt_release_date(published_at)
    if released:
        meta_bits.append(f"Released {released}")
    dur = _fmt_duration(duration_seconds)
    if dur:
        meta_bits.append(dur)
    if original_language:
        name = _LANGUAGE_NAMES.get(original_language, original_language)
        meta_bits.append(f"Translated from {_escape(name)}")
    safe_url = safe_listen_url(audio_url)
    footer_bits = []
    if safe_url:
        footer_bits.append(f'<a href="{_escape(safe_url)}">Listen to the episode</a>')
    if transcript_source == "publisher":
        footer_bits.append("Summarized from the publisher's transcript")
    listen = (
        f'<hr style="border:none;border-top:1px solid #ddd;margin:24px 0;">'
        f'<p style="color:#666;font-size:12px;">{" · ".join(footer_bits)}</p>'
        if footer_bits else ""
    )
    return (
        f'<p style="margin:0 0 4px 0;font-size:15px;"><strong>{_escape(title)}</strong></p>'
        f'<p style="color:#666;font-size:12px;margin:0 0 16px 0;">'
        f"{' · '.join(meta_bits)}</p>"
        f"{sanitize_summary_html(summary_html)}"
        f"{listen}"
    )


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
