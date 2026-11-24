"""Orchestration: one tick = poll every feed, advance every episode."""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import logging
import os
import re
import secrets
import unicodedata
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
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
    DeliveryRow,
    EpisodeRow,
    stage_lt,
)
from .dedupe import (
    CONTENT_WINDOW_DAYS,
    TranscriptFingerprint,
    find_content_match,
    fingerprint_transcript,
)
from .delivery import DeliveryHeld, Envelope, send_digest
from .emitted import EmittedLog
from .feeds import (
    SYNTHETIC_GUID_PREFIX,
    EpisodeMeta,
    FeedError,
    fetch_rss,
    parse_episodes,
    resolve_feed_url,
)
from .identity import BUILD, BuildIdentity
from .llm import SummarizerUnavailable, build_completer
from .log import err_fields, event
from .mail import MailAttachment, Mailer, _html_to_text
from .net import build_async_client
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
    if not dry_run:
        finalize_deliveries(db, emitted)
    async with build_async_client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
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
                    max_new_per_tick=feeds_file.max_new_per_tick_for(feed),
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
                stats["feed_failed"] = stats.get("feed_failed", 0) + 1
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
    if not dry_run:
        _backfill_search_index(db)
    return stats


async def _process_feed(
    *,
    feed: Feed,
    recipient: str,
    min_minutes: int,
    max_new_per_tick: int = MAX_NEW_PER_TICK,
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
            feed=feed,
            min_minutes=min_minutes,
            max_new_per_tick=max_new_per_tick,
            settings=settings,
            db=db,
            client=client,
        )
    except Exception as discovery_error:  # noqa: BLE001
        known_identity = _known_feed_identity(feed, db)
        if known_identity is None:
            raise
        identity_url = known_identity
        if stats is not None:
            stats["feed_failed"] = stats.get("feed_failed", 0) + 1
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
            if episode.audio_url:
                db.refresh_pending_episode_media(
                    feed_url=rss_url,
                    guid=episode.guid,
                    audio_url=episode.audio_url,
                    duration_seconds=episode.duration_seconds,
                )
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
    cap = 1 if first_run else max_new_per_tick
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
        delivery = db.get_delivery(rss_url, ep.guid)
        if delivery is None and emitted.contains(rss_url, ep.guid, feed.display_show, ep.title):
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
            delivery = db.get_delivery(rss_url, ep.guid)
            if delivery is not None and delivery.state == "unknown":
                # Neither episode retry budgets nor artifact GC can resolve an
                # external effect. Preserve its high watermark and evidence.
                logger.warning("%s", DeliveryHeld(delivery.operation_id))
                continue
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
    min_minutes: int = 0,
) -> list[EpisodeMeta]:
    """Replace flat-enumeration metadata with authoritative metadata for the
    newest *unseen* uploads.

    Flat channel enumeration is reliable for the video id, duration, and order,
    but carries a provisional title and no upload date. Fetch the real title +
    date via `single_video_episode` until `limit` viable unseen candidates are
    selected — which is all the pipeline will emit this tick — and leave the
    remaining back-catalog on cheap flat metadata (it's suppressed, never
    shown). Failed durationless/live candidates and enriched uploads shorter
    than ``min_minutes`` are dropped without consuming that selection budget;
    other per-video failures degrade to flat metadata.
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
                enriched = await youtube.single_video_episode(
                    meta.audio_url, cookies=cookies,
                )
                if (
                    min_minutes
                    and enriched.duration_seconds is not None
                    and enriched.duration_seconds < min_minutes * 60
                ):
                    continue
                out.append(enriched)
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
    max_new_per_tick: int = MAX_NEW_PER_TICK,
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
        # Filter known-short channel uploads before enrichment and cap assignment
        # so a 30s teaser cannot consume the cap-1 slot and pre-suppress the real
        # qualifying episode behind it.
        if min_minutes:
            episodes = [
                ep for ep in episodes
                if ep.duration_seconds is None or ep.duration_seconds >= min_minutes * 60
            ]
        episodes = await _enrich_new_youtube(
            episodes, feed_url=feed_url, db=db, cookies=cookies,
            limit=max_new_per_tick, min_minutes=min_minutes,
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
    # Unknown sends remain held. A proven non-submission retains its frozen
    # payload, but positive delivery evidence can make retrying unnecessary.
    existing_delivery = db.get_delivery(ep.feed_url, ep.guid)
    if existing_delivery is not None:
        if dry_run:
            logger.info("mail.dry_run existing operation=%s state=%s",
                        existing_delivery.operation_id, existing_delivery.state)
            return False
        if existing_delivery.state == "not_submitted":
            reason = None
            if emitted.contains(ep.feed_url, ep.guid, existing_delivery.show or feed.display_show, ep.title):
                reason = "duplicate_identity"
            else:
                fingerprint = _load_or_create_content_fingerprint(
                    ep=ep, transcript=None,
                    transcript_path=Path(ep.transcript_path) if ep.transcript_path else None,
                    db=db,
                )
                cutoff = datetime.now(UTC) - timedelta(days=CONTENT_WINDOW_DAYS)
                candidates = db.delivered_content_since(cutoff.isoformat(timespec="seconds"))
                candidates.extend(emitted.content_candidates(cutoff))
                match = find_content_match(fingerprint, candidates) if fingerprint else None
                if match is not None:
                    reason = f"duplicate_transcript_{match.kind}"
            if reason is not None:
                db.suppress_delivery(existing_delivery.operation_id, reason)
                if ep.audio_path:
                    audio_cleanup(Path(ep.audio_path))
                    if not Path(ep.audio_path).exists():
                        db.clear_audio_path(ep.feed_url, ep.guid)
                return False
        if mailer is None:
            raise RuntimeError("no mailer configured for a real tick")
        sent = await send_digest(db, mailer, ep.feed_url, ep.guid)
        if sent:
            finalize_deliveries(db, emitted)
        return sent

    feed_slug = _feed_storage_key(feed, ep.feed_url)
    safe_guid = _artifact_key(ep.guid)

    transcript_path = Path(ep.transcript_path) if ep.transcript_path else None
    segments_path = Path(ep.segments_path) if ep.segments_path else None

    # We only need audio if we still need to transcribe. After transcription
    # the audio file is disposable — skip the (re-)download entirely for
    # episodes resuming at `transcribed` or later. This matters when we resume
    # after the email step failed: we'd otherwise re-fetch 80MB for nothing.
    need_transcribe = stage_lt(ep.status, "summarized") and (
        stage_lt(ep.status, "transcribed") or not (
            transcript_path and transcript_path.exists()
            and segments_path and segments_path.exists()
        )
    )

    # ---- 0. Publisher transcript. When the feed links a usable transcript
    # (Podcasting 2.0), it replaces download + local transcription. Anything
    # unusable — a fetch error, an unknown format, a transcript too short for
    # the episode — falls through to local transcription in this same pass, so it never
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
                status="transcribed" if stage_lt(ep.status, "transcribed") else ep.status,
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
            status="downloaded" if stage_lt(ep.status, "downloaded") else ep.status,
            audio_path=str(audio_path),
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
        # Transcribe in the feed's language. English can use Parakeet and
        # Chinese can use Qwen, with the configured Whisper models as fallback.
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
            parakeet_model_dir=settings.PODCAST_DIGEST_PARAKEET_MODEL_DIR,
            parakeet_worker_ssh_config=settings.PODCAST_DIGEST_PARAKEET_WORKER_SSH_CONFIG,
            parakeet_threads=settings.PODCAST_DIGEST_PARAKEET_THREADS,
            qwen_model_dir=settings.PODCAST_DIGEST_QWEN_MODEL_DIR,
            qwen_threads=settings.PODCAST_DIGEST_QWEN_THREADS,
        )
        transcript_path = txt_path
        segments_path = json_path
        db.set_status(
            feed_url=ep.feed_url, guid=ep.guid,
            status="transcribed" if stage_lt(ep.status, "transcribed") else ep.status,
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
    content_match = (
        find_content_match(content_fingerprint, candidates) if content_fingerprint else None
    )
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
    has_transcript_file = (
        stage_lt(ep.status, "emailed")
        and transcript_path is not None
        and Path(transcript_path).exists()
    )
    can_attach = not dry_run and has_transcript_file

    if attach_mode == "raw" and can_attach:
        assert transcript_path is not None
        tpath = Path(transcript_path)
        try:
            attachments.append(MailAttachment(
                name=_attachment_filename(ep.title, ".txt"),
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
        # it to the summary (before `summarized`) and within CLEAN_MAX_TICKS.
        if (
            not _is_nonempty_file(clean_path)
            and stage_lt(ep.status, "summarized")
            and ep.clean_attempts < CLEAN_MAX_TICKS
        ):
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
            if can_attach:
                attachments.append(MailAttachment(
                    name=_attachment_filename(ep.title, ".md"),
                    content=cleaned_md.encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                ))

    # ---- 3b. Summarize (prefer the cleaned transcript when available)
    summary_html = ep.summary_html
    summary_build = ep.summary_build
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
        summary_build = BUILD
        db.set_status(
            feed_url=ep.feed_url, guid=ep.guid,
            status="summarized", summary_html=summary_html,
            summary_build=summary_build,
        )
        _index_episode_for_search(
            db,
            feed_url=ep.feed_url,
            guid=ep.guid,
            show=feed.display_show,
            title=ep.title,
            summary_html=summary_html,
            transcript_text=cleaned_md or (
                transcript.text if transcript is not None else _read_transcript_for_webhook(ep) or ""
            ),
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
            sent_now = await send_digest(
                db, mailer, ep.feed_url, ep.guid,
                Envelope(
                    to=recipient, subject=subject,
                    body_html=_wrap_email(
                        summary_html, feed.display_show, ep.title, ep.audio_url,
                        published_at=ep.published_at,
                        duration_seconds=ep.duration_seconds,
                        original_language=feed.source_language,
                        transcript_source=(
                            publisher.source_of(segments_path)
                            if segments_path and segments_path.exists() else "whisper"
                        ),
                        generated_by=summary_build,
                    ),
                    attachments=tuple(attachments),
                ),
                show=feed.display_show,
            )
            if sent_now:
                finalize_deliveries(db, emitted)

    return sent_now


def finalize_deliveries(db: Database, emitted: EmittedLog) -> None:
    """Retry accepted projections under the writer lock, even for removed feeds."""
    for delivery in db.deliveries_pending_finalization():
        try:
            _finalize_delivery(delivery, db, emitted)
        except Exception as exc:  # noqa: BLE001
            event(logger, logging.WARNING, "delivery.finalization_pending",
                  operation_id=delivery.operation_id, **err_fields(exc))


def _finalize_delivery(delivery: DeliveryRow, db: Database, emitted: EmittedLog) -> None:
    current = db.get_episode(delivery.feed_url, delivery.guid)
    assert current is not None and current.emailed_at is not None
    fingerprint = _load_or_create_content_fingerprint(
        ep=current, transcript=None,
        transcript_path=Path(current.transcript_path) if current.transcript_path else None,
        db=db,
    )
    recorded = emitted.record(current.feed_url, current.guid, current.emailed_at,
                              delivery.show, current.title, fingerprint=fingerprint)
    if current.audio_path:
        path = Path(current.audio_path)
        audio_cleanup(path)
        if path.exists():
            return
        db.clear_audio_path(current.feed_url, current.guid)
    if recorded:
        db.finalize_delivery(delivery.operation_id)


def _load_or_create_content_fingerprint(
    *,
    ep: EpisodeRow,
    transcript: Transcript | None,
    transcript_path: Path | None,
    db: Database,
) -> TranscriptFingerprint | None:
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

    if (
        not stage_lt(ep.status, "summarized")
        and transcript is None
        and not (transcript_path and transcript_path.exists())
    ):
        # A persisted summary is sufficient to deliver. Missing artifacts must
        # never move its stage backward or reset its failure budgets.
        return None
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


def _index_episode_for_search(
    db: Database,
    *,
    feed_url: str,
    guid: str,
    show: str,
    title: str,
    summary_html: str | None,
    transcript_text: str,
) -> None:
    """Best-effort FTS5 index update; never blocks or fails the digest pipeline."""
    try:
        db.upsert_search_index(
            feed_url=feed_url,
            guid=guid,
            show=show,
            title=title,
            summary_text=(
                _html_to_text(summary_html, include_links=False).strip()
                if summary_html else ""
            ),
            transcript_text=transcript_text,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("search.index_failed guid=%r err=%s", guid, exc)


def _backfill_search_index(db: Database) -> None:
    """Populate episode_search for any historical episodes not yet indexed."""
    try:
        pending = db.unindexed_searchable_episodes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("search.backfill_failed err=%s", exc)
        return
    if not pending:
        return
    try:
        with db.tx():
            for ep, show in pending:
                _index_episode_for_search(
                    db,
                    feed_url=ep.feed_url,
                    guid=ep.guid,
                    show=show,
                    title=ep.title,
                    summary_html=ep.summary_html,
                    transcript_text=_read_transcript_for_webhook(ep) or "",
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("search.backfill_tx_failed err=%s", exc)


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


def _attachment_filename(title: str, ext: str) -> str:
    """Human-readable, filesystem-safe attachment filename truncated at word boundaries.

    Preserves Unicode letters and digits (e.g. CJK titles) while collapsing
    punctuation and whitespace into single hyphens.
    """
    raw = "".join(c if c.isalnum() else "-" for c in title)
    collapsed = re.sub(r"-+", "-", raw).strip("-")
    if len(collapsed) > 80:
        cut = collapsed[:80]
        if "-" in cut:
            cut = cut.rsplit("-", 1)[0]
        collapsed = cut.rstrip("-") or collapsed[:80]
    return f"{collapsed or 'episode'}{ext}"


def _artifact_key(guid: str) -> str:
    """Collision-resistant ASCII artifact id accepted by local-transcriber."""
    ascii_guid = unicodedata.normalize("NFKD", guid).encode(
        "ascii", "ignore",
    ).decode("ascii")
    readable = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in ascii_guid
    ).lstrip("-_").rstrip("-")
    readable = (readable or "episode")[:64].rstrip("-") or "episode"
    if not readable[0].isalnum():
        readable = "episode"
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
    """Durably publish UTF-8 text without exposing a partial final file.

    Uses ``os.open(..., O_CREAT | O_EXCL | O_WRONLY, 0o666)`` so the kernel
    applies the process umask atomically without mutating process-wide state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
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


_MAX_SUBJECT_LEN = 240


def _build_subject(podcast: str, title: str, duration: int | None) -> str:
    clean_podcast = " ".join(podcast.split())
    clean_title = " ".join(title.split())
    dur = _fmt_duration(duration)
    suffix = f" ({dur})" if dur else ""
    if not clean_podcast or clean_podcast.casefold() == clean_title.casefold():
        base = f"[Podcast] {clean_title}"
    else:
        base = f"[Podcast] {clean_podcast} — {clean_title}"
    budget = _MAX_SUBJECT_LEN - len(suffix)
    if len(base) > budget:
        base = base[: max(1, budget - 1)].rstrip() + "…"
    return f"{base}{suffix}"


# Display names for the language codes we expect to see in feeds.toml; any
# other code falls back to itself in the email note.
_LANGUAGE_NAMES = {
    "zh": "Mandarin",
    "yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}


_YOUTUBE_TS_TEXT_RE = re.compile(r"\[(?:(\d{1,2}):)?(\d{2}):(\d{2})\]")

_EMAIL_STYLE = (
    "<style>"
    ":root{color-scheme:light dark;supported-color-schemes:light dark;}"
    "body{margin:0;padding:0;}"
    ".pd-wrap{max-width:680px;margin:0 auto;padding:16px;font-family:-apple-system,BlinkMacSystemFont,"
    "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.6;color:#1f2937;}"
    ".pd-body h2{font-size:16px;font-weight:700;margin:22px 0 8px 0;padding-bottom:4px;"
    "border-bottom:1px solid #e5e7eb;}"
    ".pd-body h3{font-size:14px;font-weight:600;margin:16px 0 6px 0;}"
    ".pd-body ul,.pd-body ol{margin:8px 0 14px 0;padding-left:20px;}"
    ".pd-body li{margin:0 0 6px 0;}"
    ".pd-body p{margin:0 0 12px 0;}"
    ".pd-body a{color:#4f46e5;text-decoration:none;}"
    "@media (prefers-color-scheme:dark){"
    ".pd-wrap{color:#e5e7eb !important;}"
    ".pd-body h2{border-bottom-color:#374151 !important;color:#f3f4f6 !important;}"
    ".pd-body h3{color:#f3f4f6 !important;}"
    ".pd-meta,.pd-foot{color:#9ca3af !important;}"
    ".pd-eyebrow,.pd-body a,.pd-listen{color:#818cf8 !important;}"
    ".pd-head,.pd-hr{border-color:#374151 !important;}"
    "}"
    "</style>"
)

_INLINE_TAG_STYLES: dict[str, str] = {
    "h2": "font-size:16px;font-weight:700;margin:22px 0 8px 0;padding-bottom:4px;border-bottom:1px solid #e5e7eb;",
    "h3": "font-size:14px;font-weight:600;margin:16px 0 6px 0;",
    "ul": "margin:8px 0 14px 0;padding-left:20px;",
    "ol": "margin:8px 0 14px 0;padding-left:20px;",
    "li": "margin:0 0 6px 0;",
    "a": "color:#4f46e5;text-decoration:none;",
}


class _SummaryEmailRewriter(HTMLParser):
    """Inline critical email styles and turn YouTube timestamps into deep links
    without ever nesting `<a>` tags."""

    def __init__(self, youtube_base: str | None) -> None:
        super().__init__(convert_charrefs=False)
        self._youtube_base = youtube_base
        self._anchor_depth = 0
        self._out: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "a":
            self._anchor_depth += 1
        attr_list = list(attrs)
        style = _INLINE_TAG_STYLES.get(lower)
        if style and not any(k.lower() == "style" for k, _ in attr_list):
            attr_list.append(("style", style))
        rendered = "".join(
            f' {k}="{_escape(v)}"' if v is not None else f" {k}"
            for k, v in attr_list
        )
        self._out.append(f"<{lower}{rendered}>")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "a" and self._anchor_depth > 0:
            self._anchor_depth -= 1
        self._out.append(f"</{lower}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        rendered = "".join(
            f' {k}="{_escape(v)}"' if v is not None else f" {k}"
            for k, v in attrs
        )
        self._out.append(f"<{lower}{rendered}/>")

    def handle_data(self, data: str) -> None:
        if self._youtube_base and self._anchor_depth == 0:
            base = self._youtube_base

            def _replace(match: re.Match[str]) -> str:
                hh = int(match.group(1) or 0)
                mm = int(match.group(2))
                ss = int(match.group(3))
                total_seconds = hh * 3600 + mm * 60 + ss
                href = f"{base}&t={total_seconds}s"
                return (
                    f'<a href="{_escape(href)}" rel="noopener noreferrer" '
                    f'style="color:#4f46e5;text-decoration:none;">'
                    f"{_escape(match.group(0))}</a>"
                )

            # Escape non-matching segments so raw text stays valid HTML.
            last = 0
            parts: list[str] = []
            for m in _YOUTUBE_TS_TEXT_RE.finditer(data):
                if m.start() > last:
                    parts.append(_escape(data[last : m.start()]))
                parts.append(_replace(m))
                last = m.end()
            if last < len(data):
                parts.append(_escape(data[last:]))
            self._out.append("".join(parts))
        else:
            self._out.append(_escape(data))

    def handle_entityref(self, name: str) -> None:
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._out.append(f"&#{name};")

    def result(self) -> str:
        return "".join(self._out)


def _link_youtube_timestamps(body_html: str, audio_url: str) -> str:
    """Turn `[hh:mm:ss]` highlight markers into clickable YouTube deep links."""
    youtube_base: str | None = None
    if youtube.is_youtube_url(audio_url):
        try:
            video_id = youtube.extract_video_id(audio_url)
            youtube_base = youtube.canonical_watch_url(video_id)
        except youtube.FeedError:
            youtube_base = None
    rewriter = _SummaryEmailRewriter(youtube_base)
    rewriter.feed(body_html)
    rewriter.close()
    return rewriter.result()


_TLDR_P_RE = re.compile(
    r"<h2[^>]*>\s*TL;DR\s*</h2>\s*<p[^>]*>(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
_FIRST_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def _extract_preheader(summary_html: str) -> str:
    """Extract a concise plain-text preheader from TL;DR (or the first paragraph)."""
    match = _TLDR_P_RE.search(summary_html) or _FIRST_P_RE.search(summary_html)
    if not match:
        return ""
    text = _html_to_text(match.group(1), include_links=False).strip()
    text = " ".join(text.split())
    if len(text) > 160:
        text = text[:159].rstrip() + "…"
    return text


def _wrap_email(
    summary_html: str,
    podcast: str,
    title: str,
    audio_url: str,
    published_at: str | None = None,
    duration_seconds: int | None = None,
    original_language: str | None = None,
    transcript_source: str = "whisper",
    generated_by: BuildIdentity | None = None,
) -> str:
    """Wrap sanitized model HTML in a readable, dark-mode-safe digest layout.

    The header carries the podcast name, title, release date, runtime, and any
    translation note. The model summary is sanitized at production and
    re-sanitized here (see sanitize.py) before adding clickable YouTube
    timestamps and inline email styles — the email is the outward surface, so
    it never trusts storage.
    """
    meta_bits: list[str] = []
    released = _fmt_release_date(published_at)
    if released:
        meta_bits.append(f"Released {released}")
    dur = _fmt_duration(duration_seconds)
    if dur:
        meta_bits.append(dur)
    if original_language:
        name = _LANGUAGE_NAMES.get(original_language, original_language)
        meta_bits.append(f"Translated from {_escape(name)}")
    meta_line = (
        f'<p class="pd-meta" style="color:#6b7280;font-size:13px;margin:0;">'
        f"{' · '.join(meta_bits)}</p>"
        if meta_bits else ""
    )

    safe_url = safe_listen_url(audio_url)
    footer_bits: list[str] = []
    if safe_url:
        cta_label = (
            "Watch on YouTube" if youtube.is_youtube_url(safe_url)
            else "Listen to the episode"
        )
        footer_bits.append(
            f'<a class="pd-listen" href="{_escape(safe_url)}" '
            f'style="color:#4f46e5;font-weight:600;text-decoration:none;">'
            f"{cta_label}</a>"
        )
    if transcript_source == "publisher":
        footer_bits.append("Summarized from the publisher's transcript")
    footer = (
        f'<p class="pd-foot" style="color:#6b7280;font-size:13px;margin:0 0 8px 0;">'
        f"{' · '.join(footer_bits)}</p>"
        if footer_bits else ""
    )
    if generated_by is not None:
        footer += (
            f'<p class="pd-foot" style="color:#6b7280;font-size:11px;margin:0;'
            f'overflow-wrap:break-word;">'
            f"Generated by {_escape(generated_by.service)} {_escape(generated_by.version)}, "
            f"source revision {_escape(generated_by.source_revision)}.</p>"
        )
    if footer:
        footer = (
            f'<hr class="pd-hr" style="border:none;border-top:1px solid #e5e7eb;'
            f'margin:24px 0 16px 0;">{footer}'
        )

    sanitized = sanitize_summary_html(summary_html)
    preheader_text = _extract_preheader(sanitized)
    preheader_html = (
        f'<div class="pd-preheader" style="display:none;max-height:0;overflow:hidden;'
        f'opacity:0;mso-hide:all;font-size:1px;line-height:1px;">'
        f"{_escape(preheader_text)}&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;</div>"
        if preheader_text else ""
    )

    show_eyebrow = bool(
        podcast.strip() and podcast.strip().casefold() != title.strip().casefold()
    )
    eyebrow_html = (
        f'<p class="pd-eyebrow" style="font-size:12px;font-weight:600;letter-spacing:0.04em;'
        f'text-transform:uppercase;color:#4f46e5;margin:0 0 4px 0;">{_escape(podcast)}</p>'
        if show_eyebrow else ""
    )

    clean_summary = _link_youtube_timestamps(sanitized, audio_url)
    return (
        f"<!DOCTYPE html><html><head>"
        f'<meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="color-scheme" content="light dark">'
        f'<meta name="supported-color-schemes" content="light dark">'
        f"{_EMAIL_STYLE}</head>"
        f'<body style="margin:0;padding:0;">'
        f"{preheader_html}"
        f'<!--[if mso]><table role="presentation" width="680" align="center" '
        f'cellpadding="0" cellspacing="0" border="0"><tr><td><![endif]-->'
        f'<div class="pd-wrap" style="max-width:680px;margin:0 auto;padding:16px;'
        f"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
        f'font-size:15px;line-height:1.6;color:#1f2937;">'
        f'<div class="pd-head" style="border-bottom:1px solid #e5e7eb;'
        f'padding-bottom:14px;margin-bottom:20px;">'
        f"{eyebrow_html}"
        f'<p style="margin:0 0 8px 0;font-size:20px;line-height:1.3;font-weight:700;">'
        f"<strong>{_escape(title)}</strong></p>"
        f"{meta_line}"
        f"</div>"
        f'<div class="pd-body">{clean_summary}</div>'
        f"{footer}"
        f"</div>"
        f"<!--[if mso]></td></tr></table><![endif]-->"
        f"</body></html>"
    )


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
