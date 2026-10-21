"""Long-running tick loop.

The container runs `serve` which calls run_forever. SIGTERM / SIGINT cancels
the current sleep but lets an in-flight tick finish — important so an episode
that's halfway through transcription isn't left in 'downloaded' status.
"""
from __future__ import annotations

import asyncio
import logging
import random
import signal
import time

import httpx

from .config import Settings, load_feeds
from .db import Database
from .log import err_fields, event
from .mail import Mailer
from .pipeline import run_tick
from .wake import read_wake

logger = logging.getLogger(__name__)

JITTER_SECONDS = 600  # up to 10 min of random jitter on each sleep


async def run_forever(settings: Settings, db: Database, mailer: Mailer) -> None:
    stop = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("scheduler.signal stop_requested=true")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Token of the most recent "process now" signal we've already acted on. The
    # startup tick below covers anything pending, so seed from the current value
    # to avoid an immediate redundant second tick.
    last_wake = read_wake(settings.PODCAST_DIGEST_CONFIG)

    while not stop.is_set():
        # Snapshot the signal *before* the tick: a signal that lands during the
        # tick stays unconsumed and triggers one more tick afterwards (so a feed
        # added mid-tick is never missed).
        pending_wake = read_wake(settings.PODCAST_DIGEST_CONFIG)

        try:
            feeds_file = load_feeds(settings.PODCAST_DIGEST_CONFIG)
        except Exception as e:  # noqa: BLE001
            event(logger, logging.ERROR, "scheduler.config_error",
                  exc_info=True, **err_fields(e))
            # Wait a minute and try again — bad config shouldn't kill the loop.
            await _wait_for_next_tick(60, stop, settings, last_wake)
            continue

        event(
            logger, logging.INFO, "tick.start",
            feeds=len(feeds_file.feeds), poll_seconds=feeds_file.poll_seconds,
        )
        skipped = False
        tick_ran = False
        stats: dict = {}
        started = time.perf_counter()
        try:
            stats = await run_tick(
                settings=settings, feeds_file=feeds_file,
                db=db, mailer=mailer, dry_run=False,
            )
            tick_ran = True
            skipped = bool(stats.get("skipped"))
            event(
                logger, logging.INFO, "tick.done",
                feeds=stats["feeds"], new=stats["new_episodes"],
                emailed=stats["emailed"], failed=stats["failed"],
                summarize_failed=stats.get("summarize_failed", 0),
                skipped=skipped,
                elapsed_s=round(time.perf_counter() - started, 1),
            )
        except Exception as e:  # noqa: BLE001
            event(logger, logging.ERROR, "tick.error",
                  elapsed_s=round(time.perf_counter() - started, 1),
                  exc_info=True, **err_fields(e))

        if settings.PODCAST_DIGEST_HEARTBEAT_URL and _should_heartbeat(
            tick_ran=tick_ran, skipped=skipped,
            summarize_failed=stats.get("summarize_failed", 0),
            summarize_ok=stats.get("summarize_ok", 0),
        ):
            await _ping_heartbeat(settings.PODCAST_DIGEST_HEARTBEAT_URL)

        # Consume the signal once a tick actually ran (success or error). Only a
        # lock-skip leaves it unconsumed, so it retries on the next short poll
        # rather than busy-looping or being lost.
        if not skipped:
            last_wake = max(last_wake, pending_wake)

        sleep_for = feeds_file.poll_seconds + random.uniform(0, JITTER_SECONDS)
        event(logger, logging.INFO, "tick.sleep", seconds=round(sleep_for))
        await _wait_for_next_tick(sleep_for, stop, settings, last_wake)


def _should_heartbeat(
    *, tick_ran: bool, skipped: bool, summarize_failed: int, summarize_ok: int
) -> bool:
    """Dead-man's-switch gate: heartbeat only on a clean tick (it ran, wasn't
    lock-skipped) with a healthy summarizer *transport*. A summarizer outage
    zeroes the pings → the monitor alerts after its grace. Idle ticks
    (nothing to summarize) still ping = healthy. Keyed on summarize_failed, NOT
    the generic `failed`, so a feed-resolution error or a one-off content/email
    hiccup never mutes the heartbeat.

    A tick with at least one successful summarize pings even if another episode
    failed transport: one success proves the summarizer is reachable, so that
    failure is episode-scoped (e.g. an oversized transcript riding the canary
    slot — see db.episodes_in_progress) and must not hold the whole switch
    hostage. During a real outage nothing succeeds — the canary keeps failing,
    summarize_ok stays 0, and the heartbeat stays honestly muted throughout."""
    return tick_ran and not skipped and (summarize_failed == 0 or summarize_ok > 0)


async def _ping_heartbeat(url: str) -> None:
    """Best-effort GET to the heartbeat URL. Never raises — a failed heartbeat
    must not disrupt the tick loop; a *missing* one is the whole point (the
    monitor alerts on the absence)."""
    try:
        await asyncio.to_thread(lambda: httpx.get(url, timeout=10.0))
        logger.debug("heartbeat.ping ok")
    except Exception as e:  # noqa: BLE001
        logger.warning("heartbeat.ping failed err=%s", e)


async def _wait_for_next_tick(
    seconds: float, stop: asyncio.Event, settings: Settings, last_wake: int
) -> None:
    """Sleep up to `seconds`, returning early on shutdown or a new wake signal.

    Polls the wake sentinel every `WAKE_POLL_SECONDS` so a "process now" signal
    kicks off a tick within seconds instead of waiting out the poll interval.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while not stop.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        if read_wake(settings.PODCAST_DIGEST_CONFIG) > last_wake:
            logger.info("tick.wake process-now signal received — running early")
            return
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=min(settings.PODCAST_DIGEST_WAKE_POLL_SECONDS, remaining),
            )
        except TimeoutError:
            continue
