"""Tests for the process-now wake protocol + the scheduler's responsive wait."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from podcast_digest.scheduler import _wait_for_next_tick
from podcast_digest.wake import read_wake, signal_wake, wake_path


def test_wake_round_trip(tmp_path):
    cfg = tmp_path / "feeds.toml"
    assert read_wake(cfg) == 0  # missing sentinel → 0
    t1 = signal_wake(cfg)
    # `.tmp` suffix so a `*.tmp` ignore rule keeps the sentinel out of any
    # sync of the config directory (its token churn must not be replicated).
    assert wake_path(cfg).name == ".process-now.tmp"
    assert read_wake(cfg) == t1
    t2 = signal_wake(cfg)
    assert t2 > t1  # monotonic token advances
    assert read_wake(cfg) == t2


def test_read_wake_tolerates_garbage(tmp_path):
    cfg = tmp_path / "feeds.toml"
    wake_path(cfg).write_text("not-a-number", encoding="utf-8")
    assert read_wake(cfg) == 0


def _settings(cfg, poll=0.05):
    return SimpleNamespace(
        PODCAST_DIGEST_CONFIG=cfg, PODCAST_DIGEST_WAKE_POLL_SECONDS=poll
    )


async def test_wait_returns_early_on_new_wake(tmp_path):
    cfg = tmp_path / "feeds.toml"
    last = read_wake(cfg)  # 0
    signal_wake(cfg)  # bump past `last`
    # A 10s nominal wait must return promptly because the signal is newer.
    await asyncio.wait_for(
        _wait_for_next_tick(10, asyncio.Event(), _settings(cfg), last), timeout=2
    )


async def test_wait_ignores_already_consumed_wake(tmp_path):
    cfg = tmp_path / "feeds.toml"
    consumed = signal_wake(cfg)  # this token is already consumed
    t0 = time.monotonic()
    # last_wake == consumed, so the existing signal must NOT wake it early.
    await _wait_for_next_tick(0.2, asyncio.Event(), _settings(cfg), consumed)
    assert time.monotonic() - t0 >= 0.15  # waited out the interval


async def test_wait_returns_on_stop(tmp_path):
    cfg = tmp_path / "feeds.toml"
    stop = asyncio.Event()
    stop.set()
    await asyncio.wait_for(
        _wait_for_next_tick(10, stop, _settings(cfg, poll=1.0), 0), timeout=1
    )


