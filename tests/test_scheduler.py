"""Tests for the dead-man's-switch heartbeat in the tick loop.

The heartbeat pings the monitor once per *healthy* tick; a summarizer
outage zeroes the pings so the monitor alerts after its grace window. We
test the gate predicate (`_should_heartbeat`) and the best-effort ping helper.
"""
from __future__ import annotations

import httpx

from podcast_digest import scheduler

# ---- gate predicate ---------------------------------------------------------


def test_heartbeat_on_healthy_tick():
    # Ran, not skipped, summarizer healthy → ping.
    assert scheduler._should_heartbeat(
        tick_ran=True, skipped=False, summarize_failed=0, summarize_ok=1) is True


def test_heartbeat_on_idle_tick():
    # Nothing to summarize is still healthy (no failures, no successes) → ping.
    assert scheduler._should_heartbeat(
        tick_ran=True, skipped=False, summarize_failed=0, summarize_ok=0) is True


def test_no_heartbeat_on_summarizer_outage():
    # Transport failures with ZERO successes = a real outage → mute (alert).
    # This holds all outage long: the canary keeps attempting and failing, so
    # summarize_failed never goes silent while episodes are pending.
    assert scheduler._should_heartbeat(
        tick_ran=True, skipped=False, summarize_failed=1, summarize_ok=0) is False


def test_heartbeat_when_one_episode_stuck_but_transport_healthy():
    # A success in the same tick proves the transport is up — the lone failure
    # is episode-scoped (e.g. an oversized transcript riding the canary slot)
    # and must not hold the whole dead-man's-switch hostage forever.
    assert scheduler._should_heartbeat(
        tick_ran=True, skipped=False, summarize_failed=1, summarize_ok=1) is True


def test_no_heartbeat_when_lock_skipped():
    # Another tick holds the lock; it will ping instead.
    assert scheduler._should_heartbeat(
        tick_ran=True, skipped=True, summarize_failed=0, summarize_ok=0) is False


def test_no_heartbeat_when_tick_errored():
    # run_tick raised → don't claim health.
    assert scheduler._should_heartbeat(
        tick_ran=False, skipped=False, summarize_failed=0, summarize_ok=0) is False


# ---- ping helper ------------------------------------------------------------


async def test_ping_calls_url(monkeypatch):
    called = {}

    def fake_get(url, **kw):
        called["url"] = url
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", fake_get)
    await scheduler._ping_heartbeat("https://heartbeat.example.com/ping/abc")
    assert called["url"] == "https://heartbeat.example.com/ping/abc"


async def test_ping_swallows_errors(monkeypatch):
    def fake_get(url, **kw):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "get", fake_get)
    # Must not raise — a failed heartbeat cannot disrupt the loop.
    await scheduler._ping_heartbeat("https://heartbeat.example.com/ping/abc")
