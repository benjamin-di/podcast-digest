"""Tests for the subprocess helper: capture, timeout, and kill-on-timeout."""
from __future__ import annotations

import asyncio
import time

import pytest

from podcast_digest.proc import SubprocessTimeout, run_capture


async def test_run_capture_returns_output():
    rc, out, err = await run_capture(
        ["sh", "-c", "printf hello; printf oops 1>&2; exit 3"],
        timeout=10, label="t",
    )
    assert rc == 3
    assert out == b"hello"
    assert err == b"oops"


async def test_run_capture_times_out_and_kills_quickly():
    """A wedged child must be killed and surface a timeout fast — not hang the
    tick for the child's natural lifetime."""
    started = time.monotonic()
    with pytest.raises(SubprocessTimeout):
        # `sleep 30` would otherwise block the whole tick.
        await run_capture(["sleep", "30"], timeout=0.5, label="sleeper")
    elapsed = time.monotonic() - started
    assert elapsed < 5  # killed near the timeout, not 30s later


async def test_run_capture_kills_child_on_cancel():
    """Cancelling the awaiting task must kill the child (not orphan it)."""
    task = asyncio.create_task(
        run_capture(["sleep", "30"], timeout=30, label="sleeper")
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
