"""Subprocess helper with a hard timeout and kill-on-cancel.

Every external tool the pipeline shells out to (yt-dlp, ffmpeg, whisper-cli,
ffprobe, curl) is wrapped here so a single stalled child can never wedge the
serial tick forever. Two guarantees the bare ``create_subprocess_exec`` +
``communicate()`` pattern does NOT give us:

  1. **Timeout.** ``asyncio.wait_for`` bounds the wait; on expiry we SIGKILL the
     child's whole process group and raise :class:`SubprocessTimeout`.
  2. **Kill-on-cancel.** If the tick is cancelled (SIGTERM during shutdown, or an
     outer ``wait_for``), a plain ``communicate()`` leaves the child orphaned and
     still burning CPU/GPU. We start each child in its own session
     (``start_new_session=True`` → pid == pgid) and kill the group on any exit
     path — timeout, cancellation, or error.

The group kill (vs a single-process ``proc.kill()``) matters because yt-dlp and
ffmpeg spawn their own children; killing only the leader would orphan those.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

logger = logging.getLogger(__name__)


class SubprocessTimeout(RuntimeError):
    """A wrapped subprocess exceeded its timeout and was killed."""


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group (started with its own session,
    so pid == pgid). Best-effort: the group may already be gone."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Group owned elsewhere (rare) — fall back to a single-process kill.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def run_capture(
    cmd: list[str],
    *,
    timeout: float,
    label: str,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    """Run ``cmd`` to completion, capturing stdout/stderr, bounded by ``timeout``.

    Returns ``(returncode, stdout, stderr)``. Raises :class:`SubprocessTimeout`
    if the child doesn't finish within ``timeout`` seconds (after killing it),
    and kills the child on cancellation before re-raising ``CancelledError``.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        logger.warning("proc.timeout label=%s timeout_s=%s cmd=%s",
                       label, timeout, cmd[0])
        raise SubprocessTimeout(f"{label} timed out after {timeout}s") from None
    finally:
        # Whatever the exit path — timeout, cancellation, or any other exception
        # out of communicate() — never leave the child running/un-reaped (the
        # whole reason this helper exists). On the success path returncode is
        # already set, so this is a no-op. Swallow only Exception during the reap
        # (a CancelledError must still propagate).
        if proc.returncode is None:
            _kill_group(proc)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=10)
    # communicate() returned without raising, so the child has exited and
    # returncode is set (every exception path re-raises past the finally).
    assert proc.returncode is not None
    return proc.returncode, stdout, stderr
