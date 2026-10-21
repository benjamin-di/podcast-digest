"""Bounded subprocess execution with process-group cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

logger = logging.getLogger(__name__)


class SubprocessTimeout(RuntimeError):
    pass


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def run_capture(
    cmd: list[str],
    *,
    timeout: float,
    label: str,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        logger.warning("proc.timeout label=%s timeout_s=%s cmd=%s", label, timeout, cmd[0])
        raise SubprocessTimeout(f"{label} timed out after {timeout}s") from None
    finally:
        if proc.returncode is None:
            _kill_group(proc)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=10)
    assert proc.returncode is not None
    return proc.returncode, stdout, stderr
