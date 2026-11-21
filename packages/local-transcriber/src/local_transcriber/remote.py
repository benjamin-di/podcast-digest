"""Bounded stdio transport for a Parakeet worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .models import PARAKEET_MODEL_NAME
from .proc import _kill_group

_MAX_HEADER = 4096
_MAX_WAV = 768 * 1024 * 1024
_MAX_JSON = 32 * 1024 * 1024
_CHUNK = 256 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteConfig:
    command: tuple[str, ...]
    timeout_s: float = 3700.0


@dataclass(frozen=True)
class WorkerProvenance:
    engine_version: str
    model_sha256: str
    threads: int
    source_revision: str


def _encode_header(header: dict[str, object]) -> bytes:
    return json.dumps(header, separators=(",", ":"), allow_nan=False).encode() + b"\n"


async def _read_header(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        line = await reader.readuntil(b"\n")
        if len(line) > _MAX_HEADER:
            raise WorkerError("worker header exceeds limit")
        header = json.loads(line)
    except (ValueError, RecursionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        raise WorkerError("invalid worker header") from None
    if not isinstance(header, dict) or type(header.get("version")) is not int or header["version"] != 1:
        raise WorkerError("unsupported worker protocol")
    return header


def _bounded_size(header: dict[str, Any], key: str, limit: int) -> int:
    size = header.get(key)
    if type(size) is not int or not 0 < size <= limit:
        raise WorkerError(f"invalid worker field {key}")
    return size


async def _read_body(reader: asyncio.StreamReader, dest: BinaryIO, remaining: int) -> None:
    try:
        while remaining:
            chunk = await reader.readexactly(min(remaining, _CHUNK))
            dest.write(chunk)
            remaining -= len(chunk)
    except asyncio.IncompleteReadError:
        raise WorkerError("incomplete worker body") from None


async def _exchange(
    proc: asyncio.subprocess.Process,
    wav_path: Path,
    output_path: Path,
    *,
    wav_size: int,
) -> WorkerProvenance:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(_encode_header({
        "version": 1, "model": PARAKEET_MODEL_NAME, "wav_size": wav_size,
    }))
    with wav_path.open("rb") as media:
        while chunk := media.read(_CHUNK):
            proc.stdin.write(chunk)
            await proc.stdin.drain()
    # Keep stdin open until the full reply. EOF before completion cancels a worker.
    header = await _read_header(proc.stdout)
    if header.get("model") != PARAKEET_MODEL_NAME:
        raise WorkerError("worker returned a different model")
    digest = header.get("model_sha256")
    engine_version = header.get("engine_version")
    revision = header.get("source_revision")
    if (not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            or not isinstance(engine_version, str) or not engine_version
            or not re.fullmatch(r"[0-9A-Za-z.+_-]{1,80}", engine_version)
            or not isinstance(revision, str) or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", revision)):
        raise WorkerError("invalid worker provenance")
    threads = _bounded_size(header, "threads", 32)
    json_size = _bounded_size(header, "json_size", _MAX_JSON)
    with output_path.open("wb") as segments:
        await _read_body(proc.stdout, segments, json_size)
    if await proc.stdout.read(1) or await proc.wait():
        raise WorkerError("worker returned an invalid response")
    return WorkerProvenance(engine_version, digest, threads, revision)


async def run_remote_parakeet(
    wav_path: Path,
    output_path: Path,
    *,
    config: RemoteConfig,
) -> WorkerProvenance:
    """Send normalized audio and stage the decoder output in the caller's workdir."""
    wav_size = wav_path.stat().st_size
    if not 44 < wav_size <= _MAX_WAV:
        raise WorkerError("WAV size outside worker limit")
    try:
        proc = await asyncio.create_subprocess_exec(
            *config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=_MAX_HEADER,
        )
    except (OSError, ValueError):
        raise WorkerError("unable to start worker command") from None
    assert proc.stdin is not None and proc.stdout is not None
    try:
        async with asyncio.timeout(config.timeout_s):
            return await _exchange(
                proc, wav_path, output_path,
                wav_size=wav_size,
            )
    except TimeoutError:
        raise WorkerError("worker command timed out") from None
    except OSError:
        raise WorkerError("worker transport failed") from None
    finally:
        if not proc.stdin.is_closing():
            proc.stdin.transport.abort()
        try:
            # Give the worker time to observe EOF and kill its own inference.
            await _drain_and_wait(proc, timeout_s=2)
        except TimeoutError:
            _kill_group(proc)
            with contextlib.suppress(TimeoutError):
                await _drain_and_wait(proc, timeout_s=5)


async def _drain_and_wait(proc: asyncio.subprocess.Process, *, timeout_s: float) -> None:
    assert proc.stdout is not None
    async with asyncio.timeout(timeout_s):
        while await proc.stdout.read(_CHUNK):
            pass
        await proc.wait()
