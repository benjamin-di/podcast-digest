"""One-request Parakeet worker for authenticated stdio transports."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import resource
import signal
import sys
import tempfile
import wave
from pathlib import Path

from .identity import BUILD
from .models import PARAKEET_MODEL_NAME, ParakeetConfig
from .remote import (
    _CHUNK,
    _MAX_HEADER,
    _MAX_JSON,
    _MAX_WAV,
    WorkerError,
    _bounded_size,
    _encode_header,
    _read_body,
    _read_header,
)
from .whisper import TranscriptionError, _run_sherpa, load_whisper_json


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--timeout", type=_positive, default=3600)
    return parser


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as media:
            if (media.getnchannels(), media.getsampwidth(), media.getframerate()) != (1, 2, 16_000):
                raise WorkerError("worker requires mono 16 kHz PCM audio")
            frames = media.getnframes()
            if not 0 < frames * 2 <= path.stat().st_size - 44:
                raise WorkerError("invalid PCM data length")
    except (wave.Error, EOFError):
        raise WorkerError("invalid WAV input") from None


async def _infer(
    reader: asyncio.StreamReader, wav_path: Path, output_path: Path, config: ParakeetConfig
) -> tuple[str, str]:
    # A closed transport must not leave an orphan inference occupying the slot.
    inference = asyncio.create_task(_run_sherpa(wav_path, output_path, "parakeet", config))
    disconnect = asyncio.create_task(reader.read(1))
    try:
        await asyncio.wait((inference, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if disconnect.done():
            raise WorkerError("worker request cancelled or has trailing data")
        return await inference
    finally:
        for task in (inference, disconnect):
            task.cancel()
        await asyncio.gather(inference, disconnect, return_exceptions=True)


async def _write_reply_part(fd: int, data: bytes) -> None:
    loop = asyncio.get_running_loop()
    pending = memoryview(data)
    while pending:
        try:
            pending = pending[os.write(fd, pending):]
        except BlockingIOError:
            ready = asyncio.Event()
            loop.add_writer(fd, ready.set)
            try:
                await ready.wait()
            finally:
                loop.remove_writer(fd)


async def _serve(args: argparse.Namespace, reader: asyncio.StreamReader) -> None:
    with tempfile.TemporaryDirectory(prefix="parakeet-worker-") as directory:
        wav_path = Path(directory) / "input.wav"
        json_path = Path(directory) / "result.json"
        async with asyncio.timeout(180):
            header = await _read_header(reader)
            if header.get("model") != PARAKEET_MODEL_NAME:
                raise WorkerError("model unavailable")
            with wav_path.open("wb") as dest:
                await _read_body(reader, dest, _bounded_size(header, "wav_size", _MAX_WAV))
        _validate_wav(wav_path)
        config = ParakeetConfig(args.model_dir, args.threads)
        version_used, digest = await _infer(reader, wav_path, json_path, config)
        json_size = json_path.stat().st_size
        if not 0 < json_size <= _MAX_JSON:
            raise WorkerError("artifact size outside worker limit")
        load_whisper_json(json_path)
        reply = _encode_header({
            "version": 1, "model": PARAKEET_MODEL_NAME,
            "engine_version": version_used, "threads": args.threads,
            "model_sha256": digest, "source_revision": BUILD.source_revision,
            "json_size": json_size,
        })
        fd = sys.stdout.fileno()
        os.set_blocking(fd, False)
        await _write_reply_part(fd, reply)
        with json_path.open("rb") as source:
            while chunk := source.read(_CHUNK):
                await _write_reply_part(fd, chunk)


async def _run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_MAX_HEADER)
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer,
    )
    task = asyncio.current_task()
    assert task is not None
    for signum in (signal.SIGHUP, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        async with asyncio.timeout(args.timeout):
            await _serve(args, reader)
    finally:
        transport.close()
        for signum in (signal.SIGHUP, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def _limit(kind: int, desired: int) -> None:
    soft, hard = resource.getrlimit(kind)
    bounded = min(limit for limit in (desired, soft, hard) if limit != resource.RLIM_INFINITY)
    resource.setrlimit(kind, (bounded, bounded))


def main() -> int:
    args = _parser().parse_args()
    try:
        os.umask(0o077)
        os.nice(10)
        _limit(resource.RLIMIT_AS, 8 * 1024**3)
        _limit(resource.RLIMIT_FSIZE, _MAX_WAV)
        _limit(resource.RLIMIT_CORE, 0)
        with args.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(_run(args))
    except BlockingIOError:
        return 75
    except (WorkerError, TranscriptionError, OSError, asyncio.CancelledError,
            TimeoutError, TypeError, ValueError, RecursionError):
        print("transcription worker failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
