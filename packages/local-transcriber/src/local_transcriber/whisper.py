"""Atomic whisper.cpp transcription of a local media file."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .models import Segment, Transcript, TranscriptionResult, WhisperCppConfig
from .proc import SubprocessTimeout, run_capture

logger = logging.getLogger(__name__)

_WAV_BYTES_PER_SECOND = 32_000
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


class TranscriptionError(RuntimeError):
    pass


@contextlib.asynccontextmanager
async def _device_lock(config: WhisperCppConfig):
    """Coordinate access to one accelerator across sibling containers."""
    if config.lock_path is None:
        yield
        return
    lock_path = Path(config.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    deadline = time.monotonic() + config.lock_timeout_s
    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TranscriptionError(
                        f"timed out waiting for transcription lock: {lock_path}"
                    ) from None
                await asyncio.sleep(1)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _validate_artifact_id(value: str) -> str:
    if not _ARTIFACT_ID.fullmatch(value) or value in {".", ".."}:
        raise TranscriptionError(
            "artifact_id must be 1-200 safe filename characters (letters, digits, ._- )"
        )
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_sha256(prompt: str | None) -> str | None:
    return hashlib.sha256(prompt.encode()).hexdigest() if prompt else None


def _atomic_publish(src: Path, dst: Path) -> None:
    """Publish *src* atomically, including across container mount boundaries."""
    try:
        os.replace(src, dst)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    # A bind-mounted output directory can be a different filesystem from the
    # work directory. Copy into a hidden file on the destination filesystem,
    # fsync it, then atomically replace the public path there.
    fd, raw_staged = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    staged = Path(raw_staged)
    try:
        with os.fdopen(fd, "wb") as target, src.open("rb") as source:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(staged, dst)
        src.unlink()
    finally:
        staged.unlink(missing_ok=True)


def _environment(config: WhisperCppConfig) -> dict[str, str]:
    env = dict(os.environ)
    if config.library_path:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{config.library_path}:{existing}" if existing else config.library_path
        )
    return env


def _whisper_timeout(wav_bytes: int, config: WhisperCppConfig) -> float:
    audio_seconds = wav_bytes / _WAV_BYTES_PER_SECOND
    return max(
        config.whisper_timeout_floor_s,
        audio_seconds * config.whisper_seconds_per_audio_second,
    )


async def transcribe_file(
    input_path: Path,
    *,
    output_dir: Path,
    artifact_id: str,
    config: WhisperCppConfig,
    work_root: Path | None = None,
) -> TranscriptionResult:
    """Transcribe a local media file and publish validated artifacts atomically.

    Whisper and ffmpeg write only inside a unique work directory. The final text
    and JSON files are individually replaced atomically; the manifest is written
    last and is the completion marker for non-database consumers.
    """
    artifact_id = _validate_artifact_id(artifact_id)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise TranscriptionError(f"media not found: {input_path}")
    if not config.model_path.is_file():
        raise TranscriptionError(f"whisper model not found: {config.model_path}")
    if not shutil.which(config.ffmpeg_bin):
        raise TranscriptionError(f"ffmpeg binary not found: {config.ffmpeg_bin}")
    if not shutil.which(config.whisper_bin) and not Path(config.whisper_bin).is_file():
        raise TranscriptionError(f"whisper binary not found: {config.whisper_bin}")

    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(work_root) if work_root else output_dir.parent / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{artifact_id}-", dir=root))
    wav_path = work_dir / "input.wav"
    output_base = work_dir / "transcript"
    started = time.perf_counter()
    try:
        await _normalize(input_path, wav_path, config=config)
        duration_seconds = max(0.0, (wav_path.stat().st_size - 44) / _WAV_BYTES_PER_SECOND)
        async with _device_lock(config):
            await _run_whisper(wav_path, output_base=output_base, config=config)
        staged_json = output_base.with_suffix(".json")
        staged_txt = output_base.with_suffix(".txt")
        transcript = load_whisper_json(staged_json, fallback_text_path=staged_txt)
        if not staged_txt.exists():
            staged_txt.write_text(transcript.text, encoding="utf-8")

        txt_path = output_dir / f"{artifact_id}.txt"
        json_path = output_dir / f"{artifact_id}.json"
        manifest_path = output_dir / f"{artifact_id}.manifest.json"
        _atomic_publish(staged_txt, txt_path)
        _atomic_publish(staged_json, json_path)

        input_sha256 = _hash_file(input_path)
        model_sha256 = _hash_file(config.model_path)
        source_revision = os.environ.get("LOCAL_TRANSCRIBER_SOURCE_REV") or None
        manifest = {
            "schema_version": 2,
            "artifact_id": artifact_id,
            "input_sha256": input_sha256,
            "duration_seconds": round(duration_seconds, 3),
            "language": config.language,
            "engine": "whisper.cpp",
            "engine_version": config.engine_version,
            "model_name": config.model_path.name,
            "model_sha256": model_sha256,
            "local_transcriber_source_revision": source_revision,
            "whisper_binary": config.whisper_bin,
            "ffmpeg_binary": config.ffmpeg_bin,
            "threads": config.threads,
            "device": config.device,
            "initial_prompt": config.initial_prompt,
            "initial_prompt_sha256": _prompt_sha256(config.initial_prompt),
            "segments": [asdict(segment) for segment in transcript.segments],
            "text_file": txt_path.name,
            "json_file": json_path.name,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        staged_manifest = work_dir / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _atomic_publish(staged_manifest, manifest_path)
        return TranscriptionResult(
            transcript=transcript,
            txt_path=txt_path,
            json_path=json_path,
            manifest_path=manifest_path,
            input_sha256=input_sha256,
            duration_seconds=duration_seconds,
            engine="whisper.cpp",
            engine_version=config.engine_version,
            model_name=config.model_path.name,
            language=config.language,
            model_sha256=model_sha256,
            source_revision=source_revision,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def _normalize(src: Path, dst: Path, *, config: WhisperCppConfig) -> None:
    cmd = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    try:
        rc, _stdout, stderr = await run_capture(
            cmd, timeout=config.ffmpeg_timeout_s, label="ffmpeg"
        )
    except SubprocessTimeout as exc:
        raise TranscriptionError(str(exc)) from exc
    if rc != 0:
        raise TranscriptionError(
            f"ffmpeg failed (rc={rc}): {stderr.decode(errors='replace')[-800:]}"
        )


async def _run_whisper(
    wav_path: Path,
    *,
    output_base: Path,
    config: WhisperCppConfig,
) -> None:
    cmd = [
        config.whisper_bin,
        "-m",
        str(config.model_path),
        "-f",
        str(wav_path),
        "-l",
        config.language,
        "-t",
        str(config.threads),
        "-of",
        str(output_base),
        "-oj",
        "-otxt",
    ]
    if config.initial_prompt:
        # `-p` is whisper.cpp's integer `--processors` option.  The prompt has
        # no short form, so spelling it out is required (and avoids stoi aborts).
        cmd += ["--prompt", config.initial_prompt]
    if config.device:
        cmd += ["-oved", config.device]
    timeout = _whisper_timeout(wav_path.stat().st_size, config)
    try:
        rc, _stdout, stderr = await run_capture(
            cmd,
            timeout=timeout,
            label="whisper-cli",
            env=_environment(config),
        )
    except SubprocessTimeout as exc:
        raise TranscriptionError(str(exc)) from exc
    if rc != 0:
        raise TranscriptionError(
            f"whisper-cli failed (rc={rc}): {stderr.decode(errors='replace')[-1200:]}"
        )


def load_whisper_json(
    json_path: Path, *, fallback_text_path: Path | None = None
) -> Transcript:
    if not json_path.exists():
        raise TranscriptionError(f"whisper JSON output missing: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"invalid whisper JSON output: {json_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TranscriptionError(f"invalid whisper JSON root (expected object): {json_path}")

    segments: list[Segment] = []
    for raw in data.get("transcription") or []:
        if not isinstance(raw, dict):
            continue
        offsets = raw.get("offsets") or {}
        if not isinstance(offsets, dict):
            offsets = {}
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(offsets.get("from", 0)) / 1000.0
            end = float(offsets.get("to", 0)) / 1000.0
        except (TypeError, ValueError):
            start = end = 0.0
        segments.append(Segment(start=max(0.0, start), end=max(start, end), text=text))
    if segments:
        return Transcript(text=" ".join(segment.text for segment in segments), segments=segments)

    if fallback_text_path and fallback_text_path.exists():
        text = fallback_text_path.read_text(encoding="utf-8").strip()
        if text:
            return Transcript(text=text, segments=[])
    raise TranscriptionError(f"whisper produced no text: {json_path}")
