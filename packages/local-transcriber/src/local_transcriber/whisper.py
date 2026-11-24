"""Atomic transcription of a local media file."""

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
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .identity import BUILD
from .models import (
    PARAKEET_MODEL_FILES,
    PARAKEET_MODEL_NAME,
    QWEN_MODEL_FILES,
    QWEN_MODEL_NAME,
    ParakeetConfig,
    QwenConfig,
    Segment,
    Transcript,
    TranscriptionResult,
    WhisperCppConfig,
)
from .proc import SubprocessTimeout, run_capture
from .remote import RemoteConfig, WorkerError, run_remote_parakeet

logger = logging.getLogger(__name__)

_SHERPA_MODELS = {
    "parakeet": (PARAKEET_MODEL_NAME, PARAKEET_MODEL_FILES),
    "qwen": (QWEN_MODEL_NAME, QWEN_MODEL_FILES),
}

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


_MODEL_HASH_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_stat_key(path: Path) -> tuple[str, int, int, int, int, int]:
    st = path.stat()
    return (
        str(path.resolve()), st.st_dev, st.st_ino,
        st.st_ctime_ns, st.st_mtime_ns, st.st_size,
    )


def _hash_model_file(path: Path) -> str:
    """Hash a model asset, reusing the digest while its file identity is unchanged."""
    for _ in range(3):
        before = _model_stat_key(path)
        cached = _MODEL_HASH_CACHE.get(before)
        if cached is not None:
            return cached
        digest = _hash_file(path)
        after = _model_stat_key(path)
        if after == before:
            _MODEL_HASH_CACHE[before] = digest
            return digest
    raise TranscriptionError(f"model file changed during hashing: {path}")


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
    # work directory. Copy into a hidden file on the destination filesystem
    # created with 0o666 so the kernel applies the process umask consistently
    # with same-device publication, fsync it, then atomically replace dst.
    staged = dst.parent / f".{dst.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
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
    parakeet: ParakeetConfig | None = None,
    qwen: QwenConfig | None = None,
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
    local_engine, local_config = {
        "en": ("parakeet", parakeet),
        "zh": ("qwen", qwen),
        "yue": ("qwen", qwen),
    }.get(config.language, ("", None))
    if config.initial_prompt:
        local_config = None
    if local_config is None:
        _validate_whisper(config)
    if not shutil.which(config.ffmpeg_bin):
        raise TranscriptionError(f"ffmpeg binary not found: {config.ffmpeg_bin}")

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
        staged_json = output_base.with_suffix(".json")
        staged_txt = output_base.with_suffix(".txt")
        engine, engine_version = "whisper.cpp", config.engine_version
        model_name, threads, device = config.model_path.name, config.threads, config.device
        transcript: Transcript | None = None
        remote_revision = None
        if isinstance(local_config, ParakeetConfig) and local_config.worker_ssh_config:
            try:
                worker = await run_remote_parakeet(
                    wav_path, staged_json,
                    config=RemoteConfig(("ssh", "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes",
                                         "-F", str(local_config.worker_ssh_config), "parakeet-worker")),
                )
                transcript = load_whisper_json(staged_json, strict_parakeet=True)
            except (WorkerError, TranscriptionError, OSError, ValueError, TypeError, RecursionError, OverflowError) as exc:
                logger.warning("transcribe.parakeet_worker_fallback reason=%s", str(exc) if isinstance(exc, WorkerError) else type(exc).__name__)
                staged_json.unlink(missing_ok=True)
            else:
                engine, engine_version = "parakeet", worker.engine_version
                model_sha256, remote_revision = worker.model_sha256, worker.source_revision
                model_name, threads, device = PARAKEET_MODEL_NAME, worker.threads, "CPU"
                logger.info("transcribe.parakeet_worker_complete threads=%d", threads)
        if transcript is None and local_config is not None:
            try:
                version_used, model_sha256 = await _run_sherpa(wav_path, staged_json, local_engine, local_config)
                transcript = load_whisper_json(staged_json)
            except (TranscriptionError, OSError) as exc:
                logger.warning("transcribe.%s_fallback reason=%s", local_engine, exc)
                staged_json.unlink(missing_ok=True)
            else:
                engine, engine_version = local_engine, version_used
                model_name, threads, device = _SHERPA_MODELS[local_engine][0], local_config.threads, "CPU"
                logger.info("transcribe.%s_complete threads=%d", local_engine, threads)
        if transcript is None:
            if local_config is not None:
                _validate_whisper(config)
            async with _device_lock(config):
                await _run_whisper(wav_path, output_base=output_base, config=config)
            transcript = load_whisper_json(staged_json, fallback_text_path=staged_txt)
            model_sha256 = await asyncio.to_thread(_hash_model_file, config.model_path)
        if not staged_txt.exists():
            staged_txt.write_text(transcript.text, encoding="utf-8")

        txt_path = output_dir / f"{artifact_id}.txt"
        json_path = output_dir / f"{artifact_id}.json"
        manifest_path = output_dir / f"{artifact_id}.manifest.json"
        _atomic_publish(staged_txt, txt_path)
        _atomic_publish(staged_json, json_path)

        input_sha256 = _hash_file(input_path)
        source_revision = BUILD.source_revision
        manifest = {
            "schema_version": 2,
            "artifact_id": artifact_id,
            "input_sha256": input_sha256,
            "duration_seconds": round(duration_seconds, 3),
            "language": config.language,
            "engine": engine,
            "engine_version": engine_version,
            "model_name": model_name,
            "model_sha256": model_sha256,
            "local_transcriber_source_revision": source_revision,
            "whisper_binary": config.whisper_bin if engine == "whisper.cpp" else None,
            "ffmpeg_binary": config.ffmpeg_bin,
            "threads": threads,
            "device": device,
            "initial_prompt": config.initial_prompt,
            "initial_prompt_sha256": _prompt_sha256(config.initial_prompt),
            "segments": [asdict(segment) for segment in transcript.segments],
            "text_file": txt_path.name,
            "json_file": json_path.name,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if remote_revision is not None:
            manifest["worker_source_revision"] = remote_revision
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
            engine=engine,
            engine_version=engine_version,
            model_name=model_name,
            language=config.language,
            model_sha256=model_sha256,
            source_revision=source_revision,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _validate_whisper(config: WhisperCppConfig) -> None:
    if not config.model_path.is_file():
        raise TranscriptionError(f"whisper model not found: {config.model_path}")
    if not shutil.which(config.whisper_bin) and not Path(config.whisper_bin).is_file():
        raise TranscriptionError(f"whisper binary not found: {config.whisper_bin}")


async def _run_sherpa(
    wav_path: Path, output_path: Path, engine: str, config: ParakeetConfig | QwenConfig,
) -> tuple[str, str]:
    try:
        engine_version = version("sherpa-onnx")
    except PackageNotFoundError:
        raise TranscriptionError("sherpa-onnx is unavailable") from None
    model_files = _SHERPA_MODELS[engine][1]
    if any(not (config.model_dir / name).is_file() for name in model_files):
        raise TranscriptionError(f"{engine} model files are unavailable")
    cmd = [sys.executable, "-m", f"local_transcriber.{engine}", str(wav_path),
           str(config.model_dir), str(output_path), str(config.threads)]
    audio_seconds = wav_path.stat().st_size / _WAV_BYTES_PER_SECOND
    try:
        rc, _, _ = await run_capture(
            cmd, timeout=max(120.0, 2.0 * audio_seconds), label=engine,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        )
    except SubprocessTimeout as exc:
        raise TranscriptionError(str(exc)) from exc
    if rc:
        raise TranscriptionError(f"{engine} inference failed (rc={rc})")
    digests = await asyncio.to_thread(
        lambda: b"".join(bytes.fromhex(_hash_model_file(config.model_dir / name)) for name in model_files)
    )
    return engine_version, hashlib.sha256(digests).hexdigest()


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
    json_path: Path, *, fallback_text_path: Path | None = None, strict_parakeet: bool = False
) -> Transcript:
    if not json_path.exists():
        raise TranscriptionError(f"whisper JSON output missing: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"invalid whisper JSON output: {json_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TranscriptionError(f"invalid whisper JSON root (expected object): {json_path}")

    if strict_parakeet:
        rows = data.get("transcription")
        if data.get("source") != {"kind": "parakeet"} or not isinstance(rows, list) or not rows:
            raise TranscriptionError("invalid Parakeet transcript")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise TranscriptionError("invalid Parakeet segment text")
            row["text"].encode("utf-8")  # Reject unpaired surrogates before accepting the reply.
            offsets = row.get("offsets")
            if not isinstance(offsets, dict):
                raise TranscriptionError("missing Parakeet offsets")
            start_ms, end_ms = offsets.get("from"), offsets.get("to")
            if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms <= end_ms <= 86_400_000:
                raise TranscriptionError("invalid Parakeet segment offsets")

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
