"""Podcast adapter over the reusable local whisper.cpp transcriber."""

from __future__ import annotations

import os
from pathlib import Path

from local_transcriber import (
    Segment,
    Transcript,
    TranscriptionError,
    WhisperCppConfig,
    load_whisper_json,
    transcribe_file,
)

# Backward-compatible policy constants used by the podcast tests and runtime.
_WHISPER_TIMEOUT_FLOOR = 7200.0
_WHISPER_SECONDS_PER_AUDIO_SECOND = 4.0
_WAV_BYTES_PER_SECOND = 32_000


def _whisper_timeout(wav_bytes: int) -> float:
    audio_seconds = wav_bytes / _WAV_BYTES_PER_SECOND
    return max(_WHISPER_TIMEOUT_FLOOR, audio_seconds * _WHISPER_SECONDS_PER_AUDIO_SECOND)


async def transcribe(
    audio_path: Path,
    *,
    out_dir: Path,
    basename: str,
    whisper_bin: str,
    model_path: Path,
    language: str = "en",
    threads: int = 8,
) -> tuple[Transcript, Path, Path]:
    """Transcribe an episode and return the podcast pipeline's legacy tuple."""
    result = await transcribe_file(
        audio_path,
        output_dir=out_dir,
        artifact_id=basename,
        config=WhisperCppConfig(
            whisper_bin=whisper_bin,
            model_path=model_path,
            language=language,
            threads=threads,
            device="GPU",
            library_path=os.environ.get(
                "PODCAST_DIGEST_WHISPER_LD_PATH",
                "/usr/local/lib/whisper:/opt/intel/openvino/runtime/lib/intel64",
            ),
            engine_version=os.environ.get("WHISPER_CPP_VERSION", "unknown"),
            lock_path=(
                Path(os.environ["WHISPER_GPU_LOCK_PATH"])
                if os.environ.get("WHISPER_GPU_LOCK_PATH")
                else None
            ),
        ),
    )
    return result.transcript, result.txt_path, result.json_path


__all__ = [
    "Segment",
    "Transcript",
    "TranscriptionError",
    "load_whisper_json",
    "transcribe",
]
