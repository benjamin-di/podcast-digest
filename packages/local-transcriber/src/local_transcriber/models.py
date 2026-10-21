"""Stable public contracts for the local transcription boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment]


@dataclass(frozen=True)
class WhisperCppConfig:
    whisper_bin: str
    model_path: Path
    language: str = "en"
    threads: int = 8
    device: str | None = "GPU"
    library_path: str | None = None
    initial_prompt: str | None = None
    engine_version: str = "unknown"
    ffmpeg_bin: str = "ffmpeg"
    ffmpeg_timeout_s: float = 900.0
    whisper_timeout_floor_s: float = 7200.0
    whisper_seconds_per_audio_second: float = 4.0
    lock_path: Path | None = None
    lock_timeout_s: float = 21_600.0


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: Transcript
    txt_path: Path
    json_path: Path
    manifest_path: Path
    input_sha256: str
    duration_seconds: float
    engine: str
    engine_version: str
    model_name: str
    language: str
    model_sha256: str | None = None
    source_revision: str | None = None
