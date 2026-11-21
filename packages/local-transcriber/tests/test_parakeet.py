from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from local_transcriber import (
    ParakeetConfig,
    QwenConfig,
    WhisperCppConfig,
    load_whisper_json,
    whisper,
)
from local_transcriber.models import PARAKEET_MODEL_FILES, QWEN_MODEL_FILES
from local_transcriber.parakeet import _chunks


@pytest.mark.parametrize("frames", [35 * 16_000 + 1, 71 * 16_000])
def test_chunks_cover_the_wave_across_quiet_boundaries(tmp_path: Path, frames: int) -> None:
    pytest.importorskip("numpy")
    media = tmp_path / "sample.wav"
    with wave.open(str(media), "wb") as wav:
        wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * frames)
    consumed = 0
    with wave.open(str(media), "rb") as wav:
        for offset, samples in _chunks(wav):
            assert offset == consumed
            assert 0 < len(samples) <= 35 * 16_000
            consumed += len(samples)
    assert consumed == frames


@pytest.mark.parametrize(("language", "condition", "commands", "backend"), [
    ("en", "ok", ["ffmpeg", "parakeet"], "parakeet"),
    ("en", "failed", ["ffmpeg", "parakeet", "whisper-cli"], "whisper.cpp"),
    ("en", "empty", ["ffmpeg", "parakeet", "whisper-cli"], "whisper.cpp"),
    ("en", "missing", ["ffmpeg", "whisper-cli"], "whisper.cpp"),
    ("zh", "ok", ["ffmpeg", "qwen"], "qwen"),
    ("yue", "ok", ["ffmpeg", "qwen"], "qwen"),
    ("zh", "failed", ["ffmpeg", "qwen", "whisper-cli"], "whisper.cpp"),
    ("zh", "empty", ["ffmpeg", "qwen", "whisper-cli"], "whisper.cpp"),
    ("zh", "missing", ["ffmpeg", "whisper-cli"], "whisper.cpp"),
    ("zh", "disabled", ["ffmpeg", "whisper-cli"], "whisper.cpp"),
    ("fr", "ok", ["ffmpeg", "whisper-cli"], "whisper.cpp"),
])
async def test_local_engine_routing_reuses_normalized_audio_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    language: str, condition: str, commands: list[str], backend: str,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    medium = model_dir / "medium.bin"
    medium.write_bytes(b"whisper fixture")
    if condition != "missing":
        for name in (*PARAKEET_MODEL_FILES, *QWEN_MODEL_FILES):
            model_file = model_dir / name
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes(name.encode())
    item = tmp_path / "episode.mp3"
    item.write_bytes(b"source fixture")
    local_lock = tmp_path / "whisper.lock"
    calls: list[str] = []
    monkeypatch.setattr(whisper.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(whisper, "version", lambda _: "2.0.1")

    async def fake_run(cmd, *, timeout, label, env=None):
        calls.append(label)
        segment = {"offsets": {"from": 250, "to": 1000}, "text": "hello world"}
        if label == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"0" * (44 + 64_000))
        elif label in ("parakeet", "qwen"):
            assert cmd[-1] == "4"
            assert cmd[2] == f"local_transcriber.{label}"
            data = {"transcription": [] if condition != "ok" else [segment]}
            Path(cmd[-2]).write_text(json.dumps(data))
            if condition == "failed":
                return 1, b"", b"stub process failure"
        else:
            output = Path(cmd[cmd.index("-of") + 1])
            output.with_suffix(".json").write_text(json.dumps({"transcription": [segment]}))
        return 0, b"", b""

    monkeypatch.setattr(whisper, "run_capture", fake_run)
    result = await whisper.transcribe_file(
        item, output_dir=tmp_path / "out", artifact_id="episode",
        config=WhisperCppConfig(whisper_bin="whisper-cli", model_path=medium,
                                language=language, lock_path=local_lock),
        parakeet=ParakeetConfig(model_dir),
        qwen=QwenConfig(model_dir) if condition != "disabled" else None,
    )
    assert calls == commands
    assert result.engine == backend
    assert result.transcript == load_whisper_json(result.json_path)
    assert result.txt_path.read_text() == "hello world"
    assert result.transcript.segments[0].start == 0.25
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["engine"] == backend and manifest["language"] == language
    assert manifest["device"] == ("GPU" if backend == "whisper.cpp" else "CPU")
    assert local_lock.exists() == (backend == "whisper.cpp")
