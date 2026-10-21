from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from local_transcriber import TranscriptionError, WhisperCppConfig, load_whisper_json, whisper


def test_atomic_publish_handles_cross_device_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "work" / "artifact.txt"
    dst = tmp_path / "mounted-output" / "artifact.txt"
    src.parent.mkdir()
    dst.parent.mkdir()
    src.write_text("complete artifact", encoding="utf-8")
    real_replace = whisper.os.replace
    first_call = True

    def replace_with_cross_device_once(source: Path, target: Path) -> None:
        nonlocal first_call
        if first_call:
            first_call = False
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(source, target)

    monkeypatch.setattr(whisper.os, "replace", replace_with_cross_device_once)
    whisper._atomic_publish(src, dst)

    assert dst.read_text(encoding="utf-8") == "complete artifact"
    assert not src.exists()
    assert list(dst.parent.iterdir()) == [dst]


def test_load_whisper_json_parses_timestamped_segments(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text(
        json.dumps(
            {
                "transcription": [
                    {"offsets": {"from": 1250, "to": 2750}, "text": " CET1 guidance "},
                    {"offsets": {"from": 2750, "to": 4000}, "text": "remains unchanged"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = load_whisper_json(path)
    assert result.text == "CET1 guidance remains unchanged"
    assert result.segments[0].start == 1.25
    assert result.segments[0].end == 2.75
    assert result.segments[0].speaker is None


def test_load_whisper_json_falls_back_to_text(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    fallback = tmp_path / "out.txt"
    path.write_text('{"transcription": []}', encoding="utf-8")
    fallback.write_text("fallback transcript", encoding="utf-8")
    result = load_whisper_json(path, fallback_text_path=fallback)
    assert result.text == "fallback transcript"
    assert result.segments == []


@pytest.mark.parametrize("payload", ["not json", "[]"])
def test_load_whisper_json_rejects_invalid_output(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "out.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(TranscriptionError):
        load_whisper_json(path)


async def test_transcribe_file_uses_unique_staging_and_publishes_manifest_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "call.mp3"
    model = tmp_path / "model.bin"
    media.write_bytes(b"source-media")
    model.write_bytes(b"model")
    calls: list[list[str]] = []

    monkeypatch.setattr(whisper.shutil, "which", lambda _name: "/bin/tool")
    monkeypatch.setenv("LOCAL_TRANSCRIBER_SOURCE_REV", "abc123")

    async def fake_run(cmd, *, timeout, label, env=None):
        calls.append(cmd)
        if label == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"0" * (44 + 64_000))
        else:
            output_base = Path(cmd[cmd.index("-of") + 1])
            output_base.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "transcription": [
                            {"offsets": {"from": 0, "to": 2000}, "text": "hello bank"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_base.with_suffix(".txt").write_text("hello bank", encoding="utf-8")
        return 0, b"", b""

    monkeypatch.setattr(whisper, "run_capture", fake_run)
    result = await whisper.transcribe_file(
        media,
        output_dir=tmp_path / "out",
        artifact_id="examplebank-q1-2026",
        config=WhisperCppConfig(
            whisper_bin="whisper-cli",
            model_path=model,
            initial_prompt="Example Bank, CET1, NII",
            engine_version="v1.7.4",
        ),
    )

    assert result.duration_seconds == 2.0
    assert result.txt_path.read_text(encoding="utf-8") == "hello bank"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["engine_version"] == "v1.7.4"
    assert manifest["input_sha256"] == result.input_sha256
    assert manifest["model_sha256"] == result.model_sha256
    assert manifest["local_transcriber_source_revision"] == "abc123"
    assert result.source_revision == "abc123"
    assert manifest["initial_prompt"] == "Example Bank, CET1, NII"
    assert len(manifest["initial_prompt_sha256"]) == 64
    whisper_cmd = calls[1]
    assert whisper_cmd[whisper_cmd.index("--prompt") + 1] == "Example Bank, CET1, NII"
    assert "-p" not in whisper_cmd
    assert not list((tmp_path / "tmp").iterdir())


async def test_transcribe_file_rejects_unsafe_artifact_id(tmp_path: Path) -> None:
    media = tmp_path / "call.mp3"
    model = tmp_path / "model.bin"
    media.write_bytes(b"x")
    model.write_bytes(b"x")
    with pytest.raises(TranscriptionError, match="artifact_id"):
        await whisper.transcribe_file(
            media,
            output_dir=tmp_path,
            artifact_id="../escape",
            config=WhisperCppConfig(whisper_bin="whisper-cli", model_path=model),
        )
