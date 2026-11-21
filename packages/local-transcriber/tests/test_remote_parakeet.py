from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from local_transcriber import ParakeetConfig, WhisperCppConfig, whisper
from local_transcriber.models import PARAKEET_MODEL_NAME
from local_transcriber.remote import RemoteConfig, WorkerError, run_remote_parakeet
from local_transcriber.worker import _infer


@pytest.mark.parametrize("condition", ["ok", "bad_json", "surrogate", "bad_version", "wrong_model", "truncated", "failed"])
async def test_ssh_reply_or_local_retry(monkeypatch, tmp_path: Path, condition: str):
    # Exercise the actual transport process and final artifact publication.
    ssh = tmp_path / "ssh"
    row = {"offsets": {"from": 10, "to": 20}, "text": "remote decode"}
    if condition == "surrogate":
        row["text"] = chr(0xd800)
    payload = json.dumps({"source": {"kind": "parakeet"}, "transcription": [row]}).encode()
    if condition == "bad_json":
        payload = b'{"source":{"kind":"parakeet"},"transcription":[{"text":[] }]}'
    header = json.dumps({
        "version": 1, "model": PARAKEET_MODEL_NAME if condition != "wrong_model" else "other",
        "engine_version": chr(0xd800) if condition == "bad_version" else "2.0.1", "threads": 4, "source_revision": "a" * 40,
        "model_sha256": "b" * 64, "json_size": len(payload),
    }).encode() + b"\n"
    response = header + (payload[:-1] if condition == "truncated" else payload)
    ssh.write_text(
        f"#!{sys.executable}\nimport sys,json\n"
        "size=json.loads(sys.stdin.buffer.readline())['wav_size']\n"
        "assert len(sys.stdin.buffer.read(size)) == size\n"
        f"sys.stdout.buffer.write({response!r})\nsys.exit({int(condition == 'failed')})\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    media = tmp_path / "media"
    media.write_bytes(b"sample" * 20)
    monkeypatch.setattr(whisper.shutil, "which", lambda _: "fixture")
    calls = []

    async def normalize(src, dst, *, config):
        calls.append("normalize")
        dst.write_bytes(src.read_bytes())

    async def local(wav, output, engine, config):
        calls.append(engine)
        output.write_text(json.dumps({"transcription": [{**row, "text": "local decode"}]}))
        return "2.0.1", "c" * 64

    monkeypatch.setattr(whisper, "_normalize", normalize)
    monkeypatch.setattr(whisper, "_run_sherpa", local)
    result = await whisper.transcribe_file(
        media, output_dir=tmp_path / "out", artifact_id="item",
        config=WhisperCppConfig("whisper-cli", tmp_path / "unused"),
        parakeet=ParakeetConfig(tmp_path, worker_ssh_config=tmp_path / "config"),
    )
    remote = condition == "ok"
    assert calls == (["normalize"] if remote else ["normalize", "parakeet"])
    assert result.txt_path.read_text() == ("remote decode" if remote else "local decode")
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest.get("worker_source_revision") == ("a" * 40 if remote else None)
    assert manifest["model_sha256"] == ("b" if remote else "c") * 64


async def test_transport_timeout(tmp_path: Path):
    media = tmp_path / "input.wav"
    media.write_bytes(bytes(128))
    cmd = (sys.executable, "-c", "import sys;sys.stdin.buffer.read()")
    with pytest.raises(WorkerError, match="timed out"):
        await run_remote_parakeet(media, tmp_path / "result", config=RemoteConfig(cmd, 0.1))


async def test_disconnect_cancels_decoder(monkeypatch, tmp_path: Path):
    cancelled = asyncio.Event()

    async def decode(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("local_transcriber.worker._run_sherpa", decode)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    with pytest.raises(WorkerError, match="cancelled"):
        await _infer(reader, tmp_path / "in", tmp_path / "out", ParakeetConfig(tmp_path))
    assert cancelled.is_set()
