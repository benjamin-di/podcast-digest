"""Decode normalized Chinese WAV audio with Qwen3-ASR 0.6B INT8."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from .audio_chunks import RATE, quiet_chunks

_MAX_NEW_TOKENS = 256


def _transcribe(wav_path: Path, model_dir: Path, threads: int) -> list[dict]:
    import sherpa_onnx

    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(model_dir / "conv_frontend.onnx"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        tokenizer=str(model_dir / "tokenizer"),
        num_threads=threads,
        provider="cpu",
        max_total_len=768,
        max_new_tokens=_MAX_NEW_TOKENS,
    )
    segments = []
    with wave.open(str(wav_path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise ValueError("Qwen requires 16 kHz mono signed 16-bit PCM")
        for offset, samples in quiet_chunks(wav, target_seconds=20, margin_seconds=3):
            stream = recognizer.create_stream()
            stream.accept_waveform(RATE, samples)
            recognizer.decode_stream(stream)
            result = stream.result
            if len(result.tokens) >= _MAX_NEW_TOKENS:
                raise ValueError("Qwen hit its output token limit")
            text = result.text.strip()
            if text:
                # The ONNX export provides text without native timestamps.
                segments.append({
                    "offsets": {"from": round(1000 * offset / RATE),
                                "to": round(1000 * (offset + len(samples)) / RATE)},
                    "text": text,
                })
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("threads", type=int)
    args = parser.parse_args()
    segments = _transcribe(args.wav, args.model_dir, args.threads)
    if not segments:
        raise SystemExit("Qwen produced no text")
    args.output.write_text(
        json.dumps({"source": {"kind": "qwen", "timing": "chunk"}, "transcription": segments},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
