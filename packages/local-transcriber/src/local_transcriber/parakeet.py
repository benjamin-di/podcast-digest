"""Decode normalized English WAV audio with an offline Parakeet INT8 export."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

from .audio_chunks import RATE as _RATE
from .audio_chunks import quiet_chunks as _chunks
from .models import PARAKEET_MODEL_FILES


def _transcribe(wav_path: Path, model_dir: Path, threads: int) -> list[dict]:
    import sherpa_onnx

    encoder, decoder, joiner, tokens = (str(model_dir / name) for name in PARAKEET_MODEL_FILES)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens,
        num_threads=threads, provider="cpu", model_type="nemo_transducer",
        decoding_method="greedy_search",
    )
    segments = []
    with wave.open(str(wav_path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, _RATE):
            raise ValueError("Parakeet requires 16 kHz mono signed 16-bit PCM")
        for offset, samples in _chunks(wav):
            stream = recognizer.create_stream()
            stream.accept_waveform(_RATE, samples)
            recognizer.decode_stream(stream)
            result = stream.result
            text = result.text.strip()
            if not text:
                continue
            duration = len(samples) / _RATE
            times = [stamp for stamp in result.timestamps if math.isfinite(stamp) and 0 <= stamp < duration]
            start = min(times) if times else 0.0
            end = min(duration, max(times) + 0.08) if times else duration
            segments.append({
                "offsets": {"from": round(1000 * (offset / _RATE + start)),
                            "to": round(1000 * (offset / _RATE + end))},
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
        raise SystemExit("Parakeet produced no text")
    args.output.write_text(
        json.dumps({"source": {"kind": "parakeet"}, "transcription": segments}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
