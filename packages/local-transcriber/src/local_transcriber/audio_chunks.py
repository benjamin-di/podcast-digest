"""Split normalized PCM near quiet points using bounded working memory."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

RATE = 16_000
_FRAME = RATE // 50


def quiet_chunks(
    wav: wave.Wave_read, *, target_seconds: int = 30, margin_seconds: int = 5,
) -> Iterator[tuple[int, NDArray]]:
    import numpy as np

    target_frames = target_seconds * 50
    margin_frames = margin_seconds * 50
    max_chunk = (target_frames + margin_frames) * _FRAME
    length = wav.getnframes()
    start = 0
    while start < length:
        wav.setpos(start)
        # The extra frames give the smoothing window full context at the latest cut.
        payload = wav.readframes(min(length - start, max_chunk + 4 * _FRAME))
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)
        if len(samples) == 0:
            raise ValueError("truncated normalized WAV")
        cut = len(samples)
        if len(samples) > max_chunk:
            frames = samples[:len(samples) // _FRAME * _FRAME].reshape(-1, _FRAME)
            energy = np.mean(frames * frames, axis=1)
            smooth = np.convolve(energy, np.ones(9) / 9, mode="same")
            latest = min(len(smooth), target_frames + margin_frames + 1)
            choices = np.arange(target_frames - margin_frames, latest)
            score = 10 * np.log10(1e-10 + smooth[choices]) + 0.3 * abs(choices - target_frames) / 50
            cut = int(choices[np.argmin(score)]) * _FRAME
        yield start, samples[:cut]
        start += cut
