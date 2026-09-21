"""Render learner crops consistently for playback and recognition checks."""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


PAD_BEFORE_S = 0.06
PAD_AFTER_S = 0.05
PAD_QUIET_DB = 18.0  # relative to the crop's own loudest 20 ms
FADE_S = 0.01
TARGET_PEAK = 0.7
MAX_GAIN = 8.0


def sample_bounds(
    audio: np.ndarray,
    start: float,
    end: float,
    sr: int = SAMPLE_RATE,
    *,
    pad_before: bool = True,
    pad_after: bool = True,
) -> tuple[int, int]:
    """Return the exact sample range rendered by :func:`clip`.

    Padding extends only through quiet audio. A corrected edge can disable its
    padding so rendering does not add the removed speech back into the crop.
    """
    a = min(len(audio), max(0, int(start * sr)))
    b = max(a, min(len(audio), int(end * sr)))
    if a == b or not (pad_before or pad_after):
        return a, b
    win = max(1, int(0.005 * sr))

    def rms(x) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float32)))) if len(x) else 0.0

    core = np.asarray(audio[a:b], dtype=np.float32)
    n = len(core) // win
    if n >= 4:
        frames = np.sqrt(np.mean(np.square(core[: n * win]).reshape(n, win), axis=1))
        loud = float(np.max(np.convolve(frames, np.ones(4) / 4, mode="valid")))
    else:
        loud = rms(core)
    thr = loud * 10 ** (-PAD_QUIET_DB / 20)
    if pad_before:
        lim = max(0, a - int(PAD_BEFORE_S * sr))
        while a - win >= lim and rms(audio[a - win : a]) < thr:
            a -= win
    if pad_after:
        lim = min(len(audio), b + int(PAD_AFTER_S * sr))
        while b + win <= lim and rms(audio[b : b + win]) < thr:
            b += win
    return a, b


def clip(
    audio: np.ndarray,
    start: float,
    end: float,
    sr: int = SAMPLE_RATE,
    *,
    pad_before: bool = True,
    pad_after: bool = True,
) -> tuple[int, np.ndarray]:
    """Crop original audio, pad through quiet sound, fade clicks and boost level."""
    a, b = sample_bounds(audio, start, end, sr, pad_before=pad_before, pad_after=pad_after)
    out = np.array(audio[a:b], dtype=np.float32)
    n = min(int(FADE_S * sr), len(out) // 2)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0:
        out *= min(TARGET_PEAK / peak, MAX_GAIN)
    return sr, out
