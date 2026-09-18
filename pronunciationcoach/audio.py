"""Audio loading and normalisation. Everything downstream expects float32 mono at 16 kHz."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from . import SAMPLE_RATE


def to_mono_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """Convert any (n,) or (n, channels) buffer at any rate to float32 mono 16 kHz."""
    x = np.asarray(samples)
    if x.dtype.kind in "iu":  # browsers and Gradio hand us int16
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = AF.resample(torch.from_numpy(x), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(x)


def load_audio(path: str | Path) -> np.ndarray:
    """Read wav/flac/ogg via libsndfile; fall back to ffmpeg for webm, mp3, m4a, ..."""
    path = str(path)
    try:
        samples, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_mono_16k(samples, sr)
    except (RuntimeError, sf.LibsndfileError):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            check=True,
            capture_output=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()


def duration_s(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE
