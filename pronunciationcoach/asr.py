"""Word recognition for free-speech mode.

Whisper is deliberately *not* used for phonemes: it is trained to produce the
words a speaker meant, which is exactly what we want here (a reference text)
and exactly what hides pronunciation errors if used for anything else.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# ctranslate2 keeps roughly ten times the checkpoint size resident on CPU (measured:
# tiny.en 0.75 GB, base.en 1.2 GB, small.en 2.3 GB). base.en fits next to the phoneme
# model on an 8 GB laptop; the Dockerfile picks small.en for the 16 GB Space.
DEFAULT_ASR = os.environ.get("PC_ASR_MODEL", "base.en")


@lru_cache(maxsize=1)
def _model(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(audio_16k: np.ndarray, size: str = DEFAULT_ASR) -> str:
    segments, _info = _model(size).transcribe(audio_16k, language="en", beam_size=5, vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()
