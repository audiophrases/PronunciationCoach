"""Word recognition, and the listener's confidence in each word.

Whisper is deliberately *not* used for phonemes: it is trained to produce the
words a speaker meant, which is exactly what we want here (a reference text)
and exactly what hides pronunciation errors if used for anything else.

It is, however, a fair stand-in for a native listener: the probability it
assigns to each word says how easily the word was caught. listener.py turns
that into "understood / hesitated / missed" per reference word.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ctranslate2 keeps roughly ten times the checkpoint size resident on CPU (measured:
# tiny.en 0.75 GB, base.en 1.2 GB, small.en 2.3 GB). base.en fits next to the phoneme
# model on an 8 GB laptop; the Dockerfile picks small.en for the 16 GB Space.
DEFAULT_ASR = os.environ.get("PC_ASR_MODEL", "base.en")


@dataclass
class HeardWord:
    text: str
    probability: float
    start: float
    end: float


@lru_cache(maxsize=1)
def _model(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe_words(audio_16k: np.ndarray, size: str = DEFAULT_ASR) -> tuple[str, list[HeardWord]]:
    """The transcript, and each word with the probability Whisper gave it."""
    segments, _info = _model(size).transcribe(
        audio_16k, language="en", beam_size=5, vad_filter=False, word_timestamps=True
    )
    text_parts: list[str] = []
    words: list[HeardWord] = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append(HeardWord(w.word.strip(), float(w.probability), float(w.start), float(w.end)))
    return " ".join(text_parts).strip(), words


def transcribe(audio_16k: np.ndarray, size: str = DEFAULT_ASR) -> str:
    return transcribe_words(audio_16k, size)[0]
