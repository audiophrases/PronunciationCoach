"""A listener in the loop: was each word understood?

Pronunciation only matters through what a listener makes of it. Whisper,
trained on how native listeners resolve speech, recognises the words and says
how confident it was about each one. Matched against the reference words
that gives, per word, whether it was caught, caught with hesitation, or
missed - which is what separates an accent from a problem.

Whisper is generous (it uses context to fill words in), so this is a floor,
not a verdict on its own; scoring.py combines it with the phone-level
evidence and the importance of each sound.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from .asr import HeardWord, transcribe_words

UNDERSTOOD_P = 0.6  # at or above: caught without effort
HESITANT_P = 0.3  # below UNDERSTOOD_P and at or above this: caught, with effort; below: missed


@dataclass
class Listened:
    word: str
    probability: float  # 0 when the listener produced something else, or nothing
    matched: bool  # the listener's word was the reference word

    @property
    def understood(self) -> bool:
        return self.matched and self.probability >= HESITANT_P

    @property
    def confident(self) -> bool:
        return self.matched and self.probability >= UNDERSTOOD_P


def _norm(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def match_words(reference: list[str], heard: list[HeardWord]) -> list[Listened]:
    """Line the listener's words up with the reference words (lower-cased, punctuation
    stripped) and carry each matched word's probability across."""
    ref = [_norm(w) for w in reference]
    got = [_norm(h.text) for h in heard]
    out = [Listened(w, 0.0, False) for w in reference]
    sm = SequenceMatcher(a=ref, b=got, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            continue
        for k in range(i2 - i1):
            out[i1 + k] = Listened(reference[i1 + k], heard[j1 + k].probability, True)
    return out


def listen(audio_16k: np.ndarray, reference: list[str], heard: list[HeardWord] | None = None) -> list[Listened]:
    """Per reference word: did the listener get it, and how easily?"""
    if heard is None:
        _text, heard = transcribe_words(audio_16k)
    return match_words(reference, heard)
