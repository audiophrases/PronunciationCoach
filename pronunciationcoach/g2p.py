"""Reference text -> expected phones, using espeak-ng through phonemizer.

espeak-ng is used because the default engine's label set *is* espeak's IPA
inventory, so expected and recognised phones share one alphabet and can be
compared symbol for symbol. `en-us` and `en-gb` differ where a teacher would
expect them to (water: w ɔː ɾ ɚ vs w ɔː t ə).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from .espeak import ensure_espeak

ACCENTS = {"American": "en-us", "British": "en-gb"}  # General American is the project default
DEFAULT_ACCENT = os.environ.get("PC_ACCENT", "American")
if DEFAULT_ACCENT not in ACCENTS:
    raise ValueError(f"PC_ACCENT must be one of {list(ACCENTS)}, got {DEFAULT_ACCENT!r}")

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@lru_cache(maxsize=4)
def _backend(lang: str):
    ensure_espeak()
    from phonemizer.backend import EspeakBackend

    # with_stress=False matches how the recogniser's labels were produced.
    return EspeakBackend(lang, language_switch="remove-flags", with_stress=False, words_mismatch="ignore")


@dataclass
class WordPhones:
    word: str
    phones: list[str]


def words_of(text: str) -> list[str]:
    """Plain words only; digits and symbols are dropped (espeak would read '21' aloud)."""
    return _WORD_RE.findall(text)


def text_to_phones(text: str, lang: str = "en-us") -> list[WordPhones]:
    """Phonemise word by word so every phone stays attached to the word it belongs to."""
    from phonemizer.separator import Separator

    words = words_of(text)
    if not words:
        return []
    sep = Separator(phone=" ", word="", syllable="")
    phonemised = _backend(lang).phonemize(words, separator=sep, strip=True)
    return [WordPhones(word, phones.split()) for word, phones in zip(words, phonemised)]
