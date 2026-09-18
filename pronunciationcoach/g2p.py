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
    import logging

    from phonemizer.backend import EspeakBackend

    # with_stress=False matches how the recogniser's labels were produced. phonemizer logs a
    # "words count mismatch" warning even when told to ignore it, so give it a muted logger.
    quiet = logging.getLogger("phonemizer.quiet")
    quiet.setLevel(logging.ERROR)
    return EspeakBackend(lang, language_switch="remove-flags", with_stress=False, words_mismatch="ignore", logger=quiet)


@dataclass
class WordPhones:
    word: str
    phones: list[str]


def words_of(text: str) -> list[str]:
    """Plain words only; digits and symbols are dropped (espeak would read '21' aloud)."""
    return _WORD_RE.findall(text)


def text_to_phones(text: str, lang: str = "en-us") -> list[WordPhones]:
    """Phonemise the sentence in context, keeping every phone attached to its word.

    Context matters for function words: espeak gives the weak forms a learner
    actually hears (a → ɐ, the → ð ɪ before a vowel, to → t ə), whereas in
    isolation it reads "a" as the letter name eɪ. Word boundaries come back as
    "|". espeak sometimes fuses a pair ("for a" → f ɚ ɹ ə), which breaks the
    one-chunk-per-word mapping; then each word is phonemised inside its own
    three-word window instead, so context survives everywhere except the fused pair.
    """
    words = words_of(text)
    if not words:
        return []
    chunks = _phonemize_chunks(" ".join(words), lang)
    if len(chunks) == len(words):
        return _tidy([WordPhones(word, phones) for word, phones in zip(words, chunks)])
    return _tidy([WordPhones(word, _in_context(words, i, lang)) for i, word in enumerate(words)])


def _phonemize_chunks(text: str, lang: str) -> list[list[str]]:
    from phonemizer.separator import Separator

    out = _backend(lang).phonemize([text], separator=Separator(phone=" ", word=" | ", syllable=""), strip=True)[0]
    return [chunk.split() for chunk in out.split("|") if chunk.strip()]


# The one word whose isolated pronunciation is simply wrong as a fallback.
_ISOLATED_OVERRIDES = {"a": ["ə"]}


def _in_context(words: list[str], i: int, lang: str) -> list[str]:
    window = words[max(0, i - 1) : i + 2]
    chunks = _phonemize_chunks(" ".join(window), lang)
    if len(chunks) == len(window):
        return chunks[0 if i == 0 else 1]
    if words[i].lower() in _ISOLATED_OVERRIDES:
        return list(_ISOLATED_OVERRIDES[words[i].lower()])
    return _phonemize_chunks(words[i], lang)[0]


_R_COLOURED = ("ɹ", "ɚ", "ɝ")


def _tidy(words: list[WordPhones]) -> list[WordPhones]:
    """Drop espeak's doubled r: it writes ʊɹ ɹ inside 'curiosity' and ɚ ɹ for 'water is'
    (American), but there is only one r sound. A British linking r (ə ɹ) is kept."""
    prev: str | None = None
    for wp in words:
        kept: list[str] = []
        for phone in wp.phones:
            if phone == "ɹ" and prev is not None and prev.endswith(_R_COLOURED):
                continue
            kept.append(phone)
            prev = phone
        wp.phones = kept
    return words
