"""Native reference by example.

The dictionary says how a word is pronounced in isolation; fluent speakers say
it differently in a sentence (weak forms, linking, dropped or merged sounds),
and the recogniser has label habits of its own (it may hear a native's "so" as
s oː). Scoring a learner against the dictionary punishes both.

So the sentence is synthesised with a couple of natural voices, each rendering
is run through *our own* recogniser, and what it hears for each word - Edge
reports where every word lies - becomes an accepted way of saying that word.
Mapped back onto the canonical phones, that yields, per phone, extra symbols
that count as correct and a flag for sounds natives drop in this position.
Everything is cached next to the synthesised audio, so a sentence costs this
once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from .align import align_words, greedy_decode
from .audio import load_audio
from .g2p import WordPhones, text_to_phones, words_of
from .tts import REFERENCE_VOICES, synthesize, word_boundaries
from .variants import apply_rules, casual_renderings

log = logging.getLogger("pronunciationcoach")


@dataclass
class NativeReference:
    words: list[str]
    renderings: dict[str, list[list[str]]]  # voice -> per word, the phones the recogniser heard


@dataclass
class Acceptance:
    """What the scorer may accept for one canonical phone besides the phone itself."""

    also: set[str] = field(default_factory=set)  # symbols a native rendering produced here
    optional: bool = False  # a native rendering dropped this sound
    source: str = ""  # voice that attested it, for the feedback text


def _heard_per_word(engine, audio_path, canonical: list[WordPhones]) -> list[list[str]] | None:
    """What the recogniser hears in one native rendering, split by word.

    Words are located with our own forced alignment of the canonical phones (the
    ordering constraint keeps a word-initial sound from being credited to the
    previous word, which happens if Edge's timings are used with a fixed spike
    lag); every heard spike then goes to the word of the nearest aligned spike.
    Edge's word count is only used as a sanity check on the synthesis.
    """
    bounds = word_boundaries(audio_path)
    if bounds is None:
        return None
    if len([b for b in bounds if any(ch.isalpha() for ch in b.text)]) != len(canonical):
        log.info("native reference: Edge split %s differently from the sentence - skipping", audio_path.name)
        return None
    em = engine.emissions(load_audio(audio_path))
    word_ids = [[engine.phone_id(p) if engine.phone_id(p) is not None else engine.unk_id for p in wp.phones] for wp in canonical]
    aligned = align_words(em, word_ids).segments
    owner = np.concatenate([[wi] * len(wp.phones) for wi, wp in enumerate(canonical)])
    centres = np.array([(s.start + s.end) / 2 for s in aligned])
    heard: list[list[str]] = [[] for _ in canonical]
    for seg in greedy_decode(em):
        c = (seg.start + seg.end) / 2
        heard[int(owner[int(np.abs(centres - c).argmin())])].append(seg.phone)
    return heard


def native_reference(text: str, lang: str, engine, voices: list[str] | None = None) -> NativeReference | None:
    """Listen to native renderings of `text`; None if none could be produced."""
    words = words_of(text)
    if not words:
        return None
    canonical = text_to_phones(text, lang)
    renderings: dict[str, list[list[str]]] = {}
    for voice in voices or REFERENCE_VOICES.get(lang, REFERENCE_VOICES["en-us"]):
        try:
            path = synthesize(text, lang, "normal", voice=voice, with_words=True)
        except Exception as exc:
            log.warning("native reference: could not synthesise with %s (%s)", voice, exc)
            continue
        cache = path.with_suffix(f".heard.{engine.model_id.replace('/', '_')}.json")
        if cache.exists():
            heard = json.loads(cache.read_text(encoding="utf-8"))
        else:
            heard = _heard_per_word(engine, path, canonical)
            if heard is None:
                continue
            cache.write_text(json.dumps(heard, ensure_ascii=False), encoding="utf-8")
        renderings[voice] = heard
    return NativeReference(words, renderings) if renderings else None


def acceptances(
    canonical: list[WordPhones], ref: NativeReference | None, lang: str = "en-us", casual: bool = True
) -> list[list[Acceptance]]:
    """Per word, per canonical phone: what counts as correct in that position.

    Three sources feed it: native voices (`ref`), the casual forms and merges in
    variants.py, and the connected-speech rules. Each alternative rendering is
    aligned to the canonical phones with a sequence matcher: equal stretches
    confirm the phone, substitutions add the heard symbol, and canonical phones
    with no counterpart become optional (natives drop them). Inserted sounds are
    ignored - a learner is not penalised for omitting a glide the recogniser
    heard in a native's diphthong.
    """
    out = [[Acceptance() for _ in wp.phones] for wp in canonical]
    renderings: dict[str, list[list[str]]] = dict(ref.renderings) if ref else {}
    if casual:
        renderings.update(casual_renderings(canonical, lang))
    for voice, rendering in renderings.items():
        for wi, (wp, heard) in enumerate(zip(canonical, rendering)):
            sm = SequenceMatcher(a=wp.phones, b=heard, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal" or op == "insert":
                    continue
                if op == "delete":
                    for i in range(i1, i2):
                        out[wi][i].optional = True
                        out[wi][i].source = out[wi][i].source or voice
                    continue
                # replace: pair phones off in order; leftover canonical phones are optional
                n = min(i2 - i1, j2 - j1)
                for k in range(n):
                    out[wi][i1 + k].also.add(heard[j1 + k])
                    out[wi][i1 + k].source = out[wi][i1 + k].source or voice
                if i2 - i1 > n:
                    for i in range(i1 + n, i2):
                        out[wi][i].optional = True
                        out[wi][i].source = out[wi][i].source or voice
    if casual:
        apply_rules(canonical, out, lang)
    return out
