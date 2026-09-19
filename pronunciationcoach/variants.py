"""How fluent speakers actually say the words of a sentence.

The dictionary form is the careful, isolated one. Connected speech reduces,
links, drops and merges - "I'm going to go" comes out as "I'm'ona go" and
nobody is mispronouncing anything. This module lists those forms so the scorer
can accept them, in three layers:

* LEXICAL  - casual forms of single words (gonna, and, of, you, him, ...)
* PAIRS    - two words that merge (going to, don't know, did you, ...), with the
             merged phones split back over the two words
* rules    - processes that apply anywhere the context allows: final t/d dropped
             before a consonant, nasal assimilation, yod coalescence, glottal t,
             g-dropping, th assimilating to a preceding n/l/z

All symbols are espeak-ng IPA as the recogniser labels them. The tables are
General American; British gets ɚ -> ə. Everything here is standard ELT
connected-speech material and is meant to be edited by the teacher.
"""

from __future__ import annotations

from .g2p import WordPhones

VOWELS = set("iɪeɛæaɑɒɔoʊuʌəɐɜɚɝ") | {"iː", "uː", "ɑː", "ɔː", "ɜː", "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "əʊ", "ᵻ", "ɨ"}


def is_vowel(p: str) -> bool:
    return p in VOWELS or p[0] in "iɪeɛæaɑɒɔoʊuʌəɐɜɚɝ"


# Casual single-word forms (lowercase headword -> alternative renderings).
LEXICAL: dict[str, list[list[str]]] = {
    "gonna": [["ɡ", "ʌ", "n", "ə"], ["ɡ", "ɔ", "n", "ə"], ["ə", "n", "ə"]],
    "wanna": [["w", "ʌ", "n", "ə"], ["w", "ɑː", "n", "ə"]],
    "gotta": [["ɡ", "ɑː", "ɾ", "ə"]],
    "dunno": [["d", "ə", "n", "oʊ"]],
    "lemme": [["l", "ɛ", "m", "i"]],
    "gimme": [["ɡ", "ɪ", "m", "i"]],
    "kinda": [["k", "aɪ", "n", "d", "ə"]],
    "sorta": [["s", "ɔːɹ", "ɾ", "ə"]],
    "outta": [["aʊ", "ɾ", "ə"]],
    "hafta": [["h", "æ", "f", "t", "ə"]],
    "because": [["k", "ə", "z"], ["b", "ɪ", "k", "ə", "z"]],
    "cause": [["k", "ə", "z"]],
    "cuz": [["k", "ə", "z"]],
    "probably": [["p", "ɹ", "ɑː", "b", "l", "i"], ["p", "ɹ", "ɑː", "l", "i"]],
    "and": [["ə", "n"], ["n̩"], ["ə", "n", "d"]],
    "of": [["ə", "v"], ["ə"]],
    "you": [["j", "ə"], ["j", "ʊ"]],
    "your": [["j", "ɚ"]],
    "them": [["ð", "ə", "m"], ["ə", "m"]],
    "him": [["ɪ", "m"]],
    "her": [["ɚ"]],
    "his": [["ɪ", "z"]],
    "he": [["i"]],
    "have": [["ə", "v"], ["h", "ə", "v"]],
    "has": [["ə", "z"], ["h", "ə", "z"]],
    "had": [["ə", "d"], ["h", "ə", "d"]],
    "for": [["f", "ɚ"]],
    "can": [["k", "ə", "n"], ["k", "n̩"]],
    "will": [["ə", "l"], ["l̩"]],
    "would": [["w", "ə", "d"], ["ə", "d"]],
    "could": [["k", "ə", "d"]],
    "should": [["ʃ", "ə", "d"]],
    "that": [["ð", "ə", "t"]],
    "at": [["ə", "t"]],
    "as": [["ə", "z"]],
    "but": [["b", "ə", "t"]],
    "or": [["ɚ"]],
    "are": [["ɚ"]],
    "was": [["w", "ə", "z"]],
    "were": [["w", "ɚ"]],
    "from": [["f", "ɹ", "ə", "m"]],
    "than": [["ð", "ə", "n"]],
    "there": [["ð", "ɚ"]],
    "some": [["s", "ə", "m"]],
    "just": [["dʒ", "ə", "s", "t"], ["dʒ", "ə", "s"]],
    "to": [["t", "ə"], ["ɾ", "ə"]],
    "the": [["ð", "ə"], ["ð", "ɪ"]],
    "a": [["ə"]],
    "an": [["ə", "n"]],
    "us": [["ə", "s"]],
    "our": [["ɑːɹ"]],
    "do": [["d", "ə"]],
    "does": [["d", "ə", "z"]],
    "am": [["ə", "m"], ["m"]],
    "what": [["w", "ə", "t"]],
    "yeah": [["j", "ɛ", "ə"], ["j", "æ"]],
}

# Two words that merge: (merged phones for word 1, merged phones for word 2).
PAIRS: dict[tuple[str, str], list[tuple[list[str], list[str]]]] = {
    ("going", "to"): [(["ɡ", "ə", "n"], ["ə"]), (["ɡ", "ʌ", "n"], ["ə"]), (["ɡ", "oʊ", "ɪ", "n"], ["ə"])],
    ("want", "to"): [(["w", "ɑː", "n"], ["ə"]), (["w", "ʌ", "n"], ["ə"])],
    ("got", "to"): [(["ɡ", "ɑː", "ɾ"], ["ə"])],
    ("have", "to"): [(["h", "æ", "f"], ["t", "ə"])],
    ("has", "to"): [(["h", "æ", "s"], ["t", "ə"])],
    ("ought", "to"): [(["ɔː", "ɾ"], ["ə"])],
    ("don't", "know"): [(["d", "ə"], ["n", "oʊ"]), (["d", "oʊ", "n"], ["oʊ"]), (["d", "ə", "n"], ["oʊ"])],
    ("let", "me"): [(["l", "ɛ"], ["m", "i"])],
    ("give", "me"): [(["ɡ", "ɪ"], ["m", "i"])],
    ("kind", "of"): [(["k", "aɪ", "n", "d"], ["ə"])],
    ("sort", "of"): [(["s", "ɔːɹ", "ɾ"], ["ə"])],
    ("out", "of"): [(["aʊ", "ɾ"], ["ə"])],
    ("lot", "of"): [(["l", "ɑː", "ɾ"], ["ə"])],
    ("did", "you"): [(["d", "ɪ", "dʒ"], ["ə"])],
    ("would", "you"): [(["w", "ʊ", "dʒ"], ["ə"])],
    ("could", "you"): [(["k", "ʊ", "dʒ"], ["ə"])],
    ("don't", "you"): [(["d", "oʊ", "n", "tʃ"], ["ə"])],
    ("what", "are"): [(["w", "ʌ", "ɾ"], ["ɚ"])],
    ("what", "do"): [(["w", "ʌ", "ɾ"], ["ə"])],
    ("i'm", "gonna"): [(["aɪ", "m"], ["ə", "n", "ə"])],
}


def _british(phones: list[str], lang: str) -> list[str]:
    return [p.replace("ɚ", "ə") for p in phones] if lang == "en-gb" else phones


def casual_renderings(canonical: list[WordPhones], lang: str = "en-us") -> dict[str, list[list[str]]]:
    """Alternative renderings of the whole sentence from LEXICAL and PAIRS, one per
    variant slot, each a per-word list (canonical where nothing applies). They are
    consumed exactly like a native voice's rendering by reference.acceptances."""
    base = [list(wp.phones) for wp in canonical]
    lower = [wp.word.lower() for wp in canonical]
    out: dict[str, list[list[str]]] = {}
    for i, w in enumerate(lower):
        for k, alt in enumerate(LEXICAL.get(w, [])):
            key = f"casual:{k}"
            out.setdefault(key, [list(p) for p in base])[i] = _british(alt, lang)
    for i in range(len(lower) - 1):
        for k, (a, b) in enumerate(PAIRS.get((lower[i], lower[i + 1]), [])):
            key = f"merge:{k}"
            r = out.setdefault(key, [list(p) for p in base])
            r[i], r[i + 1] = _british(a, lang), _british(b, lang)
    return out


def apply_rules(canonical: list[WordPhones], acc: list[list], lang: str = "en-us") -> None:
    """Context rules across word boundaries, written straight into the acceptances
    (`acc[word][phone].also` / `.optional`, see reference.Acceptance)."""
    n = len(canonical)
    for i, wp in enumerate(canonical):
        ph = wp.phones
        if not ph:
            continue
        nxt = canonical[i + 1].phones[0] if i + 1 < n and canonical[i + 1].phones else None
        prev = canonical[i - 1].phones[-1] if i > 0 and canonical[i - 1].phones else None
        last, first = ph[-1], ph[0]
        a_last, a_first = acc[i][-1], acc[i][0]

        # final t/d after a consonant, before a consonant: dropped ("nex' day", "don' know")
        if last in ("t", "d") and len(ph) > 1 and not is_vowel(ph[-2]) and nxt and not is_vowel(nxt):
            a_last.optional = True
            a_last.source = a_last.source or "dropped t/d"
        # glottal t: word-final t before a consonant or pause, and t before syllabic n
        if last == "t" and (nxt is None or not is_vowel(nxt)):
            a_last.also.add("ʔ")
            a_last.source = a_last.source or "glottal t"
        for j, p in enumerate(ph[:-1]):
            if p == "t" and ph[j + 1] == "n̩":
                acc[i][j].also.add("ʔ")
        # nasal / stop assimilation to the next word's first sound
        if nxt:
            if last == "n" and nxt in ("p", "b", "m"):
                a_last.also.add("m")
            if last == "n" and nxt in ("k", "ɡ"):
                a_last.also.add("ŋ")
            if last == "t" and nxt in ("p", "b", "m"):
                a_last.also.add("p")
            if last == "t" and nxt in ("k", "ɡ"):
                a_last.also.add("k")
            if last == "d" and nxt in ("p", "b", "m"):
                a_last.also.add("b")
            if last == "d" and nxt in ("k", "ɡ"):
                a_last.also.add("ɡ")
            if last in ("n", "t", "d") and a_last.also:
                a_last.source = a_last.source or "assimilation"
        # yod coalescence: t/d/s/z + y -> ch/j/sh/zh ("did you", "this year")
        if nxt == "j" and last in ("t", "d", "s", "z"):
            a_last.also.add({"t": "tʃ", "d": "dʒ", "s": "ʃ", "z": "ʒ"}[last])
            acc[i + 1][0].optional = True
            acc[i + 1][0].source = acc[i + 1][0].source or "linking"
            a_last.source = a_last.source or "linking"
        # th taking the colour of a preceding n / l / z ("in the", "all the", "was the")
        if first == "ð" and prev in ("n", "l", "z"):
            a_first.also.add(prev)
            a_first.source = a_first.source or "linking"
        # -ing said as -in'
        if wp.word.lower().endswith("ing") and last == "ŋ":
            a_last.also.add("n")
            a_last.source = a_last.source or "casual -in'"
        # h dropped in unstressed pronouns/auxiliaries mid-sentence
        if i > 0 and first == "h" and wp.word.lower() in {"he", "him", "his", "her", "have", "has", "had", "here"}:
            a_first.optional = True
            a_first.source = a_first.source or "dropped h"
