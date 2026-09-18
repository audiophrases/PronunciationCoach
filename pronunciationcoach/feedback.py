"""Turn phone scores into words a learner understands.

Nobody in a classroom reads IPA. Every symbol the recogniser can report gets a
plain description ("th as in *this*"), and every flagged phone becomes one
sentence of advice. The IPA is still there for the teacher, one click away.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .scoring import PhoneScore, WordScore

# Learner-facing names for the scoring bands.
BAND_LABEL = {"good": "clear", "unsure": "almost", "off": "work on this"}
BAND_COLOR = {"clear": "#2e8b57", "almost": "#e0a800", "work on this": "#c0392b"}

# IPA -> (spelling, example word). Covers everything espeak produces for English plus the
# sounds Catalan, Spanish and French speakers bring with them.
SOUNDS: dict[str, tuple[str, str]] = {
    # consonants
    "p": ("p", "pen"), "b": ("b", "bad"), "t": ("t", "ten"), "d": ("d", "dog"), "k": ("k", "cat"),
    "ɡ": ("g", "go"), "tʃ": ("ch", "chip"), "dʒ": ("j", "job"), "f": ("f", "fun"), "v": ("v", "very"),
    "θ": ("th", "think"), "ð": ("th", "this"), "s": ("s", "sun"), "z": ("z", "zoo"), "ʃ": ("sh", "ship"),
    "ʒ": ("s", "measure"), "h": ("h", "hat"), "m": ("m", "man"), "n": ("n", "no"), "ŋ": ("ng", "sing"),
    "l": ("l", "leg"), "ɫ": ("l", "feel"), "ɹ": ("r", "red"), "r": ("r", "red"), "j": ("y", "yes"),
    "w": ("w", "we"), "ɾ": ("t", "water (American, quick t)"), "əl": ("le", "little"), "n̩": ("n", "button"),
    # vowels (American)
    "ɪ": ("i", "ship"), "iː": ("ee", "sheep"), "i": ("y", "happy"), "ɛ": ("e", "bed"), "æ": ("a", "cat"),
    "ʌ": ("u", "cup"), "ə": ("a", "about"), "ɐ": ("a", "about"), "ᵻ": ("i", "wanted"),
    "ʊ": ("u", "put"), "uː": ("oo", "food"), "ɑː": ("a", "father"), "ɔː": ("aw", "law"), "ɔ": ("o", "long"),
    "ɒ": ("o", "hot"), "ɜː": ("er", "bird"), "ɝ": ("er", "bird"), "ɚ": ("er", "teacher"),
    "eɪ": ("ay", "day"), "aɪ": ("i", "my"), "ɔɪ": ("oy", "boy"), "oʊ": ("o", "go"), "əʊ": ("o", "go"),
    "aʊ": ("ow", "now"), "ɪɹ": ("ear", "near"), "ɛɹ": ("air", "hair"), "ʊɹ": ("ure", "sure"),
    "ɑːɹ": ("ar", "car"), "ɔːɹ": ("or", "for"), "ɪə": ("ear", "near"), "eə": ("air", "hair"), "ʊə": ("ure", "sure"),
    "aɪɚ": ("ire", "fire"), "aʊɚ": ("our", "hour"), "aɪə": ("ire", "fire"), "aʊə": ("our", "hour"),
}

# Weak (unstressed) vowels deserve a hint that the point is the weakness, not the letter.
WEAK = {
    "ə": "the weak 'uh' sound (as in the first syllable of *about*)",
    "ɐ": "the weak 'uh' sound (as in the first syllable of *about*)",
    "ᵻ": "the weak 'i' sound (as in the last syllable of *wanted*)",
}

# Sounds learners bring from other languages, described by where they come from.
FOREIGN: dict[str, str] = {
    "a": "the Catalan/Spanish 'a'", "e": "the Catalan/Spanish 'e'", "o": "the Catalan/Spanish 'o'",
    "u": "the Catalan/Spanish 'u'", "β": "the soft Spanish 'b'", "x": "the Spanish 'j' (as in *jamón*)",
    "ʁ": "the French 'r'", "y": "the French 'u' (as in *tu*)", "ɲ": "the Catalan 'ny'", "ʎ": "the Catalan 'll'",
    "ɛ̃": "a French nasal vowel", "ɑ̃": "a French nasal vowel", "œ": "the French 'eu' (as in *peur*)",
    "ø": "the French 'eu' (as in *peu*)", "ʲ": "a soft (palatalised) consonant",
}


def describe(phone: str) -> str:
    """'th as in *this*' for ð, 'the French r' for ʁ; the bare symbol if we have no name for it."""
    if phone in WEAK:
        return WEAK[phone]
    if phone in FOREIGN:
        return FOREIGN[phone]
    if phone in SOUNDS:
        spelling, example = SOUNDS[phone]
        return f"'{spelling}' as in *{example}*"
    return f"'{phone}'"


def band(score: PhoneScore | WordScore) -> str:
    return BAND_LABEL[score.category]


def phone_tip(p: PhoneScore, word: str) -> str:
    if p.dropped:
        return f"The {describe(p.expected)} sound in *{word}* was not heard - make sure you say it."
    if p.category == "unsure":
        return f"The {describe(p.expected)} sound in *{word}* was not quite clear (it sounded a bit like {describe(p.heard)})."
    return f"In *{word}*, {describe(p.expected)} came out as {describe(p.heard)}."


@dataclass
class WordFeedback:
    word: str
    band: str
    tips: list[str]

    @property
    def clear(self) -> bool:
        return self.band == "clear"


def word_feedback(w: WordScore) -> WordFeedback:
    tips = [phone_tip(p, w.word) for p in w.phones if p.category != "good"]
    return WordFeedback(w.word, band(w), tips)


# How much a recurring problem matters for being understood. Consonant contrasts that
# separate many word pairs (th, v/b, s/z, ship/sheep) rank above accent colouring such as
# unreduced weak vowels, which sound foreign but rarely cause misunderstanding. This is a
# first, hand-set version of intelligibility weighting; it grows with the teacher's judgement.
PRIORITY: dict[str, float] = {
    "ð": 3, "θ": 3, "v": 3, "b": 2, "ɪ": 2, "iː": 2, "æ": 2, "ʃ": 2, "s": 2, "z": 2, "h": 2, "dʒ": 2, "tʃ": 2,
    "ŋ": 1.5, "ʌ": 1.5, "ɜː": 1.5, "ɝ": 1.5,
    "ə": 0.5, "ɐ": 0.5, "ᵻ": 0.5, "ɚ": 0.5,
}


def summary(words: list[WordScore], max_items: int = 3) -> str:
    """One paragraph a learner can act on: how many words were clear, and the sounds
    that went wrong most often across the whole sentence."""
    feedback = [word_feedback(w) for w in words]
    clear = sum(f.clear for f in feedback)
    n = len(feedback)
    if n == 0:
        return "Nothing to score yet."
    if clear == n:
        head = f"All {n} words were clear. Nice work!"
    else:
        head = f"{clear} of {n} words were clear."

    # Recurring problems, grouped by how the expected sound is described (so ə and ɐ are one
    # line) and ranked by count x importance: "th" is one lesson whether it came out as z in
    # one word and d in the next, and it outranks four unreduced vowels.
    by_sound: dict[str, Counter[str]] = {}
    weight: dict[str, float] = {}
    for w in words:
        for p in w.phones:
            if p.category == "off":
                key = describe(p.expected)
                by_sound.setdefault(key, Counter())["(not heard)" if p.dropped else describe(p.heard)] += 1
                weight[key] = max(weight.get(key, 0.0), PRIORITY.get(p.expected, 1.0))
    if not by_sound:
        return head
    ranked = sorted(by_sound.items(), key=lambda kv: -sum(kv[1].values()) * weight[kv[0]])
    lines = []
    for sound, heards in ranked[:max_items]:
        total = sum(heards.values())
        times = f" ({total} times)" if total > 1 else ""
        missing = heards.pop("(not heard)", 0)
        parts = [h for h, _ in heards.most_common(2)]
        if parts and missing:
            lines.append(f"• {sound} came out as {' or '.join(parts)}, or was missing{times}")
        elif parts:
            lines.append(f"• {sound} came out as {' or '.join(parts)}{times}")
        else:
            lines.append(f"• {sound} was missing{times}")
    return head + "\n\nSounds to practise:\n" + "\n".join(lines)
