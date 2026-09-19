"""Goodness of Pronunciation (GOP) read straight off the emission grid.

Classic GOP (Witt & Young, 2000): for the frames where the expected phone was
aligned, compare the log-probability of that phone against the best competing
phone. 0 means the recogniser agreed completely; the more negative, the more it
heard something else – and *which* something else is the substitution we report.

The CTC blank is excluded from the competition: it carries "nothing new here"
mass and would otherwise win almost every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import Segment
from .engine import Emissions

# Placeholder bands until we calibrate against real learner recordings.
GOP_GOOD = -0.5
GOP_UNSURE = -2.0

# Symbols the reference may write differently from what the recogniser reports, without
# there being any pronunciation difference a teacher would flag. Each group is scored as
# one sound: the expected phone gets the probability mass of the whole group and none of
# the group counts as a competitor. Curated from real recordings; grows with the teacher.
EQUIVALENT: list[set[str]] = [
    {"ə", "ɐ", "ᵻ"},  # reduced central vowels: espeak's "a" (ɐ), "-ed" (ᵻ) and plain schwa
    {"i", "iː"},  # happY vowel vs FLEECE: espeak marks length, speech in unstressed positions doesn't
]


# One-way tolerance: an expected flap may be realised as a clear t or d (nobody marks that),
# and an expected t or d may come out flapped between vowels - but t must never accept d.
ALLOPHONES: dict[str, set[str]] = {"ɾ": {"t", "d"}, "t": {"ɾ"}, "d": {"ɾ"}}


def equivalents(phone: str) -> set[str]:
    for group in EQUIVALENT:
        if phone in group:
            return group
    return {phone}


def accepted(phone: str) -> set[str]:
    """Every symbol that counts as a correct realisation of the expected phone."""
    return equivalents(phone) | ALLOPHONES.get(phone, set()) | {phone}


# A phone whose aligned frames sit this far after the previous phone of the same word was
# not produced where it belongs; the aligner parked it on the nearest similar sound.
DROPPED_GAP_S = 0.25


def category(gop: float) -> str:
    if gop >= GOP_GOOD:
        return "good"
    if gop >= GOP_UNSURE:
        return "unsure"
    return "off"


@dataclass
class PhoneScore:
    expected: str
    start_s: float
    end_s: float
    gop: float
    posterior: float  # share of non-blank probability mass on the expected phone
    heard: str  # strongest non-blank competitor (may equal `expected`)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # top-k (phone, prob)
    dropped: bool = False  # the phone was not produced at all (see mark_dropped)
    natural: str = ""  # non-empty when a native rendering does the same thing (see reference.py)

    @property
    def heard_label(self) -> str:
        return "(not heard)" if self.dropped else self.heard

    @property
    def category(self) -> str:
        if self.natural:
            return "good"  # natives do it too
        return "off" if self.dropped else category(self.gop)  # a sound not said at all is always an error


# How much a deviation in a sound matters for being understood. Consonant contrasts that
# separate many word pairs (th, v/b, s/z, ship/sheep) outrank accent colouring such as
# unreduced weak vowels, which sound foreign but rarely cause misunderstanding. This is a
# first, hand-set version of intelligibility weighting; it grows with the teacher's judgement.
PRIORITY: dict[str, float] = {
    "ð": 3, "θ": 3, "v": 3, "b": 2, "ɪ": 2, "iː": 2, "æ": 2, "ʃ": 2, "s": 2, "z": 2, "h": 2, "dʒ": 2, "tʃ": 2,
    "ŋ": 1.5, "ʌ": 1.5, "ɜː": 1.5, "ɝ": 1.5,
    "ə": 0.5, "ɐ": 0.5, "ᵻ": 0.5, "ɚ": 0.5,
}

VERDICTS = ("clear", "accent", "almost", "work on this")


@dataclass
class WordScore:
    word: str
    phones: list[PhoneScore]
    listener_p: float | None = None  # the listener's confidence in this word, None if no listener ran
    understood: bool | None = None  # the listener produced this word at all

    @property
    def gop_mean(self) -> float:
        return float(np.mean([p.gop for p in self.phones])) if self.phones else 0.0

    @property
    def gop_min(self) -> float:
        return float(min(p.gop for p in self.phones)) if self.phones else 0.0

    @property
    def severity(self) -> float:
        """Worst deviation, weighted by how much that sound matters (0 = nothing flagged)."""
        flagged = [p for p in self.phones if p.category != "good"]
        if not flagged:
            return 0.0
        return max(PRIORITY.get(p.expected, 1.0) * (1.0 if p.category == "off" else 0.5) for p in flagged)

    @property
    def verdict(self) -> str:
        """What the learner should take away about this word.

        clear        - nothing to note
        accent       - understood without effort; the deviations are colouring, not errors
        almost       - understood, but with effort or a sound that matters
        work on this - missed by the listener, or a top-priority contrast (th, v) broken
        """
        sev = self.severity
        p = 0.7 if self.listener_p is None else self.listener_p  # no listener: neither confident nor lost
        understood = True if self.understood is None else self.understood
        if not understood or p < 0.3:
            return "work on this"
        if sev == 0.0:
            return "clear"
        if sev >= 3.0 or (sev >= 2.0 and p < 0.8):
            return "work on this"
        if sev >= 1.5 or p < 0.6 or (sev >= 1.0 and p < 0.8):
            return "almost"
        return "accent"

    @property
    def category(self) -> str:
        """Phone-style band of the word, kept for the technical views."""
        return {"clear": "good", "accent": "good", "almost": "unsure", "work on this": "off"}[self.verdict]


def score_segment(em: Emissions, seg: Segment, top_k: int = 3, also: set[str] | None = None) -> PhoneScore:
    block = em.log_probs[seg.start : seg.end]  # (n, C)
    group = accepted(seg.phone) | (also or set())
    group_ids = [i for i, label in enumerate(em.labels) if label in group] or [seg.phone_id]

    competitors = block.copy()
    competitors[:, em.blank_id] = -np.inf
    probs = np.exp(competitors)  # blank is now exactly 0
    mass = probs.sum(axis=-1, keepdims=True)
    mass[mass == 0] = 1.0
    mean_nb = (probs / mass).mean(axis=0)  # distribution over non-blank phones
    order = np.argsort(mean_nb)[::-1][:top_k]

    # log P(any symbol of the group) per frame, then averaged; the group is not its own competitor.
    lp_expected = float(np.logaddexp.reduce(block[:, group_ids], axis=-1).mean())
    competitors[:, group_ids] = -np.inf
    lp_best = float(competitors.max(axis=-1).mean())
    top = int(order[0])

    return PhoneScore(
        expected=seg.phone,
        start_s=em.frame_to_s(seg.start),
        end_s=em.frame_to_s(seg.end),
        gop=min(0.0, lp_expected - lp_best),
        posterior=float(mean_nb[group_ids].sum()),
        heard=seg.phone if top in group_ids else em.labels[top],
        candidates=[(em.labels[int(i)], float(mean_nb[i])) for i in order],
    )


def score_words(
    em: Emissions,
    segments: list[Segment],
    words: list[tuple[str, int]],
    acceptances: list[list] | None = None,
) -> list[WordScore]:
    """`words` is [(word, phone_count), ...] in order; segments are flat in the same order.
    `acceptances` (reference.Acceptance per phone, per word) widens what counts as correct."""
    out: list[WordScore] = []
    cursor = 0
    for wi, (word, n) in enumerate(words):
        chunk = segments[cursor : cursor + n]
        accs = acceptances[wi] if acceptances else [None] * n
        scores = [score_segment(em, s, also=(a.also if a else None)) for s, a in zip(chunk, accs)]
        mark_dropped(scores, chunk, em)
        for score, a in zip(scores, accs):
            if a is None or score.category == "good":
                continue
            if a.optional:
                score.natural = f"drop|{a.source}"
            elif score.heard in a.also:
                score.natural = f"also|{a.source}"
        out.append(WordScore(word, scores))
        cursor += n
    return out


def mark_dropped(scores: list[PhoneScore], segments: list[Segment], em: Emissions) -> None:
    """Flag expected phones that were not produced, so feedback says "not heard" rather than
    reporting whatever sound the aligner had to park them on.

    Two signatures, both only within a word (pauses between words are legitimate):
    * the phone sits far after the previous phone - it was found somewhere later instead;
    * its frames really belong to a neighbour: the competitor that won is the previous or
      next expected phone of the same word, and the expected phone had next to no mass.
    """
    for i, (score, seg) in enumerate(zip(scores, segments)):
        if score.category == "good":
            continue
        if i > 0 and em.frame_to_s(seg.start - segments[i - 1].end) > DROPPED_GAP_S:
            score.dropped = True
            continue
        neighbours = {scores[j].expected for j in (i - 1, i + 1) if 0 <= j < len(scores)}
        if score.heard in neighbours and score.posterior < 0.1:
            score.dropped = True
