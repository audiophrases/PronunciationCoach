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


def equivalents(phone: str) -> set[str]:
    for group in EQUIVALENT:
        if phone in group:
            return group
    return {phone}


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

    @property
    def category(self) -> str:
        return category(self.gop)


@dataclass
class WordScore:
    word: str
    phones: list[PhoneScore]

    @property
    def gop_mean(self) -> float:
        return float(np.mean([p.gop for p in self.phones])) if self.phones else 0.0

    @property
    def gop_min(self) -> float:
        return float(min(p.gop for p in self.phones)) if self.phones else 0.0

    @property
    def category(self) -> str:
        return category(self.gop_min)


def score_segment(em: Emissions, seg: Segment, top_k: int = 3) -> PhoneScore:
    block = em.log_probs[seg.start : seg.end]  # (n, C)
    group = equivalents(seg.phone)
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


def score_words(em: Emissions, segments: list[Segment], words: list[tuple[str, int]]) -> list[WordScore]:
    """`words` is [(word, phone_count), ...] in order; segments are flat in the same order."""
    out: list[WordScore] = []
    cursor = 0
    for word, n in words:
        chunk = segments[cursor : cursor + n]
        out.append(WordScore(word, [score_segment(em, s) for s in chunk]))
        cursor += n
    return out
