"""Conservative two-word playback groups; pronunciation scores remain per word.

Function-word membership suggests a candidate, not a measurement of stress.
Pauses, punctuation, extra speech and likely emphasis take precedence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE
from .boundaries import HOP, Span, energy_db
from .g2p import _WORD_RE
from .variants import PAIRS

SHORT_S = 0.12
STRONG_S = 0.30
PAUSE_S = 0.10
MAX_PAIR_S = 1.5
EPS = 1e-8
ARTICLES = set("a an the".split())
PREPOSITIONS = set("to from in on at for of with by as than into onto over under about before after".split())
SUBJECTS = set("i you he she it we they".split())
OBJECTS = set("me you him her it us them".split())
POSSESSIVES = set("my your his her its our their".split())
CONJUNCTIONS = set("and but or because if that".split())
AUXILIARIES = set("am is are was were be been being can could will would shall should may might must".split())
AMBIGUOUS_VERBS = set("do does did have has had".split())
CONTRACTIONS = set("i'm you're he's she's it's we're they're i've you've we've they've i'll you'll he'll she'll it'll we'll they'll i'd you'd he'd she'd we'd they'd".split())
FUNCTION_WORDS = ARTICLES | PREPOSITIONS | SUBJECTS | OBJECTS | POSSESSIVES | CONJUNCTIONS | AUXILIARIES | CONTRACTIONS
PUNCTUATION = re.compile(r"[.,;:!?\u2014\u2013\-\n]")


@dataclass
class PlaybackChunk:
    members: list[int]
    text: str
    span: Span | None
    reason: str
    blocked_before: str = ""
    blocked_after: str = ""


def safe_spans(spans: list[Span], duration: float, dropped: list[bool]) -> list[Span]:
    """Bound live crops in reading order; missing words contribute no timing evidence.

    This is a logged safety fallback, not an acoustic correction. Empty spans are
    allowed: inventing a minimum duration can borrow a neighbor's speech.
    """
    if len(spans) != len(dropped) or not np.isfinite(duration) or duration < 0:
        raise ValueError("Invalid crop lengths or audio duration")
    out, cursor = [], 0.0
    for i, (span, missing) in enumerate(zip(spans, dropped)):
        if missing:
            out.append(Span(cursor, cursor))
            continue
        if not np.isfinite([span.start, span.end]).all():
            start = end = cursor
        else:
            start = min(duration, max(cursor, span.start))
            end = min(duration, max(start, span.end))
        if abs(start - span.start) > EPS or abs(end - span.end) > EPS or not np.isfinite([span.start, span.end]).all():
            logging.getLogger("pronunciationcoach").warning("invalid/overlapping crop %d repaired: %s -> %.3f-%.3f", i, span, start, end)
        out.append(Span(start, end))
        cursor = end
    return out


def build_chunks(text: str, words: list[str], spans: list[Span], audio: np.ndarray,
                 dropped: list[bool] | None = None, extras: list[tuple[float, float]] | None = None,
                 enabled: bool = True) -> list[PlaybackChunk]:
    """Return disjoint singletons/pairs, with a playable union of live members.

    Prefer grammatical pairs (can you, hear me, the store). Maximum-weight
    adjacent matching keeps a word in exactly one pair, without chaining pairs
    into longer phrases. Ties prefer the earlier pair.
    """
    n = len(words)
    dropped = [False] * n if dropped is None else dropped
    if len(spans) != n or len(dropped) != n:
        raise ValueError("Words, spans and dropped flags must have equal lengths")
    if not n:
        return []
    spans = safe_spans(spans, len(audio) / SAMPLE_RATE, dropped)
    live = [not missing and sp.duration > EPS for sp, missing in zip(spans, dropped)]
    lower = [w.lower() for w in words]
    matches = list(_WORD_RE.finditer(text))
    matched = [m.group().lower() for m in matches] == lower
    db = energy_db(audio)
    step = HOP / SAMPLE_RATE
    silent = float(np.percentile(db, 10)) + 6.0

    def levels(a, b):
        lo = min(len(db) - 1, max(0, int(a / step)))
        hi = min(len(db), max(lo + 1, int(np.ceil(b / step))))
        return db[lo:hi]

    energy = [float(np.percentile(levels(sp.start, sp.end), 75)) for sp in spans]
    strong = []
    for i, sp in enumerate(spans):
        neighbors = [energy[j] for j in (i - 1, i + 1) if 0 <= j < n and live[j]]
        strong.append(live[i] and sp.duration >= STRONG_S - EPS and
                      bool(neighbors) and energy[i] >= max(neighbors) - 3.0)

    def auxiliary(i):
        return lower[i] in AUXILIARIES or (lower[i] in AMBIGUOUS_VERBS and i + 1 < n and
                                           lower[i + 1] in SUBJECTS | {"to", "not", "been", "got"})

    weak = [(w in FUNCTION_WORDS or auxiliary(i)) and not strong[i] for i, w in enumerate(lower)]
    barriers = []
    for i in range(n - 1):
        separator = text[matches[i].end():matches[i + 1].start()] if matched else ""
        if not matched or PUNCTUATION.search(separator) or re.search(r"[\d]", separator):
            barriers.append("punctuation" if matched else "text mismatch")
            continue
        # Project the nearest live boundary across missing words, never their parked spans.
        left = next((j for j in range(i, -1, -1) if live[j]), None)
        right = next((j for j in range(i + 1, n) if live[j]), None)
        if left is None or right is None:
            barriers.append("")
            continue
        a, b = spans[left].end, spans[right].start
        run = longest = 0
        for quiet in levels(a - .04, b + .04) < silent:
            run = run + 1 if quiet else 0
            longest = max(longest, run)
        if any(x < b + .02 and y > a - .02 for x, y in extras or []):
            barriers.append("extra speech")
        elif b - a >= PAUSE_S - EPS or longest * step >= PAUSE_S - EPS:
            barriers.append("pause")
        else:
            barriers.append("")

    candidates: list[tuple[int, str]] = []
    for i in range(n - 1):
        j = i + 1
        members = [spans[k] for k in (i, j) if live[k]]
        if not enabled or barriers[i] or (members and members[-1].end - members[0].start > MAX_PAIR_S + EPS):
            candidates.append((0, ""))
            continue
        a, b = lower[i], lower[j]
        choices = [(0, "")]
        # Emphasized grammar words stand alone; content-word hosts may be stressed.
        if (a in FUNCTION_WORDS | AMBIGUOUS_VERBS and strong[i]) or (b in FUNCTION_WORDS | AMBIGUOUS_VERBS and strong[j]):
            candidates.append((0, ""))
            continue
        if weak[i] and auxiliary(i) and b in SUBJECTS:
            choices.append((110, "auxiliary + pronoun"))
        if weak[i] and a in ARTICLES | POSSESSIVES and not weak[j]:
            choices.append((105, "determiner + word"))
        if weak[j] and b in OBJECTS and not weak[i]:
            choices.append((100, "word + pronoun"))
        if weak[i] and a in PREPOSITIONS and not weak[j]:
            choices.append((90, "preposition + word"))
        if (a, b) in PAIRS and (weak[i] or weak[j]):
            choices.append((85, "connected pair"))
        if weak[i] and not weak[j]:
            choices.append((80, "function word + word"))
        if weak[i] and weak[j]:
            choices.append((65, "function words"))
        if weak[j]:
            choices.append((40, "word + function word"))
        if not live[i] or not live[j]:
            choices.append((30, "missing word context"))
        elif min(spans[i].duration, spans[j].duration) < SHORT_S - EPS:
            choices.append((20, "short crop context"))
        candidates.append(max(choices, key=lambda choice: choice[0]))

    best, pair = [0] * (n + 1), [False] * n
    for i in range(n - 1, -1, -1):
        best[i] = best[i + 1]
        if i + 1 < n and candidates[i][0] and candidates[i][0] + best[i + 2] >= best[i]:
            best[i] = candidates[i][0] + best[i + 2]
            pair[i] = True
    result, i = [], 0
    while i < n:
        ids = [i, i + 1] if pair[i] else [i]
        live_spans = [spans[k] for k in ids if live[k]]
        span = Span(live_spans[0].start, live_spans[-1].end) if live_spans else None
        label = " ".join(words[k] for k in ids)
        if matched:
            last = ids[-1]
            tail = text[matches[last].end():matches[last + 1].start() if last + 1 < n else len(text)]
            terminal = re.match(r"\s*([?!])", tail)
            if terminal:
                label += terminal[1]
        result.append(PlaybackChunk(ids, label, span, candidates[i][1] if pair[i] else "single word",
                                    barriers[i - 1] if i else "",
                                    barriers[ids[-1]] if ids[-1] < n - 1 else ""))
        i += len(ids)
    return result
