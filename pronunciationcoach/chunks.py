"""Playback groups of one to three words spoken as one flow; scores remain per word.

Each gap between words is scored for how joined it sounds: grammar words leaning
on a neighbour, a linking sound (kind‿of), and no break in the audio. Pauses,
punctuation, extra speech and likely emphasis always separate words.
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
MAX_CHUNK_S = 1.5
EPS = 1e-8
ARTICLES = set("a an the".split())
PREPOSITIONS = set("to from in on at for of with by as than into onto over under about before after".split())
SUBJECTS = set("i you he she it we they".split())
OBJECTS = set("me you him her it us them".split())
POSSESSIVES = set("my your his her its our their".split())
CONJUNCTIONS = set("and but or because if that".split())
# Negatives are stressed as a rule, so their length and loudness say nothing about emphasis.
NEGATIVES = set("can't couldn't won't wouldn't shouldn't don't doesn't didn't isn't aren't wasn't weren't "
                "haven't hasn't hadn't".split())
AUXILIARIES = set("am is are was were be been being can could will would shall should may might must".split()) | NEGATIVES
AMBIGUOUS_VERBS = set("do does did have has had".split())
CONTRACTIONS = set("i'm you're he's she's it's we're they're i've you've we've they've i'll you'll he'll she'll it'll we'll they'll i'd you'd he'd she'd we'd they'd".split())
FUNCTION_WORDS = ARTICLES | PREPOSITIONS | SUBJECTS | OBJECTS | POSSESSIVES | CONJUNCTIONS | AUXILIARIES | CONTRACTIONS
QUESTION_WORDS = set("what where when why how who".split())
# Grammar words that attach to the word after them rather than the one before.
FORWARD_LEANERS = ARTICLES | PREPOSITIONS | SUBJECTS | POSSESSIVES | CONJUNCTIONS | CONTRACTIONS
VOWEL_LETTERS = set("aeiouɑæʌɔəɛɪʊɜɝɚɒɐ")

# Join scores for the gap between two words (build_chunks.join_score). A gap needs
# JOIN_MIN to sit inside a group: grammar alone (the‿store) or linking alone
# (pick‿up) is enough; two content words run together without linking are not.
JOIN_MIN = 0.75
JOIN_WEAK_AFTER = 0.5  # an unstressed grammar word after a word, but leaning forward
EDGE_S = 0.06  # how much of each word's edge sets the reference level
VALLEY_S = 0.03  # how far either side of the gap to look for a break
JOIN_DIP_DB = 10.0  # stop closures dip this much even in connected speech; more is a break


def vowel(phone: str) -> bool:
    return any(ch in VOWEL_LETTERS for ch in phone)
PUNCTUATION = re.compile(r"[.,;:!?\u2014\u2013\-\n]")


@dataclass
class PlaybackChunk:
    members: list[int]
    text: str
    span: Span | None
    reason: str
    blocked_before: str = ""
    blocked_after: str = ""
    pad_before: bool = True
    pad_after: bool = True


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
                 enabled: bool = True, sounds: list[tuple[str, str]] | None = None) -> list[PlaybackChunk]:
    """Return disjoint groups of one to three words that were spoken as one flow,
    with a playable union of live members.

    `sounds` gives each word's first and last pronounced phone, the evidence of
    linking (kind‿of); without it only grammar and the audio decide.
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
        # Inspect enough audio to detect a 100 ms pause even when the aligner
        # allocated its two halves to touching word crops.
        for quiet in levels(a - PAUSE_S, b + PAUSE_S) < silent:
            run = run + 1 if quiet else 0
            longest = max(longest, run)
        if any(x < b + .02 and y > a - .02 for x, y in extras or []):
            barriers.append("extra speech")
        elif b - a >= PAUSE_S - EPS or longest * step >= PAUSE_S - EPS:
            barriers.append("pause")
        else:
            barriers.append("")

    energy = [float(np.percentile(levels(sp.start, sp.end), 75)) for sp in spans]
    strong = []
    for i, sp in enumerate(spans):
        neighbors = [energy[j] for j in (i - 1, i + 1) if 0 <= j < n and live[j]]
        # A word before a pause is lengthened whether or not it is stressed ("kind of...
        # gets": a 370 ms "of"), so its duration is no evidence of emphasis there.
        phrase_final = i < n - 1 and barriers[i] == "pause"
        strong.append(live[i] and not phrase_final and lower[i] not in NEGATIVES and sp.duration >= STRONG_S - EPS and
                      bool(neighbors) and energy[i] >= max(neighbors) - 3.0)

    def auxiliary(i):
        return lower[i] in AUXILIARIES or (lower[i] in AMBIGUOUS_VERBS and i + 1 < n and
                                           lower[i + 1] in SUBJECTS | {"to", "not", "been", "got"})

    function = [w in FUNCTION_WORDS or auxiliary(i) for i, w in enumerate(lower)]
    weak = [candidate and not strong[i] for i, candidate in enumerate(function)]
    # How joined each gap between neighbours sounds, as a score (see JOIN_* above).
    # Grammar says which words lean on a neighbour; the phones say whether the words
    # were linked; the audio says whether the flow was broken. A gap scoring at
    # least JOIN_MIN can sit inside a group, and groups keep the most joined gaps.
    def leans_forward(i):
        return weak[i] and (lower[i] in FORWARD_LEANERS or auxiliary(i))

    def join_score(i) -> tuple[float, list[str]]:
        j = i + 1
        a, b = lower[i], lower[j]
        score, why = 0.0, []
        if (a, b) in PAIRS and (weak[i] or weak[j]):
            score, why = score + 2.0, why + ["connected pair"]
        elif weak[i] and auxiliary(i) and b in SUBJECTS:
            score, why = score + 2.0, why + ["auxiliary + pronoun"]
        elif weak[i] and a in SUBJECTS and i > 0 and auxiliary(i - 1):
            # "do you like", "could it be": the pronoun belongs to its auxiliary first
            score, why = score + 1.0, why + [f"{a!r} + verb"]
        elif a in QUESTION_WORDS and auxiliary(j) and weak[j]:
            score, why = score + 1.5, why + ["question opening"]
        elif leans_forward(i) and not weak[j]:
            score, why = score + 1.5, why + [f"{a!r} leans on the next word"]
        elif weak[j] and b in OBJECTS and not weak[i]:
            score, why = score + 1.5, why + [f"{b!r} leans on the word before"]
        elif leans_forward(i):
            score, why = score + 1.0, why + ["grammar words"]
        elif weak[j]:
            score, why = score + JOIN_WEAK_AFTER, why + [f"unstressed {b!r}"]
        if sounds is not None and sounds[i][1] and sounds[j][0]:
            last, first = sounds[i][1], sounds[j][0]
            if not vowel(last) and vowel(first):
                score, why = score + 1.0, why + [f"linked {last}‿{first}"]
            elif last == first:
                score, why = score + 1.0, why + [f"shared {last}"]
            elif vowel(last) and vowel(first):
                score, why = score + 0.5, why + [f"glide {last}‿{first}"]
        if live[i] and live[j]:
            end, start = spans[i].end, spans[j].start
            edge = min(float(np.percentile(levels(end - EDGE_S, end), 75)),
                       float(np.percentile(levels(start, start + EDGE_S), 75)))
            dip = edge - float(np.min(levels(end - VALLEY_S, start + VALLEY_S)))
            if dip > JOIN_DIP_DB:
                score, why = score - min(1.5, (dip - JOIN_DIP_DB) / JOIN_DIP_DB), why + [f"{dip:.0f} dB break"]
        return score, why

    joins: list[tuple[float, list[str]]] = []
    for i in range(n - 1):
        pair = [spans[k] for k in (i, i + 1) if live[k]]
        if not enabled or barriers[i]:
            joins.append((-1.0, [barriers[i] or "grouping off"]))
        elif (function[i] and strong[i]) or (function[i + 1] and strong[i + 1]):
            joins.append((-1.0, ["emphasized grammar word"]))  # emphasis stands alone
        elif pair and pair[-1].end - pair[0].start > MAX_CHUNK_S + EPS:
            joins.append((-1.0, ["too long"]))
        else:
            joins.append(join_score(i))

    # Split into groups of one to three words, keeping the gaps with the most margin
    # over JOIN_MIN inside groups. Ties favour shorter groups.
    best = [0.0] * (n + 1)
    lengths, reasons = [1] * n, ["single word"] * n
    for i in range(n - 1, -1, -1):
        best[i] = best[i + 1]
        for length in (2, 3):
            ids = range(i, i + length)
            if i + length > n or any(joins[k][0] < JOIN_MIN for k in ids[:-1]):
                break
            live_ids = [k for k in ids if live[k]]
            if length == 3 and len(live_ids) < 3:
                break  # a missing word may borrow one neighbour's context, not build a phrase
            if live_ids and spans[live_ids[-1]].end - spans[live_ids[0]].start > MAX_CHUNK_S + EPS:
                break
            total = sum(joins[k][0] - JOIN_MIN for k in ids[:-1]) + best[i + length]
            if total > best[i] + EPS:
                best[i] = total
                lengths[i] = length
                reasons[i] = "; ".join(r for k in ids[:-1] for r in joins[k][1]) or "smooth join"
    result, i = [], 0
    while i < n:
        ids = list(range(i, i + lengths[i]))
        live_spans = [spans[k] for k in ids if live[k]]
        span = Span(live_spans[0].start, live_spans[-1].end) if live_spans else None
        label = " ".join(words[k] for k in ids)
        if matched:
            last = ids[-1]
            tail = text[matches[last].end():matches[last + 1].start() if last + 1 < n else len(text)]
            terminal = re.match(r"\s*([?!])", tail)
            if terminal:
                label += terminal[1]
        result.append(PlaybackChunk(ids, label, span, reasons[i],
                                    barriers[i - 1] if i else "",
                                    barriers[ids[-1]] if ids[-1] < n - 1 else ""))
        i += len(ids)
    return result
