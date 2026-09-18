"""Two ways to read the emission grid.

* `greedy_decode` – what the recogniser thinks it heard, with no reference text.
* `forced_align`  – where each *expected* phone sits, given a reference text.
* `align_words`   – forced alignment with a wildcard between words, so hesitations,
                    repeats and false starts are absorbed instead of dragging the
                    expected phones onto them. This is what the pipeline uses.

All return `Segment`s in frame units; `Emissions.frame_to_s` converts to seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as AF

from .engine import Emissions


@dataclass
class Segment:
    phone: str
    phone_id: int
    start: int  # frame, inclusive
    end: int  # frame, exclusive
    score: float  # mean log-prob of `phone` over [start, end)


@dataclass
class ExtraRun:
    """Consecutive phones the recogniser heard where the sentence has no word."""

    phones: list[str]
    start: int
    end: int


@dataclass
class Alignment:
    segments: list[Segment]  # one per expected phone, in order
    extra: list[ExtraRun]  # speech that matched no word of the sentence


# Log-probability granted to the wildcard on every frame. 0 (probability 1) means "between
# words, anything goes for free": each word is then matched wherever its phones fit best, in
# order, and everything else - silence, hesitations, a repeated word - goes to the wildcard.
# Measured on real recordings: clean sentences are unaffected, repeats are absorbed.
WILDCARD_LOG_PROB = 0.0


def greedy_decode(em: Emissions) -> list[Segment]:
    """Standard CTC collapse: drop blanks, merge repeats not separated by a blank."""
    best = em.log_probs.argmax(axis=-1)
    segments: list[Segment] = []
    prev = em.blank_id
    for t, c in enumerate(best.tolist()):
        if c == em.blank_id:
            prev = c
            continue
        if c == prev:
            segments[-1].end = t + 1
        else:
            segments.append(Segment(em.labels[c], c, t, t + 1, 0.0))
        prev = c
    for seg in segments:
        seg.score = float(em.log_probs[seg.start : seg.end, seg.phone_id].mean())
    return segments


def forced_align(em: Emissions, target_ids: list[int]) -> list[Segment]:
    """Viterbi-align the expected phone sequence to the grid; one Segment per target.

    The span of a segment is the run of frames the aligner labelled with that
    phone (CTC is peaky, so typically 1-3 frames). When the learner did not
    produce the phone at all, the aligner still has to put it somewhere and
    will pick the least-bad frame – the scorer sees that as a very low score.
    """
    if not target_ids:
        return []
    needed = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
    if em.n_frames < needed:
        raise ValueError(
            f"Audio too short: {em.n_frames} frames for {len(target_ids)} expected phones. "
            "Is the recording really this sentence?"
        )
    log_probs = torch.from_numpy(np.ascontiguousarray(em.log_probs, dtype=np.float32))[None]
    targets = torch.tensor([target_ids], dtype=torch.int32)
    labels, scores = AF.forced_align(log_probs, targets, blank=em.blank_id)
    spans = AF.merge_tokens(labels[0], scores[0], blank=em.blank_id)
    return [Segment(em.labels[s.token], s.token, s.start, s.end, float(s.score)) for s in spans]


def align_words(em: Emissions, words: list[list[int]]) -> Alignment:
    """Forced alignment of word phone sequences with a wildcard allowed between words."""
    if not words:
        return Alignment([], [])
    star = em.log_probs.shape[1]  # one extra column, used only by the aligner
    targets: list[int] = [star]
    for ids in words:
        targets += list(ids) + [star]
    needed = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
    if em.n_frames < needed:
        raise ValueError(
            f"Audio too short: {em.n_frames} frames for {sum(map(len, words))} expected phones. "
            "Is the recording really this sentence?"
        )
    padded = np.concatenate(
        [
            np.ascontiguousarray(em.log_probs, dtype=np.float32),
            np.full((em.n_frames, 1), WILDCARD_LOG_PROB, dtype=np.float32),
        ],
        axis=1,
    )
    labels, scores = AF.forced_align(
        torch.from_numpy(padded)[None], torch.tensor([targets], dtype=torch.int32), blank=em.blank_id
    )
    spans = AF.merge_tokens(labels[0], scores[0], blank=em.blank_id)

    segments = [Segment(em.labels[s.token], s.token, s.start, s.end, float(s.score)) for s in spans if s.token != star]
    gaps = [(s.start, s.end) for s in spans if s.token == star]
    extra: list[ExtraRun] = []
    for heard in greedy_decode(em):
        if any(a <= heard.start and heard.end <= b for a, b in gaps):
            if extra and heard.start - extra[-1].end <= 15:  # same run if within 300 ms
                extra[-1].phones.append(heard.phone)
                extra[-1].end = heard.end
            else:
                extra.append(ExtraRun([heard.phone], heard.start, heard.end))
    return Alignment(segments, extra)
