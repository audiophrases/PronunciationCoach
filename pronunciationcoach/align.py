"""Two ways to read the emission grid.

* `greedy_decode` – what the recogniser thinks it heard, with no reference text.
* `forced_align`  – where each *expected* phone sits, given a reference text.

Both return `Segment`s in frame units; `Emissions.frame_to_s` converts to seconds.
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
