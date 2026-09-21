"""Targeted playback-edge checks using the existing, full-context Charsiu output.

This is another alignment search, not another recognizer or an independent
correctness verdict. Only small outward extensions into unclaimed audio are
eligible; ambiguous joins and suspected repeats retain the original crop.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import numpy as np

from .boundaries import SPIKE_LAG_S, Span
from .chunks import MAX_CHUNK_S, PlaybackChunk, SHORT_S
from .segmenter import FRAME_S, OFFSET_S, CharsiuSegmenter

MIN_CHANGE_S = .02
MAX_CHANGE_S = .08
AGREEMENT_S = .011  # one Charsiu frame
MAX_REGIONS = 8
MAX_CONTEXT_S = 8.0


@dataclass
class CropEvidence:
    segmenter: CharsiuSegmenter
    log_probs: np.ndarray
    phones: list[list[str]]
    windows: list[tuple[float, float]]
    raw_spans: list[Span]
    spikes: list[Span]
    extras: list[tuple[float, float]]


@dataclass
class CropCheck:
    chunk: int
    text: str
    before: Span
    proposed: Span | None
    after: Span
    flags: list[str]
    status: str
    detail: str
    accepted: Span | None = None

    def summary(self) -> str:
        change = (f"; {self.before.start:.2f}-{self.before.end:.2f} -> "
                  f"{self.accepted.start:.2f}-{self.accepted.end:.2f}s") if self.accepted else ""
        return f"[{self.text}] {self.status}: {', '.join(self.flags)}; {self.detail}{change}"


def _local_alignment(e: CropEvidence, first: int, last: int, neighbors: int) -> list[Span]:
    """Retain neighboring words and full-recording neural context; relax search windows.

    Slice already-computed posteriors, not raw audio. The frame origin must be
    translated in both directions; Charsiu's latency correction is applied once.
    """
    a, b = max(0, first - neighbors), min(len(e.phones), last + neighbors + 1)
    if any(not phones or any(p not in e.segmenter.vocab for p in phones) for phones in e.phones[a:b]):
        raise ValueError("unsupported phones in surrounding words")
    lo = max(0., min(w[0] for w in e.windows[a:b]) - .12)
    hi = max(w[1] for w in e.windows[a:b]) + .12
    if hi - lo > MAX_CONTEXT_S:
        raise ValueError("surrounding speech exceeds recheck limit")
    f0 = max(0, int(np.floor(lo / FRAME_S)))
    f1 = min(len(e.log_probs), int(np.ceil((hi + OFFSET_S) / FRAME_S)))
    origin = f0 * FRAME_S
    windows = [(l - origin - MAX_CHANGE_S, h - origin + MAX_CHANGE_S) for l, h in e.windows[a:b]]
    extras = [(l - origin, h - origin) for l, h in e.extras]
    spans = e.segmenter.align(e.log_probs[f0:f1], e.phones[a:b], windows, extras)
    if len(spans) != b - a or any(s.end <= s.start for s in spans):
        raise ValueError("no complete contextual alignment")
    return [Span(s.start + origin, s.end + origin) for s in spans[first-a:last-a+1]]


def _support(e: CropEvidence, word: int, edge: str, lo: float, hi: float) -> bool:
    """Require the added frames to favor the edge phone over silence and its neighbor.

    Equal adjacent phones cannot identify ownership, so they are ambiguous.
    This protects quiet onsets: the check never trims existing speech.
    """
    phone = e.phones[word][0 if edge == "start" else -1]
    neighbor = word - 1 if edge == "start" else word + 1
    competitors = [e.segmenter.sil]
    if 0 <= neighbor < len(e.phones) and e.phones[neighbor]:
        other = e.phones[neighbor][-1 if edge == "start" else 0]
        if other == phone:
            return False
        if other in e.segmenter.vocab:
            competitors.append(e.segmenter.vocab[other])
    times = (np.arange(len(e.log_probs)) + .5) * FRAME_S - OFFSET_S
    block = e.log_probs[(times >= lo) & (times < hi)]
    if len(block) < 2:
        return False
    target = e.segmenter.vocab[phone]
    margin = block[:, target] - np.max(block[:, competitors], axis=1)
    return bool(np.mean(margin) >= np.log(2) and np.mean(margin > 0) >= .75
                and np.mean(np.argmax(block, axis=1) == target) >= .75)


def recheck_chunks(chunks: list[PlaybackChunk], e: CropEvidence, duration: float,
                   dropped: list[bool], *, apply: bool = False) -> tuple[list[PlaybackChunk], list[CropCheck]]:
    """Audit suspicious chunk edges; apply only conservative, supported extensions.

    Two window-relaxed alignments use one and two neighboring words. Agreement
    alone is insufficient: added frames must support the target phone, avoid
    extra speech, and stay inside the gap between original playback chunks.
    Missing words, failed searches and budget limits keep the baseline intact.
    """
    result = [replace(c, span=Span(c.span.start, c.span.end) if c.span else None) for c in chunks]
    checks: list[CropCheck] = []
    attempted = 0
    for ci, chunk in enumerate(chunks):
        sp = chunk.span
        if sp is None or sp.duration <= 0:
            continue
        live = [i for i in chunk.members if not dropped[i]]
        if not live:
            continue
        first, last = live[0], live[-1]
        flags = []
        if sp.start - (e.spikes[first].start - SPIKE_LAG_S) > .04:
            flags.append("possibly clipped onset")
        if e.spikes[last].end - sp.end > .04:
            flags.append("possibly clipped tail")
        if abs(e.raw_spans[first].start - sp.start) > .05:
            flags.append("onset estimates disagree")
        if sp.duration < SHORT_S:
            flags.append("very short crop")
        near_extra = any(a < sp.end + MAX_CHANGE_S and b > sp.start - MAX_CHANGE_S for a, b in e.extras)
        if near_extra:
            flags.append("near extra speech")
        if not flags:
            continue
        check = CropCheck(ci, chunk.text, Span(sp.start, sp.end), None,
                          Span(sp.start, sp.end), flags, "uncertain", "")
        checks.append(check)
        # A forced transcript can steal a repeat or invent a missing word. There
        # is not enough independent evidence to fix these automatically.
        if any(dropped[max(0, first-2):last+3]):
            check.detail = "missing word in surrounding speech"
            continue
        if attempted >= MAX_REGIONS:
            check.status, check.detail = "skipped", "recheck limit reached"
            continue
        attempted += 1
        try:
            local = _local_alignment(e, first, last, 1)
            wider = _local_alignment(e, first, last, 2)
            check.proposed = Span(local[0].start, local[-1].end)
            if near_extra:
                check.detail = "context re-aligned; extra speech makes ownership uncertain"
                continue
            candidate = Span(sp.start, sp.end)
            notes = []
            previous_end = max((c.span.end for c in result[:ci] if c.span), default=0.)
            next_start = min((c.span.start for c in chunks[ci+1:] if c.span), default=duration)
            for edge, word, pos in (("start", first, 0), ("end", last, -1)):
                old = getattr(sp, edge)
                new = getattr(local[pos], edge)
                other = getattr(wider[pos], edge)
                extension = old - new if edge == "start" else new - old
                if extension < MIN_CHANGE_S - 1e-8:
                    continue
                if extension > MAX_CHANGE_S + 1e-8 or abs(new - other) > AGREEMENT_S:
                    notes.append(f"{edge}: searches disagree or change too large")
                elif not previous_end - 1e-8 <= new <= next_start + 1e-8:
                    notes.append(f"{edge}: would enter neighboring playback")
                elif not _support(e, word, edge, min(old, new), max(old, new)):
                    notes.append(f"{edge}: insufficient phone evidence")
                else:
                    setattr(candidate, edge, new)
                    notes.append(f"{edge}: supported extension")
            changed = candidate != sp
            if changed and len(chunk.members) > 1 and candidate.duration > MAX_CHUNK_S:
                changed = False
                notes.append("extension would exceed playback group duration limit")
            if changed:
                # Reserve accepted audio in audit mode too, so proposals cannot
                # compete for the same gap and differ from an apply-mode run.
                result[ci].span = candidate
                check.accepted = Span(candidate.start, candidate.end)
                check.status = "adjusted" if apply else "proposed"
                check.after = Span(candidate.start, candidate.end) if apply else Span(sp.start, sp.end)
            else:
                check.status = "unchanged" if all(abs(getattr(check.proposed, k) - getattr(sp, k)) < MIN_CHANGE_S
                                                   for k in ("start", "end")) else "uncertain"
            check.detail = "; ".join(notes) or "no supported outward change"
        except Exception as exc:
            logging.getLogger("pronunciationcoach").warning("crop recheck kept [%s]: %s", chunk.text, exc)
            check.status, check.detail = "failed", str(exc)
    return (result if apply else chunks), checks
