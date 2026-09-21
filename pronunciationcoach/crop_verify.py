"""Hear a playback crop, trim corroborated edge words, then hear the result.

The expected text is a comparison target, never an ASR prompt. Only unwanted
prefix/suffix words can be removed. Substitutions, omissions and internal extras
are pronunciation evidence, not a reason to shave audio until a transcript fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import re
from typing import Callable

import numpy as np

from . import SAMPLE_RATE
from .asr import HeardWord, transcribe_crop
from .boundaries import SPIKE_LAG_S, Span
from .chunks import PlaybackChunk
from .crop_recheck import CropEvidence
from .playback import clip, sample_bounds
from .listener import HESITANT_P, UNDERSTOOD_P

MAX_CROPS = 8
MAX_CLIP_S = 3.0
MIN_CLIP_S = .06
EDGE_SLACK_S = .02
CONTEXT_TOL_S = .18  # Whisper timestamps are supporting evidence, not sample-accurate cuts.


@dataclass
class CropAttempt:
    span: Span
    heard: str
    probabilities: list[float]


@dataclass
class TranscriptCheck:
    chunk: int
    text: str
    before: Span
    after: Span
    status: str = "uncertain"
    detail: str = ""
    heard_before: str = ""
    heard_after: str = ""
    proposed: Span | None = None
    attempts: list[CropAttempt] = field(default_factory=list)
    words_before: list[HeardWord] = field(default_factory=list)

    def summary(self) -> str:
        heard = f"; heard {self.heard_before!r}" if self.heard_before else ""
        changed = (f" -> {self.heard_after!r}; {self.before.start:.2f}-{self.before.end:.2f}"
                   f" -> {self.proposed.start:.2f}-{self.proposed.end:.2f}s") if self.proposed and self.heard_after else ""
        return f"[{self.text}] {self.status}: {self.detail}{heard}{changed}"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower().replace("’", "'"))


def _words(words: list[HeardWord]) -> list[HeardWord]:
    return [HeardWord(t, w.probability, w.start, w.end) for w in words for t in tokens(w.text)]


def _matches(haystack: list[str], needle: list[str]) -> list[int]:
    return [i for i in range(len(haystack)-len(needle)+1) if haystack[i:i+len(needle)] == needle] if needle else []


def _valid(words: list[HeardWord], duration: float) -> bool:
    return bool(words) and all(np.isfinite([w.start, w.end, w.probability]).all()
        and 0 <= w.start < w.end <= duration + .04 and 0 <= w.probability <= 1
        and (i == 0 or w.start >= words[i-1].end - .02) for i, w in enumerate(words))


def _context_match(observed: list[HeardWord], context: list[HeardWord], offset: float,
                   start: int, count: int) -> list[HeardWord] | None:
    """Extra words must also occur here in the full-recording recognition."""
    got = [w.text for w in observed]
    candidates = []
    for i in _matches([w.text for w in context], got):
        run = context[i:i+len(got)]
        if not _valid(run, max(w.end for w in run)):
            continue
        a, b = observed[start], observed[start+count-1]
        c, d = run[start], run[start+count-1]
        extra_indices = list(range(start)) + list(range(start+count, len(observed)))
        # A real neighboring word outside the clip cannot corroborate a
        # hallucinated copy of that word inside it. Require acoustic overlap.
        if any(min(offset+observed[k].end, run[k].end) - max(offset+observed[k].start, run[k].start)
               < min(.02, (observed[k].end-observed[k].start)/2, (run[k].end-run[k].start)/2)
               for k in extra_indices):
            continue
        if (abs(offset + a.start - c.start) <= CONTEXT_TOL_S
                and abs(offset + b.end - d.end) <= CONTEXT_TOL_S):
            candidates.append(run)
    return candidates[0] if len(candidates) == 1 else None


def _contains_target(candidate: Span, chunk: PlaybackChunk, anchors: list[Span] | None) -> bool:
    if anchors is None:
        return False
    first, last = chunk.members[0], chunk.members[-1]
    # Retain the scoring model's first/last kept phone spikes. They need not
    # locate the exact boundary, but trimming past them could erase a sound.
    a, b = anchors[first], anchors[last]
    return candidate.start <= max(0., a.start - SPIKE_LAG_S) + 1e-8 and candidate.end >= b.end - 1e-8


def verify_chunks(chunks: list[PlaybackChunk], audio: np.ndarray, heard_words: list[HeardWord],
                  *, evidence: CropEvidence | None = None, suspicious: set[int] | None = None,
                  anchors: list[Span] | None = None,
                  protected: set[int] | None = None, apply: bool = True,
                  recognize: Callable = transcribe_crop) -> tuple[list[PlaybackChunk], list[TranscriptCheck]]:
    """Bounded detect/trim/recheck loop over rendered clips; failures preserve playback.

    Prioritize suspect alignments and crops overlapping neighboring recognized
    words. Check remaining short clips within the same eight-crop budget. A crop
    can have at most two trim proposals, each transcribed independently. Reuse
    the sentence's recognizer/model and its transcript for corroboration.
    """
    result = [replace(c, span=replace(c.span) if c.span else None) for c in chunks]
    context = _words(heard_words)
    anchors = anchors if anchors is not None else evidence.spikes if evidence is not None else None
    priority = set(suspicious or ())
    protected = protected or set()
    for ci, chunk in enumerate(chunks):
        if not chunk.span:
            continue
        expected = tokens(chunk.text)
        for j in _matches([w.text for w in context], expected):
            target = context[j:j+len(expected)]
            if target[-1].end <= chunk.span.start or target[0].start >= chunk.span.end:
                continue
            neighbors = context[max(0, j-1):j] + context[j+len(expected):j+len(expected)+1]
            if any(min(w.end, chunk.span.end) - max(w.start, chunk.span.start) > .035 for w in neighbors):
                priority.add(ci)
    order = sorted(range(len(chunks)), key=lambda i: (i not in priority, i))
    checks, used, unavailable = [], 0, False
    for ci in order:
        chunk = chunks[ci]
        if chunk.span is None or chunk.span.duration <= 0:
            continue
        original = chunk.span
        check = TranscriptCheck(ci, chunk.text, replace(original), replace(original))
        checks.append(check)
        if unavailable or used >= MAX_CROPS or original.duration > MAX_CLIP_S:
            check.status = "skipped"
            check.detail = "recognizer unavailable" if unavailable else "crop verification limit reached"
            continue
        if any(i in protected for i in chunk.members):
            check.status, check.detail = "skipped", "missing or inserted pronunciation sounds preserved"
            continue
        try:
            a, b = sample_bounds(audio, original.start, original.end, pad_before=chunk.pad_before, pad_after=chunk.pad_after)
            if (b-a)/SAMPLE_RATE < MIN_CLIP_S:
                check.status, check.detail = "skipped", "too little audio for verification"
                continue
            used += 1
            _, rendered = clip(audio, original.start, original.end, pad_before=chunk.pad_before, pad_after=chunk.pad_after)
            transcript, recognized = recognize(rendered)
            check.heard_before = transcript
            check.words_before = recognized
            observed = _words(recognized)
            expected = tokens(chunk.text)
            raw_tokens = [w.text for w in observed]
            # Whisper sometimes adds a word with no acoustic interval. Never
            # trim on that word's authority. Other, timed edge words can still
            # justify a correction ("I put the..." with a zero-duration "the").
            unlocalized = [w for w in observed if w.start == w.end]
            if any(w.text in expected for w in unlocalized):
                check.detail = "target word has no usable acoustic interval"
                continue
            observed = [w for w in observed if w.start != w.end]
            got = [w.text for w in observed]
            if tokens(transcript) != raw_tokens or not _valid(observed, len(rendered)/SAMPLE_RATE):
                check.detail = "unusable recognition or word timestamps"
                continue
            if got == expected:
                if unlocalized:
                    check.detail = "extra recognition has no acoustic interval; original crop retained"
                else:
                    check.status, check.detail = "matched", "crop transcript matches expected words"
                continue
            positions = _matches(got, expected)
            if len(positions) != 1:
                check.detail = "target missing, repeated or changed; no edge-only correction"
                continue
            start, count = positions[0], len(expected)
            stop = start + count
            extras = observed[:start] + observed[stop:]
            if not extras or len(extras) > 3 or any(w.probability < HESITANT_P for w in extras):
                check.detail = "extra edge words are not clear enough"
                continue
            corroborated = _context_match(observed, context, a/SAMPLE_RATE, start, count)
            if corroborated is None:
                check.detail = "extra words not corroborated at this position in the sentence"
                continue
            if any(w.probability < UNDERSTOOD_P for w in corroborated[:start] + corroborated[stop:]):
                check.detail = "surrounding recognition is uncertain about the extra words"
                continue
            # Two estimates: crop timestamps and full-sentence timestamps. Each
            # must retain the complete target and pass an actual playback recheck.
            estimates = [(a/SAMPLE_RATE + observed[start].start, a/SAMPLE_RATE + observed[stop-1].end),
                         (corroborated[start].start, corroborated[stop-1].end)]
            seen = set()
            for lo, hi in estimates:
                candidate = Span(max(original.start, lo - EDGE_SLACK_S) if start else original.start,
                                 min(original.end, hi + EDGE_SLACK_S) if stop < len(observed) else original.end)
                before_pad = chunk.pad_before and start == 0
                after_pad = chunk.pad_after and stop == len(observed)
                key = (round(candidate.start*SAMPLE_RATE), round(candidate.end*SAMPLE_RATE), before_pad, after_pad)
                if key in seen:
                    continue
                seen.add(key)
                if (candidate.duration < MIN_CLIP_S or candidate.duration < original.duration*.2
                        or not _contains_target(candidate, chunk, anchors)):
                    check.detail = "proposed trim would remove target sounds"
                    continue
                if candidate == original and before_pad == chunk.pad_before and after_pad == chunk.pad_after:
                    continue
                check.proposed = candidate
                _, adjusted = clip(audio, candidate.start, candidate.end, pad_before=before_pad, pad_after=after_pad)
                after_text, after_words = recognize(adjusted)
                check.heard_after = after_text
                check.attempts.append(CropAttempt(candidate, after_text, [w.probability for w in after_words]))
                # Tiny isolated function words often have low ASR probability.
                # Exact text plus contextual ownership and retained phone spikes
                # establish acceptance; probability is recorded, not an accuracy claim.
                if tokens(after_text) != expected or [w.text for w in _words(after_words)] != expected:
                    check.detail = "trimmed crop did not recheck as the expected words"
                    continue
                if not _valid(_words(after_words), len(adjusted)/SAMPLE_RATE):
                    check.detail = "trimmed crop returned unusable timestamps"
                    continue
                check.status = "trimmed" if apply else "proposed"
                check.detail = "removed corroborated edge words; adjusted crop transcript matches"
                if apply:
                    result[ci] = replace(chunk, span=candidate, pad_before=before_pad, pad_after=after_pad)
                    check.after = replace(candidate)
                break
            else:
                check.detail = check.detail or "no supported inward boundary change"
        except Exception as exc:
            logging.getLogger("pronunciationcoach").warning("playback verification kept [%s]: %s", chunk.text, exc)
            check.status, check.detail = "failed", str(exc)
            unavailable = True  # avoid repeating a model/OOM failure for every crop
    return result, sorted(checks, key=lambda c: c.chunk)
