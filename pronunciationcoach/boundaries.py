"""Where each word and phone really is in the recording, for cropping and replay.

A CTC recogniser marks a phone with a brief spike somewhere inside it, not at
its edges. Measured against Edge TTS word boundaries on synthesised speech
(scripts/calibrate_spikes, 29 words): the first spike of a word starts a very
steady ~80 ms after the word begins (p10 63 ms, p90 106 ms), and the audio of
a word ends within ~60 ms after its last spike ends. Inside a word the spikes
follow one another every ~60 ms, so "spike start minus 80 ms" is a usable onset
for every phone.

Connected speech has no silence between words. When two words' estimated spans
overlap, the cut goes at the quietest 10 ms between the last spike of one word
and the first spike of the next - the closure of a consonant or the trough of a
liaison - rather than at an arbitrary midpoint. Silence around a word after a
pause is trimmed with an energy threshold set from the recording itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE
from .align import Segment

SPIKE_LAG_S = 0.08  # spike start is this long after the sound's onset (measured)
ONSET_MARGIN_S = 0.02  # a little slack before the estimated onset
END_MARGIN_S = 0.06  # word audio ends this long after the last spike ends (p90, measured)
LAST_WORD_TAIL_S = 0.15  # the final word may ring out; silence trimming takes back the excess
CUT_SEARCH_S = 0.04  # how far around the estimated boundary to look for the quietest point

HOP = 160  # 10 ms energy frames
WIN = 320


@dataclass
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def energy_db(audio: np.ndarray) -> np.ndarray:
    """RMS energy in dB per 10 ms frame."""
    if len(audio) < WIN:
        return np.full(1, -120.0)
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP]
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-6)


def speech_threshold(db: np.ndarray) -> float:
    """Between the recording's noise floor and its loud speech, nearer the floor so
    quiet fricatives (s, f, th, h) still count as speech."""
    floor = np.percentile(db, 10)
    loud = np.percentile(db, 95)
    return floor + 0.35 * (loud - floor)


def _t(frame: int, frame_ms: float) -> float:
    return frame * frame_ms / 1000.0


def word_spans(
    audio: np.ndarray,
    words: list[list[Segment]],
    dropped: list[list[bool]],
    frame_ms: float,
) -> list[Span]:
    """One span per word. `words` holds each word's aligned phone segments in order;
    `dropped` marks phones that were never produced (their positions are meaningless)."""
    duration = len(audio) / SAMPLE_RATE
    if not words:
        return []
    db = energy_db(audio)
    thr = speech_threshold(db)

    def db_at(t: float) -> float:
        i = min(max(int(t * SAMPLE_RATE / HOP), 0), len(db) - 1)
        return float(db[i])

    firsts, lasts = [], []
    for segs, drop in zip(words, dropped):
        kept = [s for s, d in zip(segs, drop) if not d] or segs
        firsts.append(kept[0])
        lasts.append(kept[-1])

    starts = [max(0.0, _t(f.start, frame_ms) - SPIKE_LAG_S - ONSET_MARGIN_S) for f in firsts]
    ends = [_t(l.end, frame_ms) + END_MARGIN_S for l in lasts]
    ends[-1] = min(duration, _t(lasts[-1].end, frame_ms) + LAST_WORD_TAIL_S)

    # Neighbours that run into each other: the next word's onset estimate (spike - lag) is
    # the tighter of the two estimates (p10-p90 spread of ~40 ms), so search for the quietest
    # 10 ms within +-40 ms of it and cut there - a consonant closure or the trough of a liaison.
    for i in range(len(words) - 1):
        if ends[i] <= starts[i + 1]:
            continue
        guess = _t(firsts[i + 1].start, frame_ms) - SPIKE_LAG_S
        lo = max(guess - CUT_SEARCH_S, _t(lasts[i].end, frame_ms) - SPIKE_LAG_S)  # never cut into word i's last sound
        hi = min(guess + CUT_SEARCH_S, _t(firsts[i + 1].start, frame_ms))
        a, b = int(lo * SAMPLE_RATE / HOP), int(hi * SAMPLE_RATE / HOP)
        cut = (a + int(np.argmin(db[a:b]))) * HOP / SAMPLE_RATE if b - a >= 2 else guess
        ends[i] = starts[i + 1] = float(min(max(cut, lo), hi))

    # Only the final word gets a generous tail, so trim the silence it may have swallowed.
    step = HOP / SAMPLE_RATE
    limit = _t(lasts[-1].end, frame_ms)
    while ends[-1] - step > limit and db_at(ends[-1] - step) < thr:
        ends[-1] -= step
    for i in range(len(words)):
        if ends[i] - starts[i] < 0.05:
            ends[i] = min(duration, starts[i] + 0.05)
    return [Span(s, e) for s, e in zip(starts, ends)]


def phone_spans(word_span: Span, segs: list[Segment], drop: list[bool], frame_ms: float) -> list[Span | None]:
    """Phone spans inside one word: each runs from its own estimated onset to the next
    phone's onset; the last one to the end of the word. Dropped phones get None."""
    onsets: list[float | None] = []
    for s, d in zip(segs, drop):
        onsets.append(None if d else min(max(_t(s.start, frame_ms) - SPIKE_LAG_S, word_span.start), word_span.end))
    out: list[Span | None] = []
    for i, on in enumerate(onsets):
        if on is None:
            out.append(None)
            continue
        nxt = next((o for o in onsets[i + 1 :] if o is not None), word_span.end)
        out.append(Span(on, max(nxt, on + 0.03)))
    return out
