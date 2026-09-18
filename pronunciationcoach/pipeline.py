"""One call from audio to per-phone scores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .align import ExtraRun, Segment, align_words, greedy_decode
from .asr import transcribe
from .audio import duration_s, to_mono_16k
from .engine import DEFAULT_MODEL, Emissions, get_engine
from .g2p import text_to_phones
from .scoring import WordScore, score_words


@dataclass
class Assessment:
    text: str  # reference text actually used (typed by the teacher, or transcribed)
    transcribed: bool
    lang: str
    duration_s: float
    heard: list[Segment]  # text-independent greedy decode
    words: list[WordScore]  # per-word, per-phone scores against the reference
    segments: list[Segment]  # the forced alignment behind `words`, flat, one per expected phone
    extra: list[ExtraRun]  # speech heard where the sentence has no word (repeats, hesitations)
    emissions: Emissions
    unknown_phones: list[str] = field(default_factory=list)  # expected phones the model has no label for
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def phones(self):
        return [p for w in self.words for p in w.phones]

    @property
    def heard_text(self) -> str:
        return " ".join(s.phone for s in self.heard)

    def word_spans(self, pad_before: float = 0.10, pad_after: float = 0.25) -> list[tuple[float, float]]:
        """(start, end) in seconds of each word in the recording, for replaying it.

        CTC marks a phone near its onset, so the last phone's span ends before the
        sound does: pad after, but never into the next word. Dropped phones were
        aligned somewhere arbitrary and are ignored when placing the word.
        """
        em = self.emissions
        starts: list[float] = []
        ends: list[float] = []
        cursor = 0
        for w in self.words:
            segs = self.segments[cursor : cursor + len(w.phones)]
            kept = [s for s, p in zip(segs, w.phones) if not p.dropped] or segs
            starts.append(em.frame_to_s(min(s.start for s in kept)))
            ends.append(em.frame_to_s(max(s.end for s in kept)))
            cursor += len(w.phones)
        spans = []
        for i in range(len(starts)):
            lo = max(starts[i] - pad_before, ends[i - 1] if i else 0.0, 0.0)
            hi = min(ends[i] + pad_after, starts[i + 1] if i + 1 < len(starts) else self.duration_s, self.duration_s)
            spans.append((lo, max(hi, lo + 0.05)))
        return spans

    def extra_text(self, min_phones: int = 2) -> str:
        """Human-readable list of the extra runs, e.g. 'ɔ z ə z (7.5–8.4 s)'."""
        em = self.emissions
        runs = [r for r in self.extra if len(r.phones) >= min_phones]
        return "; ".join(f"{' '.join(r.phones)} ({em.frame_to_s(r.start):.1f}–{em.frame_to_s(r.end):.1f} s)" for r in runs)


def assess(
    samples: np.ndarray,
    sr: int,
    text: str | None = None,
    lang: str = "en-us",
    model_id: str = DEFAULT_MODEL,
) -> Assessment:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    audio = to_mono_16k(samples, sr)

    engine = get_engine(model_id)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    em = engine.emissions(audio)
    timings["emissions"] = time.perf_counter() - t0
    heard = greedy_decode(em)

    transcribed = False
    if not text or not text.strip():
        t0 = time.perf_counter()
        text = transcribe(audio)
        transcribed = True
        timings["asr"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    word_phones = text_to_phones(text, lang)
    word_ids: list[list[int]] = []
    expected: list[str] = []
    unknown: list[str] = []
    counts: list[tuple[str, int]] = []
    for wp in word_phones:
        ids = []
        for phone in wp.phones:
            pid = engine.phone_id(phone)
            if pid is None:
                unknown.append(phone)
                pid = engine.unk_id
            ids.append(pid)
            expected.append(phone)
        word_ids.append(ids)
        counts.append((wp.word, len(wp.phones)))

    alignment = align_words(em, word_ids)
    segments = alignment.segments
    if len(segments) != len(expected):
        raise RuntimeError(f"alignment returned {len(segments)} spans for {len(expected)} phones")
    for seg, phone in zip(segments, expected):
        seg.phone = phone  # show the expected spelling even where the model only had <unk>
    words = score_words(em, segments, counts)
    timings["align+score"] = time.perf_counter() - t0

    return Assessment(
        text=text,
        transcribed=transcribed,
        lang=lang,
        duration_s=duration_s(audio),
        heard=heard,
        words=words,
        segments=segments,
        extra=alignment.extra,
        emissions=em,
        unknown_phones=unknown,
        timings=timings,
    )
