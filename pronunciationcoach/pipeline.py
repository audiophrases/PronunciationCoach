"""One call from audio to per-phone scores."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .align import ExtraRun, Segment, align_words, greedy_decode
from .asr import transcribe_words
from .listener import match_words
from .audio import duration_s, to_mono_16k
from .boundaries import Span, word_spans
from .engine import DEFAULT_MODEL, Emissions, get_engine
from .g2p import text_to_phones
from .reference import acceptances, native_reference
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
    audio: np.ndarray  # the 16 kHz mono signal that was scored
    spans: list[Span] = field(default_factory=list)  # where each word is, for replay
    span_source: str = "spikes"  # "charsiu" (frame aligner) or "spikes" (fallback)
    reference_voices: list[str] = field(default_factory=list)  # native renderings the scorer listened to
    unknown_phones: list[str] = field(default_factory=list)  # expected phones the model has no label for
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def phones(self):
        return [p for w in self.words for p in w.phones]

    @property
    def heard_text(self) -> str:
        return " ".join(s.phone for s in self.heard)

    def segments_by_word(self) -> list[list[Segment]]:
        out, cursor = [], 0
        for w in self.words:
            out.append(self.segments[cursor : cursor + len(w.phones)])
            cursor += len(w.phones)
        return out

    def word_spans(self) -> list[Span]:
        """Where each word is in the recording (seconds), for replaying it."""
        return self.spans

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

    # The listener: Whisper's words and its confidence in each. It is the reference in
    # free-speech mode and the intelligibility judge in both modes.
    transcribed = False
    heard_words = None
    want_listener = os.environ.get("PC_LISTENER", "1") == "1"
    if not text or not text.strip() or want_listener:
        t0 = time.perf_counter()
        try:
            asr_text, heard_words = transcribe_words(audio)
            if not text or not text.strip():
                text, transcribed = asr_text, True
        except Exception as exc:  # out of memory, model missing, ...
            logging.getLogger("pronunciationcoach").warning("listener unavailable (%s)", exc)
            if not text or not text.strip():
                raise
        timings["listen"] = time.perf_counter() - t0

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
    timings["align"] = time.perf_counter() - t0

    # Natives as the yardstick: whatever a natural voice does in this sentence is not an error.
    t0 = time.perf_counter()
    ref = None
    if os.environ.get("PC_NATIVE_REF", "1") == "1":
        try:
            ref = native_reference(text, lang, engine)
        except Exception as exc:
            logging.getLogger("pronunciationcoach").warning("native reference unavailable (%s)", exc)
    timings["reference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    casual = os.environ.get("PC_CASUAL", "1") == "1"  # accept connected-speech forms (variants.py)
    words = score_words(em, segments, counts, acceptances(word_phones, ref, lang, casual))
    if heard_words is not None:
        for w, listened in zip(words, match_words([wp.word for wp in word_phones], heard_words)):
            w.listener_p, w.understood = listened.probability, listened.matched
    timings["score"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spans, span_source = locate_words(audio, word_phones, words, segments, alignment.extra, em.frame_ms)
    timings["crop"] = time.perf_counter() - t0

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
        audio=audio,
        spans=spans,
        span_source=span_source,
        reference_voices=list(ref.renderings) if ref else [],
        unknown_phones=unknown,
        timings=timings,
    )


# How far a word may extend beyond its spikes in the frame aligner: spikes lag onsets by
# ~80 ms and a final consonant can outlast its last spike.
WINDOW_BEFORE_S = 0.25
WINDOW_AFTER_S = 0.25


def locate_words(audio, word_phones, words, segments, extra, frame_ms) -> tuple[list[Span], str]:
    """Word crops from Charsiu's frame aligner, anchored to the scoring aligner's spikes so
    repeats and hesitations (the wildcard runs) cannot be swallowed by a neighbouring word.
    Falls back to the spike-based estimate if the aligner is unavailable."""
    by_word, cursor = [], 0
    for w in words:
        by_word.append(segments[cursor : cursor + len(w.phones)])
        cursor += len(w.phones)
    dropped = [[p.dropped for p in w.phones] for w in words]
    frame_s = frame_ms / 1000.0

    def kept(i):
        segs = [s for s, d in zip(by_word[i], dropped[i]) if not d]
        return segs or by_word[i]

    try:
        from .segmenter import get_segmenter, to_arpabet

        seg = get_segmenter()
        windows = []
        for i in range(len(by_word)):
            lo = kept(i)[0].start * frame_s - WINDOW_BEFORE_S
            hi = kept(i)[-1].end * frame_s + WINDOW_AFTER_S
            if i > 0:
                lo = max(lo, kept(i - 1)[-1].start * frame_s)  # not before the previous word's last spike
            if i + 1 < len(by_word):
                hi = min(hi, kept(i + 1)[0].end * frame_s)  # not past the next word's first spike
            windows.append((max(0.0, lo), hi))
        extras = [(r.start * frame_s, r.end * frame_s) for r in extra if len(r.phones) >= 2]
        spans = seg.align(seg.frame_log_probs(audio), [to_arpabet(wp.phones) for wp in word_phones], windows, extras)
        return spans, "charsiu"
    except Exception as exc:  # model not downloadable, out of memory, ...
        logging.getLogger("pronunciationcoach").warning("Charsiu segmenter unavailable (%s); using spike-based crops", exc)
    return word_spans(audio, by_word, dropped, frame_ms), "spikes"
