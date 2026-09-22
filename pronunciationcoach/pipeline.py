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
from . import SAMPLE_RATE
from .boundaries import HOP, SPIKE_LAG_S, Span, energy_db, word_spans
from .chunks import PlaybackChunk, build_chunks, safe_spans
from .crop_recheck import CropCheck, CropEvidence, recheck_chunks
from .crop_verify import TranscriptCheck, verify_chunks
from .engine import DEFAULT_MODEL, Emissions, get_engine
from .g2p import text_to_phones
from .reference import acceptances, native_reference
from .scoring import PhoneScore, WordScore, score_words
from .variants import VOWELS


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
    span_source: str = "spikes"  # "mfa", "charsiu" (frame aligner) or "spikes" (fallback)
    span_cases: list[str] = field(default_factory=list)  # how each start was settled: onset / dip / join
    span_reject: str = ""  # why MFA was not used, when it was not
    reference_voices: list[str] = field(default_factory=list)  # native renderings the scorer listened to
    unknown_phones: list[str] = field(default_factory=list)  # expected phones the model has no label for
    timings: dict[str, float] = field(default_factory=dict)
    chunks: list[PlaybackChunk] = field(default_factory=list)
    crop_checks: list[CropCheck] = field(default_factory=list)
    crop_recheck_mode: str = "disabled"
    transcript_checks: list[TranscriptCheck] = field(default_factory=list)

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
    extra = attach_insertions(words, word_phones, segments, alignment.extra, em, audio)
    if heard_words is not None:
        for w, listened in zip(words, match_words([wp.word for wp in word_phones], heard_words)):
            w.listener_p, w.understood = listened.probability, listened.matched
    timings["score"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # A word the learner never said gives MFA nothing to align, so this has to be
    # known before the crops are chosen, not after them.
    dropped_words = [bool(w.phones) and all(p.dropped for p in w.phones) and not w.insertions for w in words]
    crop_evidence: list[CropEvidence] = []
    recheck_mode = os.environ.get("PC_CROP_RECHECK", "0").lower()
    spans, span_source, span_cases, span_reject = locate_words(
        audio, word_phones, words, segments, extra, em.frame_ms, crop_evidence,
        dropped_words=dropped_words, timings=timings,
        want_evidence=recheck_mode != "0")  # an audit still needs Charsiu's evidence
    spans = safe_spans(spans, duration_s(audio), dropped_words)
    timings["crop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = build_chunks(text, [w.word for w in words], spans, audio, dropped_words,
                          [(r.start * em.frame_ms / 1000, r.end * em.frame_ms / 1000) for r in extra],
                          enabled=os.environ.get("PC_CHUNKS", "1") != "0", sounds=edge_sounds(words))
    timings["chunk"] = time.perf_counter() - t0

    # Retired from normal processing: across 24 archived playback groups the second
    # pass accepted no correction at all, for 2.7-9.9 s per assessment. It stays
    # available as an audit (PC_CROP_RECHECK=audit) and its modules are unchanged.
    t0 = time.perf_counter()
    crop_checks = []
    transcript_checks = []
    if recheck_mode != "0" and crop_evidence:
        # Alignment proposals supply suspicion/evidence. Playback adjustments
        # now require detection and verification by the existing word listener.
        chunks, crop_checks = recheck_chunks(chunks, crop_evidence[0], duration_s(audio), dropped_words,
                                            apply=False)
    if recheck_mode != "0" and heard_words is not None:
        # Scoring anchors survive a missing Charsiu model. Cropping must not
        # remove the target's kept phone spikes just to satisfy an ASR transcript.
        anchors = []
        for w, sp in zip(words, spans):
            kept_phones = [p for p in w.phones if not p.dropped]
            anchors.append(Span(kept_phones[0].start_s, kept_phones[-1].end_s) if kept_phones else sp)
        chunks, transcript_checks = verify_chunks(
            chunks, audio, heard_words, evidence=crop_evidence[0] if crop_evidence else None,
            anchors=anchors,
            suspicious={c.chunk for c in crop_checks},
            protected={i for i, w in enumerate(words) if dropped_words[i] or w.insertions},
            apply=recheck_mode != "audit")
        # Say which halves actually ran: recheck_chunks needs Charsiu's evidence, and
        # reporting "apply" when only the verifier ran made past audits incomparable.
        recheck_mode = "audit" if recheck_mode == "audit" else "apply"
        if not crop_evidence:
            recheck_mode += " (verify only)"
    elif recheck_mode == "0":
        recheck_mode = "retired"
    else:
        recheck_mode = "listener-unavailable"
    timings["crop_recheck"] = time.perf_counter() - t0

    return Assessment(
        text=text,
        transcribed=transcribed,
        lang=lang,
        duration_s=duration_s(audio),
        heard=heard,
        words=words,
        segments=segments,
        extra=extra,
        emissions=em,
        audio=audio,
        spans=spans,
        span_source=span_source,
        span_cases=span_cases,
        span_reject=span_reject,
        reference_voices=list(ref.renderings) if ref else [],
        unknown_phones=unknown,
        timings=timings,
        chunks=chunks,
        crop_checks=crop_checks,
        crop_recheck_mode=recheck_mode,
        transcript_checks=transcript_checks,
    )


# How far a word may extend beyond its spikes in the frame aligner: spikes lag onsets by
# ~80 ms and a final consonant can outlast its last spike.
WINDOW_BEFORE_S = 0.25
WINDOW_AFTER_S = 0.25


# The wildcard between words absorbs anything that is not in the sentence - repeats, "uh",
# a breath - and reports it as an extra run. One kind of extra is a pronunciation error and
# must stay with its word: a vowel glued to a word that starts with s + consonant ("e-speak",
# "e-Spain" - Spanish and Catalan have no such word onsets) or to a word-final consonant
# ("English-e"). Glued means the next sound follows within INSERT_GAP_S with no silence.
INSERT_GAP_S = 0.16  # spike to spike
INSERT_VOWELS = VOWELS - {"eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "əʊ"}  # a diphthong is a word ("I", "a"), not an epenthetic vowel


def attach_insertions(words, word_phones, segments, extra, em, audio) -> list:
    """Move epenthetic vowels from the extra runs into the neighbouring word's `insertions`;
    return the extra runs that remain."""
    frame_s = em.frame_ms / 1000.0
    db = energy_db(audio)
    silent = float(np.percentile(db, 10)) + SILENCE_ABOVE_FLOOR_DB
    step = HOP / SAMPLE_RATE

    def pause_between(a: float, b: float) -> bool:
        return any(db[min(int(t / step), len(db) - 1)] < silent for t in np.arange(a, max(a, b), step))

    by_word, cursor = [], 0
    for w in words:
        by_word.append(segments[cursor : cursor + len(w.phones)])
        cursor += len(w.phones)

    def score(run, where: str) -> PhoneScore:
        block = np.exp(em.log_probs[run.start : run.end + 1])
        block[:, em.blank_id] = 0
        mass = block.sum(axis=0)
        p = float(mass[em.labels.index(run.phones[0])] / mass.sum()) if run.phones[0] in em.labels and mass.sum() > 0 else 0.0
        return PhoneScore(expected="", start_s=run.start * frame_s, end_s=run.end * frame_s, gop=-10.0, posterior=p,
                          heard=run.phones[0], inserted=where)

    remaining = []
    for run in extra:
        if len(run.phones) != 1 or run.phones[0] not in INSERT_VOWELS:
            remaining.append(run)
            continue
        run_start, run_end = run.start * frame_s, run.end * frame_s
        attached = False
        for w, wp, segs in zip(words, word_phones, by_word):
            first, last = segs[0].start * frame_s, segs[-1].end * frame_s
            starts_with_cluster = len(wp.phones) >= 2 and wp.phones[0] == "s" and wp.phones[1] not in VOWELS
            if starts_with_cluster and 0 <= first - run_end <= INSERT_GAP_S and not pause_between(run_end, first - SPIKE_LAG_S):
                w.insertions.append(score(run, "before"))
                attached = True
                break
            ends_with_consonant = wp.phones[-1] not in VOWELS
            if ends_with_consonant and 0 <= run_start - last <= INSERT_GAP_S and not pause_between(last, run_start):
                w.insertions.append(score(run, "after"))
                attached = True
                break
        if not attached:
            remaining.append(run)
    return remaining


# MFA aligns the whole utterance against the sentence, which is the best crop available
# when the learner really did read that sentence. It cannot report that they did not:
# forced alignment always emits an interval for every transcript word, so an omitted or
# reordered word does not raise - it silently shifts its neighbours (measured: up to
# 250 ms, enough that tapping "like" plays "my"). These gates, not MFA's own errors, are
# what decide whether its answer is usable.
# |calibrated MFA start - spike onset|, measured over 78 correctly-transcribed words:
# median 0 ms, p95 40 ms, worst 110 ms. With one word omitted from the reading the same
# figure reaches 210 ms, so 150 ms separates the two - but only by 40 ms on the good side.
# A false reject costs nothing worse than falling back to Charsiu, so err on rejecting.
MFA_ANCHOR_TOL_S = 0.15


def mfa_reject_reason(words, word_phones, extra, dropped_words: list[bool] | None) -> str:
    """Why MFA must not be asked for these crops, or "" when it can be.

    Each case is one where the sentence is not what was actually said, and forced
    alignment would answer confidently anyway. Charsiu's wildcard state handles
    them; it is cheaper to check here than to align and then distrust the result.
    """
    if os.environ.get("PC_MFA", "1") == "0":
        return "disabled"
    if not word_phones:
        return "no words"
    if any(not w.phones for w in words):
        return "a word has no phones to anchor against"  # kept() would be empty
    if dropped_words and any(dropped_words):
        return "a word was not said"
    if any(w.understood is False for w in words):
        return "a word was not recognised"
    if any(w.insertions for w in words):
        return "an inserted vowel has no dictionary entry"
    if any(len(run.phones) >= 2 for run in extra):
        return "speech outside the sentence"
    from . import mfa

    spelling = [getattr(wp, "word", "") for wp in word_phones]
    if not all(spelling):
        return "no spelling to align against"
    known = mfa.lexicon()
    unknown = [word for word in spelling if known and word.lower() not in known]
    if unknown:
        return f"not in the dictionary: {' '.join(unknown[:3])}"
    return ""


def edge_sounds(words) -> list[tuple[str, str]]:
    """Each word's first and last sound as pronounced: dropped phones are skipped and
    an inserted vowel counts, since that is what links (or fails to link) to a neighbour."""
    out = []
    for w in words:
        said = ([p.heard for p in w.insertions if p.inserted == "before"]
                + [p.expected for p in w.phones if not p.dropped]
                + [p.heard for p in w.insertions if p.inserted == "after"])
        out.append((said[0], said[-1]) if said else ("", ""))
    return out


def mfa_spans(audio, words_to_align: list[str], duration: float) -> list[Span]:
    """Word crops from the warm MFA worker, calibrated. Raises if MFA cannot be used."""
    from . import mfa
    from .mfa_worker import align

    reply, _ = align(audio, " ".join(words_to_align))
    spans = mfa.word_spans(reply, words_to_align, reply.get("duration", duration))
    return mfa.calibrate(spans)


def locate_words(audio, word_phones, words, segments, extra, frame_ms,
                 evidence: list[CropEvidence] | None = None, dropped_words: list[bool] | None = None,
                 timings: dict | None = None, want_evidence: bool = False) -> tuple[list[Span], str, list[str], str]:
    """Word crops from MFA, falling back to Charsiu's frame aligner and then to the
    scoring aligner's spikes. Every source is anchored to those spikes so that repeats
    and hesitations cannot be swallowed by a neighbouring word."""
    by_word, cursor = [], 0
    for w in words:
        by_word.append(segments[cursor : cursor + len(w.phones)])
        cursor += len(w.phones)
    dropped = [[p.dropped for p in w.phones] for w in words]
    frame_s = frame_ms / 1000.0

    def inserted(i, where):
        return [Segment(p.heard, 0, round(p.start_s / frame_s), round(p.end_s / frame_s) + 1, 0.0)
                for p in words[i].insertions if p.inserted == where]

    def kept(i):
        """The word's spikes as pronounced: an inserted vowel belongs to the word's crop."""
        segs = [s for s, d in zip(by_word[i], dropped[i]) if not d]
        return inserted(i, "before") + (segs or by_word[i]) + inserted(i, "after")

    def pronounced(i):
        w = words[i]
        return ([p.heard for p in w.insertions if p.inserted == "before"] + list(word_phones[i].phones)
                + [p.heard for p in w.insertions if p.inserted == "after"])

    def charsiu() -> tuple[list[Span], list[str]]:
        """Charsiu's crops, and the evidence the audit pass reads."""
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
        phones = [to_arpabet(pronounced(i)) for i in range(len(by_word))]
        log_probs = seg.frame_log_probs(audio)
        raw_spans = seg.align(log_probs, phones, windows, extras)
        spans, cases = settle_starts(raw_spans, [kept(i) for i in range(len(by_word))], audio, frame_s)
        if evidence is not None:
            evidence.append(CropEvidence(seg, log_probs, phones, windows, raw_spans,
                                        [Span(kept(i)[0].start * frame_s, kept(i)[-1].end * frame_s)
                                         for i in range(len(by_word))],
                                        [(r.start * frame_s, r.end * frame_s) for r in extra]))
        return spans, cases

    log = logging.getLogger("pronunciationcoach")
    reject = mfa_reject_reason(words, word_phones, extra, dropped_words)
    if not reject:
        t0 = time.perf_counter()
        try:
            spans = mfa_spans(audio, [wp.word for wp in word_phones], len(audio) / SAMPLE_RATE)
            # MFA has no wildcard state, so a hesitation after a word is absorbed into it
            # (measured: a 1.30 s crop for "So"). Its own spikes bound how far it can run.
            for i, sp in enumerate(spans):
                sp.end = max(sp.start, min(sp.end, kept(i)[-1].end * frame_s + WINDOW_AFTER_S))
            late = [abs(sp.start - (kept(i)[0].start * frame_s - SPIKE_LAG_S)) for i, sp in enumerate(spans)]
            if max(late, default=0.0) > MFA_ANCHOR_TOL_S:
                # One shifted word moves every boundary after it, so this is all or nothing.
                reject = f"disagrees with the spikes by {max(late) * 1000:.0f} ms"
            else:
                if timings is not None:
                    timings["mfa"] = time.perf_counter() - t0
                if want_evidence:
                    try:
                        charsiu()  # audit only: MFA keeps the crops, Charsiu supplies the evidence
                    except Exception as exc:
                        log.warning("no Charsiu evidence for the audit pass (%s)", exc)
                spans, cases = settle_starts(spans, [kept(i) for i in range(len(by_word))], audio, frame_s)
                return spans, "mfa", cases, ""
        except Exception as exc:  # worker missing, timed out, or transcript mismatch
            reject = str(exc)
        log.info("MFA not used (%s); falling back", reject)

    try:
        spans, cases = charsiu()
        return spans, "charsiu", cases, reject
    except Exception as exc:  # model not downloadable, out of memory, ...
        log.warning("Charsiu segmenter unavailable (%s); using spike-based crops", exc)
    return word_spans(audio, by_word, dropped, frame_ms), "spikes", [], reject


# Where exactly does a word start? Two estimates disagree in opposite ways. Charsiu's frame
# labels place the transition well when the sound changes cleanly, but on real recordings they
# run late for quiet onsets (a stop's closure and burst, a fricative, h, a nasal): the previous
# vowel is extended over them, and the learner's clip loses the very sound they tap the word
# for. The scoring model's spikes (a sound starts ~SPIKE_LAG_S before its spike) are robust
# there but vague after a vowel, and a start pulled back that far takes the previous word's
# vowel with it - and the previous word's own last spike can fire *after* our first sound has
# begun, so it is no floor either. The audio is the referee, read with the phone sequence:
#   onset - silence before the word: it starts where sound resumes (a burst, a fricative);
#   dip   - the word starts with a quiet consonant after a vowel and the audio shows the valley
#           Charsiu skipped: take the valley's start;
#   join  - anything else: Charsiu's boundary, never later than the spike allows.
ONSET_SLACK_S = 0.02
SEARCH_BEFORE_S = 0.10  # how far before the earlier candidate a word's first sound may begin
SILENCE_ABOVE_FLOOR_DB = 6.0  # below this the frame is background, not a quiet consonant
DIP_DB = 6.0  # a quiet consonant's valley is at least this deep on both sides
RISE_TOL_S = 0.05  # ...and Charsiu started the word about where the valley ends
LATE_TOL_S = 0.06  # a start later than the spike onset plus this is certainly late
QUIET_ONSETS = set("pbtdkgɡfvθðszʃʒhmnŋɾ") | {"tʃ", "dʒ"}


def settle_starts(spans: list[Span], kept: list[list[Segment]], audio: np.ndarray, frame_s: float) -> tuple[list[Span], list[str]]:
    """Settle each word's start between Charsiu's boundary and the spike-based onset using
    the audio's energy (see above); the previous word ends where the next one starts."""
    db = energy_db(audio)
    step = HOP / SAMPLE_RATE
    silent = float(np.percentile(db, 10)) + SILENCE_ABOVE_FLOOR_DB

    def level(t: float) -> float:
        return float(db[min(max(int(round(t / step)), 0), len(db) - 1)])

    def ticks(a: float, b: float) -> list[float]:
        return [k * step for k in range(int(np.ceil(a / step - 1e-6)), int(np.floor(b / step + 1e-6)) + 1)]

    def valley_start(lo: float, hi: float, c: float) -> float | None:
        """Start of the valley in [lo, hi] that is at least DIP_DB deep after a peak and whose
        rise is where Charsiu began the word - the quiet consonant it handed to the previous
        word. Of several (a closure, then the word's own fricative) the one at Charsiu's boundary."""
        times = ticks(lo, hi)
        levels = [level(t) for t in times]
        best: tuple[float, float] | None = None
        for k in range(1, len(times) - 1):
            if not (levels[k] <= levels[k - 1] and levels[k] <= levels[k + 1]):
                continue  # not a local minimum
            peak = int(np.argmax(levels[:k]))
            if levels[peak] - levels[k] < DIP_DB:
                continue
            half = (levels[peak] + levels[k]) / 2
            rise = next((t for t, lv in zip(times[k:], levels[k:]) if lv >= half), None)
            if rise is None or abs(rise - c) > RISE_TOL_S:
                continue
            start = next(t for t, lv in zip(times[peak:], levels[peak:]) if lv < half)
            if best is None or abs(rise - c) < best[0]:
                best = (abs(rise - c), start)
        return best[1] if best else None

    out = [Span(sp.start, sp.end) for sp in spans]
    cases: list[str] = []
    for i, (sp, segs) in enumerate(zip(out, kept)):
        c = sp.start
        s = segs[0].start * frame_s - SPIKE_LAG_S - ONSET_SLACK_S
        early, late = min(c, s), max(c, s)
        lo = max(0.0, early - SEARCH_BEFORE_S)
        if i > 0:
            lo = max(lo, kept[i - 1][0].start * frame_s)
        quiet = [t for t in ticks(lo, s) if level(t) < silent]
        if quiet:
            start, case = quiet[-1] + step, "onset"  # sound resumes here; a near-silent h is kept
        else:
            start, case = c, "join"
            ours = segs[0].phone in QUIET_ONSETS and (i == 0 or kept[i - 1][-1].phone not in QUIET_ONSETS)
            if ours:
                v = valley_start(lo, late + RISE_TOL_S, c)
                if v is not None:
                    start, case = v, "dip"
        start = min(start, s + LATE_TOL_S)
        sp.start = min(max(start, lo, 0.0), max(late, lo))
        sp.start = min(len(audio) / SAMPLE_RATE, max(sp.start, out[i - 1].start if i else 0.0))
        if i > 0 and out[i - 1].end > sp.start:
            out[i - 1].end = sp.start
        sp.end = min(len(audio) / SAMPLE_RATE, max(sp.end, sp.start))
        cases.append(case)
    return out, cases
