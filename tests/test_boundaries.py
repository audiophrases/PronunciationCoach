"""Word/phone crop boundaries on synthetic signals (no model needed)."""

import numpy as np
import pytest

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.align import Segment
from pronunciationcoach.boundaries import SPIKE_LAG_S, word_spans

FRAME_MS = 20.0


def seg(start_frame: int, end_frame: int | None = None, phone: str = "ə") -> Segment:
    return Segment(phone, 1, start_frame, end_frame or start_frame + 1, 0.0)


def tone(seconds: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_isolated_words_start_a_lag_before_the_first_spike_and_end_just_after_the_last():
    audio = np.concatenate([silence(0.1), tone(0.4), silence(0.3), tone(0.4), silence(0.5)])  # words at 0.10-0.50, 0.80-1.20
    # spikes ~80 ms after each onset, last spike near the end of the word
    a = [seg(9), seg(14), seg(23)]  # 0.18, 0.28, 0.46 s
    b = [seg(44), seg(50), seg(58)]  # 0.88, 1.00, 1.16 s
    spans = word_spans(audio, [a, b], [[False] * 3, [False] * 3], FRAME_MS)
    assert spans[0].start == pytest.approx(0.18 - SPIKE_LAG_S - 0.02, abs=1e-6)
    assert spans[0].end == pytest.approx(0.48 + 0.06, abs=1e-6)
    assert spans[1].start == pytest.approx(0.88 - SPIKE_LAG_S - 0.02, abs=1e-6)
    # last word: generous tail, then silence trimmed back to roughly the end of the tone
    assert 1.18 <= spans[1].end <= 1.26


def test_touching_words_are_cut_at_the_quietest_point_between_them():
    # continuous sound with a 40 ms dip centred at 0.46 s
    audio = np.concatenate([silence(0.1), tone(0.34), tone(0.04, level=0.02), tone(0.4), silence(0.3)])
    a = [seg(9), seg(14), seg(20)]  # last spike 0.40-0.42 -> naive end 0.48
    b = [seg(27), seg(33), seg(40)]  # first spike 0.54 -> naive start 0.44 (overlap)
    spans = word_spans(audio, [a, b], [[False] * 3, [False] * 3], FRAME_MS)
    assert spans[0].end == spans[1].start
    assert 0.44 <= spans[0].end <= 0.48, spans


def test_dropped_phones_do_not_move_the_word():
    audio = np.concatenate([silence(0.1), tone(0.4), silence(1.0)])
    segs = [seg(9), seg(14), seg(60)]  # third phone parked far away: it was never said
    with_drop = word_spans(audio, [segs], [[False, False, True]], FRAME_MS)[0]
    assert with_drop.end < 0.7, with_drop


def test_settle_starts_after_silence_at_the_point_where_sound_resumes():
    from pronunciationcoach.boundaries import Span
    from pronunciationcoach.pipeline import settle_starts

    # word 1 at 0.30-0.80, a pause, word 2 at 0.90-1.30 (a stop's burst at 0.90)
    audio = np.concatenate([silence(0.3), tone(0.5), silence(0.1), tone(0.4), silence(0.5)])
    kept = [[seg(19), seg(37, 39)], [seg(49), seg(60)]]  # word 2's first spike at 0.98 -> spike onset 0.88
    spans = [Span(0.30, 0.96), Span(0.96, 1.30)]  # Charsiu let word 1 run over the burst
    out, cases = settle_starts(spans, kept, audio, 0.02)
    assert cases[1] == "onset" and 0.89 <= out[1].start <= 0.91  # where sound resumes, not the vaguer spike onset
    assert out[0].end == out[1].start


def test_settle_starts_a_quiet_consonant_at_the_start_of_its_dip():
    from pronunciationcoach.boundaries import Span
    from pronunciationcoach.pipeline import settle_starts

    # vowel 0.30-0.80, a quiet consonant 0.80-0.88 (20 dB down), vowel 0.88-1.30: no silence anywhere
    audio = np.concatenate([silence(0.3), tone(0.5), tone(0.08, 0.03), tone(0.42), silence(0.5)])
    kept = [[seg(19), seg(37, 39)], [seg(46, phone="ð"), seg(60)]]  # word 2 starts with th; first spike 0.92 -> onset 0.82
    spans = [Span(0.30, 0.88), Span(0.88, 1.30)]  # Charsiu gave the consonant to word 1
    out, cases = settle_starts(spans, kept, audio, 0.02)
    assert cases[1] == "dip" and 0.79 <= out[1].start <= 0.81
    assert out[0].end == out[1].start
    # the same valley when it is the previous word's final consonant (word 2 starts with a vowel): not ours
    kept2 = [[seg(19), seg(37, 39, phone="s")], [seg(46, phone="ɪ"), seg(60)]]
    out2, cases2 = settle_starts([Span(0.30, 0.88), Span(0.88, 1.30)], kept2, audio, 0.02)
    assert cases2[1] == "join" and out2[1].start == pytest.approx(0.88)


def test_settle_starts_trusts_charsiu_within_tolerance_when_voicing_is_continuous():
    from pronunciationcoach.boundaries import Span
    from pronunciationcoach.pipeline import LATE_TOL_S, settle_starts

    audio = np.concatenate([silence(0.3), tone(1.0), silence(0.5)])  # vowel into vowel, nothing to see
    kept = [[seg(19), seg(37, 39)], [seg(45), seg(60)]]  # word 2's first spike at 0.90 -> spike onset 0.80
    out, cases = settle_starts([Span(0.30, 0.84), Span(0.84, 1.30)], kept, audio, 0.02)
    assert cases[1] == "join" and out[1].start == pytest.approx(0.84)  # not pulled back into the vowel
    out2, _ = settle_starts([Span(0.30, 0.95), Span(0.95, 1.30)], kept, audio, 0.02)
    assert out2[1].start == pytest.approx(0.80 + LATE_TOL_S)  # ...but Charsiu cannot start arbitrarily late


def test_playback_pads_extend_only_through_quiet_audio():
    import os

    os.environ.setdefault("PC_OPEN_BROWSER", "0")
    from app import PAD_BEFORE_S, clip

    audio = np.concatenate([silence(0.2), tone(0.3), tone(0.3, 0.3)])  # word 0.20-0.50, a loud neighbour right after
    _, out = clip(audio, 0.20, 0.50, SAMPLE_RATE)
    assert len(out) / SAMPLE_RATE == pytest.approx(0.30 + PAD_BEFORE_S, abs=0.006)  # silence before: yes; the neighbour: no
