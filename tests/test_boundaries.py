"""Word/phone crop boundaries on synthetic signals (no model needed)."""

import numpy as np
import pytest

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.align import Segment
from pronunciationcoach.boundaries import SPIKE_LAG_S, word_spans

FRAME_MS = 20.0


def seg(start_frame: int, end_frame: int | None = None) -> Segment:
    return Segment("x", 1, start_frame, end_frame or start_frame + 1, 0.0)


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
