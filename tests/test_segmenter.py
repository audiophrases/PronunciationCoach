"""Viterbi word alignment over frame posteriors, on synthetic posteriors (no model)."""

import numpy as np

from pronunciationcoach.segmenter import FRAME_S, MIN_FRAMES, OFFSET_S, CharsiuSegmenter, to_arpabet


class FakeSegmenter(CharsiuSegmenter):
    """The aligner without the model: a 4-symbol vocabulary."""

    def __init__(self):
        self.vocab = {"[SIL]": 0, "A": 1, "B": 2, "[UNK]": 3}
        self.sil = 0


def posteriors(labels: list[int], n_classes: int = 4, p: float = 0.9) -> np.ndarray:
    probs = np.full((len(labels), n_classes), (1 - p) / (n_classes - 1))
    probs[np.arange(len(labels)), labels] = p
    return np.log(probs)


def frames(span):
    return round((span.start + OFFSET_S) / FRAME_S), round((span.end + OFFSET_S) / FRAME_S)


def test_words_get_their_frames_and_silence_between_is_excluded():
    seq = [0] * 10 + [1] * 8 + [2] * 6 + [0] * 5 + [2] * 7 + [0] * 10  # sil A B sil B sil
    spans = FakeSegmenter().align(posteriors(seq), [["A", "B"], ["B"]])
    assert frames(spans[0]) == (10, 24)
    assert frames(spans[1]) == (29, 36)


def test_one_phone_word_between_silences_is_not_skipped():
    # regression: a single-phone word after an optional silence used to be hoppable
    seq = [0] * 5 + [1] * 6 + [0] * 4 + [2] * 5 + [0] * 4 + [1] * 6 + [0] * 5
    spans = FakeSegmenter().align(posteriors(seq), [["A"], ["B"], ["A"]])
    assert all(sp.duration >= MIN_FRAMES * FRAME_S - 1e-9 for sp in spans)
    assert frames(spans[1]) == (15, 20)


def test_extras_and_windows_keep_a_repeat_out_of_the_neighbour():
    # "A B" said as "A B B": the second B is an extra; the first word must not absorb it
    seq = [0] * 5 + [1] * 6 + [2] * 6 + [0] * 3 + [2] * 6 + [0] * 5
    seg = FakeSegmenter()
    t = lambda f: f * FRAME_S - OFFSET_S
    windows = [(t(0), t(18)), (t(4), t(20))]
    extras = [(t(20), t(26))]
    spans = seg.align(posteriors(seq), [["A"], ["B"]], windows, extras)
    assert frames(spans[1])[1] <= 20, spans


def test_ipa_to_arpabet_covers_espeak_english():
    assert to_arpabet(["ð", "ɪ", "s"]) == ["DH", "IH", "S"]
    assert to_arpabet(["w", "ɔː", "ɾ", "ɚ"]) == ["W", "AO", "T", "ER"]
    assert to_arpabet(["k", "j", "ʊɹ", "i"]) == ["K", "Y", "UH", "R", "IY"]
    assert to_arpabet(["ʡ"]) == []  # unknown symbols vanish rather than crash
