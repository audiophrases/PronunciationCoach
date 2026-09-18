"""Model-free checks of the alignment and scoring maths on a hand-built emission grid."""

import numpy as np
import pytest

from pronunciationcoach.align import forced_align, greedy_decode
from pronunciationcoach.engine import Emissions
from pronunciationcoach.scoring import category, score_segment, score_words

LABELS = ["<pad>", "a", "b", "c"]
BLANK = 0


def grid(frames: list[dict[int, float]]) -> Emissions:
    """Build log-probs from per-frame {phone_id: prob}; the rest of the mass goes to blank."""
    probs = np.full((len(frames), len(LABELS)), 1e-6, dtype=np.float32)
    for t, spec in enumerate(frames):
        for idx, p in spec.items():
            probs[t, idx] = p
        probs[t, BLANK] = max(1e-6, 1.0 - sum(spec.values()))
    probs /= probs.sum(axis=1, keepdims=True)
    return Emissions(np.log(probs), LABELS, BLANK, 20.0)


def test_greedy_decode_collapses_repeats_and_blanks():
    em = grid([{1: 0.9}, {1: 0.9}, {}, {1: 0.9}, {2: 0.9}, {2: 0.9}, {}])
    phones = [(s.phone, s.start, s.end) for s in greedy_decode(em)]
    assert phones == [("a", 0, 2), ("a", 3, 4), ("b", 4, 6)]


def test_forced_align_finds_the_obvious_path():
    em = grid([{}, {1: 0.9}, {}, {2: 0.9}, {}, {3: 0.9}, {}])
    segs = forced_align(em, [1, 2, 3])
    assert [(s.phone, s.start) for s in segs] == [("a", 1), ("b", 3), ("c", 5)]


def test_forced_align_rejects_audio_shorter_than_the_sentence():
    em = grid([{1: 0.9}, {2: 0.9}])
    with pytest.raises(ValueError):
        forced_align(em, [1, 2, 3])


def test_gop_is_zero_when_expected_phone_wins_and_negative_otherwise():
    em = grid([{1: 0.8, 2: 0.1}])
    good = score_segment(em, greedy_decode(em)[0])
    assert good.gop == pytest.approx(0.0)
    assert good.heard == "a"
    assert good.posterior == pytest.approx(0.8 / 0.9, abs=1e-3)

    # The learner said "b" where "a" was expected: align "a" onto that frame.
    em = grid([{1: 0.1, 2: 0.8}])
    seg = forced_align(em, [1])[0]
    bad = score_segment(em, seg)
    assert bad.expected == "a" and bad.heard == "b"
    assert bad.gop == pytest.approx(np.log(0.1) - np.log(0.8))
    assert category(bad.gop) == "off"


def test_score_words_groups_phones_by_word():
    em = grid([{1: 0.9}, {}, {2: 0.9}, {}, {3: 0.9}])
    segs = forced_align(em, [1, 2, 3])
    words = score_words(em, segs, [("ab", 2), ("c", 1)])
    assert [(w.word, [p.expected for p in w.phones]) for w in words] == [("ab", ["a", "b"]), ("c", ["c"])]
    assert all(w.category == "good" for w in words)
