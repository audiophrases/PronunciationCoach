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


def test_equivalent_variants_are_not_penalised():
    from pronunciationcoach.scoring import equivalents

    assert equivalents("ɐ") == {"ə", "ɐ", "ᵻ"} and equivalents("t") == {"t"}
    labels = ["<pad>", "ə", "ɐ", "t"]
    probs = np.array([[0.05, 0.69, 0.25, 0.01]], dtype=np.float32)
    em = Emissions(np.log(probs), labels, 0, 20.0)
    seg = forced_align(em, [2])[0]  # expected ɐ, but the model prefers ə
    score = score_segment(em, seg)
    assert score.gop == pytest.approx(0.0)
    assert score.heard == "ɐ"
    assert score.posterior == pytest.approx((0.69 + 0.25) / 0.95, abs=1e-3)


def test_wildcard_absorbs_a_repeated_word():
    from pronunciationcoach.align import align_words

    # "a b" said as "a b a b": the wildcard should swallow one repeat, not smear the phones.
    em = grid([{1: 0.9}, {}, {2: 0.9}, {}, {}, {1: 0.9}, {}, {2: 0.9}, {}])
    result = align_words(em, [[1], [2]])
    assert [s.phone for s in result.segments] == ["a", "b"]
    assert len(result.segments) == 2
    assert result.extra and [p for run in result.extra for p in run.phones] == ["a", "b"]


def test_wildcard_leaves_a_clean_sentence_alone():
    from pronunciationcoach.align import align_words

    em = grid([{}, {1: 0.9}, {}, {2: 0.9}, {}, {3: 0.9}, {}])
    result = align_words(em, [[1, 2], [3]])
    assert [(s.phone, s.start) for s in result.segments] == [("a", 1), ("b", 3), ("c", 5)]
    assert result.extra == []


def test_dropped_phone_is_reported_as_not_heard():
    from pronunciationcoach.scoring import mark_dropped

    # word "a c b": the c was never said; the aligner squeezes it onto a's frame where a still wins
    em = grid([{1: 0.9}, {1: 0.9}, {2: 0.9}])
    segs = forced_align(em, [1, 3, 2])
    scores = [score_segment(em, s) for s in segs]
    mark_dropped(scores, segs, em)
    assert [s.dropped for s in scores] == [False, True, False]
    assert scores[1].heard_label == "(not heard)"


def test_flap_accepts_t_but_t_does_not_accept_d():
    from pronunciationcoach.scoring import accepted

    assert accepted("ɾ") >= {"ɾ", "t", "d"}
    assert "ɾ" in accepted("t") and "d" not in accepted("t")
