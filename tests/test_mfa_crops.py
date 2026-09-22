"""MFA is the crop aligner, but only where the sentence is what was actually said.

Forced alignment cannot refuse: it emits an interval for every transcript word
whatever the learner did, so an omitted or mangled word comes back as confident,
wrong timings rather than an error. These tests pin the gates that catch that.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from pronunciationcoach.align import Segment
from pronunciationcoach.boundaries import Span
from pronunciationcoach.scoring import PhoneScore, WordScore

FRAME_MS = 20.0
AUDIO = np.ones(16000, dtype=np.float32)


def sentence():
    """Two words whose spikes sit at 0.24-0.34 s and 0.54-0.70 s."""
    words = [WordScore(w, [PhoneScore(p, s, t, 0., .9, p)])
             for w, p, s, t in (("one", "A", .24, .34), ("two", "B", .54, .70))]
    phones = [SimpleNamespace(word="one", phones=["A"]), SimpleNamespace(word="two", phones=["B"])]
    segments = [Segment("A", 1, 12, 17, .9), Segment("B", 2, 27, 35, .9)]
    return words, phones, segments


def locate(monkeypatch, spans, **kwargs):
    from pronunciationcoach import pipeline, segmenter

    words, phones, segments = sentence()
    monkeypatch.setattr(segmenter, "get_segmenter",
                        lambda: (_ for _ in ()).throw(RuntimeError("offline")))  # so the fallback is visible
    if spans is not None:
        monkeypatch.setattr(pipeline, "mfa_spans", lambda *a, **k: [Span(*s) for s in spans])
    return pipeline.locate_words(AUDIO, phones, words, segments, kwargs.pop("extra", []), FRAME_MS,
                                 None, dropped_words=kwargs.pop("dropped_words", [False, False]), **kwargs)


def test_mfa_supplies_the_crops_when_the_sentence_matches(monkeypatch):
    spans, source, cases, reject = locate(monkeypatch, [(.16, .34), (.46, .70)])
    assert source == "mfa" and reject == ""
    assert len(spans) == 2 and len(cases) == 2


def test_a_word_shifted_off_its_spikes_rejects_the_whole_alignment(monkeypatch):
    # An omitted word moves every later boundary, so one bad word discards all of them.
    spans, source, _, reject = locate(monkeypatch, [(.16, .34), (.90, 1.10)])
    assert source == "spikes" and "disagrees with the spikes" in reject


def test_a_hesitation_cannot_extend_a_word_past_its_own_spikes(monkeypatch):
    # MFA has no wildcard state and absorbs trailing speech; its spikes bound it.
    spans, source, _, _ = locate(monkeypatch, [(.16, .34), (.46, 3.00)])
    assert source == "mfa"
    assert spans[1].end == pytest.approx(.70 + .25, abs=1e-6)  # last spike + WINDOW_AFTER_S


def test_an_unavailable_worker_falls_back_instead_of_failing(monkeypatch):
    from pronunciationcoach import pipeline

    monkeypatch.setattr(pipeline, "mfa_spans",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("worker gone")))
    spans, source, _, reject = locate(monkeypatch, None)
    assert source == "spikes" and "worker gone" in reject
    assert len(spans) == 2


@pytest.mark.parametrize("kwargs,expected", [
    ({"dropped_words": [True, False]}, "was not said"),
    ({"extra": [SimpleNamespace(phones=["a", "b"], start=1, end=4)]}, "outside the sentence"),
])
def test_mfa_is_not_asked_when_the_sentence_is_not_what_was_said(monkeypatch, kwargs, expected):
    spans, source, _, reject = locate(monkeypatch, [(.16, .34), (.46, .70)], **kwargs)
    assert source == "spikes" and expected in reject


def test_mfa_is_not_asked_for_a_word_the_listener_missed(monkeypatch):
    from pronunciationcoach import pipeline, segmenter

    words, phones, segments = sentence()
    words[1].understood = False
    monkeypatch.setattr(segmenter, "get_segmenter", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(pipeline, "mfa_spans", lambda *a, **k: [Span(.16, .34), Span(.46, .70)])
    _, source, _, reject = pipeline.locate_words(AUDIO, phones, words, segments, [], FRAME_MS,
                                                 None, dropped_words=[False, False])
    assert source == "spikes" and "not recognised" in reject


def test_an_inserted_vowel_keeps_the_crop_away_from_mfa(monkeypatch):
    # "e-speak": the dictionary has no such word, so MFA would place the vowel anywhere.
    from pronunciationcoach import pipeline

    words, phones, _ = sentence()
    words[0].insertions.append(PhoneScore("", .20, .24, -10., .5, "e", inserted="before"))
    assert "inserted vowel" in pipeline.mfa_reject_reason(words, phones, [], [False, False])


def test_disabling_mfa_is_honoured(monkeypatch):
    from pronunciationcoach import pipeline

    monkeypatch.setenv("PC_MFA", "0")
    words, phones, _ = sentence()
    assert pipeline.mfa_reject_reason(words, phones, [], [False, False]) == "disabled"


def test_out_of_vocabulary_words_are_caught_before_aligning(monkeypatch):
    from pronunciationcoach import mfa, pipeline

    monkeypatch.setattr(mfa, "lexicon", lambda: frozenset({"one"}))
    words, phones, _ = sentence()
    assert "not in the dictionary: two" in pipeline.mfa_reject_reason(words, phones, [], [False, False])


def test_the_calibration_offset_moves_starts_but_never_ends(monkeypatch):
    """Moving ends earlier clips the word's final sound, the most audible crop error."""
    from pronunciationcoach import mfa, mfa_worker, pipeline

    monkeypatch.setattr(mfa, "OFFSET_S", .04)
    monkeypatch.setattr(mfa_worker, "align",
                        lambda *a, **k: ({"duration": 1.0, "tiers": {"words": {"entries": [
                            [.5, .8, "one"]]}}}, .01))
    spans = pipeline.real_mfa_spans(AUDIO, ["one"], 1.0)  # conftest blocks the patched name
    assert spans[0].start == pytest.approx(.5 - mfa.OFFSET_S)
    assert spans[0].end == pytest.approx(.8)
