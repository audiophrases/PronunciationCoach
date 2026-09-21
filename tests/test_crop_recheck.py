"""Known frame labels exercise re-alignment, ownership and safe playback changes."""
import numpy as np
import pytest

from pronunciationcoach.boundaries import Span
from pronunciationcoach.chunks import PlaybackChunk
from pronunciationcoach.crop_recheck import CropEvidence, _local_alignment, recheck_chunks
from pronunciationcoach.segmenter import CharsiuSegmenter


@pytest.fixture
def evidence():
    seg = object.__new__(CharsiuSegmenter)
    seg.vocab = {"[SIL]": 0, "A": 1, "B": 2, "[UNK]": 3}
    seg.sil = 0
    # A: .16-.36, B: .46-.66. The original B crop loses 40 ms at
    # each edge; both sides have unclaimed audio with strong B evidence.
    labels = [0]*20 + [1]*20 + [0]*10 + [2]*20 + [0]*30
    probs = np.full((len(labels), 4), .01)
    probs[np.arange(len(labels)), labels] = .97
    return CropEvidence(seg, np.log(probs), [["A"], ["B"]], [(0., .42), (.4, .8)],
                        [Span(.16, .36), Span(.46, .66)], [Span(.24, .34), Span(.54, .70)], [])


def chunks():
    return [PlaybackChunk([0], "one", Span(.16, .36), "single"),
            PlaybackChunk([1], "two", Span(.50, .62), "single")]


def test_contextual_second_pass_recovers_known_frames_without_mutating_word_crops(evidence):
    original = chunks()
    actual, checks = recheck_chunks(original, evidence, 1., [False]*2, apply=True)
    assert actual[1].span.start == pytest.approx(.46)
    assert actual[1].span.end == pytest.approx(.66)
    assert original[1].span == Span(.50, .62)
    assert evidence.raw_spans[1] == Span(.46, .66)
    assert checks[0].status == "adjusted"
    assert checks[0].after == actual[1].span


def test_audit_proposes_same_change_but_does_not_apply_it(evidence):
    original = chunks()
    actual, checks = recheck_chunks(original, evidence, 1., [False]*2)
    assert actual == original
    assert checks[0].status == "proposed"
    assert checks[0].accepted.start == pytest.approx(.46)
    assert checks[0].after == original[1].span


@pytest.mark.parametrize("blocked", ["extra", "missing", "unsupported"])
def test_uncertain_ownership_keeps_original(evidence, blocked):
    dropped = [False]*2
    if blocked == "extra":
        evidence.extras = [(.38, .47)]
    elif blocked == "missing":
        dropped[0] = True
    else:
        evidence.phones[0] = ["UNKNOWN"]
    actual, checks = recheck_chunks(chunks(), evidence, 1., dropped, apply=True)
    assert actual == chunks()
    assert checks[0].accepted is None
    if blocked == "extra":
        assert checks[0].proposed is not None  # still inspect with context, never apply


def test_extension_cannot_take_neighboring_chunk_audio(evidence):
    original = chunks()
    original[0].span.end = .50
    actual, checks = recheck_chunks(original, evidence, 1., [False]*2, apply=True)
    assert actual[1].span.start == .50
    assert "neighboring playback" in checks[-1].detail


def test_disagreement_between_contexts_keeps_original(evidence, monkeypatch):
    from pronunciationcoach import crop_recheck
    monkeypatch.setattr(crop_recheck, "_local_alignment", lambda e, a, b, n:
                        [Span(.46 if n == 1 else .4, .66 if n == 1 else .75)])
    actual, checks = recheck_chunks(chunks(), evidence, 1., [False]*2, apply=True)
    assert actual == chunks()
    assert "searches disagree" in checks[0].detail


def test_alignment_agreement_without_phone_support_is_not_enough(evidence):
    evidence.log_probs[:] = np.log(.25)
    actual, checks = recheck_chunks(chunks(), evidence, 1., [False]*2, apply=True)
    assert actual == chunks()
    assert checks[0].accepted is None


def test_recheck_never_trims_a_quiet_onset(evidence):
    original = chunks()
    original[1].span = Span(.40, .70)
    evidence.spikes[1].end = .80  # trigger inspection
    actual, checks = recheck_chunks(original, evidence, 1., [False]*2, apply=True)
    assert actual == original
    assert checks[0].proposed.start > original[1].span.start


def test_local_frame_origin_is_translated_once(evidence):
    # Shift the entire utterance 2 s so slicing starts well after frame zero.
    silence = np.tile(np.log([.97, .01, .01, .01]), (200, 1))
    evidence.log_probs = np.concatenate([silence, evidence.log_probs])
    evidence.windows = [(a+2, b+2) for a, b in evidence.windows]
    spans = _local_alignment(evidence, 1, 1, 1)
    assert spans[0].start == pytest.approx(2.46)
    assert spans[0].end == pytest.approx(2.66)


def test_infeasible_alignment_fails_instead_of_returning_fabricated_crops(evidence):
    with pytest.raises(ValueError, match="feasible"):
        evidence.segmenter.align(evidence.log_probs[:4], [["A"], ["B"]])
    with pytest.raises(ValueError, match="frames"):
        evidence.segmenter.align(evidence.log_probs[:0], [["A"]])


def test_recheck_budget_keeps_original_and_reports_skip(evidence, monkeypatch):
    from pronunciationcoach import crop_recheck
    monkeypatch.setattr(crop_recheck, "MAX_REGIONS", 0)
    actual, checks = recheck_chunks(chunks(), evidence, 1., [False]*2, apply=True)
    assert actual == chunks()
    assert checks[0].status == "skipped"


def test_locate_words_retains_evidence_without_another_model_call(evidence, monkeypatch):
    from types import SimpleNamespace
    from pronunciationcoach import pipeline, segmenter
    from pronunciationcoach.align import Segment
    from pronunciationcoach.scoring import PhoneScore, WordScore
    calls = []
    monkeypatch.setattr(evidence.segmenter, "frame_log_probs", lambda audio: calls.append(len(audio)) or evidence.log_probs)
    monkeypatch.setattr(segmenter, "get_segmenter", lambda: evidence.segmenter)
    monkeypatch.setattr(segmenter, "to_arpabet", lambda phones: phones)
    words = [WordScore(w, [PhoneScore(p, s, t, 0., .9, p)])
             for w, p, s, t in (("one", "A", .24, .34), ("two", "B", .54, .70))]
    phones = [SimpleNamespace(phones=["A"]), SimpleNamespace(phones=["B"])]
    segments = [Segment("A", 1, 12, 17, .9), Segment("B", 2, 27, 35, .9)]
    retained = []
    spans, source, _ = pipeline.locate_words(np.ones(16000, dtype=np.float32), phones, words, segments, [], 20., retained)
    assert source == "charsiu" and len(retained) == 1 and len(spans) == 2
    recheck_chunks(chunks(), retained[0], 1., [False]*2)
    assert calls == [16000]


def test_failed_baseline_alignment_leaves_no_recheck_evidence(evidence, monkeypatch):
    from types import SimpleNamespace
    from pronunciationcoach import pipeline, segmenter
    from pronunciationcoach.align import Segment
    from pronunciationcoach.scoring import PhoneScore, WordScore
    monkeypatch.setattr(segmenter, "get_segmenter", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    retained = []
    spans, source, _ = pipeline.locate_words(np.ones(16000, dtype=np.float32), [SimpleNamespace(phones=["a"])],
        [WordScore("one", [PhoneScore("a", .24, .34, 0., .9, "a")])], [Segment("a", 1, 12, 17, .9)], [], 20., retained)
    assert source == "spikes" and len(spans) == 1 and retained == []
