"""Recognition can trim corroborated edge words without shaving pronunciation errors."""

from types import SimpleNamespace

import numpy as np
import pytest

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.asr import HeardWord
from pronunciationcoach.boundaries import Span
from pronunciationcoach.chunks import PlaybackChunk
from pronunciationcoach.crop_verify import verify_chunks
from pronunciationcoach.playback import clip


def heard(*words):
    """Positive, ordered timestamps in seconds; optionally supply probability."""
    return [HeardWord(text, rest[0] if rest else .95, start, end)
            for text, start, end, *rest in words]


class Recognizer:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.clips = []

    def __call__(self, audio):
        self.clips.append(audio.copy())
        if not self.responses:
            raise AssertionError("Unexpected additional recognition")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def audio():
    # Continuous sound prevents automatic quiet padding from obscuring the
    # timestamp origin in these recognizer behavior tests.
    return np.full(2 * SAMPLE_RATE, .2, dtype=np.float32)


@pytest.fixture
def target():
    return PlaybackChunk([1], "I", Span(.2, .9), "single word")


@pytest.fixture
def context():
    return heard(("If", .2, .34), ("I", .38, .78), ("want", .95, 1.3))


def leading_extra():
    return "If I", heard(("If", .0, .14), ("I", .18, .58))


def corrected_i():
    return "I", heard(("I", .02, .42))


def test_detect_trim_and_recheck_i_from_if_i(audio, target, context):
    recognizer = Recognizer(leading_extra(), corrected_i())
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)

    assert result[0].span.start == pytest.approx(.36)
    assert result[0].span.end == .9
    assert result[0].pad_before is False
    assert result[0].pad_after is True
    assert result[0].members == target.members
    assert result[0].text == target.text
    assert target.span == Span(.2, .9), "Verification must not mutate the original crop"
    assert target.pad_before is True
    assert checks[0].status == "trimmed"
    assert checks[0].heard_before == "If I"
    assert checks[0].heard_after == "I"
    assert checks[0].after == result[0].span
    assert len(checks[0].attempts) == 1
    np.testing.assert_array_equal(recognizer.clips[0], clip(audio, .2, .9)[1])
    np.testing.assert_array_equal(recognizer.clips[1], clip(audio, .36, .9, pad_before=False)[1])


def test_both_leading_and_trailing_extras_disable_both_pads(audio):
    target = PlaybackChunk([1], "I", Span(.2, 1.4), "single word")
    context = heard(("if", .22, .4), ("I", .44, .86), ("want", .9, 1.2))
    recognizer = Recognizer(
        ("if I want", heard(("if", .02, .2), ("I", .24, .66), ("want", .7, 1.0))),
        ("I", heard(("I", .02, .44))),
    )
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert checks[0].status == "trimmed"
    assert result[0].span.start == pytest.approx(.42)
    assert result[0].span.end == pytest.approx(.88)
    assert result[0].pad_before is result[0].pad_after is False


def test_trailing_extra_preserves_original_before_padding(audio):
    target = PlaybackChunk([0], "I", Span(.2, 1), "single word")
    context = heard(("I", .22, .62), ("want", .66, .94))
    recognizer = Recognizer(
        ("I want", heard(("I", .02, .42), ("want", .46, .74))),
        ("I", heard(("I", .02, .42))),
    )
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert checks[0].status == "trimmed"
    assert result[0].span.start == .2
    assert result[0].span.end == pytest.approx(.64)
    assert result[0].pad_before is True
    assert result[0].pad_after is False


def test_exact_match_leaves_crop_unchanged_without_context(audio, target):
    recognizer = Recognizer(("I.", heard(("I.", .04, .55, .3))))
    result, checks = verify_chunks([target], audio, [], recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "matched"
    assert len(recognizer.clips) == 1


def test_audit_returns_verified_proposal_without_changing_playback(audio, target, context):
    recognizer = Recognizer(leading_extra(), corrected_i())
    result, checks = verify_chunks([target], audio, context, apply=False, recognize=recognizer)
    assert result == [target]
    assert result[0].pad_before is result[0].pad_after is True
    assert checks[0].status == "proposed"
    assert checks[0].proposed.start == pytest.approx(.36)
    assert checks[0].after == checks[0].before == target.span


def test_zero_duration_extra_is_rejected_even_with_high_probability(audio, target, context):
    recognizer = Recognizer(("If I", heard(("If", .0, .0, .999), ("I", .18, .58))))
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "no acoustic interval" in checks[0].detail
    assert len(recognizer.clips) == 1


@pytest.mark.parametrize("transcript,words", [
    ("eye want", [("eye", .02, .3), ("want", .32, .58)]),
    ("I I", [("I", .02, .25), ("I", .28, .58)]),
])
def test_missing_or_repeated_target_is_not_corrected(audio, target, context, transcript, words):
    recognizer = Recognizer((transcript, heard(*words)))
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "no edge-only correction" in checks[0].detail
    assert len(recognizer.clips) == 1


def test_internal_extra_is_not_removed_to_force_a_match(audio):
    target = PlaybackChunk([0, 1], "I want", Span(.2, 1), "pair")
    words = heard(("I", .02, .2), ("really", .22, .46), ("want", .48, .72))
    recognizer = Recognizer(("I really want", words))
    result, checks = verify_chunks([target], audio, [], recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert len(recognizer.clips) == 1


@pytest.mark.parametrize("context", [[], heard(("if", 1.2, 1.34), ("I", 1.38, 1.78))])
def test_extra_requires_full_sentence_corroboration_at_same_position(audio, target, context):
    recognizer = Recognizer(leading_extra())
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "not corroborated" in checks[0].detail
    assert len(recognizer.clips) == 1


def test_failed_recheck_keeps_original_crop_and_padding(audio, target, context):
    recognizer = Recognizer(leading_extra(), ("I want", heard(("I", .02, .23), ("want", .25, .46))))
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "did not recheck" in checks[0].detail
    assert checks[0].after == target.span
    assert len(checks[0].attempts) == 1
    assert len(recognizer.clips) == 2


def test_sentence_estimate_gets_its_own_recheck_if_crop_estimate_still_has_extra(audio, target):
    context = heard(("if", .24, .38), ("I", .42, .82))
    recognizer = Recognizer(
        leading_extra(),
        ("if I", heard(("if", .0, .04), ("I", .06, .46))),
        corrected_i(),
    )
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert checks[0].status == "trimmed"
    assert result[0].span.start == pytest.approx(.4)
    assert len(checks[0].attempts) == 2
    assert checks[0].attempts[0].span.start == pytest.approx(.36)
    assert checks[0].attempts[1].span.start == pytest.approx(.4)
    assert len(recognizer.clips) == 3


def test_exact_recheck_text_with_zero_duration_target_does_not_authorize_trim(audio, target, context):
    recognizer = Recognizer(leading_extra(), ("I", heard(("I", .0, .0, .99))))
    result, checks = verify_chunks([target], audio, context, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "unusable timestamps" in checks[0].detail


def test_preserves_first_phone_anchor_even_when_transcript_suggests_trim(audio, target, context):
    evidence = SimpleNamespace(spikes=[Span(.22, .3), Span(.3, .7)])
    recognizer = Recognizer(leading_extra())
    result, checks = verify_chunks([target], audio, context, evidence=evidence, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "remove target sounds" in checks[0].detail
    assert len(recognizer.clips) == 1


def test_preserves_last_phone_anchor_when_trailing_word_is_recognized(audio):
    target = PlaybackChunk([0], "I", Span(.2, 1), "single word")
    context = heard(("I", .22, .62), ("want", .66, .94))
    evidence = SimpleNamespace(spikes=[Span(.3, .8)])
    recognizer = Recognizer(("I want", heard(("I", .02, .42), ("want", .46, .74))))
    result, checks = verify_chunks([target], audio, context, evidence=evidence, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "uncertain"
    assert "remove target sounds" in checks[0].detail
    assert len(recognizer.clips) == 1


def test_supported_trim_preserves_first_and_last_phone_anchors(audio, target, context):
    evidence = SimpleNamespace(spikes=[Span(.22, .3), Span(.44, .76)])
    recognizer = Recognizer(leading_extra(), corrected_i())
    result, checks = verify_chunks([target], audio, context, evidence=evidence, recognize=recognizer)
    assert checks[0].status == "trimmed"
    assert result[0].span.start <= evidence.spikes[1].start - .02
    assert result[0].span.end >= evidence.spikes[1].end - .02


def test_protected_pronunciation_is_skipped(audio, target, context):
    recognizer = Recognizer()
    result, checks = verify_chunks([target], audio, context, protected={1}, recognize=recognizer)
    assert result == [target]
    assert checks[0].status == "skipped"
    assert recognizer.clips == []


def test_crop_budget_prioritizes_suspect_and_keeps_remaining_original(audio, monkeypatch):
    import pronunciationcoach.crop_verify as verifier

    monkeypatch.setattr(verifier, "MAX_CROPS", 2)
    chunks = [PlaybackChunk([i], "I", Span(.2 + i*.4, .5 + i*.4), "single word") for i in range(3)]
    response = ("I", heard(("I", .02, .26)))
    recognizer = Recognizer(response, response)
    result, checks = verify_chunks(chunks, audio, [], suspicious={2}, recognize=recognizer)
    assert result == chunks
    assert [c.status for c in checks] == ["matched", "skipped", "matched"]
    assert "limit" in checks[1].detail
    assert len(recognizer.clips) == 2


def test_recognizer_failure_keeps_all_crops_and_stops_repeated_calls(audio, target, context):
    next_target = PlaybackChunk([2], "want", Span(.95, 1.3), "single word")
    recognizer = Recognizer(RuntimeError("recognizer unavailable"))
    result, checks = verify_chunks([target, next_target], audio, context, recognize=recognizer)
    assert result == [target, next_target]
    assert [c.status for c in checks] == ["failed", "skipped"]
    assert len(recognizer.clips) == 1
