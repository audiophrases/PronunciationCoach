"""Playback policy and safety, independent of speech model inference."""
import numpy as np
import pytest

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.boundaries import Span
from pronunciationcoach.chunks import build_chunks, safe_spans


def audio(duration=3):
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    x = (.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x[:int(.2 * SAMPLE_RATE)] = 0
    x[-int(.2 * SAMPLE_RATE):] = 0
    return x


def chunks(text, words, bounds, **kw):
    return build_chunks(text, words.split(), [Span(*s) for s in bounds], audio(), **kw)


def test_can_you_and_hear_me_are_two_separate_pairs():
    result = chunks("Can you hear me?", "Can you hear me", [(0.2, .42), (.42, .49), (.49, .7), (.7, .99)])
    assert [c.members for c in result] == [[0, 1], [2, 3]]
    assert [c.text for c in result] == ["Can you", "hear me?"]
    assert [(c.span.start, c.span.end) for c in result] == [(.2, .49), (.49, .99)]


@pytest.mark.parametrize("text", ["the store", "to school", "I want", "and then", "is good", "hear me", "can you", "her book"])
def test_function_word_categories_pair_with_a_neighbor(text):
    result = chunks(text, text, [(.2, .35), (.35, .6)])
    assert [c.members for c in result] == [[0, 1]]


def test_ordinary_function_word_run_still_prefers_pairs():
    words = "I want to go to the store"
    spans = [(.2 + i * .15, .35 + i * .15) for i in range(7)]
    result = chunks(words, words, spans)
    assert [c.text for c in result] == ["I want", "to go", "to", "the store"]
    assert [i for c in result for i in c.members] == list(range(7))
    assert all(len(c.members) <= 2 for c in result)


def test_no_merging_plain_content_words_or_main_verb_have():
    for text in ("bright birds", "have books", "do work"):
        assert len(chunks(text, text, [(.2, .4), (.4, .6)])) == 2


def test_emphasized_function_word_stays_separate():
    assert len(chunks("we CAN go", "we CAN go", [(.2, .3), (.3, .7), (.7, .95)])) == 3


def test_main_verb_can_host_a_pronoun_and_a_quiet_slow_auxiliary_can_pair():
    assert [c.text for c in chunks("I have books", "I have books", [(.2, .35), (.35, .75), (.75, 1.)])] == ["I have", "books"]
    samples = audio()
    samples[int(.3 * SAMPLE_RATE):int(.7 * SAMPLE_RATE)] *= .1
    result = build_chunks("we can go", ["we", "can", "go"], [Span(.2, .3), Span(.3, .7), Span(.7, .95)], samples)
    assert [c.text for c in result] == ["we", "can go"]


def test_pause_inside_touching_crops_still_blocks_a_pair():
    samples = audio()
    samples[int(.44 * SAMPLE_RATE):int(.56 * SAMPLE_RATE)] = 0
    result = build_chunks("can you", ["can", "you"], [Span(.2, .5), Span(.5, .8)], samples)
    assert len(result) == 2 and result[0].blocked_after == "pause"


@pytest.mark.parametrize("text,bounds,extras", [
    ("can, you", [(.2, .4), (.4, .6)], []),
    ("can you", [(.2, .4), (.6, .8)], []),
    ("can you", [(.2, .4), (.4, .6)], [(.38, .43)]),
])
def test_punctuation_gaps_and_overlapping_extras_block_pairing(text, bounds, extras):
    assert len(chunks(text, "can you", bounds, extras=extras)) == 2


def test_missing_word_cannot_bridge_a_pause_or_play_parked_audio():
    result = chunks("can you hear", "can you hear", [(.2, .4), (8, 9), (.8, 1.1)], dropped=[False, True, False])
    assert [c.members for c in result] == [[0], [1], [2]]
    assert result[1].span is None
    all_missing = chunks("the book", "the book", [(3, 4), (4, 5)], dropped=[True, True])
    assert all(c.span is None for c in all_missing)


def test_missing_article_can_use_present_word_context():
    result = chunks("the book", "the book", [(4, 5), (.2, .5)], dropped=[True, False])
    assert [c.members for c in result] == [[0, 1]]
    assert result[0].span == Span(.2, .5)


def test_span_safety_is_bounded_monotone_and_does_not_reverse_invalid_crops():
    result = safe_spans([Span(-1, .5), Span(.6, .4), Span(.5, 8), Span(float("nan"), 9)], 1.0, [False] * 4)
    assert result == [Span(0, .5), Span(.6, .6), Span(.6, 1), Span(1, 1)]


def test_long_pair_is_not_created_and_switch_restores_single_words():
    assert len(chunks("the mountain", "the mountain", [(.2, .35), (.35, 2.2)])) == 2
    assert len(chunks("can you", "can you", [(.2, .35), (.35, .5)], enabled=False)) == 2


def test_empty_or_mismatched_inputs():
    assert build_chunks("", [], [], audio()) == []
    with pytest.raises(ValueError):
        build_chunks("a", ["a"], [], audio())
    assert len(chunks("something else", "can you", [(.2, .35), (.35, .5)])) == 2


def test_connected_question_and_want_to_stay_together():
    words = "what do you want to watch"
    bounds = [(.2 + i * .15, .35 + i * .15) for i in range(6)]
    result = chunks(words + "?", words, bounds)
    assert [c.text for c in result] == ["what do you", "want to", "watch?"]
    assert [c.members for c in result] == [[0, 1, 2], [3, 4], [5]]
    assert result[0].span.start == pytest.approx(.2)
    assert result[0].span.end == pytest.approx(.65)
    assert len(chunks(words + "?", words, bounds, enabled=False)) == 6


@pytest.mark.parametrize("text", ["how are you", "where did she", "what does he", "why would they", "what have you"])
def test_connected_question_pattern_is_not_just_one_fixed_phrase(text):
    result = chunks(text, text, [(.2, .35), (.35, .5), (.5, .65)])
    assert [c.members for c in result] == [[0, 1, 2]]


@pytest.mark.parametrize("boundary", [0, 1])
@pytest.mark.parametrize("barrier", ["punctuation", "gap", "extra"])
def test_three_word_candidate_checks_both_internal_boundaries(boundary, barrier):
    words = ["what", "do", "you"]
    bounds = [[.2, .35], [.35, .5], [.5, .65]]
    extras = []
    text = " ".join(words)
    if barrier == "punctuation":
        words[boundary] += ","
        text = " ".join(words)
    elif barrier == "gap":
        for i in range(boundary + 1, 3):
            bounds[i] = [t + .2 for t in bounds[i]]
    else:
        end = bounds[boundary][1]
        extras = [(end - .01, end + .02)]
    result = chunks(text, "what do you", bounds, extras=extras)
    assert all(len(c.members) < 3 for c in result)


@pytest.mark.parametrize("emphasized", [0, 1, 2])
def test_emphasis_on_any_member_prevents_the_three_word_group(emphasized):
    bounds, start = [], .2
    for i in range(3):
        end = start + (.4 if i == emphasized else .15)
        bounds.append((start, end))
        start = end
    assert all(len(c.members) < 3 for c in chunks("what do you", "what do you", bounds))


def test_three_word_exception_needs_live_words_and_respects_duration(monkeypatch):
    result = chunks("what do you", "what do you", [(.2, .35), (8, 9), (.35, .5)], dropped=[False, True, False])
    assert all(len(c.members) < 3 for c in result)
    # Isolate the duration cap from the independent emphasis heuristic.
    monkeypatch.setattr("pronunciationcoach.chunks.STRONG_S", 3.)
    result = chunks("what do you", "what do you", [(.2, .75), (.75, 1.3), (1.3, 1.85)])
    assert all(len(c.members) < 3 for c in result)
