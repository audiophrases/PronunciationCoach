"""The default assessment path applies verified crops and retains scoring anchors."""
from types import SimpleNamespace

import numpy as np
import pytest

from pronunciationcoach.align import Segment
from pronunciationcoach.asr import HeardWord
from pronunciationcoach.boundaries import Span
from pronunciationcoach.engine import Emissions
from pronunciationcoach.scoring import PhoneScore, WordScore


@pytest.mark.parametrize("mode,listener,should_call,should_apply", [
    (None, True, True, True), ("audit", True, True, False),
    ("0", True, False, False), (None, False, False, False),
])
def test_assess_routes_verification_with_independent_scoring_anchors(monkeypatch, mode, listener, should_call, should_apply):
    from pronunciationcoach import pipeline
    monkeypatch.setenv("PC_NATIVE_REF", "0")
    monkeypatch.setenv("PC_LISTENER", "1" if listener else "0")
    if mode is None:
        monkeypatch.delenv("PC_CROP_RECHECK", raising=False)
    else:
        monkeypatch.setenv("PC_CROP_RECHECK", mode)
    em = Emissions(np.zeros((50, 2)), ["<pad>", "a"], 0, 20.)
    engine = SimpleNamespace(emissions=lambda x: em, phone_id=lambda p: 1, unk_id=0)
    monkeypatch.setattr(pipeline, "get_engine", lambda _: engine)
    monkeypatch.setattr(pipeline, "greedy_decode", lambda _: [])
    monkeypatch.setattr(pipeline, "transcribe_words", lambda _: ("I", [HeardWord("I", .9, .2, .4)]))
    monkeypatch.setattr(pipeline, "text_to_phones", lambda *_: [SimpleNamespace(word="I", phones=["a"])])
    monkeypatch.setattr(pipeline, "align_words", lambda *_: SimpleNamespace(segments=[Segment("a", 1, 14, 20, .9)], extra=[]))
    word = WordScore("I", [PhoneScore("a", .28, .4, 0., .9, "a")])
    monkeypatch.setattr(pipeline, "score_words", lambda *_: [word])
    monkeypatch.setattr(pipeline, "acceptances", lambda *_: [])
    monkeypatch.setattr(pipeline, "attach_insertions", lambda *_: [])
    # Simulate Charsiu being unavailable: verification must still receive the
    # phoneme scorer's anchors instead of silently dropping target protection.
    monkeypatch.setattr(pipeline, "locate_words", lambda *_: ([Span(.1, .45)], "spikes", []))
    calls = []

    def verify(chunks, audio, heard, **kwargs):
        calls.append(kwargs)
        assert kwargs["anchors"] == [Span(.28, .4)]
        assert kwargs["evidence"] is None
        if kwargs["apply"]:
            chunks[0].span = Span(.2, .45)
            chunks[0].pad_before = False
        return chunks, []

    monkeypatch.setattr(pipeline, "verify_chunks", verify)
    result = pipeline.assess(np.ones(16000, dtype=np.float32), 16000, "I")
    assert bool(calls) is should_call
    assert result.spans == [Span(.1, .45)]
    assert result.words == [word]
    if should_call:
        assert calls[0]["apply"] is should_apply
    assert result.chunks[0].span.start == (.2 if should_apply else .1)
    assert result.chunks[0].pad_before is not should_apply
