"""Word feedback remains individual while learner and model play the same pair."""
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.boundaries import Span
from pronunciationcoach.chunks import build_chunks
from pronunciationcoach.engine import Emissions
from pronunciationcoach.pipeline import Assessment
from pronunciationcoach.scoring import PhoneScore, WordScore


@pytest.fixture
def assessment():
    text = "can you hear me?"
    samples = np.full(SAMPLE_RATE * 2, .2, dtype=np.float32)
    samples[:int(.2 * SAMPLE_RATE)] = 0
    samples[-int(.2 * SAMPLE_RATE):] = 0
    spans = [Span(.2, .4), Span(.4, .48), Span(.48, .7), Span(.7, .95)]
    words = [WordScore(word, [PhoneScore(phone, sp.start, sp.end, -3.0 if i == 1 else 0., .8,
                                       "n" if i == 1 else phone)], listener_p=.95, understood=True)
             for i, (word, phone, sp) in enumerate(zip(text.rstrip("?").split(), ["k", "j", "h", "m"], spans))]
    return Assessment(text, False, "en-us", 2., [], words, [], [],
                      Emissions(np.zeros((100, 2)), ["<pad>", "a"], 0, 20), samples,
                      spans=spans, chunks=build_chunks(text, [w.word for w in words], spans, samples))


@pytest.fixture
def ui(monkeypatch, assessment):
    import app
    monkeypatch.setattr(app, "assess", lambda *args: assessment)
    monkeypatch.setattr(app, "log_assessment", lambda *args: None)
    monkeypatch.setattr(app, "sound_guides", lambda w: [])
    monkeypatch.setattr(app, "timeline_figure", lambda r: None)
    monkeypatch.setattr(app, "posterior_heatmap", lambda *args: None)
    original = np.full(48000 * 2, .2, dtype=np.float32)
    result = app.run((48000, original), assessment.text, "American")
    return app, result


def test_selection_plays_pairs_but_keeps_word_colors_tips_and_original_sample_rate(ui, monkeypatch):
    app, result = ui
    state = result[3]
    assert [w["chunk"] for w in state["words"]] == [0, 0, 1, 1]
    assert [entry[1] for entry in result[2][::2]] == [w["band"] for w in state["words"]]
    assert state["words"][0]["band"] != state["words"][1]["band"]
    assert all(len(row) == 9 for row in result[9])  # technical table includes playback pair
    calls = []
    monkeypatch.setattr(app, "model_audio", lambda *args: calls.append(args) or "model.mp3")

    first = app.pick_word(SimpleNamespace(index=0), state, 1., "Male")
    sr, samples = first[3]
    assert sr == 48000
    assert len(samples) == int(.48 * sr) - int(.2 * sr)
    assert calls == []

    second = app.pick_word(SimpleNamespace(index=2), state, .7, "Male")
    assert second[3] == "model.mp3"
    assert calls[-1] == ("can you", "en-us", .7, "Male")
    assert state["current"] == 1 and state["turn"] == "model"
    assert "<strong>you</strong>" in second[1]
    assert state["words"][1]["tips"][0] in second[2]

    third = app.pick_word(SimpleNamespace(index=4), state, 1., "Male")
    assert state["turn"] == "you" and third[3][0] == 48000
    app.pick_word(SimpleNamespace(index=6), state, 1., "Female")
    assert calls[-1] == ("hear me?", "en-us", 1., "Female")
    assert state["current"] == 3


def test_explicit_buttons_speed_and_missing_audio(ui, monkeypatch):
    app, result = ui
    state = result[3]
    state["current"] = 1
    calls = []
    monkeypatch.setattr(app, "retimed", lambda sr, samples, speed: calls.append((sr, len(samples), speed)) or "slow.wav")
    assert app.play_word_you(state, .7)[0] == "slow.wav"
    assert calls[-1][0] == 48000 and calls[-1][2] == .7
    state["chunks"][0]["span"] = None
    out = app.play_word_you(state, 1.)
    assert out[0] is None and "not heard" in out[2]
    monkeypatch.setattr(app, "model_audio", lambda text, *args: text)
    assert app.play_word_model(state, 1., "Male")[0] == "can you"
    assert app.play_model(state, 1., "Male") == "can you hear me?"


def test_archive_preserves_chunks_and_timing_units(assessment, monkeypatch, tmp_path):
    from pronunciationcoach import logs
    from pronunciationcoach.crop_recheck import CropCheck
    from pronunciationcoach.crop_verify import TranscriptCheck
    assessment.crop_recheck_mode = "audit"
    assessment.crop_checks = [CropCheck(0, "can you", Span(.2, .48), Span(.18, .48), Span(.2, .48),
                                      ["possibly clipped onset"], "uncertain", "insufficient evidence")]
    assessment.transcript_checks = [TranscriptCheck(0, "can you", Span(.2, .48), Span(.2, .48),
                                                   status="matched", heard_before="can you")]
    monkeypatch.setattr(logs, "SAVE_RECORDINGS", True)
    monkeypatch.setattr(logs, "REC_DIR", tmp_path)
    wav = logs.log_assessment(logging.getLogger("chunk-archive-test"), assessment, assessment.audio)
    meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["frame_ms"] == 20 and meta["sample_rate"] == SAMPLE_RATE
    assert [c["members"] for c in meta["chunks"]] == [[0, 1], [2, 3]]
    assert meta["chunks"][0]["span"] == {"start": .2, "end": .48}
    assert meta["words"][1]["listener_p"] == .95
    assert meta["crop_recheck_mode"] == "audit"
    assert meta["crop_checks"][0]["before"] == {"start": .2, "end": .48}
    assert meta["crop_checks"][0]["status"] == "uncertain"
    assert meta["transcript_checks"][0]["heard_before"] == "can you"
    assert meta["chunks"][0]["pad_before"] is True


def test_verified_trim_survives_state_packing_and_cannot_regain_padding(ui, assessment):
    app, _ = ui
    assessment.chunks[0].span = Span(.24, .48)
    assessment.chunks[0].pad_before = False
    original = np.full(48000*2, .2, dtype=np.float32)
    original[:int(.24*48000)] = 0
    result = app.run((48000, original), assessment.text, "American")
    state = result[3]
    assert state["chunks"][0]["pad_before"] is False
    assert state["words"][0]["span"] == (.2, .4)  # scoring still uses original word crop
    played = app.pick_word(SimpleNamespace(index=0), state, 1., "Male")[3]
    assert played[0] == 48000
    assert len(played[1]) == int(.48*48000) - int(.24*48000)


def test_three_word_group_keeps_individual_selection_and_routes_both_voices(ui, assessment, monkeypatch):
    app, result = ui
    state = result[3]
    state["text"] = "what do you want to watch?"
    words = state["text"].rstrip("?").split()
    spans = [Span(.2 + i * .15, .35 + i * .15) for i in range(6)]
    groups = build_chunks(state["text"], words, spans, assessment.audio)
    assert groups[0].members == [0, 1, 2]
    ids = {i: cid for cid, group in enumerate(groups) for i in group.members}
    state["chunks"] = [{"text": c.text, "members": c.members, "span": (c.span.start, c.span.end)} for c in groups]
    state["words"] = [{"word": word, "chunk": ids[i], "band": "clear", "span": (sp.start, sp.end),
                       "tips": ["Feedback for " + word], "guides": [], "expected": [], "heard": []}
                      for i, (word, sp) in enumerate(zip(words, spans))]
    state["word_index"] = list(range(6))
    calls = []
    monkeypatch.setattr(app, "model_audio", lambda *args: calls.append(args) or "model.mp3")
    first = app.pick_word(SimpleNamespace(index=0), state, 1., "Male")
    second = app.pick_word(SimpleNamespace(index=1), state, 1., "Male")
    third = app.pick_word(SimpleNamespace(index=2), state, 1., "Male")
    assert second[3] == "model.mp3" and calls[-1][0] == "what do you"
    assert first[3][0] == third[3][0] == 48000
    assert np.array_equal(first[3][1], third[3][1])
    assert "<strong>you</strong>" in third[1] and "Feedback for you" in third[2]
    app.play_word_model(state, .7, "Female")
    assert calls[-1] == ("what do you", "en-us", .7, "Female")
