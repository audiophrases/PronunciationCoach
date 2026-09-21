"""The recognizer and learner player must hear the same crop."""

import numpy as np
import pytest

from pronunciationcoach.playback import clip, sample_bounds


@pytest.mark.parametrize("sr", [16000, 48000])
def test_default_render_keeps_quiet_padding_fade_gain_and_original_audio(sr):
    audio = np.zeros(sr, dtype=np.float32)
    audio[int(.2 * sr):int(.5 * sr)] = .1
    original = audio.copy()

    bounds = sample_bounds(audio, .2, .5, sr)
    assert bounds == (int(.14 * sr), int(.55 * sr))
    returned_sr, rendered = clip(audio, .2, .5, sr)
    assert returned_sr == sr
    assert len(rendered) == bounds[1] - bounds[0]
    assert rendered[0] == rendered[-1] == 0
    assert np.max(rendered) == pytest.approx(.7)
    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("pad_before,pad_after", [(False, True), (True, False), (False, False)])
def test_corrected_edges_do_not_restore_quiet_neighbor_speech(pad_before, pad_after):
    sr = 16000
    # Quiet neighboring sound would otherwise qualify for automatic padding.
    audio = np.full(sr, .005, dtype=np.float32)
    audio[3200:8000] = .3
    bounds = sample_bounds(audio, .2, .5, sr,
                           pad_before=pad_before, pad_after=pad_after)
    assert bounds == (2240 if pad_before else 3200, 8800 if pad_after else 8000)
    _, rendered = clip(audio, .2, .5, sr,
                       pad_before=pad_before, pad_after=pad_after)
    assert len(rendered) == bounds[1] - bounds[0]
    # A corrected boundary still receives a fade, so the hard cut cannot click.
    assert rendered[0] == rendered[-1] == 0


def test_default_padding_stops_at_loud_neighbor():
    audio = np.zeros(16000, dtype=np.float32)
    audio[3200:10000] = .3
    assert sample_bounds(audio, .2, .5) == (2240, 8000)


@pytest.mark.parametrize("start,end", [(.3, .2), (2., 3.), (-2., -.1), (.2, .2)])
def test_empty_or_outside_crop_cannot_acquire_audio_through_padding(start, end):
    audio = np.ones(16000, dtype=np.float32)
    a, b = sample_bounds(audio, start, end)
    assert 0 <= a == b <= len(audio)
    assert clip(audio, start, end)[1].size == 0


def test_app_word_playback_uses_each_corrected_edge_flag(monkeypatch):
    monkeypatch.setenv("PC_OPEN_BROWSER", "0")
    import app

    sr = 48000
    audio = np.full(sr, .005, dtype=np.float32)
    audio[9600:24000] = .3
    chunk = {"span": (.2, .5), "pad_before": False, "pad_after": True}
    word = {"chunk": 0}
    state = {"audio": audio[::3], "orig": (sr, audio), "chunks": [chunk]}
    monkeypatch.setattr(app, "retimed", lambda rate, samples, speed: (rate, samples))
    actual_sr, actual = app._word_clip(state, word, 1)
    expected_sr, expected = clip(audio, .2, .5, sr, pad_before=False)
    assert actual_sr == expected_sr == sr
    np.testing.assert_array_equal(actual, expected)

    chunk.update(pad_before=True, pad_after=False)
    _, actual = app._word_clip(state, word, 1)
    _, expected = clip(audio, .2, .5, sr, pad_after=False)
    np.testing.assert_array_equal(actual, expected)
