"""An experimental aligner must not quietly shift word identities."""
import math

import pytest

from pronunciationcoach.boundaries import Span
from pronunciationcoach.mfa import word_spans


def output(entries):
    return {"tiers": {"words": {"entries": entries}}}


def test_mfa_silence_and_contraction_parts_preserve_word_identity():
    data = output([[0, .1, '<eps>'], [.1, .3, 'could'], [.3, .4, "n't"],
                   [.4, .5, 'you'], [.5, .6, '']])
    assert word_spans(data, ["Couldn't", 'you'], .6) == [Span(.1, .4), Span(.4, .5)]


@pytest.mark.parametrize('entries,words', [
    ([[0, .1, '<unk>']], ['word']),
    ([[0, .1, 'cat']], ['dog']),
    ([[0, .1, 'a']], ['a', 'cat']),
    ([[0, .1, 'a'], [.1, .2, 'cat']], ['a']),
    ([[0, .3, 'a'], [.2, .4, 'cat']], ['a', 'cat']),
    ([[0, 0, 'a']], ['a']),
    ([[0, math.nan, 'a']], ['a']),
    ([[0, 2, 'a']], ['a']),
])
def test_mfa_rejects_mismatch_missing_extra_and_invalid_intervals(entries, words):
    with pytest.raises(ValueError):
        word_spans(output(entries), words, 1.)


def test_mfa_rejects_multiple_speaker_output():
    with pytest.raises(ValueError, match='single-speaker'):
        word_spans({'tiers': {'speaker - words': {'entries': []}}}, ['word'], 1.)


def test_timeout_retains_diagnostics(monkeypatch, tmp_path):
    import subprocess
    from pronunciationcoach import mfa
    monkeypatch.setattr(mfa, 'command_prefix', lambda: ['mfa'])

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('mfa', 1, output=b'loading model')

    monkeypatch.setattr(mfa, 'run_mfa', timeout)
    work = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='timed out'):
        mfa.align_recording(tmp_path / 'audio.wav', 'word', work, timeout=1)
    assert 'loading model' in (work / 'mfa.log').read_text()
    assert 'Timed out' in (work / 'mfa.log').read_text()
    assert (work / 'command.json').exists()
