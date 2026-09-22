"""Keep the suite independent of whether MFA happens to be installed.

Without this, `locate_words` really starts the aligner on a developer machine:
the fixture audio is accepted and one call costs seconds, so the suite would be
fast on a laptop without MFA and slow (and differently exercised) on one with it.
Tests that want the MFA path patch `pipeline.mfa_spans` themselves.
"""
import pytest


@pytest.fixture(autouse=True)
def no_real_mfa(monkeypatch):
    from pronunciationcoach import pipeline

    def blocked(*args, **kwargs):
        raise RuntimeError("MFA is blocked in tests; patch pipeline.mfa_spans to exercise it")

    monkeypatch.setattr(pipeline, "real_mfa_spans", pipeline.mfa_spans, raising=False)
    monkeypatch.setattr(pipeline, "mfa_spans", blocked)
