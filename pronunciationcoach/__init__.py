"""Phoneme-level pronunciation feedback built on open speech models.

Layout:
    audio.py     – load anything, get float32 mono 16 kHz
    engine.py    – phoneme recogniser; exposes the per-frame posterior grid
    g2p.py       – reference text -> expected phones (espeak-ng)
    align.py     – greedy decode (what was heard) and forced alignment (where each expected phone is)
    scoring.py   – Goodness of Pronunciation per phone / word
    asr.py       – word recognition for free speech (faster-whisper)
    pipeline.py  – ties the above together into one `assess()` call
"""

SAMPLE_RATE = 16_000
