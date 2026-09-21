"""Application log and optional recording archive.

Always: one line per assessment in logs/app.log (what was said, what was heard,
which phones were flagged, timings), plus warnings and errors from every library.
With PC_DEBUG=1: the full per-phone table as well, and DEBUG-level library output.
With PC_SAVE_RECORDINGS=1: every recording is archived as recordings/<timestamp>.wav
next to a .json with the complete result, so it can be replayed with the smoke test.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .pipeline import Assessment

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
REC_DIR = ROOT / "recordings"

DEBUG = os.environ.get("PC_DEBUG") == "1"
SAVE_RECORDINGS = os.environ.get("PC_SAVE_RECORDINGS") == "1"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    to_file = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    to_file.setFormatter(fmt)
    to_console = logging.StreamHandler(sys.stderr)
    to_console.setFormatter(fmt)

    # Libraries: warnings and errors always land in the file. In debug mode the speech
    # libraries also report at INFO; the web/plotting plumbing stays quiet either way
    # (matplotlib's font manager alone writes hundreds of DEBUG lines per request).
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(to_file)
    root.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    if DEBUG:
        for lib in ("transformers", "faster_whisper", "huggingface_hub", "gradio"):
            logging.getLogger(lib).setLevel(logging.INFO)

    # Our own messages: INFO normally, DEBUG when debugging, on console as well as file.
    log = logging.getLogger("pronunciationcoach")
    log.handlers.clear()
    log.addHandler(to_console)
    log.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    log.propagate = True  # file handler comes from root
    return log


def _stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def log_assessment(log: logging.Logger, result: Assessment, audio_16k: np.ndarray) -> Path | None:
    """Write the one-line summary (and the details in debug mode); archive the recording if asked."""
    flagged = [
        f"{w.word}/+{p.heard}" if p.inserted else f"{w.word}/{p.expected}->{p.heard_label}({p.gop:.1f})"
        for w in result.words
        for p in w.all_phones
        if p.category != "good"
    ]
    timings = " ".join(f"{k}={v:.1f}s" for k, v in result.timings.items())
    log.info(
        "assess mode=%s accent=%s audio=%.1fs | text=%r | heard=%s | flagged=%s | %s",
        "free" if result.transcribed else "known",
        result.lang,
        result.duration_s,
        result.text,
        result.heard_text,
        " ".join(flagged) or "none",
        timings,
    )
    if result.extra:
        log.info("heard outside the sentence: %s", result.extra_text(min_phones=1))
    verdicts = " ".join(f"{w.word}={w.verdict.replace(' ', '_')}({w.listener_p:.2f})" if w.listener_p is not None else f"{w.word}={w.verdict}" for w in result.words)
    log.info("verdicts: %s", verdicts)
    natural = [f"{w.word}/{p.expected}" for w in result.words for p in w.phones if p.natural]
    if natural or result.reference_voices:
        log.info("native reference %s | accepted as natural: %s", result.reference_voices, " ".join(natural) or "none")
    if result.unknown_phones:
        log.warning("expected phones without a model label: %s", result.unknown_phones)
    cases = result.span_cases or [""] * len(result.spans)
    log.info("crops (%s): %s", result.span_source,
             " | ".join(f"{w.word} {sp.start:.2f}-{sp.end:.2f} {case}".rstrip() for w, sp, case in zip(result.words, result.spans, cases)))
    if log.isEnabledFor(logging.DEBUG):
        for w in result.words:
            for p in w.phones:
                top = " ".join(f"{ph}:{pr:.2f}" for ph, pr in p.candidates)
                log.debug("  %-12s %-6s %5.2f-%5.2fs gop=%6.2f post=%.2f heard=%-11s %s", w.word, p.expected, p.start_s, p.end_s, p.gop, p.posterior, p.heard_label, top)

    if not SAVE_RECORDINGS:
        return None
    REC_DIR.mkdir(exist_ok=True)
    stem = REC_DIR / _stamp()
    sf.write(stem.with_suffix(".wav"), audio_16k, SAMPLE_RATE, subtype="PCM_16")
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": result.text,
        "transcribed": result.transcribed,
        "accent": result.lang,
        "duration_s": result.duration_s,
        "heard": result.heard_text,
        "words": [
            {"word": w.word, "gop_min": w.gop_min, "span": [round(sp.start, 3), round(sp.end, 3)], "crop": case,
             "phones": [asdict(p) for p in w.phones], "insertions": [asdict(p) for p in w.insertions]}
            for w, sp, case in zip(result.words, result.spans, result.span_cases or [""] * len(result.spans))
        ],
        "span_source": result.span_source,
        "extra": [[r.phones, r.start, r.end] for r in result.extra],
        "timings": result.timings,
    }
    stem.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("recording archived: %s", stem.with_suffix(".wav").name)
    return stem.with_suffix(".wav")
