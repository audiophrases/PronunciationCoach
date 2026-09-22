"""Montreal Forced Aligner: the word-crop aligner, plus the offline trial runner.

Live assessments go through the warm worker in mfa_worker.py; `align_recording`
below still shells out to the CLI for scripts/mfa_compare.py, which compares
archived recordings offline and must not depend on a resident process.
"""
from __future__ import annotations

import functools
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .boundaries import Span

ROOT = Path(__file__).resolve().parents[1]
MODEL = "english_mfa"
DICTIONARY = "english_us_mfa"  # used for both accents; see README

# How far to move MFA's word starts earlier before cropping. On Edge TTS speech
# (scripts/calibrate_spikes.py, 93 words) raw MFA runs late by ~41 ms at starts and
# ~48 ms at ends, and subtracting 40 ms from both scored best. On real learner
# recordings that did not carry over: the shifted crops clipped each word's final
# sound and picked up the previous word's tail, and uncorrected MFA sounded better
# (tmp/mfa-review.html). Until it is re-measured on human speech, starts are left
# alone by default (PC_MFA_OFFSET_MS overrides it) and ends are never moved, since
# a clipped final consonant is the most audible crop error.
OFFSET_S = float(os.environ.get("PC_MFA_OFFSET_MS", "0")) / 1000.0


def calibrate(spans: list[Span]) -> list[Span]:
    """The crops the app plays from MFA's raw word intervals: starts shifted, ends kept."""
    return [Span(max(0.0, sp.start - OFFSET_S), sp.end) for sp in spans]


def model_fingerprints() -> dict:
    fingerprints = {}
    for kind, name, suffix in (("acoustic", MODEL, ".zip"), ("dictionary", DICTIONARY, ".dict")):
        path = ROOT / ".cache/mfa-models/pretrained_models" / kind / (name + suffix)
        fingerprints[kind] = {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    return fingerprints


@functools.lru_cache(maxsize=1)
def lexicon() -> frozenset[str]:
    """Every word the dictionary can pronounce, for an out-of-vocabulary preflight.

    MFA labels an unknown word `<unk>` and word_spans() then rejects the whole
    alignment. Reading the 61k-entry dictionary costs ~0.07 s once, which is far
    cheaper than discovering the problem after the alignment.
    """
    path = ROOT / ".cache/mfa-models/pretrained_models/dictionary" / (DICTIONARY + ".dict")
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as handle:
        return frozenset(line.split(maxsplit=1)[0].lower() for line in handle if line.strip())


def command_prefix() -> list[str]:
    override = os.environ.get("PC_MFA_EXE")
    if override:
        return [override]
    manager = ROOT / ".cache/mfa-tools/Library/bin/micromamba.exe"
    environment = ROOT / ".cache/mfa-env"
    if manager.exists() and (environment / "conda-meta").exists():
        return [str(manager), "run", "-p", str(environment), "mfa"]
    executable = shutil.which("mfa")
    if executable:
        return [executable]
    raise RuntimeError("MFA is not installed. Run launchers/setup.bat first, or activate an MFA environment.")


def run_mfa(arguments: list[str], *, timeout: float = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ, MFA_ROOT_DIR=str(ROOT / ".cache/mfa-models"),
               MAMBA_ROOT_PREFIX=str(ROOT / ".cache/mamba"), PYTHONIOENCODING="utf-8")
    return subprocess.run(command_prefix() + arguments, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def word_spans(data: dict, expected: list[str], duration: float) -> list[Span]:
    """Reject omissions/OOVs/reordering rather than silently assigning wrong crops.

    MFA's textgrid cleanup normally rejoins clitics. Also accept adjacent pieces
    of a contraction (could + n't), without merging unrelated transcript words.
    """
    tiers = data.get("tiers", {})
    if "words" not in tiers:
        raise ValueError("MFA output has no single-speaker word tier")
    entries = [entry for entry in tiers["words"]["entries"] if str(entry[2]).strip() not in {"", "<eps>"}]
    normalize = lambda word: word.lower().replace("’", "'")
    out, cursor, last_end = [], 0, 0.
    for word in expected:
        target, assembled, pieces = normalize(word), "", []
        while cursor < len(entries) and assembled != target:
            start, end, label = entries[cursor]
            start, end = float(start), float(end)
            if not (math.isfinite(start) and math.isfinite(end) and
                    0 <= start < end <= duration + .001 and start >= last_end - 1e-6):
                raise ValueError(f"Invalid MFA interval for {label!r}: {start}-{end}")
            assembled += normalize(label)
            if not target.startswith(assembled):
                raise ValueError(f"MFA transcript mismatch: expected {word!r}, got {assembled!r}")
            pieces.append(Span(start, min(end, duration)))
            last_end, cursor = end, cursor + 1
        if assembled != target or not pieces:
            raise ValueError(f"MFA omitted {word!r}")
        out.append(Span(pieces[0].start, pieces[-1].end))
    if cursor != len(entries):
        raise ValueError("MFA returned extra transcript words")
    return out


def align_recording(audio: Path, text: str, work: Path, *, timeout: float = 300) -> tuple[dict, float]:
    """Align a complete utterance; retain the raw output and command log."""
    from .g2p import words_of  # keeps this module importable without espeak-ng

    work.mkdir(parents=True, exist_ok=False)
    transcript = work / "transcript.lab"
    # Give MFA the same word list the scorer uses. Punctuation and digits are the
    # only things MFA's tokenizer splits differently, and a digit becomes <unk>,
    # which word_spans() would then reject for a sentence the learner read fine.
    transcript.write_text(" ".join(words_of(text)), encoding="utf-8")
    output = work / "alignment.json"
    arguments = ["align_one", str(audio.resolve()), str(transcript.resolve()), DICTIONARY, MODEL,
                 str(output.resolve()), "--output_format", "json", "--num_jobs", "1",
                 "--temporary_directory", str((work / "working").resolve())]
    (work / "command.json").write_text(json.dumps(command_prefix() + arguments, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = run_mfa(arguments, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        captured = []
        for stream in (exc.stdout, exc.stderr):
            captured.append(stream.decode('utf-8', errors='replace') if isinstance(stream, bytes) else stream or '')
        (work / "mfa.log").write_text(''.join(captured) + f'\nTimed out after {timeout}s', encoding='utf-8')
        raise RuntimeError(f"MFA timed out; see {work / 'mfa.log'}") from exc
    elapsed = time.perf_counter() - started
    (work / "mfa.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not output.exists():
        raise RuntimeError(f"MFA failed (exit {result.returncode}); see {work / 'mfa.log'}")
    return json.loads(output.read_text(encoding="utf-8")), elapsed
