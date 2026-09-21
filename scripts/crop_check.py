"""Are the word crops any good? Check every crop against the full recording's sounds.

Re-recognising an isolated 300 ms clip is meaningless (the recogniser needs
context), so instead each crop is compared with the spikes the recogniser
produced on the *whole* recording: a sound starts about SPIKE_LAG_S before its
spike and a word ends about where its last spike ends. Per word, in ms:

  * cut   - how much of the word's first sound the crop starts after (its onset clipped)
  * prev  - how much of the previous word (up to its last spike's end) the crop contains
  * next  - how much of the next word's first sound the crop contains
  * tail  - how much of the word's own last sound the crop ends before
  * lead / trail - silence at the crop's edges (energy below the speech threshold)

Archives written since the crops were archived (words carry "span" and "crop") are
checked as they are, without loading any model; older ones are re-assessed.

    uv run python scripts/crop_check.py                 # all archived recordings
    uv run python scripts/crop_check.py --only 2026-09-21 --verbose
    uv run python scripts/crop_check.py --rerun          # re-assess everything with the current code
    uv run python scripts/crop_check.py --rechunk        # audit playback pairs without models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
FRAME_S = 0.02
OK_CUT_MS, OK_PREV_MS, OK_NEXT_MS, OK_TAIL_MS = 25, 30, 30, 40  # natural overlap and spike-lag spread


def edge_silence(clip: np.ndarray, thr_db: float) -> tuple[float, float]:
    from pronunciationcoach.boundaries import HOP, SAMPLE_RATE, energy_db

    db = energy_db(clip)
    speech = np.flatnonzero(db > thr_db)
    if speech.size == 0:
        return len(clip) / SAMPLE_RATE * 1000, 0.0
    return speech[0] * HOP / SAMPLE_RATE * 1000, (len(db) - 1 - speech[-1]) * HOP / SAMPLE_RATE * 1000


def from_archive(meta: dict):
    """(words, spans, cases, source) straight from an archive that carries the crops."""
    words = [(w["word"], [(p["start_s"], p["end_s"]) for p in w["phones"] if not p.get("dropped")] or
              [(p["start_s"], p["end_s"]) for p in w["phones"]]) for w in meta["words"]]
    spans = [tuple(w["span"]) for w in meta["words"]]
    return words, spans, [w.get("crop", "") for w in meta["words"]], meta.get("span_source", "?")


def from_assessment(audio: np.ndarray, meta: dict):
    from pronunciationcoach import SAMPLE_RATE
    from pronunciationcoach.pipeline import assess

    r = assess(audio, SAMPLE_RATE, meta["text"], meta.get("accent", "en-us"))
    em = r.emissions
    words = []
    for w, segs in zip(r.words, r.segments_by_word()):
        kept = [(em.frame_to_s(s.start), em.frame_to_s(s.end)) for p, s in zip(w.phones, segs) if not p.dropped]
        words.append((w.word, kept or [(em.frame_to_s(s.start), em.frame_to_s(s.end)) for s in segs]))
    return words, [(sp.start, sp.end) for sp in r.word_spans()], r.span_cases or [""] * len(words), r.span_source, r


def audit_chunks(audio, meta, words, spans, assessment=None):
    from pronunciationcoach import SAMPLE_RATE
    from pronunciationcoach.boundaries import Span
    from pronunciationcoach.chunks import FUNCTION_WORDS, SHORT_S, build_chunks

    if assessment is not None:
        chunks = assessment.chunks
        extra = [(r.start * assessment.emissions.frame_ms / 1000, r.end * assessment.emissions.frame_ms / 1000)
                 for r in assessment.extra]
    else:
        frame_s = meta.get("frame_ms", 20.0) / 1000
        extra = [(a * frame_s, b * frame_s) for _, a, b in meta.get("extra", [])]
        dropped = [bool(w["phones"]) and all(p.get("dropped", False) for p in w["phones"]) and not w.get("insertions")
                   for w in meta["words"]]
        chunks = build_chunks(meta["text"], [w[0] for w in words], [Span(*s) for s in spans], audio, dropped, extra)
    print("   PLAYBACK (at most two words per chunk):")
    orphan = overlaps = 0
    previous_end = 0.0
    for chunk in chunks:
        sp = chunk.span
        timing = f"{sp.start:.2f}-{sp.end:.2f} ({sp.duration * 1000:.0f} ms)" if sp else "not heard"
        flags = []
        if sp and (sp.start < previous_end - 1e-8 or any(a < sp.end and b > sp.start for a, b in extra)):
            flags.append("overlap: inspect")
            overlaps += 1
        if sp:
            previous_end = sp.end
        if len(chunk.members) == 1 and (not sp or sp.duration < SHORT_S - 1e-8 or words[chunk.members[0]][0].lower() in FUNCTION_WORDS):
            orphan += 1  # not automatically wrong: barriers/emphasis can require a singleton
        print(f"     [{chunk.text}] {timing} | {chunk.reason}" + (" | " + ", ".join(flags) if flags else ""))
    invalid = sum(not np.isfinite([a, b]).all() or a < 0 or b < a or b > len(audio) / SAMPLE_RATE + 1e-8 for a, b in spans)
    print(f"   {len(chunks)} chunks | weak/short singletons {orphan} | invalid input crops {invalid} | overlaps {overlaps}")
    if meta.get("chunks") is not None:
        changed = [c.members for c in chunks] != [c["members"] for c in meta["chunks"]]
        print(f"   Membership changed from archive: {changed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring of the recording name")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="re-assess with the current code even when crops are archived")
    ap.add_argument("--rechunk", action="store_true", help="audit current playback pairs from archived audio and word crops")
    args = ap.parse_args()

    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.boundaries import SAMPLE_RATE, SPIKE_LAG_S, energy_db, speech_threshold

    files = sorted(ROOT.glob("recordings/*.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    seen_texts: set[str] = set()
    rows: list[tuple[float, float, float, float, float, float]] = []
    for meta_path in files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        wav = meta_path.with_suffix(".wav")
        if not wav.exists() or meta["text"] in seen_texts:
            continue  # several archives of the same take add nothing
        seen_texts.add(meta["text"])
        audio = load_audio(wav)
        archived = all("span" in w for w in meta["words"]) and not args.rerun
        if archived:
            words, spans, cases, source = from_archive(meta)
            assessment = None
        else:
            words, spans, cases, source, assessment = from_assessment(audio, meta)
        thr = speech_threshold(energy_db(audio))
        print(f"\n== {meta_path.stem}  ({source}{', archived' if archived else ', re-assessed'})  {meta['text'][:60]!r}")
        print(f"   {'word':<10} {'crop':<12} {'case':<5} {'cut':>4} {'prev':>4} {'next':>4} {'tail':>4} {'lead':>5} {'trail':>5}")
        for i, ((word, spikes), (a, b), case) in enumerate(zip(words, spans, cases)):
            onset = spikes[0][0] - SPIKE_LAG_S  # the first sound starts about here
            last_end = spikes[-1][1]  # the word ends about here
            cut = max(0.0, a - onset) * 1000
            tail = max(0.0, last_end - b) * 1000
            prev = max(0.0, words[i - 1][1][-1][1] - a) * 1000 if i > 0 else 0.0
            nxt = max(0.0, b - (words[i + 1][1][0][0] - SPIKE_LAG_S)) * 1000 if i + 1 < len(words) else 0.0
            clip = audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)]
            lead, trail = edge_silence(clip, thr) if len(clip) > 400 else (0.0, 0.0)
            rows.append((cut, prev, nxt, tail, lead, trail))
            flag = "" if cut <= OK_CUT_MS and prev <= OK_PREV_MS and nxt <= OK_NEXT_MS and tail <= OK_TAIL_MS and lead < 80 and trail < 120 else "  <--"
            if args.verbose or flag:
                print(f"   {word:<10} {a:5.2f}-{b:5.2f} {case:<5} {cut:4.0f} {prev:4.0f} {nxt:4.0f} {tail:4.0f} {lead:5.0f} {trail:5.0f}{flag}")
        if args.rechunk:
            audit_chunks(audio, meta, words, spans, assessment)
    if rows:
        t = np.array(rows)
        n = len(t)
        print(f"\nOVERALL {n} crops | first sound cut >{OK_CUT_MS} ms: {int((t[:,0] > OK_CUT_MS).sum())} | "
              f"previous word inside >{OK_PREV_MS} ms: {int((t[:,1] > OK_PREV_MS).sum())} | next word inside >{OK_NEXT_MS} ms: {int((t[:,2] > OK_NEXT_MS).sum())} | "
              f"last sound cut >{OK_TAIL_MS} ms: {int((t[:,3] > OK_TAIL_MS).sum())} | lead silence >80 ms: {int((t[:,4] > 80).sum())} | trail silence >120 ms: {int((t[:,5] > 120).sum())}")
        print(f"        medians (ms): cut {np.median(t[:,0]):.0f}  prev {np.median(t[:,1]):.0f}  next {np.median(t[:,2]):.0f}  tail {np.median(t[:,3]):.0f}")


if __name__ == "__main__":
    main()
