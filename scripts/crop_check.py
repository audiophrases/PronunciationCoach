"""Are the word crops any good? Check every crop against the full recording's sounds.

Re-recognising an isolated 300 ms clip is meaningless (the recogniser needs
context), so instead each crop is compared with the spikes the recogniser
produced on the *whole* recording: a sound whose onset (spike start minus the
measured ~80 ms lag) lies inside the crop is "in" it. Per word that gives:

  * missing  - the word's own sounds that the crop does not contain (clipped)
  * bleed    - the neighbours' sounds the crop does contain
  * lead/tail - silence at the crop's edges, ms (energy below the speech threshold)

    uv run python scripts/crop_check.py                 # all archived recordings
    uv run python scripts/crop_check.py --only 2026-09-19_211118 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def edge_silence(clip: np.ndarray, thr_db: float) -> tuple[float, float]:
    from pronunciationcoach.boundaries import HOP, SAMPLE_RATE, energy_db

    db = energy_db(clip)
    speech = np.flatnonzero(db > thr_db)
    if speech.size == 0:
        return len(clip) / SAMPLE_RATE * 1000, 0.0
    return speech[0] * HOP / SAMPLE_RATE * 1000, (len(db) - 1 - speech[-1]) * HOP / SAMPLE_RATE * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring of the recording name")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from pronunciationcoach import SAMPLE_RATE
    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.boundaries import SPIKE_LAG_S, energy_db, speech_threshold
    from pronunciationcoach.pipeline import assess

    files = sorted(ROOT.glob("recordings/*.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    seen_texts: set[str] = set()
    totals: list[tuple[int, int, int, float, float]] = []
    for meta_path in files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        wav = meta_path.with_suffix(".wav")
        if not wav.exists() or meta["text"] in seen_texts:
            continue  # several archives of the same take add nothing
        seen_texts.add(meta["text"])
        audio = load_audio(wav)
        r = assess(audio, SAMPLE_RATE, meta["text"], meta.get("accent", "en-us"))
        thr = speech_threshold(energy_db(audio))
        em = r.emissions
        by_word = r.segments_by_word()
        spans = r.word_spans()
        print(f"\n== {meta_path.stem}  ({r.span_source})  {meta['text'][:70]!r}")
        print(f"   {'word':<10} {'crop':<12} {'own':>3} {'miss':>4} {'bleed':>5} {'lead':>5} {'tail':>5}  missing / bleeding sounds")
        for i, (w, span, segs) in enumerate(zip(r.words, spans, by_word)):
            def onset(seg):
                return em.frame_to_s(seg.start) - SPIKE_LAG_S

            own = [(p, s) for p, s in zip(w.phones, segs) if not p.dropped]
            inside = lambda s: span.start - 0.03 <= onset(s) <= span.end - 0.04
            missing = [p.expected for p, s in own if not inside(s)]
            bleed = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(r.words):
                    bleed += [p.expected for p, s in zip(r.words[j].phones, by_word[j]) if not p.dropped and inside(s)]
            clip = audio[int(span.start * SAMPLE_RATE) : int(span.end * SAMPLE_RATE)]
            lead, tail = edge_silence(clip, thr) if len(clip) > 400 else (0.0, 0.0)
            totals.append((len(own), len(missing), len(bleed), lead, tail))
            flag = "" if not missing and not bleed and lead < 80 and tail < 120 else "  <--"
            if args.verbose or flag:
                print(f"   {w.word:<10} {span.start:5.2f}-{span.end:5.2f} {len(own):3d} {len(missing):4d} {len(bleed):5d} {lead:5.0f} {tail:5.0f}  "
                      f"{'missing ' + ' '.join(missing) if missing else ''}{'  bleed ' + ' '.join(bleed) if bleed else ''}{flag}")
    if totals:
        t = np.array(totals)
        n = len(t)
        print(f"\nOVERALL {n} crops: sounds clipped {int(t[:,1].sum())}/{int(t[:,0].sum())} ({t[:,1].sum()/max(t[:,0].sum(),1):.0%}) in {int((t[:,1]>0).sum())} crops | "
              f"crops with bleed {int((t[:,2]>0).sum())} | lead silence >80 ms: {int((t[:,3]>80).sum())} | tail silence >120 ms: {int((t[:,4]>120).sum())}")


if __name__ == "__main__":
    main()
