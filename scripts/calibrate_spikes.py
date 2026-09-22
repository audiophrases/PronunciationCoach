"""Measure word-crop accuracy against Edge TTS word boundaries.

Edge TTS reports the exact start and end of every word it synthesises, which
makes a cheap ground truth. This script synthesises a few sentences, runs the
full pipeline, and reports how far the crops are from the truth, both for the
spike-based estimate (boundaries.py) and for whichever aligner actually produced
the crops - so it also measures the calibration offsets in mfa.py and
segmenter.py. It prints where the scoring model's spikes fall inside words too,
which is where the constants in boundaries.py come from.

    uv run python scripts/calibrate_spikes.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "calib"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    "This is the expected sentence.",
    "The water is very cold in the morning.",
    "I put the book on the table and left.",
    "She thinks that other people never listen.",
]


async def synth(text: str, path: pathlib.Path) -> list[tuple[str, float, float]]:
    import edge_tts

    comm = edge_tts.Communicate(text, "en-US-JennyNeural", rate="+0%", boundary="WordBoundary")
    words = []
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append((chunk["text"], chunk["offset"] / 1e7, (chunk["offset"] + chunk["duration"]) / 1e7))
    return [w for w in words if any(ch.isalpha() for ch in w[0])]


def report(name: str, errs: list[tuple[float, float]]) -> None:
    e = np.array(errs) * 1000
    print(
        f"{name:<8} mean |error| start {np.abs(e[:, 0]).mean():4.0f} ms  end {np.abs(e[:, 1]).mean():4.0f} ms | "
        f"bias start {e[:, 0].mean():+4.0f} (p10 {np.percentile(e[:, 0], 10):+4.0f} p90 {np.percentile(e[:, 0], 90):+4.0f})  "
        f"end {e[:, 1].mean():+4.0f} (p10 {np.percentile(e[:, 1], 10):+4.0f} p90 {np.percentile(e[:, 1], 90):+4.0f})"
    )


def main() -> None:
    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.boundaries import word_spans
    from pronunciationcoach.pipeline import assess

    lead, lag, spikes_err = [], [], []
    model_err: dict[str, list] = {}
    for k, text in enumerate(SENTENCES):
        mp3 = OUT / f"s{k}.mp3"
        truth = asyncio.run(synth(text, mp3))
        json.dump({"text": text, "truth": truth}, open(OUT / f"s{k}.json", "w", encoding="utf-8"))
        r = assess(load_audio(mp3), 16000, text, "en-us")
        assert len(truth) == len(r.words), (truth, [w.word for w in r.words])
        by_word = r.segments_by_word()
        spike_spans = word_spans(r.audio, by_word, [[p.dropped for p in w.phones] for w in r.words], r.emissions.frame_ms)
        for (tw, ts, te), segs, sp_spike, sp_model in zip(truth, by_word, spike_spans, r.spans):
            lead.append(r.emissions.frame_to_s(segs[0].start) - ts)
            lag.append(te - r.emissions.frame_to_s(segs[-1].end))
            spikes_err.append((sp_spike.start - ts, sp_spike.end - te))
            model_err.setdefault(r.span_source, []).append((sp_model.start - ts, sp_model.end - te))

    print(f"{len(lead)} words from {len(SENTENCES)} sentences")
    print(f"first spike starts after the true word start by: median {np.median(lead)*1000:.0f} ms (p10 {np.percentile(lead,10)*1000:.0f}, p90 {np.percentile(lead,90)*1000:.0f})")
    print(f"true word end comes after the last spike end by: median {np.median(lag)*1000:.0f} ms (p10 {np.percentile(lag,10)*1000:.0f}, p90 {np.percentile(lag,90)*1000:.0f})")
    print()
    report("spikes", spikes_err)
    for source, errs in sorted(model_err.items()):
        if source != "spikes":
            report(source, errs)
    if not [s for s in model_err if s != "spikes"]:
        print("no model aligner was available")


if __name__ == "__main__":
    main()
