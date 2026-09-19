"""Measure where CTC spikes fall inside words, using Edge TTS word boundaries as ground truth."""

import asyncio
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(r"C:\Users\Admin\PronunciationCoach")
OUT = ROOT / "tmp" / "calib"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    "This is the expected sentence.",
    "The water is very cold in the morning.",
    "I put the book on the table and left.",
    "She thinks that other people never listen.",
]


async def synth(text, path):
    import edge_tts

    comm = edge_tts.Communicate(text, "en-US-JennyNeural", rate="+0%", boundary="WordBoundary")
    words = []
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append((chunk["text"], chunk["offset"] / 1e7, (chunk["offset"] + chunk["duration"]) / 1e7))
    return words


def main():
    from pronunciationcoach.align import align_words
    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.engine import get_engine
    from pronunciationcoach.g2p import text_to_phones

    eng = get_engine()
    lead_first, lag_last, inner = [], [], []
    rows = []
    for k, text in enumerate(SENTENCES):
        mp3 = OUT / f"s{k}.mp3"
        truth = asyncio.run(synth(text, mp3))
        audio = load_audio(mp3)
        em = eng.emissions(audio)
        wp = text_to_phones(text, "en-us")
        ids = [[eng.phone_id(p) or eng.unk_id for p in w.phones] for w in wp]
        al = align_words(em, ids)
        cursor = 0
        truth_words = [t for t in truth if any(ch.isalpha() for ch in t[0])]
        assert len(truth_words) == len(wp), (len(truth_words), len(wp), [t[0] for t in truth_words], [w.word for w in wp])
        for (tw, ts, te), w in zip(truth_words, wp):
            segs = al.segments[cursor : cursor + len(w.phones)]
            cursor += len(w.phones)
            s0 = em.frame_to_s(segs[0].start)
            e1 = em.frame_to_s(segs[-1].end)
            lead_first.append(s0 - ts)  # how far the first spike sits after the true word start
            lag_last.append(te - e1)  # how much word audio follows the last spike
            rows.append((tw, ts, te, s0, e1, te - ts))
            # per-phone: spacing between consecutive spikes vs word duration share
            if len(segs) > 1:
                inner.extend(em.frame_to_s(b.start - a.end) for a, b in zip(segs, segs[1:]))
        np.savez_compressed(OUT / f"s{k}.npz", log_probs=em.log_probs)
        json.dump({"text": text, "truth": truth_words}, open(OUT / f"s{k}.json", "w", encoding="utf-8"))

    lead, lag = np.array(lead_first) * 1000, np.array(lag_last) * 1000
    print(f"{len(rows)} words from {len(SENTENCES)} sentences")
    print(f"first spike starts AFTER true word start by: median {np.median(lead):.0f} ms  (p10 {np.percentile(lead,10):.0f}, p90 {np.percentile(lead,90):.0f})")
    print(f"true word end comes AFTER last spike end by: median {np.median(lag):.0f} ms  (p10 {np.percentile(lag,10):.0f}, p90 {np.percentile(lag,90):.0f})")
    print(f"gap between consecutive spikes inside a word: median {np.median(inner)*1000:.0f} ms, p90 {np.percentile(inner,90)*1000:.0f} ms")
    print("\nword          true_start true_end  spike_start spike_end  dur")
    for tw, ts, te, s0, e1, d in rows[:16]:
        print(f"{tw:<13} {ts:9.2f} {te:8.2f}  {s0:10.2f} {e1:9.2f}  {d:4.2f}")


if __name__ == "__main__":
    main()
