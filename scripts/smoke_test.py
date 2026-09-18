"""End-to-end check, and a quick way to inspect any recording from the terminal.

    uv run python scripts/smoke_test.py                       # bundled test clip, known text
    uv run python scripts/smoke_test.py --free                # same clip, free-speech mode (Whisper)
    uv run python scripts/smoke_test.py --audio me.wav --text "The water is cold" --accent British
    uv run python scripts/smoke_test.py --audio me.wav --plot tmp/me.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # IPA on a Windows console

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"

# Public-domain clip used in the torchaudio tutorials: a human voice, clean audio.
CLIP_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
CLIP_TEXT = "I had that curiosity beside me at this moment"


def default_clip() -> tuple[Path, str]:
    TMP.mkdir(exist_ok=True)
    path = TMP / "voices_sample.wav"
    if not path.exists():
        try:
            urllib.request.urlretrieve(CLIP_URL, path)
        except Exception as exc:  # offline: synthesise something with espeak-ng instead
            print(f"download failed ({exc}); synthesising with espeak-ng")
            path = TMP / "synth.wav"
            text = "The water is very cold"
            exe = "espeak-ng" if sys.platform != "win32" else r"C:\Program Files\eSpeak NG\espeak-ng.exe"
            subprocess.run([exe, "-v", "en-us", "-s", "150", "-w", str(path), text], check=True)
            return path, text
    return path, CLIP_TEXT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--text", help="reference sentence; omit for free-speech mode")
    ap.add_argument("--free", action="store_true", help="ignore the reference text and use Whisper")
    ap.add_argument("--accent", choices=["American", "British"], default="American")
    ap.add_argument("--plot", type=Path, help="save the posterior heatmap as PNG")
    args = ap.parse_args()

    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.g2p import ACCENTS
    from pronunciationcoach.pipeline import assess

    if args.audio:
        path, text = args.audio, args.text
    else:
        path, text = default_clip()
    if args.free:
        text = None

    audio = load_audio(path)
    print(f"audio: {path}  ({len(audio) / 16000:.2f} s)")
    t0 = time.perf_counter()
    result = assess(audio, 16000, text, ACCENTS[args.accent])
    total = time.perf_counter() - t0

    print(f"\nreference text{' (transcribed by Whisper)' if result.transcribed else ''}: {result.text}")
    print(f"heard (text-independent): {result.heard_text}")
    if result.unknown_phones:
        print(f"expected phones missing from model vocabulary: {result.unknown_phones}")

    print(f"\n{'word':<12}{'phone':<7}{'start':>6}{'end':>6}{'GOP':>7}{'post':>6}  heard  top-3")
    for w in result.words:
        for i, p in enumerate(w.phones):
            top = " ".join(f"{ph}:{pr:.2f}" for ph, pr in p.candidates)
            flag = "" if p.heard == p.expected else "  <-- " + p.category
            print(
                f"{w.word if i == 0 else '':<12}{p.expected:<7}{p.start_s:>6.2f}{p.end_s:>6.2f}"
                f"{p.gop:>7.2f}{p.posterior:>6.2f}  {p.heard:<6} {top}{flag}"
            )
    print("\nper word:", "  ".join(f"{w.word}={w.gop_min:.1f}({w.category})" for w in result.words))
    print("timings:", {k: f"{v:.2f}s" for k, v in result.timings.items()}, f"total={total:.2f}s")

    if args.plot:
        from pronunciationcoach.viz import posterior_heatmap

        fig = posterior_heatmap(result.emissions, result.segments)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"heatmap saved to {args.plot}")


if __name__ == "__main__":
    main()
