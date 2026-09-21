"""Compare baseline and proposed playback edges, with an optional local listening page.

Accepts JSON files with `text` and optional `truth: [[word, start, end], ...]`.
Audio must share the stem (prefer mp3 when truth came from TTS metadata).
Metadata boundaries are a proxy, not human-annotated ground truth. This script
disables native references and uses cached models. The existing listener checks
rendered crops unless --alignment-only is selected.

Example: python scripts/recheck_compare.py recordings/*.json --output review.json
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import glob
import html
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON files or glob patterns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-only", action="store_true", help="skip recognizer-based crop verification")
    args = parser.parse_args()
    os.environ.update(PC_NATIVE_REF="0", PC_LISTENER="0" if args.alignment_only else "1",
                      HF_HUB_OFFLINE="1", PC_CROP_RECHECK="1")
    from pronunciationcoach import SAMPLE_RATE
    from pronunciationcoach.audio import load_audio
    from pronunciationcoach.pipeline import assess
    from pronunciationcoach.playback import clip

    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        parser.error("no input files matched")
    records, pages, errors = [], [], []
    norm = lambda s: re.sub(r"[^a-z']", "", s.lower())
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        audio_path = next((path.with_suffix(ext) for ext in (".mp3", ".wav") if path.with_suffix(ext).exists()), None)
        if audio_path is None:
            raise FileNotFoundError(f"no audio for {path}")
        audio = load_audio(audio_path)
        result = assess(audio, SAMPLE_RATE, meta["text"], meta.get("accent", "en-us"))
        checks = {c.chunk: c for c in result.crop_checks}
        verified = {c.chunk: c for c in result.transcript_checks}
        truth = meta.get("truth")
        if truth and [norm(w.word) for w in result.words] != [norm(w[0]) for w in truth]:
            raise ValueError(f"metadata words do not match: {path}")
        rows = []
        for ci, chunk in enumerate(result.chunks):
            if chunk.span is None:
                continue
            old = verified[ci].before if ci in verified else checks[ci].before if ci in checks else chunk.span
            new = chunk.span
            row = {"text": chunk.text, "members": chunk.members, "before": asdict(old), "after": asdict(new),
                   "pad_before": chunk.pad_before, "pad_after": chunk.pad_after}
            if truth:
                expected = (truth[chunk.members[0]][1], truth[chunk.members[-1]][2])
                pair = [(abs(a-t)*1000, abs(b-t)*1000) for a, b, t in zip(
                    (old.start, old.end), (new.start, new.end), expected)]
                errors.extend(pair)
                row["metadata_error_ms"] = pair
            rows.append(row)
            if old != new or not chunk.pad_before or not chunk.pad_after:
                players = []
                for label, sp in (("Before", old), ("After", new)):
                    buffer = io.BytesIO()
                    _, samples = clip(audio, sp.start, sp.end,
                                      pad_before=True if label == "Before" else chunk.pad_before,
                                      pad_after=True if label == "Before" else chunk.pad_after)
                    sf.write(buffer, samples, SAMPLE_RATE, format="WAV")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    players.append(f'<p>{label} {sp.start:.3f}–{sp.end:.3f}s <audio controls src="data:audio/wav;base64,{encoded}"></audio></p>')
                pages.append(f'<section><h2>{html.escape(path.stem)}: {html.escape(chunk.text)}</h2>' + ''.join(players) + '</section>')
        record = {"source": str(audio_path), "text": result.text, "chunks": rows,
                  "checks": [asdict(c) for c in result.crop_checks],
                  "transcript_checks": [asdict(c) for c in result.transcript_checks],
                  "recheck_mode": result.crop_recheck_mode, "timings": result.timings}
        records.append(record)
        print(f"{path.stem}: {len(result.crop_checks)} flagged; "
              f"{sum(c.status == 'trimmed' for c in result.transcript_checks)} trimmed; "
              f"recheck {result.timings['crop_recheck']*1000:.0f} ms", flush=True)
        for c in result.crop_checks:
            print("  " + c.summary(), flush=True)
        for c in result.transcript_checks:
            print("  " + c.summary(), flush=True)
    summary = {"recordings": len(records), "flagged_chunks": sum(len(r["checks"]) for r in records),
               "adjusted_chunks": sum(c["status"] == "trimmed" for r in records for c in r["transcript_checks"])}
    if errors:
        values = np.array(errors)
        summary.update(metadata_edges=len(values), metadata_mae_before_ms=float(values[:, 0].mean()),
                       metadata_mae_after_ms=float(values[:, 1].mean()),
                       metadata_improved_edges=int(np.sum(values[:, 1] < values[:, 0] - 1e-6)),
                       metadata_regressed_edges=int(np.sum(values[:, 1] > values[:, 0] + 1e-6)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "recordings": records}, indent=2), encoding="utf-8")
    args.output.with_suffix(".html").write_text('<!doctype html><meta charset="utf-8"><title>Crop recheck comparison</title>'
        '<style>body{font:18px system-ui;max-width:850px;margin:40px auto}audio{vertical-align:middle}section{border-top:1px solid #aaa}</style>'
        '<h1>Proposed crop changes</h1><p>Original recording audio. Metadata measurements are not listening judgments.</p>'
        + (''.join(pages) or '<p>No changes accepted by the conservative checks.</p>'), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
