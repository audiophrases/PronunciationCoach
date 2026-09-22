"""Compare MFA with archived word crops, holding playback grouping fixed.

Example: uv run python scripts/mfa_compare.py "recordings/*.json"
No speech is uploaded. MFA models are downloaded by launchers/setup.bat.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import glob
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.boundaries import Span
from pronunciationcoach.diagnostics import runtime_snapshot
from pronunciationcoach.mfa import DICTIONARY, MODEL, OFFSET_S, align_recording, calibrate, model_fingerprints, run_mfa, word_spans
from pronunciationcoach.playback import clip


def player(samples, label):
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f'<audio controls preload="none" aria-label="{html.escape(label, quote=True)}" '
            f'src="data:audio/wav;base64,{encoded}"></audio>')


def comparison_row(label, baseline, aligned, played, samples):
    cells = []
    for name, span in (("Archived", baseline), ("Raw MFA", aligned), ("App MFA", played)):
        if span is None or span.duration <= 0:
            cells.append('<td>No playable interval</td>')
        else:
            _, rendered = clip(samples, span.start, span.end)
            cells.append(f'<td>{span.start:.3f}–{span.end:.3f}s<br>' + player(rendered, name + ' ' + label) + '</td>')
    return f'<tr><th>{html.escape(label)}</th>' + ''.join(cells) + '</tr>'


def waveform(samples, words, baseline, aligned, played):
    """Overview of the original waveform and the two word-boundary tracks."""
    duration = len(samples) / SAMPLE_RATE
    width = max(1000, len(words)*65)
    x = lambda seconds: 100 + (width-110)*seconds/duration
    peak = max(float(np.max(np.abs(samples))), 1e-6)
    path = []
    for i, block in enumerate(np.array_split(samples, 700)):
        if len(block):
            at = 100 + (width-110)*i/700
            path.append(f'M{at:.1f},{45-35*float(block.max())/peak:.1f}V{45-35*float(block.min())/peak:.1f}')
    rows = [f'<path d="{" ".join(path)}" stroke="#64748b" stroke-width="1"/>']
    for name, spans, y, color in (("Archived", baseline, 95, '#2563eb'), ("Raw MFA", aligned, 140, '#b45309'),
                                  ("App MFA", played, 185, '#15803d')):
        rows.append(f'<text x="0" y="{y+19}">{name}</text>')
        for word, span in zip(words, spans):
            rows.append(f'<g><title>{html.escape(word)} {span.start:.3f}–{span.end:.3f}s</title>'
                        f'<rect x="{x(span.start):.1f}" y="{y}" width="{max(1, x(span.end)-x(span.start)):.1f}" height="28" '
                        f'fill="{color}" fill-opacity=".15" stroke="{color}"/>'
                        f'<text x="{x(span.start)+2:.1f}" y="{y+19}" font-size="12">{html.escape(word)}</text></g>')
    for tick in np.linspace(0, duration, 6):
        rows.append(f'<text x="{x(tick):.1f}" y="235" text-anchor="end" font-size="12">{tick:.2f}s</text>')
    return f'<div style="overflow-x:auto"><svg role="img" aria-label="Waveform and word boundaries" width="{width}" height="245">' + ''.join(rows) + '</svg></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Archived recording JSON files or globs")
    parser.add_argument("--output", type=Path, default=Path("tmp/mfa-review.json"))
    parser.add_argument("--reuse", type=Path, help="Reuse a previous report's alignments only when audio, text, version and models match")
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        parser.error("No archived recordings matched")
    version = run_mfa(["version"])
    if version.returncode:
        raise RuntimeError(version.stdout + version.stderr)
    fingerprints = model_fingerprints()
    previous = json.loads(args.reuse.read_text(encoding='utf-8')) if args.reuse else {}
    reusable = {r['source']: r for r in previous.get('recordings', [])} if (
        previous.get('model_fingerprints') == fingerprints and previous.get('mfa_version') == version.stdout.strip()) else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mfa-trial-", dir=args.output.parent)).resolve()
    records, pages, failures, deltas = [], [], [], []
    def header(source):
        return ('<table><thead><tr><th>Speech</th>'
                f'<th>Archived crops ({html.escape(source or "unknown source")})</th>'
                '<th>Raw MFA</th><th>MFA as the app plays it</th></tr></thead><tbody>')
    for index, path in enumerate(paths):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get('accent', 'en-us') != 'en-us':
            raise ValueError(f"This trial uses the US dictionary; incompatible accent in {path}")
        wav = path.with_suffix('.wav')
        samples, sr = sf.read(wav, dtype="float32")
        if sr != SAMPLE_RATE or samples.ndim != 1:
            raise ValueError(f"Expected archived mono 16 kHz audio: {wav}")
        duration = len(samples) / sr
        words = [w['word'] for w in meta['words']]
        original = [Span(*w['span']) for w in meta['words']]
        audio_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
        cached = reusable.get(str(wav.resolve()))
        reuse = bool(cached and cached['audio_sha256'] == audio_hash and cached['text'] == meta['text']
                     and Path(cached['mfa_raw_output']).exists())
        print(f"Aligning {path.stem}: {meta['text']}", flush=True)
        try:
            if reuse:
                raw = json.loads(Path(cached['mfa_raw_output']).read_text(encoding='utf-8'))
                elapsed = cached['mfa_elapsed_s']
            else:
                raw, elapsed = align_recording(wav, meta['text'], work / str(index))
            aligned = word_spans(raw, words, duration)
            played = calibrate(aligned)
        except (RuntimeError, ValueError) as exc:
            failures.append({"source": str(path), "error": str(exc)})
            pages.append(f'<section><h2>{html.escape(path.stem)}</h2><p>{html.escape(str(exc))}</p></section>')
            print(f"FAILED: {exc}", flush=True)
            continue
        # Use the existing groups to isolate the alignment change from grouping.
        chunks = meta.get('chunks')
        grouping_source = 'archive'
        if chunks is None:
            chunks = [{"members": [i], "text": word, "span": asdict(span)}
                      for i, (word, span) in enumerate(zip(words, original))]
            grouping_source = 'individual words (legacy archive has no grouping data)'
        rows = []
        for word, before, after, app in zip(words, original, aligned, played):
            shift = [round((after.start-before.start)*1000, 6), round((after.end-before.end)*1000, 6)]
            deltas.extend(abs(x) for x in shift)
            rows.append({"word": word, "archived": asdict(before), "mfa": asdict(after),
                         "app_mfa": asdict(app), "shift_ms": shift})
        group_rows, group_data = [], []
        for chunk in chunks:
            ids = chunk['members']
            before = Span(**chunk['span']) if chunk['span'] else None
            after = Span(aligned[ids[0]].start, aligned[ids[-1]].end) if before else None
            app = Span(played[ids[0]].start, played[ids[-1]].end) if before else None
            group_rows.append(comparison_row(chunk['text'], before, after, app, samples))
            group_data.append({"text": chunk['text'], "members": ids, "archived": asdict(before) if before else None,
                               "mfa": asdict(after) if after else None, "app_mfa": asdict(app) if app else None})
        pages.append(f'<section><h2>{html.escape(path.stem)}</h2><p>{html.escape(meta["text"])}</p>'
                     f'<p>Whole recording: {player(samples, "Whole recording")}</p>'
                     f'<p>MFA took {elapsed:.2f}s including process/model startup. Baseline: {html.escape(meta.get("span_source", "unknown"))}. '
                     f'Grouping: {html.escape(grouping_source)}.</p>' + waveform(samples, words, original, aligned, played) + header(meta.get("span_source")) + ''.join(group_rows) + '</tbody></table>'
                     '<details><summary>Compare individual words</summary>' + header(meta.get("span_source")) + ''.join(
                         comparison_row(w, b, a, p, samples) for w, b, a, p in zip(words, original, aligned, played)) + '</tbody></table></details></section>')
        records.append({"source": str(wav.resolve()), "audio_sha256": audio_hash,
                        "text": meta['text'], "duration_s": duration, "baseline_source": meta.get('span_source'),
                        "grouping_source": grouping_source, "words": rows, "chunks": group_data,
                        "mfa_elapsed_s": elapsed, "reused_alignment": reuse,
                        "mfa_raw_output": cached['mfa_raw_output'] if reuse else str(work / str(index) / 'alignment.json'),
                        "archived_runtime": meta.get('runtime')})
        print(f"  {len(aligned)} words aligned in {elapsed:.2f}s", flush=True)
    summary = {"recordings": len(records), "failed_recordings": len(failures),
               "words": sum(len(r['words']) for r in records),
               "mean_absolute_boundary_shift_ms": float(np.mean(deltas)) if deltas else None,
               "median_signed_boundary_shift_ms": float(np.median([s for r in records for w in r['words'] for s in w['shift_ms']])) if deltas else None,
               "boundaries_shifted_over_50ms": sum(d > 50 for d in deltas),
               "note": "Boundary shifts measure disagreement, not improvement. No human ground truth supplied."}
    result = {"summary": summary, "mfa_version": version.stdout.strip(), "model": MODEL, "dictionary": DICTIONARY, "app_start_offset_s": OFFSET_S,
              "model_fingerprints": fingerprints,
              "runtime": runtime_snapshot(), "recordings": records, "failures": failures}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output.with_suffix('.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>MFA crop trial</title>'
        '<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:0 15px}table{width:100%;border-collapse:collapse}'
        'td,th{text-align:left;padding:10px;border-bottom:1px solid #ccc}audio{max-width:100%;width:240px}'
        'section{margin:40px 0}summary{cursor:pointer;margin:20px 0}td{width:28%}</style>'
        '<h1>MFA vs archived crops</h1><p>Same audio, transcript and word groups. All use the app’s quiet padding, fade and gain. '
        'Archived crops are whatever aligner the app used when the recording was made (named in each table header). '
        f'“MFA as the app plays it” applies the app’s calibration (starts {OFFSET_S*1000:.0f} ms earlier, ends unchanged) '
        'but not its spike-based clamping, which needs the full scoring pass. '
        'MFA is experimental: forced alignment can assign times to words that were omitted or mispronounced. '
        'Different timestamps do not establish better accuracy.</p>'
        f'<p>{len(records)} recordings aligned; {len(failures)} failed.</p>' + ''.join(pages) +
        '<script>document.addEventListener("play",e=>{document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>',
        encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
