"""Pictures of what the recogniser saw."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .align import Segment  # noqa: E402
from .engine import Emissions  # noqa: E402


def posterior_heatmap(em: Emissions, expected: list[Segment] | None = None, max_rows: int = 28):
    """Non-blank posterior mass per phone over time, expected phones marked with boxes.

    Rows are the phones that mattered: every expected phone plus any phone the
    model gave real weight to anywhere, ordered by when they first light up.
    """
    probs = np.exp(em.log_probs)
    probs[:, em.blank_id] = 0.0

    rows: list[int] = []
    for seg in expected or []:
        if seg.phone_id not in rows:
            rows.append(seg.phone_id)
    peak = probs.max(axis=0)
    for idx in np.argsort(peak)[::-1]:
        if peak[idx] < 0.15 or len(rows) >= max_rows:
            break
        if int(idx) not in rows:
            rows.append(int(idx))

    def first_light(idx: int) -> int:
        hits = np.flatnonzero(probs[:, idx] > 0.3)
        return int(hits[0]) if hits.size else em.n_frames

    rows.sort(key=first_light)
    row_of = {idx: r for r, idx in enumerate(rows)}

    total_s = em.frame_to_s(em.n_frames)
    fig, ax = plt.subplots(figsize=(max(7.0, total_s * 2.2), 0.3 * len(rows) + 1.4))
    ax.imshow(
        probs[:, rows].T,
        aspect="auto",
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, total_s, len(rows), 0.0),
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([em.labels[i] for i in rows], fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_title("phone posteriors (blank removed) – boxes: where each expected phone was aligned", fontsize=9)

    for seg in expected or []:
        r = row_of[seg.phone_id]
        x0, x1 = em.frame_to_s(seg.start), em.frame_to_s(seg.end)
        ax.add_patch(plt.Rectangle((x0, r), x1 - x0, 1.0, fill=False, edgecolor="cyan", linewidth=1.2))
    fig.tight_layout()
    return fig


def timeline_figure(result, max_seconds: float = 20.0):
    """Waveform with energy, every phone's spike, and the word/phone crops the learner hears.

    Reads top to bottom: the signal, the recogniser's spikes (coloured by score,
    dropped phones hollow), and the spans used for replay. If a crop looks wrong,
    this is the picture that shows why.
    """
    from . import SAMPLE_RATE
    from .boundaries import energy_db, speech_threshold, HOP

    audio = result.audio[: int(max_seconds * SAMPLE_RATE)]
    t = np.arange(len(audio)) / SAMPLE_RATE
    db = energy_db(audio)
    thr = speech_threshold(db)
    t_db = np.arange(len(db)) * HOP / SAMPLE_RATE
    colors = {"good": "#2e8b57", "unsure": "#e0a800", "off": "#c0392b"}

    fig, (ax_wave, ax_align) = plt.subplots(
        2, 1, figsize=(max(8.0, len(audio) / SAMPLE_RATE * 2.0), 4.6), sharex=True, height_ratios=[2, 1.4]
    )
    step = max(1, len(audio) // 6000)
    ax_wave.plot(t[::step], audio[::step], color="#4a6fa5", linewidth=0.6)
    ax_wave.set_ylabel("waveform")
    ax_wave.set_ylim(-1.05, 1.05)
    ax_e = ax_wave.twinx()
    ax_e.plot(t_db, db, color="#888", linewidth=0.8, alpha=0.8)
    ax_e.axhline(thr, color="#888", linestyle=":", linewidth=0.8)
    ax_e.set_ylabel("energy dB", color="#888")
    ax_e.set_ylim(min(db.min(), thr - 10), max(db.max() + 5, thr + 10))

    spans = result.word_spans()
    p_spans = result.phone_spans()
    for k, (w, span, segs, ps) in enumerate(zip(result.words, spans, result.segments_by_word(), p_spans)):
        shade = "#e8f4ea" if w.category == "good" else ("#fff4d6" if w.category == "unsure" else "#fbe3e0")
        for ax in (ax_wave, ax_align):
            ax.axvspan(span.start, span.end, color=shade, zorder=0)
        ax_align.text((span.start + span.end) / 2, 0.98 if k % 2 == 0 else 0.86, w.word, ha="center", va="top", fontsize=9, fontweight="bold")
        for p, seg, pspan in zip(w.phones, segs, ps):
            s0, s1 = result.emissions.frame_to_s(seg.start), result.emissions.frame_to_s(seg.end)
            face = "white" if p.dropped else colors[p.category]
            ax_align.bar((s0 + s1) / 2, 0.45, width=max(s1 - s0, 0.01), bottom=0.05, color=face, edgecolor=colors[p.category], linewidth=1.2, zorder=3)
            if pspan is not None:
                ax_align.plot([pspan.start, pspan.start], [0.0, 0.55], color="#555", linewidth=0.6, zorder=2)
                ax_align.text((pspan.start + pspan.end) / 2, 0.58, p.expected, ha="center", va="bottom", fontsize=7, color="#333")
    for run in result.extra:
        a, b = result.emissions.frame_to_s(run.start), result.emissions.frame_to_s(run.end)
        ax_align.axvspan(a, b, color="#ddd", alpha=0.6, zorder=0)
        ax_align.text((a + b) / 2, 0.92, "extra", ha="center", va="top", fontsize=7, color="#666")
    ax_align.set_ylim(0, 1)
    ax_align.set_yticks([])
    ax_align.set_xlabel("time (s)")
    ax_align.set_title("spikes (filled = scored sound, hollow = not heard) · ticks = phone onsets · shading = word crops", fontsize=8)
    fig.tight_layout()
    return fig
