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
