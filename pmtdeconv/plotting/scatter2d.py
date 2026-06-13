"""Per-channel 1D area/peak-height quicklook for the ``area`` command.

Lifted from ``deconvolution/deconv_area_Na22.py``. The CH1-vs-CH2 *heatmap*
that used to live here (``plot_scatter``) was retired in step 6 of the
architecture restructure — ``plot marginals --metric {area,peak_height}`` is
now the single canonical CH1-vs-CH2 correlation deliverable (heatmap + 1D
marginal projections + photopeak-finding + summary stats), a strict superset
of the old bare heatmap. Only this per-channel 1D quicklook remains, emitted
as a side effect of ``pmtdeconv area``. Pure matplotlib; byte-identity of the
``area`` CSV is unaffected.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmtdeconv.plotting.style import use_minimal_style, minimal_axes


def plot_1d(df, run_label: str, amax: float, bins: int,
            out_dir: Path, metric: str = "area", tag: str = "",
            dataset: str = "Na22") -> None:
    """Per-channel 1D log-y histogram of the area/peak_height column.

    ``dataset`` kwarg controls the title (defaults to ``"Na22"`` for
    back-compat with legacy direct callers; ``pmtdeconv area`` passes the
    resolved ``settings.dataset``).
    """
    col = metric
    label = "Deconvolved area [a.u.]" if metric == "area" else "Peak height [a.u.]"
    use_minimal_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    edges = np.linspace(0, amax, bins + 1)
    for ch, color in [("ch1", "C0"), ("ch2", "C1")]:
        v = df[df["branch"].str.endswith(ch)][col].dropna()
        v = v[v > 0].values
        ax.hist(v, bins=edges, histtype="step", color=color, lw=1.6,
                label=f"CH{ch[-1]}")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("Events")
    ax.legend()
    minimal_axes(ax)
    fig.tight_layout()
    out = out_dir / f"deconv_{metric}_1d_run{run_label}{tag}.pdf"
    fig.savefig(out, dpi=200, transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")
