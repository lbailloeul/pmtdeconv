"""ch1-vs-ch2 area heatmap + marginal histograms.

Lifted from the standalone ``plot_area_marginals.py`` so both the
legacy script and ``pmtdeconv plot marginals`` share one implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from matplotlib.colors import LogNorm
from pmtdeconv.plotting.style import (
    use_root_style, style_root_axes, minimal_axes, root_viridis,
)


def _find_top_peaks(h: np.ndarray, x: np.ndarray, n: int = 2) -> list:
    """Top-n local maxima of a 1D histogram, sorted by position."""
    idx, _ = find_peaks(h, distance=max(3, len(h) // 40))
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(h[idx])[::-1]][:n]
    return sorted(x[idx])


def render_marginals(
    area_csv: Path,
    out_path: Path,
    *,
    title: str = "",
    scale: float = 1e-3,
    xlim: Tuple[float, float] = (0.0, 200.0),
    ylim: Tuple[float, float] = (0.0, 200.0),
    bins: int = 100,
    ref_branch: str = "d0_ch1",
    scan_branch: str = "d0_ch2",
    metric: str = "area",
) -> dict:
    """Render the marginals PDF/PNG. Returns ``{n_both, med_ref, med_scan,
    peaks_ref, peaks_scan}`` so the caller can echo summary stats.

    ``metric`` selects which per-event column to correlate — ``area`` (default,
    byte-identical to the original) or ``peak_height`` (the bare-heatmap
    capability folded in from the retired ``scatter2d.plot_scatter``).
    """
    A = pd.read_csv(area_csv)
    if "interval" in A.columns:
        A = A[A.interval == 0]
    if metric not in A.columns:
        metric = "area"
    a1 = A[A.branch == ref_branch][["entry", metric]].rename(columns={metric: "a1"})
    a2 = A[A.branch == scan_branch][["entry", metric]].rename(columns={metric: "a2"})
    ac = a1.merge(a2, on="entry").dropna()
    ac["a1"] *= scale
    ac["a2"] *= scale
    both_pos = ac[(ac.a1 > 0) & (ac.a2 > 0)]
    n_both = len(both_pos)
    med1 = float(both_pos.a1.median()) if n_both else float("nan")
    med2 = float(both_pos.a2.median()) if n_both else float("nan")

    x_edges = np.linspace(*xlim, bins + 1)
    y_edges = np.linspace(*ylim, bins + 1)
    h2, _, _ = np.histogram2d(ac.a1, ac.a2, bins=(x_edges, y_edges))
    h_x, _ = np.histogram(ac.a1, bins=x_edges)
    h_y, _ = np.histogram(ac.a2, bins=y_edges)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    peaks_x = _find_top_peaks(h_x, xc, n=2)
    peaks_y = _find_top_peaks(h_y, yc, n=2)

    use_root_style()
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                          wspace=0.02, hspace=0.02)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Log color scale (2×10^n colorbar) with white empty bins: root_viridis
    # sets under/bad → white, so 0-count bins drop out of the scale and only
    # populated bins are colormapped.
    cmap = root_viridis()
    cmap.set_bad("white")
    hmax = float(h2.max())
    norm = LogNorm(vmin=1, vmax=max(hmax, 2.0)) if hmax >= 1 else None
    pcm = ax_main.imshow(h2.T, origin="lower",
                         extent=(*xlim, *ylim),
                         aspect="auto", cmap=cmap, norm=norm)
    if norm is None:
        pcm.set_clim(vmin=1)
    # Deconvolved area is genuinely in pVs (unit-area kernel × unity-DC-gain
    # filter); the default scale=1e-3 converts to nVs. Peak height stays a.u.
    if metric == "area":
        unit = ("[nVs]" if np.isclose(scale, 1e-3)
                else ("[pVs]" if scale == 1 else f"[pVs × {scale:g}]"))
    else:
        unit = f"[arb. units × {scale:g}]" if scale != 1 else "[arb. units]"
    ax_main.set_xlabel(f"CH1 {metric.replace('_', ' ')} {unit}")
    ax_main.set_ylabel(f"CH2 {metric.replace('_', ' ')} {unit}")
    style_root_axes(ax_main)

    # 1D marginal overlays: keep the simple gray-fill minimal look (per request).
    ax_top.fill_between(xc, h_x, step="mid", color="gray", alpha=0.7)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("CH1")
    ax_right.fill_betweenx(yc, h_y, step="mid", color="gray", alpha=0.7)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel("CH2")
    for a in (ax_top, ax_right):
        minimal_axes(a)
    # Counts can't be negative — cap the marginal count axes at 0 (kill the
    # default ~5% margin that dips below zero).
    ax_top.set_ylim(bottom=0)
    ax_right.set_xlim(left=0)

    cb = fig.colorbar(pcm, ax=ax_right, fraction=0.4, pad=0.04)
    cb.set_label("events / bin")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_both": n_both, "med_ref": med1, "med_scan": med2,
        "peaks_ref": peaks_x, "peaks_scan": peaks_y,
    }
