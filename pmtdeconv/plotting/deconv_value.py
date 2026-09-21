"""Deconvolution-value figures (restored 2026-07-29, 0–20 PE re-scope).

Demoted to legacy/ on 2026-06-13 pending "validated on real Na22 data with
genuine pile-up" — the re-scope's `select multihit` + PE calibration provide
exactly that validation path, and the nPE axis is now REAL photoelectrons
when `spe_charge` is calibrated.

Two figures, both from a deconv-pulse CSV (``all_results.csv`` /
``hits.csv``), that answer "what's the point of deconvolution if the
timewalk resolution doesn't improve" (vault 12c):

1. **charge vs t_peak** — a per-event charge-vs-timing 2D/scatter that exposes
   the timewalk *structure* directly (complements ``timewalk run``'s μ/σ summary).
2. **nPE / multi-hit separation** — the deconvolution-value deliverable: the
   peaks-per-waveform multiplicity (multi-hit events that raw fixed-window
   integration cannot resolve) + the per-pulse charge spectrum (the
   photoelectron-count structure deconvolution recovers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LogNorm
from pmtdeconv.plotting.style import use_root_style, style_root_axes, root_viridis


def _successful(df: pd.DataFrame) -> pd.DataFrame:
    good = df[df["success"] == True] if "success" in df.columns else df
    keep = [c for c in ("charge", "t_peak") if c in good.columns]
    if keep:
        good = good.dropna(subset=keep)
    return good


def render_deconv_value(
    pulse_csv: Path,
    out_dir: Path,
    *,
    dataset_label: str = "",
    run_label: str = "",
    branch: Optional[str] = None,
) -> dict:
    """Write the charge-vs-t_peak and nPE/multi-hit PNGs into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pulse_csv)
    if branch and "branch" in df.columns:
        df = df[df["branch"] == branch]
    good = _successful(df)

    title_suffix = " ".join(x for x in (dataset_label, run_label) if x)
    use_root_style()
    outputs = {}

    # ── Figure 1: per-event charge vs t_peak ─────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    if len(good) and {"t_peak", "charge"}.issubset(good.columns):
        t = good["t_peak"].to_numpy(float)
        q = good["charge"].to_numpy(float)
        finite = np.isfinite(t) & np.isfinite(q)
        t, q = t[finite], q[finite]
        if len(t):
            # Square-pixel 2D histogram (white empty bins + log color scale),
            # matching the marginals heatmap look.
            h2, xe, ye = np.histogram2d(t, q, bins=60)
            cmap = root_viridis()
            cmap.set_bad("white")
            hmax = float(h2.max())
            norm = LogNorm(vmin=1, vmax=max(hmax, 2.0)) if hmax >= 1 else None
            pcm = ax1.imshow(h2.T, origin="lower",
                             extent=(xe[0], xe[-1], ye[0], ye[-1]),
                             aspect="auto", cmap=cmap, norm=norm)
            if norm is None:
                pcm.set_clim(vmin=1)
            cb = fig1.colorbar(pcm, ax=ax1)
            cb.set_label("events / bin")
    ax1.set_xlabel("t_peak [ns]")
    ax1.set_ylabel("pulse charge [pVs]")
    ax1.set_title(f"Per-event charge vs t_peak — {title_suffix}".rstrip(" —"))
    style_root_axes(ax1)
    fig1.tight_layout()
    p1 = out_dir / "charge_vs_tpeak.png"
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)
    outputs["charge_vs_tpeak_png"] = p1

    # ── Figure 2: nPE / multi-hit separation ─────────────────────────
    fig2, (axm, axq) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) peaks-per-waveform multiplicity — the multi-hit story
    mult = pd.Series(dtype=int)
    if len(good) and {"entry", "branch"}.issubset(good.columns):
        mult = good.groupby(["entry", "branch"]).size()
    elif len(good) and "entry" in good.columns:
        mult = good.groupby("entry").size()
    if len(mult):
        mx = int(mult.max())
        edges = np.arange(0.5, mx + 1.5, 1.0)
        axm.hist(mult.to_numpy(), bins=edges, color="royalblue",
                 edgecolor="white")
        frac_multi = float((mult >= 2).mean())
        axm.axvline(1.5, color="crimson", ls="--", lw=1)
        axm.set_title(f"Peaks per waveform "
                      f"({frac_multi * 100:.0f}% multi-hit)")
    else:
        frac_multi = 0.0
        axm.set_title("Peaks per waveform")
    axm.set_xlabel("peaks resolved in one waveform")
    axm.set_ylabel("waveforms")
    axm.set_yscale("log")
    style_root_axes(axm)

    # (b) per-pulse charge spectrum — the nPE structure. When the run was
    # PE-calibrated (spe_charge → the `npe` column, 2026-07-28) the axis is
    # REAL photoelectrons; otherwise pVs (the honest uncalibrated unit).
    use_npe = ("npe" in good.columns and len(good)
               and good["npe"].notna().any())
    qcol, qlabel = (("npe", "pulse charge [nPE]") if use_npe
                    else ("charge", "pulse charge [pVs]"))
    if len(good) and qcol in good.columns:
        q = good[qcol].to_numpy(float)
        q = q[np.isfinite(q) & (q > 0)]
        if q.size:
            hi = float(np.percentile(q, 99.5))
            axq.hist(q, bins=np.linspace(0, hi, 120), histtype="step",
                     lw=1.6, color="crimson")
    axq.set_xlabel(qlabel)
    axq.set_ylabel("pulses / bin")
    axq.set_yscale("log")
    axq.set_title("Per-pulse charge spectrum (nPE structure)")
    style_root_axes(axq)

    fig2.suptitle(f"Deconvolution value: multi-hit separation + nPE — "
                  f"{title_suffix}".rstrip(" —"))
    fig2.tight_layout()
    p2 = out_dir / "npe_multihit.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    outputs["npe_multihit_png"] = p2

    summary = {
        "n_pulses": int(len(good)),
        "n_waveforms": int(len(mult)) if len(mult) else 0,
        "frac_multi_hit": round(float(frac_multi), 4),
        "median_charge": (round(float(good["charge"].median()), 4)
                          if len(good) and "charge" in good.columns else None),
    }
    outputs["summary"] = summary
    return outputs
