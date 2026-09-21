"""Timewalk scan — pure math, no rendering.

Pivot the area CSV + pulse CSV into per-event ``(a_ref, a_scan, dt)``,
slide a window over ``a_scan``, Gauss-fit the dt histogram in each
window, return a DataFrame in the MQAna summary schema
(``ch2_lo, ch2_hi, ch2_center, N, mean_dt, sigma_dt, mean_err,
sigma_err``). Both ``pmtdeconv timewalk run`` and ``pmtdeconv timewalk
gif`` call this. The gif command adds frame rendering on top.

The Gaussian fitter is borrowed from ``timewalk_data_gif._fit_gauss``
to keep results comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class TimewalkScan:
    """Per-bin scan result + the per-event arrays it was built from.

    The arrays are kept on the result so the gif command can re-use
    them when rendering frames (no need to re-pivot the CSVs).
    """
    summary: pd.DataFrame                # MQAna schema, one row per bin
    a_ref: np.ndarray                    # per-event CH1 area (the ref)
    a_scan: np.ndarray                   # per-event CH2 area (the scan)
    dt: np.ndarray                       # per-event Δt (ns), sign-flipped
    base_mask: np.ndarray                # in-window-on-ref + finite-dt mask
    starts: np.ndarray                   # per-bin a_scan low edges
    width: float                         # per-bin a_scan width (uniform mode)
    # Explicit (lo, hi) per bin — set in --a-scan-windows mode where bins
    # have variable widths (e.g. fine at low area, coarse above); None in
    # the uniform start/stop/step mode.
    windows: Optional[np.ndarray] = None


# ─── Gaussian fit (lifted from timewalk_data_gif._fit_gauss) ─────

def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gauss(centers: np.ndarray, counts: np.ndarray
               ) -> Tuple[float, float, float, float, float, bool]:
    """Robust Gaussian fit. Returns (mu, sigma, A, mu_err, sigma_err, ok)."""
    if counts.sum() < 20:
        return (float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), False)
    mu0 = (centers * counts).sum() / counts.sum()
    var0 = ((centers - mu0) ** 2 * counts).sum() / counts.sum()
    sig0 = max(np.sqrt(var0), (centers[1] - centers[0]))
    A0 = counts.max()
    try:
        popt, pcov = curve_fit(
            _gauss, centers, counts,
            p0=[A0, mu0, sig0], maxfev=2000,
        )
        A, mu, sig = popt
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        if sig < 0:
            sig = abs(sig)
        return float(mu), float(sig), float(A), float(perr[1]), float(perr[2]), True
    except Exception:
        return mu0, sig0, A0, float("nan"), float("nan"), False


# ─── CSV merge ────────────────────────────────────────────────────

def load_deconv_data(area_csv: Path, pulse_csv: Path,
                     ref_branch: str = "d0_ch1",
                     scan_branch: str = "d0_ch2",
                     time_col: str = "t_peak") -> pd.DataFrame:
    """Merge per-event area + per-pulse timestamp into a single DataFrame.

    Schema: ``(run, cycle, entry, a_ref, a_scan, t_ref, t_scan)``.
    Lifted from ``timewalk_deconv_gif.load_deconv_data`` to keep behavior
    identical. ``time_col`` picks the pulse-CSV timestamp column:
    ``t_peak`` (deconv apex, legacy) or ``t_half`` (CFD-50 rising-edge
    crossing — the lower-jitter MQAna `halftime` estimator).
    """
    area = pd.read_csv(area_csv)
    if "interval" in area.columns:
        area = area[area["interval"] == 0]
    present = sorted(area["branch"].unique())
    missing = [b for b in (ref_branch, scan_branch) if b not in present]
    if missing:
        raise SystemExit(
            f"area CSV {area_csv} has no rows for branch(es) {missing}. "
            f"Branches present: {present}. "
            f"Fix --ref-branch/--scan-branch.")
    area = area[area["branch"].isin([ref_branch, scan_branch])]
    area_piv = area.pivot_table(
        index=["run", "cycle", "entry"], columns="branch",
        values="area", aggfunc="first",
    ).reset_index()
    area_piv.columns.name = None
    area_piv = area_piv.rename(columns={ref_branch: "a_ref", scan_branch: "a_scan"})

    hits = pd.read_csv(pulse_csv)
    if time_col not in hits.columns:
        raise SystemExit(
            f"pulse CSV {pulse_csv} has no {time_col!r} column. "
            f"(`t_half` is emitted by `pmtdeconv hits` runs from 2026-07-01 "
            f"on — re-run `hits` to regenerate the CSV, or use "
            f"--time-col t_peak.)")
    hits = hits[(hits["success"] == True) & (hits["peak_id"] == 1.0)]
    present = sorted(hits["branch"].unique())
    missing = [b for b in (ref_branch, scan_branch) if b not in present]
    if missing:
        raise SystemExit(
            f"pulse CSV {pulse_csv} has no successful primary pulses for "
            f"branch(es) {missing}. Branches present: {present}. "
            f"Fix --ref-branch/--scan-branch.")
    hits = hits[hits["branch"].isin([ref_branch, scan_branch])]
    hit_piv = hits.pivot_table(
        index=["run", "cycle", "entry"], columns="branch",
        values=time_col, aggfunc="first",
    ).reset_index()
    hit_piv.columns.name = None
    hit_piv = hit_piv.rename(columns={ref_branch: "t_ref", scan_branch: "t_scan"})

    df = area_piv.merge(hit_piv, on=["run", "cycle", "entry"], how="inner")
    df = df.dropna(subset=["a_ref", "a_scan", "t_ref", "t_scan"])
    return df


# ─── Scan ─────────────────────────────────────────────────────────

def scan_timewalk(
    *,
    area_csv: Path,
    pulse_csv: Path,
    a_ref: Tuple[float, float],
    a_scan_start: float,
    a_scan_stop: float,
    a_scan_step: float,
    a_scan_width: Optional[float] = None,
    a_scan_windows: Optional[Sequence[Tuple[float, float]]] = None,
    ref_branch: str = "d0_ch1",
    scan_branch: str = "d0_ch2",
    time_col: str = "t_peak",
    xrange: Tuple[float, float] = (-25.0, 10.0),
    bin_width_ns: float = 0.5051,
    dt_offset: float = 0.0,
    flip_sign: bool = True,
    reverse: bool = False,
) -> TimewalkScan:
    """Sliding-window timewalk scan.

    Returns a :class:`TimewalkScan` with a per-bin DataFrame and the
    per-event arrays. The DataFrame is in the MQAna summary schema
    (see ``pmtdeconv/io/schemas.TIMEWALK_SUMMARY_SCHEMA``).
    """
    df = load_deconv_data(Path(area_csv), Path(pulse_csv),
                          ref_branch=ref_branch, scan_branch=scan_branch,
                          time_col=time_col)

    # Δt = t_scan - t_ref (MQAna convention). The legacy deconv script
    # stored t_ref - t_scan internally; flip_sign=True (default) matches
    # the published timewalk-data plots.
    sign = -1.0 if flip_sign else +1.0
    dt_all = sign * (df["t_ref"].values - df["t_scan"].values) + dt_offset
    a_ref_arr = df["a_ref"].values
    a_scan_arr = df["a_scan"].values

    aref_cut = (a_ref_arr >= a_ref[0]) & (a_ref_arr <= a_ref[1])
    base = aref_cut & np.isfinite(dt_all)

    width = a_scan_width if a_scan_width is not None else a_scan_step
    if a_scan_windows is not None:
        # Variable-width mode: explicit (lo, hi) windows override
        # start/stop/step/width (66727 request: fine bins where events
        # are dense at low area, coarse above — the 66518 4/2 cadence).
        windows = np.asarray([(float(lo), float(hi))
                              for lo, hi in a_scan_windows], dtype=float)
        if reverse:
            windows = windows[::-1]
        starts = windows[:, 0]
    else:
        windows = None
        starts = np.arange(a_scan_start, a_scan_stop + 1e-9, a_scan_step)
        if reverse:
            starts = starts[::-1]
    centers = starts + 0.5 * width

    # Histogram bin edges for the per-window Δt fit. The half-bin shift
    # (2026-06-10) keeps sample-grid Δt values at bin CENTERS: when
    # bin_width == dt_ns, every Δt is an exact multiple of the grid and
    # edge-snapped bins put the data ON the edges — float rounding then
    # flips entries between neighbours, producing a jagged comb and fake
    # spikes (seen as the "Δt=0 spike" in the 66727 study).
    xrange_disp = (xrange[0] + dt_offset, xrange[1] + dt_offset)
    bw = float(bin_width_ns)
    lo_snap = (np.floor(xrange_disp[0] / bw) - 0.5) * bw
    hi_snap = (np.ceil(xrange_disp[1] / bw) + 0.5) * bw + bw
    bin_edges = np.arange(lo_snap, hi_snap + 0.5 * bw, bw)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    rows = []
    bounds = (windows if windows is not None
              else [(lo, lo + width) for lo in starts])
    for lo, hi in bounds:
        sel = base & (a_scan_arr >= lo) & (a_scan_arr <= hi)
        dt_sel = dt_all[sel]
        in_window = (dt_sel > xrange_disp[0]) & (dt_sel < xrange_disp[1])
        n = int(in_window.sum())
        counts, _ = np.histogram(dt_sel[in_window], bins=bin_edges)
        mu, sig, _A, mu_err, sig_err, _ok = _fit_gauss(
            bin_centers, counts.astype(float))
        rows.append({
            "ch2_lo": float(lo),
            "ch2_hi": float(hi),
            "ch2_center": float(0.5 * (lo + hi)),
            "N": n,
            "mean_dt": mu,
            "sigma_dt": sig,
            "mean_err": mu_err,
            "sigma_err": sig_err,
        })
    summary = pd.DataFrame(rows)
    return TimewalkScan(
        summary=summary,
        a_ref=a_ref_arr,
        a_scan=a_scan_arr,
        dt=dt_all,
        base_mask=base,
        starts=starts,
        width=width,
        windows=windows,
    )
