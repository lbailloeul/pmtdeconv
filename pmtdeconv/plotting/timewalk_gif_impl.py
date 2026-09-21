#!/usr/bin/env python3
"""
timewalk_data_gif.py — MQAna-style timewalk animation directly from the
DRSprocessing ROOT tree (e.g. ``processed/Run65213.root``).

No re-flattening required. Per-event waveform branches
(``voltages_<board>_<ch>``, ``times_<board>_<ch>``) and per-event scalars
(``area_<board>_<ch>``) are read with ``uproot`` and the two rising-edge
timing observables are computed in vectorised numpy:

    t_half : time the rising edge crosses 0.5 * vMax    (sample interpolation)
    t_fit  : time the 20–80% rising-edge linear fit crosses 0.5 * vMax

For each frame the histogram of ``(t_scan - t_ref)`` is filled and Gaussian-fit
for events with reference-channel area in a fixed photopeak window and
scan-channel area in a sliding window. Frames are stitched into a GIF with PIL.

Mirrors the spirit of the MQAna ``timewalk.csh`` from the UCSB E41473 logbook.

Example
-------
.. code-block:: bash

    python timewalk_data_gif.py \\
        --root /Users/leobailloeul/Documents/coding/LANL/data_processing/drsProcessing/processed/Run65213.root \\
        --board 2984 --ch_ref 1 --ch_scan 2 \\
        --a_ref 165000 180000 \\
        --a_scan_start 0 --a_scan_stop 150000 --a_scan_step 5000 \\
        --outdir outputs/timewalk_gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import curve_fit
from tqdm import tqdm

MIN_VMAX_MV = 30.0
NSAMPLES_DEFAULT = 1024


def _root_kbird_cmap():
    """ROOT ``kBird`` palette as a matplotlib colormap, with empty (under) = white.

    Colour stops are the official ``kBird`` values from ROOT's TColor.
    Calling code should pair this with ``vmin`` slightly above 0 so zero-count
    bins fall under the threshold and render as white.
    """
    stops = [
        (0.0000, (0.2082, 0.1664, 0.5293)),
        (0.1250, (0.0592, 0.3599, 0.8684)),
        (0.2500, (0.0780, 0.5041, 0.8385)),
        (0.3750, (0.0232, 0.6419, 0.7914)),
        (0.5000, (0.1802, 0.7178, 0.6425)),
        (0.6250, (0.5301, 0.7491, 0.4661)),
        (0.7500, (0.8186, 0.7328, 0.3499)),
        (0.8750, (0.9956, 0.7862, 0.1968)),
        (1.0000, (0.9764, 0.9831, 0.0538)),
    ]
    cmap = LinearSegmentedColormap.from_list("root_kBird", stops, N=256)
    cmap.set_under("white")
    return cmap


_KBIRD = _root_kbird_cmap()


# -----------------------------------------------------------------------------
# Per-event timing observables
# -----------------------------------------------------------------------------
def compute_rising_edge_times(
    volts: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised half-max-crossing and 20-80% linear-fit crossings.

    Parameters
    ----------
    volts : (N, M) float array
        Sampled voltages, positive-going pulses.
    times : (N, M) float array
        Sample times (ns), same shape.

    Returns
    -------
    t_half : (N,) array — sample-interpolated time where v == 0.5*vmax.
    t_fit  : (N,) array — 0.5*vmax crossing of the 20-80% linear fit.
    vmax   : (N,) array — recorded peak voltage.
    """
    N, M = volts.shape
    idx = np.arange(M)[None, :]
    vmax = volts.max(axis=1)
    imax = volts.argmax(axis=1)
    pre = idx <= imax[:, None]

    thr_half = 0.5 * vmax

    # --- half-max sample interpolation -------------------------------------
    below = (volts < thr_half[:, None]) & pre
    has_below = below.any(axis=1)
    # last "below" index before the peak
    i_below = (below * idx).max(axis=1)
    i_below_p1 = np.minimum(i_below + 1, M - 1)

    rows = np.arange(N)
    v0 = volts[rows, i_below]
    v1 = volts[rows, i_below_p1]
    t0 = times[rows, i_below]
    t1 = times[rows, i_below_p1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_half = np.where(
            has_below & (v1 > v0),
            t0 + (thr_half - v0) * (t1 - t0) / (v1 - v0),
            np.nan,
        )

    # --- 20-80% linear-fit crossing of vmax/2 ------------------------------
    mask = (volts >= 0.2 * vmax[:, None]) & (volts <= 0.8 * vmax[:, None]) & pre
    n = mask.sum(axis=1).astype(float)
    sx = (times * mask).sum(axis=1)
    sy = (volts * mask).sum(axis=1)
    sxy = (times * volts * mask).sum(axis=1)
    sx2 = (times * times * mask).sum(axis=1)
    denom = n * sx2 - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(denom != 0, (n * sxy - sx * sy) / denom, np.nan)
        b = np.where(n > 0, (sy - a * sx) / np.maximum(n, 1), np.nan)
        t_fit = np.where(
            (n >= 2) & np.isfinite(a) & (a > 0),
            (thr_half - b) / a,
            np.nan,
        )

    return t_half, t_fit, vmax


# -----------------------------------------------------------------------------
# Tree iteration
# -----------------------------------------------------------------------------
def read_event_observables(
    rootpath: str,
    board: int,
    ch_ref: int,
    ch_scan: int,
    step_size: int = 10_000,
    max_events: int | None = None,
    tree_name: str = "Events",
) -> dict[str, np.ndarray]:
    """Stream the tree in chunks; return per-event scalars only."""
    sref = f"{board}_{ch_ref}"
    sscan = f"{board}_{ch_scan}"
    branches = [
        f"voltages_{sref}",
        f"times_{sref}",
        f"voltages_{sscan}",
        f"times_{sscan}",
        f"area_{sref}",
        f"area_{sscan}",
        f"vMax_{sref}",
        f"vMax_{sscan}",
    ]

    out = {k: [] for k in ("a_ref", "a_scan", "v_ref", "v_scan",
                            "t_half_ref", "t_half_scan",
                            "t_fit_ref", "t_fit_scan")}

    with uproot.open(rootpath) as f:
        tree = f[tree_name]
        n_total = tree.num_entries if max_events is None else min(
            max_events, tree.num_entries
        )
        pbar = tqdm(total=n_total, desc="events", unit="evt")
        seen = 0
        for chunk in tree.iterate(branches, step_size=step_size, library="np"):
            v_ref = chunk[f"voltages_{sref}"]
            t_ref = chunk[f"times_{sref}"]
            v_scan = chunk[f"voltages_{sscan}"]
            t_scan = chunk[f"times_{sscan}"]
            a_ref = chunk[f"area_{sref}"]
            a_scan = chunk[f"area_{sscan}"]
            vmax_ref_stored = chunk[f"vMax_{sref}"]
            vmax_scan_stored = chunk[f"vMax_{sscan}"]

            if max_events is not None and seen + len(a_ref) > max_events:
                k = max_events - seen
                v_ref, t_ref = v_ref[:k], t_ref[:k]
                v_scan, t_scan = v_scan[:k], t_scan[:k]
                a_ref, a_scan = a_ref[:k], a_scan[:k]
                vmax_ref_stored = vmax_ref_stored[:k]
                vmax_scan_stored = vmax_scan_stored[:k]

            th_ref, tf_ref, _ = compute_rising_edge_times(v_ref, t_ref)
            th_scan, tf_scan, _ = compute_rising_edge_times(v_scan, t_scan)

            out["a_ref"].append(a_ref)
            out["a_scan"].append(a_scan)
            out["v_ref"].append(vmax_ref_stored)
            out["v_scan"].append(vmax_scan_stored)
            out["t_half_ref"].append(th_ref)
            out["t_half_scan"].append(th_scan)
            out["t_fit_ref"].append(tf_ref)
            out["t_fit_scan"].append(tf_scan)

            seen += len(a_ref)
            pbar.update(len(a_ref))
            if max_events is not None and seen >= max_events:
                break
        pbar.close()

    return {k: np.concatenate(v) for k, v in out.items()}


# -----------------------------------------------------------------------------
# Frame rendering & GIF assembly
# -----------------------------------------------------------------------------
def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gauss(centers: np.ndarray, counts: np.ndarray):
    """Robust Gaussian fit. Returns (mu, sigma, A, fit_ok)."""
    if counts.sum() < 20:
        return np.nan, np.nan, np.nan, False
    mu0 = (centers * counts).sum() / counts.sum()
    var0 = ((centers - mu0) ** 2 * counts).sum() / counts.sum()
    sig0 = max(np.sqrt(var0), (centers[1] - centers[0]))
    A0 = counts.max()
    try:
        popt, _ = curve_fit(
            _gauss,
            centers,
            counts,
            p0=[A0, mu0, sig0],
            maxfev=2000,
        )
        A, mu, sig = popt
        if sig < 0:
            sig = abs(sig)
        return mu, sig, A, True
    except Exception:
        return mu0, sig0, A0, False


def render_frame(
    dt: np.ndarray,
    *,
    method_label: str,
    run_id: str,
    a_ref_lo: float,
    a_ref_hi: float,
    a_scan_lo: float,
    a_scan_hi: float,
    xrange: tuple[float, float],
    nbins: int,
    outpath: Path,
    scatter_data: dict | None = None,
    area_unit_scale: float = 1.0,
    area_unit_label: str = "pVs",
    dpi: int = 110,
    ref_label: str = "CH1",
    scan_label: str = "CH2",
    log_y: bool = False,
) -> tuple[float, float, int]:
    """Render one frame; return (mu, sigma, n_in_window).

    If ``scatter_data`` is provided (keys ``H``, ``xedges``, ``yedges`` from
    :func:`numpy.histogram2d` on the CH1/CH2 area plane), a second panel is
    drawn on the right with the COLZ-style 2D heatmap and a red rectangle
    showing the active (``a_ref``, ``a_scan``) selection box. Axis values are
    multiplied by ``area_unit_scale`` for display.
    """
    if isinstance(nbins, np.ndarray):
        # caller passed pre-computed bin edges
        edges = nbins
    else:
        edges = np.linspace(xrange[0], xrange[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_window = (dt > xrange[0]) & (dt < xrange[1]) & np.isfinite(dt)
    counts, _ = np.histogram(dt[in_window], bins=edges)
    mu, sig, A, ok = _fit_gauss(centers, counts.astype(float))

    from pmtdeconv.plotting.style import use_minimal_style, minimal_axes, root_viridis
    use_minimal_style()
    if scatter_data is not None:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax2 = None

    ax.step(centers, counts, where="mid", color="#1f77b4", lw=1.5, label="data")
    if ok:
        xs = np.linspace(xrange[0], xrange[1], 400)
        ax.plot(
            xs,
            _gauss(xs, A, mu, sig),
            color="#d62728",
            lw=1.8,
            label=fr"$\mu={mu:.2f}$ ns, $\sigma={sig:.2f}$ ns",
        )
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    else:
        ax.set_yscale("linear")
    ax.set_xlim(*xrange)
    ax.set_xlabel(f"{scan_label} time - {ref_label} time [ns]")
    ax.set_ylabel("entries")
    minimal_axes(ax)
    ax.legend(loc="upper right")

    if ax2 is not None:
        from matplotlib.colors import LogNorm
        H = scatter_data["H"]
        xe = scatter_data["xedges"] * area_unit_scale
        ye = scatter_data["yedges"] * area_unit_scale
        vmax = max(H.max(), 1)
        im = ax2.imshow(
            H.T,
            origin="lower",
            extent=(xe[0], xe[-1], ye[0], ye[-1]),
            aspect="auto",
            cmap=root_viridis(),
            # log z (2026-06-10): zero-count bins are masked -> white
            norm=LogNorm(vmin=0.5, vmax=vmax),
            interpolation="nearest",
        )
        ax2.add_patch(
            Rectangle(
                (a_ref_lo * area_unit_scale, a_scan_lo * area_unit_scale),
                (a_ref_hi - a_ref_lo) * area_unit_scale,
                (a_scan_hi - a_scan_lo) * area_unit_scale,
                fill=False,
                edgecolor="red",
                lw=2.0,
            )
        )
        ax2.set_xlabel(f"{ref_label} pulse area ({area_unit_label})")
        ax2.set_ylabel(f"{scan_label} pulse area ({area_unit_label})")
        minimal_axes(ax2)
        cbar = fig.colorbar(im, ax=ax2, pad=0.02)
        cbar.set_label("entries")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return (mu if ok else np.nan, sig if ok else np.nan, int(in_window.sum()))


def stitch_gif(frame_paths: Iterable[Path], outpath: Path, duration_ms: int = 300):
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        raise RuntimeError("no frames to stitch")
    frames[0].save(
        outpath,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_summary(
    a_scan_centers: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    counts: np.ndarray,
    method_label: str,
    run_id: str,
    outpath: Path,
):
    """Plot per-frame gaussian mu and sigma vs scan area: the timewalk curve."""
    from pmtdeconv.plotting.style import use_minimal_style, minimal_axes
    use_minimal_style()
    fig, (ax_mu, ax_sig) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ok = counts > 20
    ax_mu.errorbar(a_scan_centers[ok], mu[ok], yerr=sig[ok] / np.sqrt(np.maximum(counts[ok], 1)),
                    fmt="o-", color="#1f77b4")
    ax_mu.set_ylabel(r"$\mu_{\Delta t}$ [ns]")
    minimal_axes(ax_mu)

    ax_sig.plot(a_scan_centers[ok], sig[ok], "s-", color="#d62728")
    ax_sig.set_ylabel(r"$\sigma_{\Delta t}$ [ns]")
    ax_sig.set_xlabel("scan-channel area [pVs]")
    minimal_axes(ax_sig)

    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="path to processed ROOT file")
    p.add_argument("--board", type=int, default=2984, help="board id in branch names")
    p.add_argument("--ch_ref", type=int, default=1, help="reference channel")
    p.add_argument("--ch_scan", type=int, default=2, help="scan (sliding-area) channel")
    p.add_argument("--a_ref", type=float, nargs=2, default=(165_000.0, 180_000.0),
                   metavar=("LO", "HI"), help="reference-channel area window (pVs)")
    p.add_argument("--a_scan_start", type=float, default=0.0,
                   help="scan-area lower edge of first window")
    p.add_argument("--a_scan_stop", type=float, default=150_000.0,
                   help="scan-area lower edge of last window")
    p.add_argument("--a_scan_step", type=float, default=5_000.0,
                   help="step between sliding window starts")
    p.add_argument("--a_scan_width", type=float, default=None,
                   help="width of each sliding window (defaults to step)")
    p.add_argument("--xrange", type=float, nargs=2, default=(-15.0, 15.0),
                   metavar=("MIN", "MAX"), help="time-difference range (ns)")
    p.add_argument("--nbins", type=int, default=100, help="histogram bins")
    p.add_argument("--bin_width", type=float, default=None,
                   help="histogram bin width in ns; if given, overrides --nbins "
                        "and snaps bin edges to multiples of this width "
                        "(set to ~1.432 ns to match deconv t_peak quantisation)")
    p.add_argument("--outdir", default="outputs/timewalk_gif", help="output directory")
    p.add_argument("--frame_ms", type=int, default=300, help="GIF frame duration (ms)")
    p.add_argument("--max_events", type=int, default=None,
                   help="cap on events processed (for testing)")
    p.add_argument("--step_size", type=int, default=10_000,
                   help="uproot iterate chunk size")
    p.add_argument("--min_vmax", type=float, default=MIN_VMAX_MV,
                   help="minimum vMax (mV) to keep an event (timing quality)")
    p.add_argument("--reverse", action="store_true",
                   help="animate from largest scan area down to smallest (csh-style)")
    p.add_argument("--tree", default="Events", help="tree name in the ROOT file")
    p.add_argument("--no_scatter", action="store_true",
                   help="disable the CH1-vs-CH2 area heatmap overlay")
    p.add_argument("--scatter_max", type=float, default=250_000.0,
                   help="upper edge of CH1/CH2 area axes in the overlay (pVs)")
    p.add_argument("--scatter_nbins", type=int, default=50,
                   help="bins per axis in the CH1-vs-CH2 area heatmap")
    p.add_argument("--area_units", choices=("pVs", "nVs"), default="nVs",
                   help="units for area-axis labels (display only; cuts stay in pVs)")
    p.add_argument("--dt_offset", type=float, default=0.0,
                   help="constant offset (ns) added to (t_scan - t_ref); use to "
                        "absorb the per-channel cable/electronics calibration that "
                        "MQAna applies but isn't in the python-processed tree")
    p.add_argument("--area_csv", type=str, default=None,
                   help="optional path to deconvolution area CSV "
                        "(e.g. outputs/deconv_area_Na22/areas_run<RUN>.csv). "
                        "If given, per-event areas are taken from this CSV "
                        "(matched on entry index) instead of the ROOT tree's "
                        "area_<board>_<ch> branches.")
    p.add_argument("--area_csv_ref_branch", type=str, default=None,
                   help="branch label for the reference channel in --area_csv "
                        "(default: 'd0_ch<ch_ref>')")
    p.add_argument("--area_csv_scan_branch", type=str, default=None,
                   help="branch label for the scan channel in --area_csv "
                        "(default: 'd0_ch<ch_scan>')")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(exist_ok=True)

    run_id = Path(args.root).stem.replace("Run", "")

    print(f"Reading {args.root}  (tree={args.tree})")
    data = read_event_observables(
        args.root,
        board=args.board,
        ch_ref=args.ch_ref,
        ch_scan=args.ch_scan,
        step_size=args.step_size,
        max_events=args.max_events,
        tree_name=args.tree,
    )
    N = len(data["a_ref"])
    print(f"  read {N} events")

    # --- optional: override per-event areas with external CSV (e.g. deconv) ---
    if args.area_csv is not None:
        import pandas as pd
        csv_path = args.area_csv
        ref_branch = args.area_csv_ref_branch or f"d0_ch{args.ch_ref}"
        scan_branch = args.area_csv_scan_branch or f"d0_ch{args.ch_scan}"
        print(f"  loading external area CSV: {csv_path}")
        adf = pd.read_csv(csv_path)
        if "interval" in adf.columns:
            adf = adf[adf["interval"] == 0]
        # build per-entry lookup for the two branches
        adf = adf[adf["branch"].isin([ref_branch, scan_branch])]
        piv = adf.pivot_table(
            index="entry", columns="branch", values="area", aggfunc="first",
        )
        if ref_branch not in piv.columns or scan_branch not in piv.columns:
            raise SystemExit(
                f"--area_csv: expected branches {ref_branch!r} and {scan_branch!r} "
                f"in CSV; found {sorted(piv.columns)}"
            )
        ref_lookup = piv[ref_branch].to_dict()
        scan_lookup = piv[scan_branch].to_dict()
        entries = np.arange(N)
        new_ref = np.array([ref_lookup.get(int(e), np.nan) for e in entries])
        new_scan = np.array([scan_lookup.get(int(e), np.nan) for e in entries])
        n_matched = int(np.sum(np.isfinite(new_ref) & np.isfinite(new_scan)))
        print(f"  matched areas for {n_matched}/{N} events from CSV "
              f"(branches: ref={ref_branch!r}, scan={scan_branch!r})")
        data["a_ref"] = new_ref
        data["a_scan"] = new_scan

    # --- quality cuts -------------------------------------------------------
    qual = (
        (data["v_ref"] > args.min_vmax)
        & (data["v_scan"] > args.min_vmax)
        & np.isfinite(data["t_half_ref"]) & np.isfinite(data["t_half_scan"])
    )
    aref_cut = (
        np.isfinite(data["a_ref"])
        & np.isfinite(data["a_scan"])
        & (data["a_ref"] >= args.a_ref[0])
        & (data["a_ref"] <= args.a_ref[1])
    )
    base = qual & aref_cut
    print(f"  passing quality + ref-area cut: {int(base.sum())}")

    dt_half = data["t_half_scan"] - data["t_half_ref"] + args.dt_offset
    dt_fit = data["t_fit_scan"] - data["t_fit_ref"] + args.dt_offset
    # Shift the displayed x-axis window by the same offset, so a --xrange
    # chosen for the raw distribution still brackets the histogram peak.
    xrange_disp = (args.xrange[0] + args.dt_offset, args.xrange[1] + args.dt_offset)
    if args.dt_offset != 0.0:
        print(f"  applied dt_offset = {args.dt_offset:+.4f} ns to (t_scan - t_ref)")
        print(f"  displayed x-axis range shifted to "
              f"[{xrange_disp[0]:+.3f}, {xrange_disp[1]:+.3f}] ns")

    # Bin edges — either from --bin_width (snapped to multiples of bw) or --nbins.
    if args.bin_width is not None:
        bw = float(args.bin_width)
        lo_snap = np.floor(xrange_disp[0] / bw) * bw
        hi_snap = np.ceil(xrange_disp[1] / bw) * bw + bw
        bin_edges = np.arange(lo_snap, hi_snap + 0.5 * bw, bw)
        nbins_eff = len(bin_edges) - 1
        print(f"  using --bin_width {bw:.4f} ns  ({nbins_eff} bins, "
              f"edges aligned to multiples of bw)")
    else:
        bin_edges = None
        nbins_eff = args.nbins

    # --- sliding window setup ----------------------------------------------
    a_scan_width = args.a_scan_width if args.a_scan_width is not None else args.a_scan_step
    starts = np.arange(args.a_scan_start, args.a_scan_stop + 1e-9, args.a_scan_step)
    if args.reverse:
        starts = starts[::-1]
    centers = starts + 0.5 * a_scan_width

    # --- CH1-vs-CH2 area heatmap (drawn on every frame) --------------------
    scatter_data = None
    if not args.no_scatter:
        H, xedges, yedges = np.histogram2d(
            data["a_ref"],
            data["a_scan"],
            bins=args.scatter_nbins,
            range=[[0.0, args.scatter_max], [0.0, args.scatter_max]],
        )
        scatter_data = {"H": H, "xedges": xedges, "yedges": yedges}
    area_unit_scale = 1e-3 if args.area_units == "nVs" else 1.0
    area_unit_label = args.area_units

    for method, dt_all in (("half", dt_half), ("fit", dt_fit)):
        print(f"\n=== {method} timewalk: {len(starts)} frames ===")
        method_dir = frames_dir / method
        method_dir.mkdir(exist_ok=True)

        mus = np.full(len(starts), np.nan)
        sigs = np.full(len(starts), np.nan)
        cnts = np.zeros(len(starts), dtype=int)

        for i, lo in enumerate(tqdm(starts, desc=f"{method} frames")):
            hi = lo + a_scan_width
            sel = base & (data["a_scan"] >= lo) & (data["a_scan"] <= hi)
            mu, sig, n = render_frame(
                dt_all[sel],
                method_label=method,
                run_id=run_id,
                a_ref_lo=args.a_ref[0],
                a_ref_hi=args.a_ref[1],
                a_scan_lo=lo,
                a_scan_hi=hi,
                xrange=xrange_disp,
                nbins=bin_edges if bin_edges is not None else nbins_eff,
                outpath=method_dir / f"frame_{i:03d}.png",
                scatter_data=scatter_data,
                area_unit_scale=area_unit_scale,
                area_unit_label=area_unit_label,
            )
            mus[i] = mu; sigs[i] = sig; cnts[i] = n

        gif_path = outdir / f"timewalk_{method}_Run{run_id}.gif"
        frame_paths = sorted(method_dir.glob("frame_*.png"))
        stitch_gif(frame_paths, gif_path, duration_ms=args.frame_ms)
        print(f"  wrote {gif_path}")

        # animation-vs-area summary
        order = np.argsort(centers)  # plot in ascending area order regardless of reverse
        render_summary(
            centers[order], mus[order], sigs[order], cnts[order],
            method_label=method, run_id=run_id,
            outpath=outdir / f"timewalk_curve_{method}_Run{run_id}.png",
        )

        # also save the per-frame numbers
        np.savez(
            outdir / f"timewalk_{method}_Run{run_id}.npz",
            a_scan_lo=starts, a_scan_hi=starts + a_scan_width,
            a_scan_center=centers,
            mu=mus, sigma=sigs, counts=cnts,
        )

    print("done.")


if __name__ == "__main__":
    main()
