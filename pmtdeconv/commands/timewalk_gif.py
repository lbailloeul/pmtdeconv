"""``pmtdeconv timewalk gif`` — full animated timewalk visualization.

Same scan math as ``pmtdeconv timewalk run`` (shared
``pmtdeconv.pipeline.timewalk.scan_timewalk``), plus per-frame
rendering via the legacy ``timewalk_data_gif.render_frame`` /
``stitch_gif`` helpers, plus the right-panel CH1-vs-CH2 area heatmap
built from the FULL coincident-area set (not the smaller hit-matched
subset) — matching the behavior the 2026-05 session settled on.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.commands.timewalk_run import (
    autoresolve_area_csv, autoresolve_pulse_csv,
    default_out_dir, save_curve_png,
)
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.schemas import TIMEWALK_SUMMARY_COLUMNS
from pmtdeconv.pipeline.timewalk import scan_timewalk
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("timewalk gif")
    # Shared with `timewalk run`
    g.add_argument("--area-csv", type=str, default=None)
    g.add_argument("--pulse-csv", type=str, default=None)
    g.add_argument("--ref-branch", type=str, default="d0_ch1")
    g.add_argument("--scan-branch", type=str, default="d0_ch2")
    g.add_argument("--a-ref", type=float, nargs=2,
                   default=(100_000.0, 110_000.0), metavar=("LO", "HI"))
    g.add_argument("--a-scan-start", type=float, default=0.0)
    g.add_argument("--a-scan-stop", type=float, default=195_000.0)
    g.add_argument("--a-scan-step", type=float, default=5_000.0)
    g.add_argument("--a-scan-width", type=float, default=None)
    g.add_argument("--a-scan-windows", type=str, default=None,
                   help="Explicit variable-width scan windows as "
                        "'lo:hi,lo:hi,...' in pVs (e.g. fine bins at low "
                        "area, coarse above). Overrides "
                        "--a-scan-start/stop/step/width.")
    g.add_argument("--xrange", type=float, nargs=2, default=(-25.0, 10.0),
                   metavar=("MIN", "MAX"))
    g.add_argument("--bin-width", type=float, default=0.5051)
    g.add_argument("--dt-offset", type=float, default=0.0)
    g.add_argument("--no-flip-sign", action="store_true")
    g.add_argument("--reverse", action="store_true")
    # gif-only
    g.add_argument("--frame-ms", type=int, default=300)
    g.add_argument("--no-scatter", action="store_true")
    g.add_argument("--scatter-max", type=float, default=None,
                   help="Force a square heatmap range [0, MAX] on both axes "
                        "(pVs). Default: dynamic per-axis q99.5 of the data.")
    g.add_argument("--scatter-nbins", type=int, default=40,
                   help="Right-panel heatmap bins per axis (default 40, "
                        "= 5 nVs per bin over 0-200 nVs).")
    g.add_argument("--area-units", choices=("pVs", "nVs"), default="nVs")
    g.add_argument("--log-y", action="store_true",
                   help="Log-scale the Δt-histogram y-axis (entries) so the "
                        "tails of the timing distribution are visible.")
    g.add_argument("--method-label", default=r"deconv $t_{peak}$")


def _build_scatter_data(area_csv: Path, args, scatter_max: float,
                        scatter_nbins: int) -> dict:
    """Pre-compute the right-panel heatmap from the full coincident-area
    set (matches the 2026-05 session's decision)."""
    area = pd.read_csv(area_csv)
    if "interval" in area.columns:
        area = area[area["interval"] == 0]
    area = area[area["branch"].isin([args.ref_branch, args.scan_branch])]
    ap = area.pivot_table(
        index=["run", "cycle", "entry"], columns="branch",
        values="area", aggfunc="first",
    ).dropna(subset=[args.ref_branch, args.scan_branch])
    sx = ap[args.ref_branch].values
    sy = ap[args.scan_branch].values
    # Dynamic per-axis range (2026-06-10): when --scatter-max is not
    # given, size each axis to its own q99.5 so asymmetric pairs (e.g.
    # a 3 nVs ref against a 45 nVs scan) are not squashed into one
    # square range. Passing --scatter-max restores the square behavior.
    if scatter_max is None:
        x_hi = float(np.quantile(sx, 0.995)) if len(sx) else 1.0
        y_hi = float(np.quantile(sy, 0.995)) if len(sy) else 1.0
    else:
        x_hi = y_hi = scatter_max
    H, xedges, yedges = np.histogram2d(
        sx, sy, bins=scatter_nbins,
        range=[[0.0, x_hi], [0.0, y_hi]])
    return {"H": H, "xedges": xedges, "yedges": yedges, "n_coincident": len(sx)}


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    out_dir = settings.out_dir or default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    if maybe_dry_run_exit(subcommand="timewalk gif", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    area_csv = (Path(args.area_csv) if args.area_csv
                else autoresolve_area_csv(settings))
    pulse_csv = (Path(args.pulse_csv) if args.pulse_csv
               else autoresolve_pulse_csv(settings))
    if area_csv is None or not area_csv.exists():
        raise SystemExit(
            "Could not resolve --area-csv. Pass it explicitly, or first run "
            "`pmtdeconv raw-area mqana --dataset <ds> --runs <run> [...]`, "
            "`pmtdeconv raw-area drs --dataset <ds> --runs <run> [...]`, "
            "or `pmtdeconv area --dataset <ds> --runs <run> [...]`.")
    if pulse_csv is None or not pulse_csv.exists():
        raise SystemExit(
            "Could not resolve --pulse-csv. Pass it explicitly, or first run "
            "`pmtdeconv hits --dataset <ds> --runs <run> [...]`.")
    print(f"Area CSV : {area_csv}")
    print(f"Pulse CSV: {pulse_csv}")

    a_scan_windows = None
    if args.a_scan_windows:
        a_scan_windows = [tuple(float(v) for v in w.split(":"))
                          for w in args.a_scan_windows.split(",")]
        bad = [w for w in a_scan_windows if len(w) != 2 or w[1] <= w[0]]
        if bad:
            raise SystemExit(f"--a-scan-windows: bad window(s) {bad}; "
                             f"expected 'lo:hi,lo:hi,...' with hi > lo.")

    scan = scan_timewalk(
        area_csv=area_csv, pulse_csv=pulse_csv,
        a_ref=tuple(args.a_ref),
        a_scan_start=args.a_scan_start, a_scan_stop=args.a_scan_stop,
        a_scan_step=args.a_scan_step, a_scan_width=args.a_scan_width,
        a_scan_windows=a_scan_windows,
        ref_branch=args.ref_branch, scan_branch=args.scan_branch,
        xrange=tuple(args.xrange),
        bin_width_ns=args.bin_width,
        dt_offset=args.dt_offset,
        flip_sign=not args.no_flip_sign,
        reverse=args.reverse,
    )
    print(f"  merged events : {int(scan.base_mask.sum())}")

    scatter_data = None
    if not args.no_scatter:
        scatter_data = _build_scatter_data(area_csv, args,
                                           args.scatter_max, args.scatter_nbins)
        print(f"  scatter built from full area CSV: "
              f"{scatter_data['n_coincident']} coincident events")

    # Reuse the legacy renderer for visual identity with prior outputs
    from pmtdeconv.plotting.timewalk_gif_impl import render_frame, stitch_gif  # noqa

    xrange_disp = (args.xrange[0] + args.dt_offset,
                   args.xrange[1] + args.dt_offset)
    bw = float(args.bin_width)
    # Half-bin shift: keep sample-grid Δt values at bin centers (see
    # scan_timewalk — avoids the edge-collision comb when bw == dt_ns).
    lo_snap = (np.floor(xrange_disp[0] / bw) - 0.5) * bw
    hi_snap = (np.ceil(xrange_disp[1] / bw) + 0.5) * bw + bw
    bin_edges = np.arange(lo_snap, hi_snap + 0.5 * bw, bw)
    area_unit_scale = 1e-3 if args.area_units == "nVs" else 1.0

    run_id = str((settings.runs or ["unknown"])[0])
    method_dir = frames_dir / "deconv"
    method_dir.mkdir(exist_ok=True)
    a_scan_width = (args.a_scan_width if args.a_scan_width is not None
                    else args.a_scan_step)

    # Recompute per-bin selection masks and render frames
    frame_bounds = (scan.windows if scan.windows is not None
                    else [(lo, lo + a_scan_width) for lo in scan.starts])
    for i, (lo, hi) in enumerate(tqdm(frame_bounds, desc="deconv frames")):
        sel = scan.base_mask & (scan.a_scan >= lo) & (scan.a_scan <= hi)
        render_frame(
            scan.dt[sel],
            method_label=args.method_label,
            run_id=run_id,
            a_ref_lo=args.a_ref[0],
            a_ref_hi=args.a_ref[1],
            a_scan_lo=float(lo),
            a_scan_hi=float(hi),
            xrange=xrange_disp,
            nbins=bin_edges,
            outpath=method_dir / f"frame_{i:03d}.png",
            scatter_data=scatter_data,
            area_unit_scale=area_unit_scale,
            area_unit_label=args.area_units,
            ref_label=args.ref_branch.split("_")[-1].upper(),
            scan_label=args.scan_branch.split("_")[-1].upper(),
            log_y=args.log_y,
        )

    gif_path = out_dir / f"timewalk_deconv_Run{run_id}.gif"
    frame_paths = sorted(method_dir.glob("frame_*.png"))
    stitch_gif(frame_paths, gif_path, duration_ms=args.frame_ms)
    print(f"wrote {gif_path}")

    summary_csv = out_dir / f"timewalk_summary_run{run_id}.csv"
    scan.summary[list(TIMEWALK_SUMMARY_COLUMNS)].to_csv(summary_csv, index=False)
    curve_png = out_dir / f"timewalk_curve_run{run_id}.png"
    # Bug-fix 13 (2026-05-30 final scan): pass --area-units through.
    save_curve_png(scan.summary, run_id, curve_png,
                   area_units=args.area_units,
                   ref_branch=args.ref_branch, scan_branch=args.scan_branch)
    npz_path = out_dir / f"timewalk_deconv_Run{run_id}.npz"
    np.savez(npz_path,
             a_scan_lo=scan.summary["ch2_lo"].values,
             a_scan_hi=scan.summary["ch2_hi"].values,
             a_scan_center=scan.summary["ch2_center"].values,
             mu=scan.summary["mean_dt"].values,
             sigma=scan.summary["sigma_dt"].values,
             counts=scan.summary["N"].values.astype(np.int64))

    manifest_path = (Path(settings.manifest_path) if settings.manifest_path
                     else out_dir / "manifest.json")
    write_manifest(
        subcommand="timewalk gif",
        settings=settings, provenance=provenance,
        inputs=[
            {"role": "area_csv", "path": str(area_csv)},
            {"role": "pulse_csv", "path": str(pulse_csv)},
        ],
        outputs=[
            {"role": "timewalk_summary", "path": str(summary_csv)},
            {"role": "timewalk_curve_png", "path": str(curve_png)},
            {"role": "timewalk_npz", "path": str(npz_path)},
            {"role": "timewalk_gif", "path": str(gif_path)},
        ],
        stats={"n_merged": int(scan.base_mask.sum()),
               "n_scan_bins": int(len(scan.summary)),
               "n_frames": int(len(scan.starts))},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
