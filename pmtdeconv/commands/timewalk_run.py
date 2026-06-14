"""``pmtdeconv timewalk run`` — scan-only timewalk (no GIF).

Reads pmtdeconv-produced ``area_csv`` (from ``waveform`` or
``raw_area mqana`` or ``raw_area drs``) and ``pulse_csv`` (from ``pulse``).
Slides a window over ``a_scan`` (CH2 area), Gauss-fits ``Δt`` in each
window, writes the per-bin curve as a MQAna-format CSV
(``timewalk_summary_run<RUN>.csv``) + a curve PNG.

Use this for fast diagnostic scans without the multi-minute GIF render.
For the full animated view use ``pmtdeconv timewalk gif``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmtdeconv.plotting.style import use_root_style, style_root_axes

from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import (
    resolve_area_csv,
    resolve_pulse_csv,
    timewalk_output_dir,
)
from pmtdeconv.io.schemas import TIMEWALK_SUMMARY_COLUMNS
from pmtdeconv.pipeline.timewalk import scan_timewalk
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("timewalk")
    g.add_argument("--area-csv", type=str, default=None,
                   help="Per-event area CSV. Default: auto-resolved from "
                        "<dataset> + <preset> + <run> by looking under "
                        "raw_area_* and waveform output dirs.")
    g.add_argument("--pulse-csv", type=str, default=None,
                   help="Per-event deconv-pulse CSV. Default: auto-resolved from "
                        "outputs/deconv_pulse_<dataset>_lab_cosine_wavelet"
                        "[__<preset>]/.")
    g.add_argument("--ref-branch", type=str, default="d0_ch1")
    g.add_argument("--scan-branch", type=str, default="d0_ch2")
    g.add_argument("--a-ref", type=float, nargs=2, default=(100_000.0, 110_000.0),
                   metavar=("LO", "HI"),
                   help="Reference-channel area window in pVs (default "
                        "100000 110000 = the Na22 511 keV CH1 band).")
    g.add_argument("--a-scan-start", type=float, default=0.0)
    g.add_argument("--a-scan-stop", type=float, default=195_000.0)
    g.add_argument("--a-scan-step", type=float, default=5_000.0)
    g.add_argument("--a-scan-width", type=float, default=None)
    g.add_argument("--xrange", type=float, nargs=2, default=(-25.0, 10.0),
                   metavar=("MIN", "MAX"),
                   help="Δt window for the per-bin Gaussian fit (ns).")
    g.add_argument("--bin-width", type=float, default=0.5051,
                   help="dt histogram bin width in ns (default 0.5051, "
                        "the DRS4 sample quantum).")
    g.add_argument("--dt-offset", type=float, default=0.0)
    g.add_argument("--area-units", choices=("pVs", "nVs"), default="nVs",
                   help="Units for the curve PNG's CH2-area axis (default "
                        "nVs; CSVs are always pVs). Mirrors `timewalk gif`.")
    g.add_argument("--no-flip-sign", action="store_true",
                   help="Keep the deconv-script's sign convention "
                        "(t_ref - t_scan). Default flips to (t_scan - t_ref) "
                        "to match the published baseline.")
    g.add_argument("--reverse", action="store_true")


autoresolve_area_csv = resolve_area_csv
autoresolve_pulse_csv = resolve_pulse_csv
default_out_dir = timewalk_output_dir


def _branch_label(branch: str) -> str:
    """`d0_ch1` → `CH1` (display label for axis text)."""
    return branch.split("_")[-1].upper()


def save_curve_png(summary, run_id: str, out_path: Path,
                   area_units: str = "nVs",
                   ref_branch: str = "d0_ch1",
                   scan_branch: str = "d0_ch2") -> None:
    """Plot the timewalk curve (mean_dt vs a_scan_center) with error bars."""
    ref_lab, scan_lab = _branch_label(ref_branch), _branch_label(scan_branch)
    use_root_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    scale = 1e-3 if area_units == "nVs" else 1.0
    cx = summary["ch2_center"] * scale
    mu = summary["mean_dt"]
    sig = summary["sigma_dt"]
    mu_err = summary["mean_err"]
    ok = mu.notna() & sig.notna()
    ax.errorbar(cx[ok], mu[ok], yerr=mu_err[ok].fillna(0),
                fmt="o", color="tab:blue", capsize=2, label=r"$\mu_{\Delta t}$")
    ax.fill_between(cx[ok], mu[ok] - sig[ok], mu[ok] + sig[ok],
                    alpha=0.18, color="tab:blue", label=r"$\pm\sigma$")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel(f"{scan_lab} area [{area_units}]")
    ax.set_ylabel(rf"$\Delta t$ ({scan_lab} - {ref_lab}) [ns]")
    ax.set_title(f"Run {run_id} — timewalk")
    style_root_axes(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    out_dir = settings.out_dir or default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="timewalk run", settings=settings,
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

    scan = scan_timewalk(
        area_csv=area_csv, pulse_csv=pulse_csv,
        a_ref=tuple(args.a_ref),
        a_scan_start=args.a_scan_start, a_scan_stop=args.a_scan_stop,
        a_scan_step=args.a_scan_step, a_scan_width=args.a_scan_width,
        ref_branch=args.ref_branch, scan_branch=args.scan_branch,
        xrange=tuple(args.xrange),
        bin_width_ns=args.bin_width,
        dt_offset=args.dt_offset,
        flip_sign=not args.no_flip_sign,
        reverse=args.reverse,
    )
    print(f"  merged events (after a_ref cut): {int(scan.base_mask.sum())}")
    print(f"  scan bins: {len(scan.summary)}")

    run_id = (settings.runs or ["unknown"])[0]
    summary_csv = out_dir / f"timewalk_summary_run{run_id}.csv"
    scan.summary[list(TIMEWALK_SUMMARY_COLUMNS)].to_csv(summary_csv, index=False)
    print(f"  Saved summary CSV: {summary_csv}")

    curve_png = out_dir / f"timewalk_curve_run{run_id}.png"
    # Bug-fix B3 (2026-05-29 audit): honor --area-units on the curve axis
    # (previously hardcoded nVs).
    save_curve_png(scan.summary, str(run_id), curve_png,
                   area_units=args.area_units,
                   ref_branch=args.ref_branch, scan_branch=args.scan_branch)
    print(f"  Saved curve PNG: {curve_png}")

    # Bug-fix B4 (2026-05-29 audit): resolve the manifest path once so the
    # printed path matches what was actually written when --manifest is set.
    manifest_path = (Path(settings.manifest_path) if settings.manifest_path
                     else out_dir / "manifest.json")
    write_manifest(
        subcommand="timewalk run",
        settings=settings, provenance=provenance,
        inputs=[
            {"role": "area_csv", "path": str(area_csv)},
            {"role": "pulse_csv", "path": str(pulse_csv)},
        ],
        outputs=[
            {"role": "timewalk_summary", "path": str(summary_csv)},
            {"role": "timewalk_curve_png", "path": str(curve_png)},
        ],
        stats={"n_merged": int(scan.base_mask.sum()),
               "n_scan_bins": int(len(scan.summary))},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
