"""``pmtdeconv plot marginals`` — ch1-vs-ch2 area heatmap + 1D marginals.

Wraps ``pmtdeconv.plotting.marginals.render_marginals``. Auto-resolves
``--area-csv`` from ``--dataset`` + ``--preset`` + ``--runs`` when not
supplied. Outputs a single PDF/PNG plus the standard manifest.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import resolve_area_csv
from pmtdeconv.plotting.marginals import render_marginals
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plot marginals")
    g.add_argument("--area-csv", type=str, default=None,
                   help="Per-event area CSV (output of `waveform`, "
                        "`raw_area mqana`, or `raw_area drs`). Auto-resolved if "
                        "omitted.")
    g.add_argument("--ref-branch", type=str, default="d0_ch1")
    g.add_argument("--scan-branch", type=str, default="d0_ch2")
    g.add_argument("--metric", choices=("area", "peak_height"), default="area",
                   help="Per-event quantity to correlate (default area). "
                        "`peak_height` folds in the retired scatter2d heatmap.")
    g.add_argument("--title", type=str, default="")
    g.add_argument("--scale", type=float, default=1e-3,
                   help="Multiply area by this for display (default 1e-3 "
                        "= pVs→nVs).")
    g.add_argument("--xlim", type=float, nargs=2, default=(0.0, 200.0))
    g.add_argument("--ylim", type=float, nargs=2, default=(0.0, 200.0))
    g.add_argument("--bins", type=int, default=100)
    g.add_argument("--out-name", type=str, default=None,
                   help="Output filename (default: marginals_run<RUN>.png).")


def _default_out_dir(settings) -> Path:
    parts = [f"plot_marginals_{settings.dataset or 'unknown'}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    out_dir = settings.out_dir or _default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="plot marginals", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    area_csv = Path(args.area_csv) if args.area_csv else resolve_area_csv(settings)
    if area_csv is None or not area_csv.exists():
        raise SystemExit(
            "Could not resolve --area-csv. Pass it explicitly, or first "
            "produce an area CSV via `pmtdeconv area`, "
            "`pmtdeconv raw-area mqana`, or `pmtdeconv raw-area drs`.")
    print(f"Area CSV : {area_csv}")

    run_id = (settings.runs or ["unknown"])[0]
    out_name = args.out_name or f"marginals_run{run_id}.png"
    out_path = out_dir / out_name
    stats = render_marginals(
        area_csv, out_path,
        title=args.title or f"{settings.dataset} run {run_id}",
        scale=args.scale,
        xlim=tuple(args.xlim), ylim=tuple(args.ylim),
        bins=args.bins,
        ref_branch=args.ref_branch, scan_branch=args.scan_branch,
        metric=args.metric,
    )
    print(f"  N(both>0) = {stats['n_both']}  "
          f"med CH1 = {stats['med_ref']:.1f}  med CH2 = {stats['med_scan']:.1f}")
    print(f"  Saved {out_path}")

    write_manifest(
        subcommand="plot marginals",
        settings=settings, provenance=provenance,
        inputs=[{"role": "area_csv", "path": str(area_csv)}],
        outputs=[{"role": "marginals_png", "path": str(out_path)}],
        stats={
            "n_both_positive": int(stats["n_both"]),
            "median_ref": stats["med_ref"],
            "median_scan": stats["med_scan"],
            "peaks_ref": stats["peaks_ref"],
            "peaks_scan": stats["peaks_scan"],
        },
        out_path=(Path(settings.manifest_path) if settings.manifest_path
                  else out_dir / "manifest.json"),
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    return 0
