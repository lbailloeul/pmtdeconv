"""``pmtdeconv plot deconv-value`` — the deconvolution-value figures (step 6).

Two figures from a deconv-pulse CSV: per-event charge-vs-t_peak (timewalk
structure) and nPE / multi-hit separation (the visual answer to "what's the
point of deconvolution"). Input CSV resolves from ``--pulse-csv`` or the
dataset/preset/run via :func:`pmtdeconv.io.output_paths.resolve_pulse_csv`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import resolve_pulse_csv, timewalk_output_dir
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plot deconv-value")
    # Renamed from --pulse-csv on 2026-09-21: the file comes from `hits`.
    # (Internal attribute stays `pulse_csv`.)
    from pmtdeconv.cli import add_renamed_alias
    add_renamed_alias(g, "--pulse-csv", "--hits-csv", dest="pulse_csv")
    g.add_argument("--hits-csv", dest="pulse_csv", type=str, default=None,
                   help="Deconv-pulse CSV (all_results.csv / hits.csv). "
                        "Default: resolved from --dataset/--preset/--runs.")
    g.add_argument("--branch", type=str, default=None,
                   help="Restrict to one branch (e.g. d0_ch1).")


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or (timewalk_output_dir(settings).parent
                                   / f"deconv_value_{settings.dataset or 'run'}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="plot deconv-value", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    pulse_csv = (Path(args.pulse_csv) if args.pulse_csv
                 else resolve_pulse_csv(settings))
    if pulse_csv is None or not Path(pulse_csv).exists():
        raise SystemExit(
            "No deconv-pulse CSV found. Pass --pulse-csv <path>, or run "
            "`pmtdeconv hits --dataset <ds> --runs <n>` first.")

    from pmtdeconv.plotting.deconv_value import render_deconv_value
    run_label = f"Run {settings.runs[0]}" if settings.runs else ""
    result = render_deconv_value(
        Path(pulse_csv), out_dir,
        dataset_label=settings.dataset or "", run_label=run_label,
        branch=args.branch,
    )

    outputs_meta = [
        {"role": "charge_vs_tpeak_png", "path": str(result["charge_vs_tpeak_png"])},
        {"role": "npe_multihit_png", "path": str(result["npe_multihit_png"])},
    ]
    manifest_path = (Path(settings.manifest_path) if settings.manifest_path
                     else out_dir / "manifest.json")
    write_manifest(
        subcommand="plot deconv-value", settings=settings, provenance=provenance,
        inputs=[{"role": "pulse_csv", "path": str(pulse_csv)}],
        outputs=outputs_meta, stats=result["summary"],
        out_path=manifest_path, duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"All outputs in: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0
