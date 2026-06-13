"""``pmtdeconv plot analyze`` — diagnostic plots from a deconv-pulse CSV.

Delegates to ``deconvolution/analyze_deconv.py`` which has its own
preset registry (milliqan, run1000, na22, na22_wavelet) and renders
charge histograms / charge-vs-time / ch1-vs-ch2 plots. Passes through
the legacy script's flags.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.presets import resolve_settings


# Legacy `analyze_deconv.py` --preset choices (separate from pmtdeconv's
# ConfigPreset registry; the legacy preset just selects a CSV path +
# axis labels). Exposed here for help text.
_LEGACY_ANALYZE_PRESETS = ("milliqan", "run1000", "na22", "na22_wavelet")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plot analyze")
    g.add_argument("--analyze-preset", choices=_LEGACY_ANALYZE_PRESETS,
                   default="na22_wavelet",
                   help="Preset CSV selector inside analyze_deconv.py.")
    g.add_argument("--csv", type=str, default=None,
                   help="Override CSV path (skips the analyze-preset map).")
    g.add_argument("--plot", type=str, nargs="+",
                   default=["all"],
                   help="Which plots to render. See analyze_deconv.py for "
                        "the list (charge, charge_tcut, charge_vs_height, "
                        "charge_vs_time, ch1_vs_ch2, all).")


def _default_out_dir(settings, analyze_preset: str) -> Path:
    parts = [f"plot_analyze_{settings.dataset or analyze_preset}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or _default_out_dir(settings, args.analyze_preset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="plot analyze", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    from pmtdeconv.plotting import analyze as legacy  # noqa

    if args.csv is not None:
        input_csv = Path(args.csv)
    else:
        spec = legacy.PRESETS[args.analyze_preset]
        input_csv = Path(spec["csv"])

    if args.csv is not None:
        output_paths = legacy._dispatch(
            input_csv, input_csv.parent.name, channels=None, q_range=(None, None),
            plots=args.plot, ipulse=None, t_cut=None,
            dt_window=None, calib=None, emax=1500.0, split_runs=True,
            output_dir=out_dir,
        )
    else:
        output_paths = legacy._run_preset(
            args.analyze_preset,
            q_range=(None, None), plots=args.plot, ipulse=None,
            chan_override=None, t_cut=None,
            output_dir=out_dir,
        )

    outputs_meta = [
        {"role": "analyze_pdf", "path": str(p)}
        for p in sorted(output_paths)
    ]
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else out_dir / "manifest.json")

    write_manifest(
        subcommand="plot analyze",
        settings=settings, provenance=provenance,
        inputs=[{"role": "pulse_csv", "path": str(input_csv)}],
        outputs=outputs_meta,
        stats={"analyze_preset": args.analyze_preset,
               "plots": args.plot,
               "n_outputs": len(outputs_meta)},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
