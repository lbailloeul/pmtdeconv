"""``pmtdeconv plot energy`` — per-channel energy histograms.

Calls :func:`pmtdeconv.plotting.energy.render_energy_hist` with paths
resolved from the dataset preset. Works on any dataset whose
DRSprocessing tree contains the standard ``voltages_<board>_<ch>`` /
``area_<board>_<ch>`` branches (Cd109, Na22, MilliQan).

The plot overlays four methods (raw / FFT / STFT / wavelet) for
both the whole-waveform integral and qualifying-pulse integral, per
channel. The SPE template comes from the dataset's default template
CSV; override with ``--template``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plot energy")
    g.add_argument("--cutoff", type=float, default=None,
                   help="Cosine cutoff in MHz (default: 50 for Cd109, "
                        "dataset's preset value otherwise).")
    g.add_argument("--nbins", type=int, default=120)
    g.add_argument("--proc-root", type=str, default=None,
                   help="DRSprocessing ROOT file. Default: "
                        "<data-root>/Run<RUN>.root.")
    g.add_argument("--template", type=str, default=None,
                   help="SPE template CSV. Default: dataset's preset.")
    g.add_argument("--board", type=int, default=2984,
                   help="Board prefix for DRSprocessing branches (default 2984).")
    g.add_argument("--channels", type=int, nargs="+", default=(1, 2))
    g.add_argument("--exclude-saturated", action="store_true",
                   help="Exclude plateau-detected saturated events from the "
                        "energy histograms (their charge is unreliable). "
                        "Default: include all events; the `saturated` flag is "
                        "always written to energy_charges.csv either way.")


def _default_out_dir(settings) -> Path:
    parts = [f"plot_energy_{settings.dataset or 'cd109'}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or _default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="plot energy", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    # Resolve dataset-specific paths
    from pmtdeconv.plotting.energy import render_energy_hist

    runs = settings.runs or []
    if not runs:
        raise SystemExit("--runs is required for `pmtdeconv plot energy`.")
    run_num = runs[0]

    template_csv = (Path(args.template) if args.template
                    else settings.template_csv or settings.default_template_csv)
    if template_csv is None or not Path(template_csv).exists():
        raise SystemExit(
            f"No SPE template found ({template_csv}). Pass --template "
            "or first run `pmtdeconv template build && aggregate "
            f"--dataset {settings.dataset or 'cd109'}`.")

    # DRSprocessing file (Run<N>.root, with area_/voltages_ branches) sits
    # directly under the data root — same dir as the flattened_Run<N>.root
    # waveform files, no `processed/` subdir.
    if args.proc_root:
        proc_root = Path(args.proc_root)
    elif settings.data_root is not None:
        proc_root = Path(settings.data_root) / f"Run{run_num}.root"
    else:
        proc_root = PROJECT_ROOT.parent.parent.parent / "data" / f"Run{run_num}.root"
    if not proc_root.exists():
        raise SystemExit(
            f"DRSprocessing ROOT not found: {proc_root}\n"
            f"Pass --proc-root <path>, or --data-root <dir> holding "
            f"Run{run_num}.root.")

    # Cosine cutoff: explicit flag > dataset preset > 50 MHz (legacy fallback).
    # Bug-fix 8 (2026-05-30 scan): previously this branch used the dataset
    # preset only when settings.dataset == "cd109"; everything else got
    # 50 MHz, which directly contradicted the help text ("Default: 50 MHz
    # for Cd109 and the dataset preset otherwise"). Na22 in particular
    # was silently capped at 50 MHz when its preset is 450.
    cutoff_mhz = (args.cutoff if args.cutoff is not None
                  else (settings.cosine_cutoff_mhz
                        if settings.cosine_cutoff_mhz is not None else 50.0))
    # dt: 'auto' falls back to 1.432 (Cd109 default)
    dt = float(settings.dt_ns) if settings.dt_ns not in ("auto", None) else 1.432

    result = render_energy_hist(
        proc_root=proc_root,
        template_csv=Path(template_csv),
        out_dir=out_dir,
        dataset_label=settings.dataset or "",
        run_label=f"Run {run_num}",
        dt=dt, n_samp=1024,
        board=args.board,
        channels=tuple(args.channels),
        tree_name="Events",
        cutoff_mhz=cutoff_mhz,
        nbins=args.nbins,
        entry_stop=settings.entry_stop,
        exclude_saturated=args.exclude_saturated,
        # Step 6 bug-fix: the wavelet/STFT overlay curves now use the SAME
        # resolved denoiser config (via pipeline.denoise.denoise) as
        # `hits --denoiser <x>` — no longer hardcoded module constants.
        wavelet=settings.wavelet, wlevel=settings.wlevel,
        wmode=settings.wmode, wthresh_k=settings.wthresh_k,
        stft_nperseg_ns=settings.stft_nperseg_ns,
        stft_freq_cutoff_mhz=settings.stft_freq_cutoff_mhz,
        stft_amp_threshold=settings.stft_amp_threshold,
    )

    outputs_meta = [
        {"role": "energy_hist_png",   "path": str(result["png"])},
        {"role": "energy_charges_csv", "path": str(result["csv"])},
    ]

    # Bug-fix (2026-05-30 gallery run): manifest_path was printed below but
    # never defined → NameError crashed the command *after* the plot was
    # written. Resolve it once here.
    manifest_path = (Path(settings.manifest_path) if settings.manifest_path
                     else out_dir / "manifest.json")
    write_manifest(
        subcommand="plot energy",
        settings=settings, provenance=provenance,
        inputs=[
            {"role": "proc_root",    "path": str(proc_root)},
            {"role": "template_csv", "path": str(template_csv)},
        ],
        outputs=outputs_meta,
        stats={"n_outputs": len(outputs_meta),
               "cutoff_mhz": cutoff_mhz,
               "channel_summary": result["summary"]},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"All outputs in: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0
