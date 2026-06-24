"""``pmtdeconv area`` — whole-waveform deconvolution summary per trigger.

Delegates to ``deconvolution/deconv_area_Na22.process_run`` so the math
is byte-identical to the legacy pipeline. The CLI wrapper layer here
handles preset resolution, output-directory naming, the run loop, and
the manifest.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import waveform_output_dir
from pmtdeconv.io.schemas import AREA_COLUMNS
from pmtdeconv.io.templates import load_template
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("waveform")
    g.add_argument("--template", type=str, default=None,
                   help="SPE template CSV path. Default: dataset's preset.")
    g.add_argument("--sig-start", type=int, default=10)
    g.add_argument("--sig-end", type=int, default=1000)
    g.add_argument("--n-intervals", type=int, default=1)
    g.add_argument("--cosine-cutoff-mhz", type=float, default=None)
    g.add_argument("--wavelet", type=str, default=None)
    g.add_argument("--wlevel", type=int, default=None)
    g.add_argument("--wmode", type=str, default=None)
    g.add_argument("--wthresh-k", type=float, default=None)
    g.add_argument("--fft-pad", action="store_true")
    g.add_argument("--wrap-ratio", type=float, default=0.5)
    g.add_argument("--wrap-abs-k", type=float, default=0.0)
    g.add_argument("--afterpulse-window", type=float, default=0.0)
    g.add_argument("--afterpulse-ratio", type=float, default=5.0)
    g.add_argument("--no-trigger-amp", type=float, default=15.0)
    g.add_argument("--t-peak-min-ns", type=float, default=20.0)
    g.add_argument("--metric", type=str, default="peak_height",
                   choices=("area", "peak_height"))
    g.add_argument("--bins", type=int, default=80)
    g.add_argument("--amax", type=float, default=None)
    parser.set_defaults(_settings_defaults={"wthresh_k": 2.0})


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    # process_run + FALLBACK_TEMPLATE now live in the neutral pipeline
    # module (PR 2 / decision 9). process_run_object handles LANL
    # object-format ROOT (PR 4).
    from pmtdeconv.pipeline.process_area import (    # noqa
        process_run, process_run_object, FALLBACK_TEMPLATE,
    )
    # Step 6: the CH1-vs-CH2 heatmap is now the canonical `plot marginals`
    # deliverable (with --metric {area,peak_height}); the bare scatter2d
    # heatmap is retired. Only the per-channel 1D quicklook stays here.
    from pmtdeconv.plotting.scatter2d import plot_1d  # noqa

    # Templates
    template_csv = settings.template_csv
    # strict=False: warn + fall back to FALLBACK_TEMPLATE when the configured
    # template CSV is absent (e.g. a fresh clone with no built Cd109 template),
    # matching the hits/tune paths instead of hard-crashing.
    templates = load_template(template_csv, strict=False)
    # S9: per-channel (per-tube) templates override the shared CSV for the
    # listed channels — each tube uses its OWN SPE artifact. Additive: datasets
    # without `templates:` are unaffected.
    if getattr(settings, "template_csv_by_channel", None):
        from pmtdeconv.io.templates import load_templates_by_channel
        templates.update(load_templates_by_channel(
            settings.template_csv_by_channel, strict=False))
    # Always provide a fallback so unknown branches still get a kernel
    templates.setdefault("_default", FALLBACK_TEMPLATE)

    # Output dir
    out_dir = settings.out_dir or waveform_output_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="waveform", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    runs = settings.runs or [65213]
    data_dir = settings.data_root or (PROJECT_ROOT.parent.parent.parent
                                      / "data" / "Na22_lab")
    is_object_format = (settings.format == "object")

    outputs_meta = []
    inputs_meta = []
    if template_csv is not None and Path(template_csv).exists():
        inputs_meta.append({"role": "template_csv", "path": str(template_csv)})

    stats = {"runs": [], "total_rows": 0, "format": settings.format}
    n_intervals = args.n_intervals
    for run_num in runs:
        if is_object_format:
            # Object-format LANL files use sub-run naming
            # (MilliQan_Run835.<sub>_triple.root). Pick the first match
            # for this run via the dataset's file_glob.
            candidates = sorted(Path(data_dir).glob(settings.file_glob))
            candidates = [p for p in candidates if str(run_num) in p.name]
            if not candidates:
                print(f"[skip] no object-format ROOT matching run {run_num}")
                continue
            root_path = candidates[0]
        else:
            from pmtdeconv.io.data_paths import resolve_run_file
            root_path = resolve_run_file(data_dir, run_num, kind="flat")
            if root_path is None:
                print(f"[skip] run {run_num}: no flattened file under {data_dir}")
                continue
        inputs_meta.append({"role": "root_data", "path": str(root_path)})
        print(f"══ Run {run_num} ══")

        # Resolve dt_ns (object format may auto-detect from tree)
        if settings.dt_ns == "auto":
            from pmtdeconv.io.object_format import read_dt_ns
            dt_ns_resolved = read_dt_ns(root_path, tree_name=settings.tree or "Events")
            print(f"  dt_ns (auto-detected): {dt_ns_resolved} ns")
        else:
            dt_ns_resolved = float(settings.dt_ns)

        common_kwargs = dict(
            root_path=root_path, dt_ns=dt_ns_resolved,
            sig_start=args.sig_start, sig_end=args.sig_end,
            n_intervals=n_intervals,
            cosine_cutoff_mhz=settings.cosine_cutoff_mhz,
            wavelet=settings.wavelet, wlevel=settings.wlevel,
            mode=settings.wmode, thresh_k=settings.wthresh_k,
            no_trigger_amp=args.no_trigger_amp,
            t_peak_min_ns=args.t_peak_min_ns,
            afterpulse_window_ns=args.afterpulse_window,
            afterpulse_height_ratio=args.afterpulse_ratio,
            fft_pad=args.fft_pad,
            wrap_ratio=args.wrap_ratio,
            wrap_abs_k=args.wrap_abs_k,
            entry_stop=settings.entry_stop,
            templates=templates,
            branches_keep=(settings.branch_pattern
                           if isinstance(settings.branch_pattern, list)
                           else None),
        )
        if is_object_format:
            df = process_run_object(**common_kwargs,
                                    polarity=int(settings.polarity),
                                    run_num=run_num,
                                    tree_name=settings.tree or "Events")
        else:
            df = process_run(**common_kwargs, polarity=int(settings.polarity))

        # CSV output: drop left_edge-vetoed rows (interval == -1)
        # to stay byte-identical with the legacy `main()` (which writes
        # `df[df.interval >= 0]`).
        suffix = settings.out_suffix or ""
        csv_path = out_dir / f"areas_run{run_num}{suffix}.csv"
        # Deconv-area output = the raw-area schema + the per-waveform `saturated`
        # data-quality flag (plateau-detected). Kept off the shared AREA_COLUMNS
        # so raw-area mqana/drs are unaffected.
        cols = list(AREA_COLUMNS) + (
            ["saturated"] if "saturated" in df.columns else [])
        df_passed = df[df["interval"] >= 0][cols]
        df_passed.to_csv(csv_path, index=False)
        outputs_meta.append({"role": "area_csv", "path": str(csv_path)})
        stats["runs"].append(run_num)
        stats["total_rows"] += int(len(df_passed))
        print(f"  Saved area CSV: {csv_path}  "
              f"({len(df_passed)} passed / {len(df)} total)")

        # Plots — pure matplotlib helpers lifted into pmtdeconv.plotting.
        # Pass the resolved dataset so the plot title isn't hardcoded
        # to "Na22" for cd109/milliqan/LANL runs.
        only = df[df["interval"] == 0].dropna(subset=[args.metric])
        amax = args.amax or float(only[args.metric].quantile(0.99))
        dataset_label = (settings.dataset or "Na22")
        plot_1d(only, str(run_num), amax, args.bins, out_dir,
                metric=args.metric, tag=suffix,
                dataset=dataset_label)
        outputs_meta.append({"role": "marginal_pdf",
                             "path": str(out_dir / f"deconv_{args.metric}_1d_run{run_num}.pdf")})

    # Fail loudly if a flat run was requested but no file was found anywhere
    # (rather than writing an empty CSV that silently breaks downstream steps).
    if not is_object_format and runs and not stats["runs"]:
        from pmtdeconv.io.data_paths import not_found_message
        raise SystemExit(not_found_message(data_dir, runs[0], kind="flat"))

    # Manifest
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="waveform",
        settings=settings,
        provenance=provenance,
        inputs=inputs_meta,
        outputs=outputs_meta,
        stats=stats,
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"\nAll outputs in: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0
