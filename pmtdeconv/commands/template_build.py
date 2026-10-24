"""``pmtdeconv template build`` — per-(dataset, file, channel) stacked SPE templates.

Delegates to ``pipeline.template_build_impl.main``. Since 2026-07-02 the
build config comes from the ONE dataset registry (``presets/datasets.py`` +
YAML cards) via ``template_build_impl.config_from_preset`` — so any
``--dataset`` name (or card path) that works for ``hits`` works here too.
Omit ``--dataset`` to build every dataset in the registry. The per-file
templates are then aggregated per channel automatically (since 2026-10;
``--no-aggregate`` stops after the per-file step).
Previously this module kept a hand-maintained alias map onto the impl's own
parallel 8-entry registry and monkey-patched its module globals; new
datasets could not be template-built without editing that legacy list.

The retired legacy list's only entry without a registry equivalent was
``MilliQan_2150`` (``data/2150``): reproduce it with
``--dataset milliqan --data-root <...>/data/2150``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("template build")
    g.add_argument("--no-aggregate", action="store_true",
                   help="Stop after the per-file templates; skip the "
                        "per-channel aggregation (`template aggregate` "
                        "can run it later).")
    g.add_argument("--no-single-peak", action="store_true",
                   help="Disable the default single-peak veto and stack every "
                        "band-selected waveform (legacy behavior). Templates "
                        "are single-PE responses by definition, so the veto "
                        "(no secondary >=25%% of primary beyond +-5 ns) is ON "
                        "by default — 2026-06-10 finding: pulse trains were "
                        "97%% of the 66727 ch2 band and inflated tau_fall "
                        "40 ns vs the true 2.8 ns.")
    g.add_argument("--spe-band", nargs=2, type=float, metavar=("LO", "HI"),
                   default=None,
                   help="Override the SPE selection amplitude band (peak height "
                        "vMax in mV, baseline-subtracted, polarity-corrected). "
                        "The template stacks only isolated single pulses whose "
                        "vMax falls in (LO, HI]. Defaults to the dataset's "
                        "preset band. Requires --dataset. Use when a run's gain/"
                        "DAQ window puts the single-PE mode outside the preset "
                        "band (e.g. Run 66810 bipolar Range-Center-0: SPE mode "
                        "~24 mV, so `--spe-band 10 30`, not the (50,200) default).")
    g.add_argument("--model", choices=("biexp", "biexp-overshoot"),
                   default="biexp",
                   help="SPE fit model. `biexp` (default): Gaussian-smeared "
                        "biexponential. `biexp-overshoot`: the same peak "
                        "minus the Daya Bay baseline-restoration undershoot "
                        "(Eq. 4) — adds U_os0/U_os1/tau_os/t0_os/sigma_os to "
                        "the CSV and overlays the fit so you can judge whether "
                        "your detector shows the ~5%%/~150ns undershoot. "
                        "Characterization only; does not change the deconv "
                        "kernel.")


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    # Bug-fix 3 (2026-05-30 scan): honor --out-dir. Default unchanged.
    out_dir = (Path(settings.out_dir) if settings.out_dir
               else PROJECT_ROOT / "outputs" / "spe_templates_perfile").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if maybe_dry_run_exit(subcommand="template build", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0
    from pmtdeconv.pipeline import template_build_impl as impl

    if settings.dataset is not None:
        # One registry for everything: the same resolution `hits` uses
        # (in-code seeds + YAML cards). resolve_settings already applied it,
        # so the build config comes straight off the resolved Settings.
        from pmtdeconv.presets.datasets import resolve_dataset_with_source
        ds, _src, _ov = resolve_dataset_with_source(settings.dataset)
        cfg = impl.config_from_preset(ds)
        configs = [cfg]
        # CLI overrides land on the ONE config dict (no global mutation).
        if settings.data_root is not None:
            cfg["dir"] = str(settings.data_root)
            print(f"  --data-root override: {settings.data_root}")
        if settings.runs:
            cfg["runs"] = [str(r) for r in settings.runs]
            print(f"  --runs filter: {settings.runs}")
        if provenance.get("dt_ns") == "cli" and settings.dt_ns not in (None, "auto"):
            cfg["dt"] = float(settings.dt_ns)
            print(f"  --dt-ns override: {settings.dt_ns}")
        if provenance.get("branch_pattern") == "cli" and settings.branch_pattern:
            cfg["channels"] = list(settings.branch_pattern)
            print(f"  --branches override: {settings.branch_pattern}")
        if settings.entry_stop:
            cfg["max_events"] = int(settings.entry_stop)
            print(f"  --entry-stop: max_events={settings.entry_stop}")
        if getattr(args, "no_single_peak", False):
            cfg["single_peak"] = False
            print("  --no-single-peak: secondary-peak veto DISABLED")
        if getattr(args, "spe_band", None) is not None:
            lo, hi = float(args.spe_band[0]), float(args.spe_band[1])
            cfg["spe_band"] = (lo, hi)
            print(f"  --spe-band override: ({lo:g}, {hi:g}) mV")
    else:
        configs = impl.configs_from_registry()

    print(f"Delegating to template_build_impl.main() — "
          f"{[c['name'] for c in configs]}  (model={args.model})")
    impl.main(configs, model=args.model, out_dir=out_dir)

    outputs_meta = [
        {"role": "spe_templates_perfile_csv",
         "path": str(out_dir / "spe_templates_perfile.csv")},
    ]
    if not getattr(args, "no_aggregate", False):
        from pmtdeconv.commands.template_aggregate import aggregate
        print("Aggregating per-file templates per channel "
              "(--no-aggregate to skip)")
        outputs_meta += aggregate(out_dir)
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="template build",
        settings=settings,
        provenance=provenance,
        inputs=[],
        outputs=outputs_meta,
        stats={"datasets": [c["name"] for c in configs],
               "model": args.model},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
