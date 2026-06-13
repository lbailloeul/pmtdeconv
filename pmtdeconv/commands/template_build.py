"""``pmtdeconv template build`` — per-(dataset, file, channel) stacked SPE templates.

Delegates to ``build_spe_templates_perfile.main``. The legacy script's
configuration lives in its module-level ``DATASETS`` list; in this PR
the subcommand accepts no overrides and just runs the canonical build
across all configured datasets. Future PRs can wire ``--dataset``,
``--max-events``, ``--spe-band`` etc. through to the legacy module.
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
    g.add_argument("--all", dest="all_datasets", action="store_true",
                   help="Iterate every dataset in the legacy DATASETS list "
                        "(current behavior — overrides default off).")
    g.add_argument("--no-single-peak", action="store_true",
                   help="Disable the default single-peak veto and stack every "
                        "band-selected waveform (legacy behavior). Templates "
                        "are single-PE responses by definition, so the veto "
                        "(no secondary >=25%% of primary beyond +-5 ns) is ON "
                        "by default — 2026-06-10 finding: pulse trains were "
                        "97%% of the 66727 ch2 band and inflated tau_fall "
                        "40 ns vs the true 2.8 ns.")
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
    # .resolve() (2026-05-30 gallery run): the impl's main() does os.chdir()
    # into pmtdeconv/pipeline/, so a RELATIVE --out-dir would otherwise make
    # the impl's np.save(OUT/...) write to the wrong place (FileNotFound).
    out_dir = (Path(settings.out_dir) if settings.out_dir
               else PROJECT_ROOT / "outputs" / "spe_templates_perfile").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if maybe_dry_run_exit(subcommand="template build", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0
    from pmtdeconv.pipeline import template_build_impl as legacy  # noqa
    # The impl's module-level OUT path drives every file it writes;
    # patch it so --out-dir actually takes effect.
    _orig_out = legacy.OUT
    legacy.OUT = out_dir

    # Decision 11: when --dataset is supplied, filter the legacy
    # DATASETS list down to just that one entry. The names in the
    # legacy list use mixed conventions ("Cd109_lab", "Na22_lab",
    # "MilliQan_1000", "LANL_806", ...) so we match case-insensitively
    # against a few aliases.
    selected = settings.dataset
    if selected is not None:
        # Bug-fix (2026-05-30): the legacy DATASETS entries are named
        # "Cd109"/"Na22"/… (not "Cd109_lab"); the old aliases pointed at
        # the wrong names so `template build --dataset cd109` matched
        # nothing and aborted. Match the actual legacy names.
        aliases = {
            "na22":          {"Na22"},
            "na22_3ch":      {"Na22_3ch"},
            "cd109":         {"Cd109"},
            "milliqan":      {"MilliQan_1000", "MilliQan_2150"},
            "lanl_806":      {"LANL_806"},
            "lanl_835":      {"LANL_835"},
            "lanl_movedBar": {"LANL_movedBar"},
        }
        wanted = aliases.get(selected, {selected})
        # Shallow-copy so a --data-root override below doesn't mutate the
        # module-level DATASETS dicts.
        kept = [dict(d) for d in legacy.DATASETS if d["name"] in wanted]
        if not kept:
            raise SystemExit(
                f"--dataset {selected!r} matched none of the legacy "
                f"DATASETS entries. Known legacy names: "
                f"{[d['name'] for d in legacy.DATASETS]}")
        # Honor --data-root: point the kept entries at the override dir
        # (resolved absolute by the impl). Without this the impl uses the
        # legacy relative path anchored at PROJECT_ROOT.
        if settings.data_root is not None:
            for d in kept:
                d["dir"] = str(settings.data_root)
            print(f"  --data-root override: {settings.data_root}")
        # Wire --runs / --dt-ns / --branches through to the impl
        # (2026-06-10, Runs 66727-29): without these the build sweeps every
        # file matching the dataset glob at the legacy dt, but new-run
        # template extraction must use only the requested runs and the
        # run's measured sampling rate.
        if settings.runs:
            for d in kept:
                d["runs"] = [str(r) for r in settings.runs]
            print(f"  --runs filter: {settings.runs}")
        if provenance.get("dt_ns") == "cli" and settings.dt_ns not in (None, "auto"):
            for d in kept:
                d["dt"] = float(settings.dt_ns)
            print(f"  --dt-ns override: {settings.dt_ns}")
        if provenance.get("branch_pattern") == "cli" and settings.branch_pattern:
            for d in kept:
                d["channels"] = list(settings.branch_pattern)
            print(f"  --branches override: {settings.branch_pattern}")
        if getattr(args, "no_single_peak", False):
            for d in kept:
                d["single_peak"] = False
            print("  --no-single-peak: secondary-peak veto DISABLED")
        # Temporarily replace the module-level DATASETS while main() runs
        original = legacy.DATASETS
        legacy.DATASETS = kept
        print(f"Delegating to build_spe_templates_perfile.main() — "
              f"filtered to {[d['name'] for d in kept]}  (model={args.model})")
        try:
            legacy.main(model=args.model)
        finally:
            legacy.DATASETS = original
            legacy.OUT = _orig_out
    else:
        print(f"Delegating to build_spe_templates_perfile.main() — "
              f"all configured datasets  (model={args.model})")
        try:
            legacy.main(model=args.model)
        finally:
            legacy.OUT = _orig_out
    # If the conditional branch above ran legacy.main() via the
    # try/finally on filtered datasets, ensure OUT is also restored.
    if legacy.OUT is not _orig_out:
        legacy.OUT = _orig_out

    outputs_meta = [
        {"role": "spe_templates_perfile_csv",
         "path": str(out_dir / "spe_templates_perfile.csv")},
    ]
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="template build",
        settings=settings,
        provenance=provenance,
        inputs=[],
        outputs=outputs_meta,
        stats={"datasets": [d["name"] for d in legacy.DATASETS],
               "model": args.model},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
