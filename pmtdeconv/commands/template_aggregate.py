"""``pmtdeconv template aggregate`` — combine per-file templates into per-channel.

``template build`` runs this step itself since 2026-10; use this command to
re-aggregate an existing per-file CSV (e.g. after editing it, or one built
with ``--no-aggregate``). Writes weighted-average per-channel templates +
an overlay figure.
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
    g = parser.add_argument_group("template aggregate")
    g.add_argument("--input-csv", type=str, default=None,
                   help="Override the per-file CSV input "
                        "(default: outputs/spe_templates_perfile/"
                        "spe_templates_perfile.csv).")


def aggregate(out_dir: Path) -> list:
    """Aggregate ``out_dir/spe_templates_perfile.csv`` into per-channel
    templates written beside it. Returns the manifest output entries.
    Shared by ``template build`` (automatic) and ``template aggregate``."""
    from pmtdeconv.pipeline import template_aggregate_impl as impl
    orig = impl.OUT
    impl.OUT = out_dir           # the impl reads + writes under OUT
    try:
        impl.main()
    finally:
        impl.OUT = orig
    return [{"role": "per_channel_csv",
             "path": str(out_dir / "spe_templates_perchannel.csv")},
            {"role": "overlay_pdf",
             "path": str(out_dir / "aggregated_templates.png")}]


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    # Bug-fix 3 (2026-05-30 scan): honor --out-dir. Default unchanged.
    out_dir = (Path(settings.out_dir) if settings.out_dir
               else PROJECT_ROOT / "outputs" / "spe_templates_perfile")
    out_dir.mkdir(parents=True, exist_ok=True)
    if maybe_dry_run_exit(subcommand="template aggregate", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0
    # Bug-fix 3 + 4 (2026-05-30 scan): respect --out-dir AND --input-csv.
    # The impl reads ``out_dir/spe_templates_perfile.csv``; an --input-csv
    # elsewhere is staged to that name first.
    input_csv_override = args.input_csv
    if input_csv_override is not None:
        # The impl uses pandas.read_csv; the simplest swap is a
        # module-level INPUT_CSV that main() reads (if present) else
        # falls back to OUT/spe_templates_perfile.csv. To avoid editing
        # the impl, we copy/symlink the override to the expected name
        # inside out_dir before calling main().
        import shutil
        target = out_dir / "spe_templates_perfile.csv"
        try:
            if target.resolve() != Path(input_csv_override).resolve():
                shutil.copy(input_csv_override, target)
                print(f"  staged --input-csv → {target}")
        except FileNotFoundError:
            raise SystemExit(f"--input-csv {input_csv_override} not found")

    print("Delegating to aggregate_spe_templates.main()")
    outputs_meta = aggregate(out_dir)
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="template aggregate",
        settings=settings,
        provenance=provenance,
        inputs=[{"role": "per_file_csv",
                 "path": args.input_csv or str(out_dir / "spe_templates_perfile.csv")}],
        outputs=outputs_meta,
        stats={},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
