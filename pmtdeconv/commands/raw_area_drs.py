"""``pmtdeconv raw-area drs`` — pull precomputed per-event area from the DRSprocessing tree.

This is the *other* raw-area source: instead of computing the area
from each event's waveform (`raw_area mqana`), it reads the
``area_<board>_<ch>`` branch already written by the DRSprocessing
upstream. Faster and matches David's MQAna numbers exactly, but the
integration window was fixed upstream — we don't control it.

Output schema: ``run, cycle, entry, branch, interval, area, peak_height``
(same as `raw_area mqana` and `waveform`). Area in mV·ns.

Lifted from the standalone ``extract_raw_areas_to_csv.py`` and wired
through the standard preset / manifest layer.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

# Bug-fix 5 (2026-05-30 scan): defer heavy imports (numpy / pandas /
# uproot) until run() so `pmtdeconv --help` / `pmtdeconv raw-area --help`
# work in environments without uproot installed.

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import raw_area_output_dir
from pmtdeconv.io.schemas import AREA_COLUMNS
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("raw-area drs")
    g.add_argument("--file", dest="file", type=str, default=None,
                   help="Exact DRSprocessing ROOT FILE to read (wins over "
                        "the search). Default: found under --data-root, "
                        "e.g. <data-root>/Run<RUN>.root.")
    # Renamed from --root on 2026-09-18: "--root" named the file FORMAT and
    # sat next to --data-root (a DIRECTORY), so the two were easy to swap.
    from pmtdeconv.cli import add_renamed_alias
    add_renamed_alias(g, "--root", "--file", dest="file")
    g.add_argument("--board", type=str, default="2984",
                   help="Board prefix in DRSprocessing branch names "
                        "(default 2984).")
    g.add_argument("--chans", type=str, nargs="+", default=["1", "2"],
                   help="Channel suffixes in the ROOT branch names.")
    g.add_argument("--branch-out", type=str, nargs="+",
                   default=["d0_ch1", "d0_ch2"],
                   help="Branch labels written in the output CSV (match "
                        "the deconv-pulse CSV's `branch` column).")
    g.add_argument("--cycle", type=int, default=1,
                   help="Cycle number to record in the CSV (default 1).")
    # --tree comes from add_common_args; we fall back to "Events" in run()
    # when not supplied (DRSprocessing's default).


# Per-run DRSprocessing file resolution — layout search (processed/, root,
# one-level subdirs, *Run<N>*.root glob) + the actionable not-found error —
# lives in pmtdeconv.io.data_paths.resolve_runs (kind="drs"); see run().


def run(args: argparse.Namespace) -> int:
    import numpy as np  # noqa
    import pandas as pd  # noqa
    import uproot  # noqa
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    # LANL object-format datasets carry waveforms, not precomputed
    # area_<board>_<ch> branches. Point users at the dedicated
    # `raw_area offline` subcommand which reads the LANL pulse-list
    # *_offline.root files.
    if settings.format == "object" and not args.file:
        raise SystemExit(
            f"`pmtdeconv raw-area drs --dataset {settings.dataset}` is not "
            "applicable: object-format datasets (LANL_835/movedBar) have "
            "no precomputed area_<board>_<ch> branches. Use "
            f"`pmtdeconv raw-area offline --dataset {settings.dataset}` to "
            "read the LANL *_offline.root pulse list, or "
            "`pmtdeconv raw-area mqana --dataset "
            f"{settings.dataset} --full-window` to integrate the raw "
            "waveforms.")

    if len(args.chans) != len(args.branch_out):
        raise SystemExit("--chans and --branch-out must have the same length.")
    no_waveform = [f for f, key in (("--polarity", "polarity"), ("--dt-ns", "dt_ns"))
                   if provenance.get(key) == "cli"]
    if no_waveform:
        raise SystemExit(f"raw-area drs reads precomputed areas, so "
                         f"{' and '.join(no_waveform)} cannot apply. Drop them, "
                         f"or use `raw-area mqana` to integrate the waveforms.")
    pairs = list(zip(args.chans, args.branch_out))
    if settings.explicit_branches is not None:
        pairs = [(ch, br) for ch, br in pairs if br in settings.explicit_branches]
        if not pairs:
            raise SystemExit(f"--branches {settings.explicit_branches} matches none "
                             f"of --branch-out {args.branch_out}.")

    out_dir = settings.out_dir or raw_area_output_dir(settings, "drs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="raw-area drs", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    runs = settings.runs or []
    if not runs:
        raise SystemExit("--runs is required (got an empty list).")

    inputs_meta: List[dict] = []
    outputs_meta: List[dict] = []
    stats = {"runs": [], "n_entries_total": 0}

    # DRSprocessing default is "Events"; the lanl-related datasets we
    # configured already set tree="Events" in their preset, but the na22
    # preset sets tree="waveforms" (correct for flat format, wrong for
    # DRSprocessing). Default to "Events" for `raw_area drs` specifically.
    tree_name = settings.tree if settings.tree not in (None, "", "waveforms") else "Events"

    from pmtdeconv.io.data_paths import resolve_runs
    data_root = (Path(settings.data_root) if settings.data_root
                 else PROJECT_ROOT.parent.parent.parent / "data")
    for run_num, root_path in resolve_runs(data_root, runs, kind="drs",
                                           explicit=args.file):
        inputs_meta.append({"role": "root_data", "path": str(root_path)})

        with uproot.open(root_path) as ufile:
            tree = ufile[tree_name]
            n = min(tree.num_entries, settings.entry_stop or tree.num_entries)
            print(f"══ Run {run_num} ══")
            print(f"  Reading {n} entries from {root_path}::{tree_name}")
            branches = []
            for ch, _ in pairs:
                branches.append(f"area_{args.board}_{ch}")
                branches.append(f"vMax_{args.board}_{ch}")
            arrs = tree.arrays(branches, library="np", entry_stop=n)
            entries = np.arange(n)
            rows = []
            for ch, br in pairs:
                a = arrs[f"area_{args.board}_{ch}"]
                vm = arrs[f"vMax_{args.board}_{ch}"]
                for ent, ar, h in zip(entries, a, vm):
                    rows.append((run_num, args.cycle, int(ent), br, 0,
                                 float(ar), float(h)))

        df = pd.DataFrame(rows, columns=list(AREA_COLUMNS))
        csv_path = out_dir / f"area_drs_run{run_num}.csv"
        df.to_csv(csv_path, index=False)
        outputs_meta.append({"role": "area_drs_csv", "path": str(csv_path)})
        stats["runs"].append(run_num)
        stats["n_entries_total"] += int(n)
        print(f"  Saved {csv_path}  ({len(df)} rows, {n} unique entries)")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="raw-area drs",
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
