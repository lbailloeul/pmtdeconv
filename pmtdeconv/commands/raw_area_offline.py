"""``pmtdeconv raw-area offline`` — LANL pulse-list area + per-channel timing.

Reads the LANL ``*_offline.root`` files (tree ``t``, one entry per
event, with a jagged pulse list per event). Aggregates to one row
per (event, channel) and writes **two** CSVs:

- ``area_offline_run<RUN>.csv`` — AREA_COLUMNS schema. ``area`` is the
  Σ over pulse areas (total integrated charge per channel); ``peak_height``
  is max(pulse_height) across pulses.
- ``offline_hit_results_run<RUN>.csv`` — HIT_RESULTS_COLUMNS schema.
  One row per (event, channel) for the primary pulse (``ipulse == 0``)
  with ``t_peak`` from the configurable ``--time-field``. Feeds
  downstream ``timewalk run`` / ``timewalk gif`` directly.

Multi-subfile runs are concatenated with a **global** entry counter
(decision 5), so ``(run, cycle, entry, branch)`` is unique within the
output CSVs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd

from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.offline_format import (
    ALLOWED_TIME_FIELDS, DEFAULT_TIME_FIELD,
    count_offline_events, detect_offline_channels, iter_offline_run,
)
from pmtdeconv.io.output_paths import raw_area_output_dir
from pmtdeconv.io.schemas import AREA_COLUMNS, HIT_RESULTS_COLUMNS
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("raw_area offline")
    g.add_argument("--time-field", choices=ALLOWED_TIME_FIELDS,
                   default=DEFAULT_TIME_FIELD,
                   help=f"Which time observable to write as t_peak in the "
                        f"pulse-timing CSV (default: {DEFAULT_TIME_FIELD}).")
    # --tree comes from add_common_args (defaults to None); we use 't'
    # internally when not overridden (the LANL offline tree name).
    g.add_argument("--offline-glob", type=str, default=None,
                   help="Override the dataset's offline_file_glob.")
    g.add_argument("--root", type=str, nargs="+", default=None,
                   help="Explicit offline ROOT file(s). Skips the "
                        "preset's offline_data_dir lookup.")


def _resolve_offline_files(args, settings, run_num: int) -> List[Path]:
    """Return sorted sub-file paths matching ``run_num``."""
    if args.root:
        return [Path(p) for p in args.root]
    data_dir = settings.offline_data_dir
    if data_dir is None:
        raise SystemExit(
            f"Dataset {settings.dataset!r} has no offline_data_dir "
            "configured. Pass --root <path(s)> explicitly, or pick a "
            "dataset preset with offline files (lanl_835, lanl_movedBar).")
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"offline_data_dir {data_dir} does not exist.")
    glob = args.offline_glob or settings.offline_file_glob or "*_offline.root"
    cands = sorted(data_dir.glob(glob))
    matched = [p for p in cands if str(run_num) in p.name]
    return matched


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    out_dir = settings.out_dir or raw_area_output_dir(settings, "offline")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if maybe_dry_run_exit(subcommand="raw_area offline", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    runs = settings.runs or []
    if not runs:
        raise SystemExit("--runs is required (got an empty list).")

    # LANL offline files use tree 't'. We do NOT inherit settings.tree
    # (which is the waveform-format tree, "Events" for object format),
    # because the offline file has a different tree. Only honor an
    # explicit --tree CLI flag if the user passed one.
    explicit_tree = getattr(args, "tree", None)
    tree_name = explicit_tree if explicit_tree else "t"

    inputs_meta: List[dict] = []
    outputs_meta: List[dict] = []
    stats = {"runs": [], "n_events_total": 0, "n_subfiles_total": 0,
             "time_field": args.time_field}

    for run_num in runs:
        paths = _resolve_offline_files(args, settings, run_num)
        if not paths:
            print(f"[skip] no offline files matching run {run_num}")
            continue

        print(f"══ Run {run_num} ══")
        print(f"  {len(paths)} sub-file(s) to read; time_field={args.time_field}")

        area_rows = []
        hit_rows = []
        global_entry = 0   # decision 5: monotonically across sub-files
        n_events_this_run = 0
        run_channels: set = set()

        for sub_path in paths:
            inputs_meta.append({"role": "offline_root",
                                "path": str(sub_path)})
            n_evt_file = count_offline_events(sub_path, tree_name=tree_name)
            cap = settings.entry_stop
            stop = (min(cap, n_evt_file) if cap is not None else n_evt_file)
            print(f"   • {sub_path.name}  ({stop}/{n_evt_file} events, "
                  f"global entries {global_entry}..{global_entry + stop - 1})")

            for row in iter_offline_run(
                sub_path, tree_name=tree_name,
                entry_stop=stop, time_field=args.time_field,
                start_entry=global_entry,
            ):
                det, _, chan_str = row.branch.partition("_ch")
                chan_n = int(chan_str)
                # Area CSV row (AREA_COLUMNS)
                area_rows.append((
                    run_num, row.cycle, row.entry, row.branch,
                    row.interval, row.area_sum, row.peak_height,
                ))
                # Pulse-timing CSV row (HIT_RESULTS_COLUMNS) — only when we have
                # a primary pulse. Without one, downstream timewalk
                # would skip the event anyway.
                if row.has_primary:
                    hit_rows.append({
                        "run": run_num, "cycle": row.cycle,
                        "det": det, "chan": chan_n, "branch": row.branch,
                        "entry": row.entry, "success": True,
                        "reason": "offline_primary",
                        "peak_id": 1.0,
                        "t_peak": row.primary_t_peak,
                        "height": row.primary_height,
                        "charge": row.primary_area,
                        "win_lo_ns": float("nan"),
                        "win_hi_ns": float("nan"),
                        "baseline_ped": float("nan"),
                        "baseline_sig": float("nan"),
                        "baseline_t0": float("nan"),
                        "baseline_t1": float("nan"),
                        "u_rec_noise": float("nan"),
                    })
                run_channels.add(row.branch)

            global_entry += stop
            n_events_this_run += stop

        if not area_rows:
            print(f"  [skip] no rows for run {run_num}")
            continue

        area_csv = out_dir / f"area_offline_run{run_num}.csv"
        pd.DataFrame(area_rows, columns=list(AREA_COLUMNS)).to_csv(
            area_csv, index=False)
        outputs_meta.append({"role": "area_offline_csv",
                             "path": str(area_csv)})

        pulse_csv = out_dir / f"offline_hit_results_run{run_num}.csv"
        pd.DataFrame(hit_rows, columns=list(HIT_RESULTS_COLUMNS)).to_csv(
            pulse_csv, index=False)
        outputs_meta.append({"role": "offline_hit_results",
                             "path": str(pulse_csv)})

        stats["runs"].append(run_num)
        stats["n_events_total"] += n_events_this_run
        stats["n_subfiles_total"] += len(paths)
        print(f"  Saved {area_csv}  ({len(area_rows)} area rows)")
        print(f"  Saved {pulse_csv}   ({len(hit_rows)} primary-pulse timing rows, "
              f"{len(run_channels)} channels)")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="raw_area offline",
        settings=settings, provenance=provenance,
        inputs=inputs_meta, outputs=outputs_meta,
        stats=stats,
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"\nAll outputs in: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0
