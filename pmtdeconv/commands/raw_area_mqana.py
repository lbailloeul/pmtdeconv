"""``pmtdeconv raw-area mqana`` — MQAna-style narrow-window raw area.

Computes the raw area directly from each event's waveform using a
narrow window (default 35 ns centered on the per-event sample-max),
matching David's MQAna / ELOG E41473 convention. No deconvolution.

The companion subcommand ``raw_area drs`` reads the precomputed
``area_<board>_<ch>`` branch from the DRSprocessing tree instead — same
CSV schema, different source. The manifest's ``settings`` records which
path produced each output.

Pass ``--full-window`` to revert to the legacy fixed-sample window
(`--signal-start` to `--signal-end`) for back-compat with the older
``deconvolution/raw_area_Na22.py`` outputs.

Output schema: ``run, cycle, entry, branch, interval, area, peak_height``.
Area is in mV·ns, positive for real positive-going pulses; LANL
polarity is flipped internally so values are still positive.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

# Bug-fix 5 (2026-05-30 scan): defer heavy imports until run() so
# `pmtdeconv --help` works without uproot installed.

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.object_format import (
    detect_populated_channels, iter_object_run, read_dt_ns,
)
from pmtdeconv.core.dsp import detect_saturation_plateau
from pmtdeconv.io.root_loader import select_tree_key, to_numpy2d
from pmtdeconv.io.output_paths import raw_area_output_dir
from pmtdeconv.io.schemas import AREA_COLUMNS
from pmtdeconv.pipeline.area_mqana import fixed_window_area, mqana_area
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("raw-area mqana")
    g.add_argument("--mqana-window-ns", type=float, default=35.0,
                   help="Total integration window centered on per-event "
                        "sample-max (default 35 ns = ±17.5 ns). MQAna "
                        "convention. Ignored if --full-window is passed.")
    g.add_argument("--full-window", action="store_true",
                   help="Use the legacy fixed-sample window "
                        "(--signal-start to --signal-end) instead of MQAna.")
    g.add_argument("--signal-start", type=int, default=50,
                   help="Wide-window start sample (legacy mode only).")
    g.add_argument("--signal-end", type=int, default=950,
                   help="Wide-window end sample (legacy mode only).")
    g.add_argument("--n-pre", type=int, default=50,
                   help="Pre-trigger samples used for baseline.")
    g.add_argument("--step", type=int, default=500,
                   help="uproot chunk size (events).")


def _resolve_branches(tree, requested) -> List[str]:
    if isinstance(requested, list):
        return [b for b in requested if b in tree.keys()]
    return [k for k in sorted(tree.keys()) if k.startswith("d0_ch")]


def _plateau_saturated(w2d, polarity: int):
    """Per-event plateau saturation flag for a (n, N) waveform chunk
    (polarity-corrected). Universal — no per-detector rail threshold."""
    import numpy as np
    pw = polarity * np.asarray(w2d, dtype=float)
    out = np.zeros(len(pw), dtype=bool)
    for j in range(len(pw)):
        sig = float(np.nanstd(pw[j][:100])) or 1.0
        out[j] = bool(detect_saturation_plateau(pw[j], sig)[0])
    return out


def run(args: argparse.Namespace) -> int:
    import numpy as np  # noqa
    import pandas as pd  # noqa
    import uproot  # noqa
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    polarity = int(settings.polarity)
    # dt_ns may be 'auto' for object-format datasets — resolved per-file
    # inside the run loop below.

    out_dir = settings.out_dir or raw_area_output_dir(settings, "mqana")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="raw-area mqana", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    runs = settings.runs or []
    if not runs:
        raise SystemExit("--runs is required (got an empty list).")

    data_dir = settings.data_root or (
        PROJECT_ROOT.parent.parent.parent / "data" / "Na22_lab")
    data_dir = Path(data_dir)

    inputs_meta = []
    outputs_meta = []
    stats = {"runs": [], "n_entries_total": 0, "n_saturated_total": 0,
             "format": settings.format, "channels_used": None}

    is_object_format = (settings.format == "object")

    for run_num in runs:
        # File path: flat datasets use the canonical Run<N>.root naming;
        # object-format LANL files use sub-run naming so we pick the
        # first match via the dataset's file_glob.
        if is_object_format:
            candidates = sorted(data_dir.glob(
                settings.file_glob.replace("Run<N>", f"Run{run_num}")
                if "<N>" in settings.file_glob else settings.file_glob))
            candidates = [p for p in candidates if str(run_num) in p.name]
            if not candidates:
                print(f"[skip] no object-format ROOT matching run {run_num} "
                      f"under {data_dir}")
                continue
            root_path = candidates[0]
        else:
            from pmtdeconv.io.data_paths import resolve_run_file
            root_path = resolve_run_file(data_dir, run_num, kind="flat")
            if root_path is None:
                print(f"[skip] run {run_num}: no flattened file under {data_dir}")
                continue
        inputs_meta.append({"role": "root_data", "path": str(root_path)})

        # Resolve dt_ns (object format may be 'auto' = read from tree)
        if settings.dt_ns == "auto":
            dt_ns = read_dt_ns(root_path, tree_name=settings.tree or "Events")
            print(f"  dt_ns (auto-detected): {dt_ns} ns")
        else:
            dt_ns = float(settings.dt_ns)

        rows = []
        if is_object_format:
            # Resolve channel set
            if isinstance(settings.branch_pattern, list):
                branches = settings.branch_pattern
            else:
                branches = detect_populated_channels(root_path,
                                                     tree_name=settings.tree or "Events")
                print(f"  Auto-detected {len(branches)} populated channels: "
                      f"{branches[:4]}{'...' if len(branches) > 4 else ''}")
            stats["channels_used"] = branches
            print(f"══ Run {run_num} (object format) ══")
            print(f"  ROOT: {root_path.name}")

            base_entry = 0
            for chunk_base, chunk in iter_object_run(
                root_path, branches=branches,
                entry_stop=settings.entry_stop,
                step_size=args.step,
                tree_name=settings.tree or "Events",
            ):
                n_chunk = len(next(iter(chunk.values())))
                for b in branches:
                    w = chunk[b]
                    # Per-channel saturation: this channel only.
                    # Multi-channel LANL files often have unrelated
                    # channels clipped on the same event, so per-event
                    # OR-flagging would NaN every channel.
                    sat = _plateau_saturated(w, polarity)
                    if args.full_window:
                        area, peak_height = fixed_window_area(
                            w, dt_ns, n_pre=args.n_pre,
                            signal_start=args.signal_start,
                            signal_end=args.signal_end,
                            polarity=polarity)
                    else:
                        area, peak_height = mqana_area(
                            w, dt_ns, n_pre=args.n_pre,
                            window_ns=args.mqana_window_ns,
                            polarity=polarity)
                    for j in range(n_chunk):
                        rows.append((
                            run_num, 1, chunk_base + j, b, 0,
                            float("nan") if sat[j] else float(area[j]),
                            float("nan") if sat[j] else float(peak_height[j]),
                        ))
        else:
            with uproot.open(root_path) as ufile:
                tree_key, per_cycle_mult = select_tree_key(
                    ufile, tree_name=settings.tree)
                tree = ufile[tree_key]
                cycle_num = int(tree_key.split(";")[-1])
                branches = _resolve_branches(tree, settings.branch_pattern)
                if not branches:
                    print(f"[skip] no usable branches in {root_path}::{tree_key}")
                    continue
                stats["channels_used"] = branches
                n_total = tree.num_entries
                stop = (settings.entry_stop * per_cycle_mult
                        if settings.entry_stop else n_total)
                print(f"══ Run {run_num} (flat format) ══")
                print(f"  Tree: {tree_key}  branches: {branches}  "
                      f"entries: {min(stop, n_total)}")

                base_entry = 0
                for arrays in tree.iterate(branches, step_size=args.step,
                                           library="ak",
                                           entry_start=0, entry_stop=stop):
                    n_chunk = len(arrays[branches[0]])
                    br_arrays = {b: to_numpy2d(arrays[b]) for b in branches}

                    for b in branches:
                        w = br_arrays[b]
                        # Per-channel saturation (2026-07-28; matches the
                        # object path above). The old per-event OR meant one
                        # clipped channel NaN'd every channel of that event —
                        # cross-channel contamination.
                        sat = _plateau_saturated(w, polarity)
                        if args.full_window:
                            area, peak_height = fixed_window_area(
                                w, dt_ns, n_pre=args.n_pre,
                                signal_start=args.signal_start,
                                signal_end=args.signal_end,
                                polarity=polarity)
                        else:
                            area, peak_height = mqana_area(
                                w, dt_ns, n_pre=args.n_pre,
                                window_ns=args.mqana_window_ns,
                                polarity=polarity)
                        for j in range(n_chunk):
                            rows.append((
                                run_num, cycle_num, base_entry + j, b, 0,
                                float("nan") if sat[j] else float(area[j]),
                                float("nan") if sat[j] else float(peak_height[j]),
                            ))
                    base_entry += n_chunk

        df = pd.DataFrame(rows, columns=list(AREA_COLUMNS))
        csv_path = out_dir / f"area_mqana_run{run_num}.csv"
        df.to_csv(csv_path, index=False)
        outputs_meta.append({"role": "area_mqana_csv", "path": str(csv_path)})
        # Per-channel saturation flags (2026-07-28): count NaN'd channel-rows,
        # not "events" (a single event can be clipped on one channel only).
        n_sat = int(df.area.isna().sum())
        stats["runs"].append(run_num)
        stats["n_entries_total"] += int(len(df) // max(len(branches), 1))
        stats["n_saturated_total"] += n_sat
        print(f"  Saved {csv_path}  ({len(df)} rows, {n_sat} saturated channel-rows)")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="raw-area mqana",
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
