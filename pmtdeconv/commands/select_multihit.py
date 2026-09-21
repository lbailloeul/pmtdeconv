"""``pmtdeconv select multihit`` — find 0–20 PE multi-hit bunches.

The vault-11e clump recipe, finally landed as a first-class provenanced step:
from an existing hits ``all_results.csv``, select the events where one
waveform carries ≥ ``--min-pulses`` successful (and, when PE-calibrated,
in-regime) pulses — the multi-hit bunches that are the 0–20 PE re-scope's
preferred benchmark population — and write a reusable ``event_list.csv``:

    run, entry, branch, n_pulses, npe_sum, clump_lo_ns, clump_hi_ns, paired

``paired`` marks entries whose bunches appear on BOTH of the first two
channels within ``--pair-window-ns`` (the 11e ch1↔ch2 coincidence, greedy
nearest-time). Feed the list back with ``hits --entry-list event_list.csv``
to re-process exactly the selected bunches (e.g. under
``--veto-profile benchmark_hits``) or into ``benchmark denoisers --real``.

Selection is propose-and-emit: nothing upstream is mutated, and the list +
its criteria land in this command's own manifest.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.presets import resolve_settings

CLUMP_PAD_NS = 30.0        # vault 11e: clump bounds = onset − 30 ns


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("select multihit")
    # Renamed from --pulse-csv on 2026-09-21: the file comes from `hits`.
    # (Internal attribute stays `pulse_csv`.)
    from pmtdeconv.cli import add_renamed_alias
    add_renamed_alias(g, "--pulse-csv", "--hits-csv", dest="pulse_csv")
    g.add_argument("--hits-csv", dest="pulse_csv", type=str, default=None,
                   help="Hits all_results.csv / hits.csv (default: "
                        "auto-resolve latest for --dataset/--runs).")
    g.add_argument("--min-pulses", type=int, default=2,
                   help="Minimum successful pulses per waveform (default 2).")
    g.add_argument("--pair-window-ns", type=float, default=40.0,
                   help="ch1↔ch2 pairing window (vault 11e; default 40).")
    g.add_argument("--include-out-of-regime", action="store_true",
                   default=False,
                   help="Count pulses regardless of the in_regime flag "
                        "(default: only in-regime pulses count when the run "
                        "was PE-calibrated).")


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.io.output_paths import resolve_pulse_csv, suffixed_output_dir

    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)
    out_dir = Path(settings.out_dir
                   or suffixed_output_dir("select_multihit", settings))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (Path(args.pulse_csv) if args.pulse_csv
                else resolve_pulse_csv(settings))
    if csv_path is None or not Path(csv_path).exists():
        raise SystemExit("select multihit: no hits CSV found — run "
                         "`pmtdeconv hits` first or pass --pulse-csv.")

    if maybe_dry_run_exit(subcommand="select multihit", settings=settings,
                          provenance=provenance, out_dir=out_dir):
        return 0

    df = pd.read_csv(csv_path)
    ok = df[df["success"] == True] if "success" in df.columns else df  # noqa: E712
    ok = ok.dropna(subset=["t_peak"])

    calibrated = "in_regime" in ok.columns and ok["in_regime"].notna().any()
    if calibrated and not args.include_out_of_regime:
        ok = ok[ok["in_regime"] == True]                   # noqa: E712
    elif not calibrated:
        print("[note] run is not PE-calibrated (no in_regime values) — "
              "counting ALL successful pulses; calibrate with "
              "`gain calibrate` to restrict to the 0–20 PE regime.")

    n_scanned = df.groupby(["run", "entry", "branch"]).ngroups if len(df) \
        else 0

    rows = []
    for (run_no, entry, branch), grp in ok.groupby(["run", "entry",
                                                    "branch"]):
        if len(grp) < args.min_pulses:
            continue
        t = grp["t_peak"].to_numpy(float)
        npe_sum = (float(grp["npe"].sum())
                   if "npe" in grp.columns and grp["npe"].notna().any()
                   else float("nan"))
        rows.append({"run": int(run_no), "entry": int(entry),
                     "branch": str(branch), "n_pulses": int(len(grp)),
                     "npe_sum": npe_sum,
                     "clump_lo_ns": float(t.min() - CLUMP_PAD_NS),
                     "clump_hi_ns": float(t.max() + CLUMP_PAD_NS),
                     "paired": False})
    sel = pd.DataFrame(rows)

    # ch1↔ch2 pairing (11e): bunches on the first two branches of the same
    # entry whose nearest pulse times fall within the pairing window.
    # Times come from ONE upfront groupby (a per-row DataFrame filter here is
    # quadratic and effectively hangs on real runs — 2026-07-29 fix).
    if len(sel):
        branches = sorted(sel["branch"].unique())[:2]
        if len(branches) == 2:
            times = {key: grp["t_peak"].to_numpy(float)
                     for key, grp in ok.groupby(["run", "entry", "branch"])}
            for (run_no, entry), grp in sel.groupby(["run", "entry"]):
                have = set(grp["branch"])
                if not set(branches) <= have:
                    continue
                t1 = times.get((run_no, entry, branches[0]),
                               np.empty(0))
                t2 = times.get((run_no, entry, branches[1]),
                               np.empty(0))
                if t1.size and t2.size and \
                        np.min(np.abs(t1[:, None] - t2[None, :])) \
                        <= args.pair_window_ns:
                    sel.loc[grp.index, "paired"] = True

    list_path = out_dir / "event_list.csv"
    sel.to_csv(list_path, index=False)

    n_entries = sel.groupby(["run", "entry"]).ngroups if len(sel) else 0
    n_paired = (sel[sel["paired"]].groupby(["run", "entry"]).ngroups
                if len(sel) else 0)
    print(f"scanned {n_scanned} (run, entry, branch) waveforms → "
          f"{len(sel)} multi-hit bunches on {n_entries} entries "
          f"({n_paired} ch-paired)")
    print(f"event list: {list_path}")
    print(f"re-process with:  pmtdeconv hits --dataset <ds> --runs <N> "
          f"--entry-list {list_path}")

    write_manifest(
        subcommand="select multihit",
        settings=settings, provenance=provenance,
        inputs=[{"role": "pulse_csv", "path": str(csv_path)}],
        outputs=[{"role": "event_list", "path": str(list_path)}],
        stats={"n_scanned": int(n_scanned), "n_bunches": int(len(sel)),
               "n_entries": int(n_entries), "n_paired_entries": int(n_paired),
               "min_pulses": int(args.min_pulses),
               "pair_window_ns": float(args.pair_window_ns),
               "in_regime_only_pulses": bool(calibrated
                                             and not args.include_out_of_regime)},
        out_path=out_dir / "manifest.json",
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    return 0
