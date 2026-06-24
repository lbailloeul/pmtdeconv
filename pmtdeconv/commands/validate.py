"""``pmtdeconv validate`` — comparative FoM report across area sources.

Computes a card-declared figure of merit (e.g. photopeak σ/μ) on the SAME
quantity from each method — deconvolved area, raw area, and optionally the DRS
precomputed area — and prints a side-by-side, per-channel table. It is a
REPORT, not a gate: it never fails a run; it shows the expert which method wins
on THEIR chosen physics number.

Inputs are the area CSVs the existing commands already emit, so validate
composes with them::

    pmtdeconv area        --dataset mydet --runs 42 --out-dir out/deconv
    pmtdeconv raw-area mqana --dataset mydet --runs 42 --out-dir out/raw
    pmtdeconv validate --dataset mydet \\
        --deconv-csv out/deconv/areas_run42.csv \\
        --raw-csv    out/raw/area_mqana_run42.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deconv-csv", required=True,
                        help="Deconvolved area CSV (from `area`).")
    parser.add_argument("--raw-csv", required=True,
                        help="Raw area CSV (from `raw-area mqana`).")
    parser.add_argument("--drs-csv", default=None,
                        help="Optional DRS precomputed area CSV (raw-area drs).")
    parser.add_argument("--fom", default=None,
                        help="FoM name (default: the card's `fom:`). See "
                             "`pmtdeconv dataset lint` for the catalog.")
    parser.add_argument("--area-col", default="area",
                        help="Charge column to score (default: area).")
    parser.add_argument("--branch-col", default="branch",
                        help="Channel column (default: branch).")
    parser.add_argument("--window", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Photopeak window [lo hi] for the FoM fit.")
    parser.add_argument("--out-name", default="validate_report",
                        help="Report JSON basename.")


def _fom_per_branch(csv: str, fom_fn, area_col: str, branch_col: str,
                    window) -> dict:
    df = pd.read_csv(csv)
    if "interval" in df.columns:
        df = df[df["interval"] == 0]
    if area_col not in df.columns or branch_col not in df.columns:
        raise SystemExit(
            f"{csv}: needs columns {area_col!r} and {branch_col!r}; "
            f"has {list(df.columns)}")
    out = {}
    for branch, grp in df.groupby(branch_col):
        try:
            res = fom_fn(grp[area_col].values, window=tuple(window)) \
                if window is not None else fom_fn(grp[area_col].values)
        except TypeError:
            # FoM does not accept a window (e.g. timing_sigma)
            res = fom_fn(grp[area_col].values)
        out[str(branch)] = res
    return out


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.fom import get_fom, FOM_LOWER_IS_BETTER

    fom_name = args.fom
    if fom_name is None and getattr(args, "dataset", None):
        from pmtdeconv.presets.datasets import resolve_dataset
        fom_name = getattr(resolve_dataset(args.dataset), "fom", None)
    if fom_name is None:
        print("no FoM: pass --fom or declare `fom:` in the dataset card")
        return 1
    fom_fn = get_fom(fom_name)
    lower_better = FOM_LOWER_IS_BETTER.get(fom_name, True)

    sources = {"deconv": args.deconv_csv, "raw": args.raw_csv}
    if args.drs_csv:
        sources["drs"] = args.drs_csv

    per = {src: _fom_per_branch(csv, fom_fn, args.area_col, args.branch_col,
                                args.window)
           for src, csv in sources.items()}

    branches = sorted({b for s in per.values() for b in s})
    print(f"FoM = {fom_name}  "
          f"({'lower' if lower_better else 'higher'} is better)"
          + (f"  window={tuple(args.window)}" if args.window else ""))
    header = f"{'channel':<12}" + "".join(f"{s:>14}" for s in sources)
    print(header)
    print("-" * len(header))
    for br in branches:
        vals = {s: per[s].get(br, {}).get("value", math.nan) for s in sources}
        row = f"{br:<12}" + "".join(f"{vals[s]:>14.4g}" for s in sources)
        finite = {s: v for s, v in vals.items() if v == v}
        if finite:
            best = (min if lower_better else max)(finite, key=finite.get)
            row += f"   best={best}"
        print(row)

    report = {"fom": fom_name, "lower_is_better": lower_better,
              "window": args.window, "area_col": args.area_col,
              "sources": per}
    out_dir = Path(getattr(args, "out_dir", None) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.out_name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0
