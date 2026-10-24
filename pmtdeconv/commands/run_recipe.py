"""``pmtdeconv run`` — execute a complete named analysis workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from pmtdeconv import PROJECT_ROOT


RECIPES = {
    "na22-timewalk": (
        "Raw CFD-20 timing plus raw area, then the timewalk scan for Na22 "
        "Run 65213 — the E41473 benchmark. No deconvolution: the run is "
        "mostly saturated."
    ),
    "na22-energy": (
        "Render the four-method Na22 energy comparison for Run 65212."
    ),
    "build-template": (
        "Build per-file SPE templates for one dataset (Cd109 by default)."
    ),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("recipe", nargs="?", choices=tuple(RECIPES))
    parser.add_argument(
        "--raw-source",
        choices=("drs", "mqana"),
        default="drs",
        help="Raw-area source for na22-timewalk (default: drs).",
    )
    parser.add_argument(
        "--area-file",
        dest="area_file",
        default=None,
        help="Exact DRSprocessing ROOT FILE for na22-timewalk.",
    )
    parser.add_argument(
        "--cfd-frac",
        dest="cfd_frac",
        type=float,
        default=None,
        help="CFD fraction for na22-timewalk's raw timing (raw-area mqana; "
             "default there 0.2).",
    )
    parser.add_argument(
        "--proc-file",
        dest="proc_file",
        default=None,
        help="Exact DRSprocessing ROOT FILE for na22-energy.",
    )
    # Renamed from --area-root / --proc-root on 2026-09-21: "-root" named the
    # file FORMAT and read like a sibling of --data-root, which is a DIRECTORY.
    from pmtdeconv.cli import add_renamed_alias
    add_renamed_alias(parser, "--area-root", "--area-file", dest="area_file")
    add_renamed_alias(parser, "--proc-root", "--proc-file", dest="proc_file")


def run(args: argparse.Namespace) -> int:
    if args.recipe is None:
        print("Available recipes:")
        for name, description in RECIPES.items():
            print(f"  {name:<18} {description}")
        return 0

    if getattr(args, "manifest", None):
        raise SystemExit(
            "`pmtdeconv run` writes one manifest per workflow step; "
            "--manifest is only valid on individual commands."
        )

    if args.recipe == "na22-timewalk":
        return _run_na22_timewalk(args)
    if args.recipe == "na22-energy":
        return _run_na22_energy(args)
    if args.recipe == "build-template":
        return _run_build_template(args)
    raise SystemExit(f"Unknown recipe {args.recipe!r}.")


def _base_common(args: argparse.Namespace,
                 *,
                 dataset: str,
                 runs: Iterable[int],
                 preset: str | None = None) -> List[str]:
    selected_dataset = getattr(args, "dataset", dataset)
    selected_runs = list(getattr(args, "runs", list(runs)))
    selected_preset = getattr(args, "preset", preset)

    argv = ["--dataset", selected_dataset, "--runs", *map(str, selected_runs)]
    if selected_preset:
        argv += ["--preset", selected_preset]

    scalar_flags = (
        ("entry_stop", "--entry-stop"),
        ("data_root", "--data-root"),
        ("tree", "--tree"),
        ("dt_ns", "--dt-ns"),
        ("polarity", "--polarity"),
        ("out_suffix", "--out-suffix"),
        ("log_level", "--log-level"),
    )
    for attribute, flag in scalar_flags:
        if hasattr(args, attribute):
            argv += [flag, str(getattr(args, attribute))]

    if hasattr(args, "branches"):
        argv += ["--branches", *map(str, args.branches)]
    hash_inputs = getattr(args, "hash_inputs", None)
    if hash_inputs is not None:
        argv.append("--hash-inputs" if hash_inputs else "--no-hash-inputs")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    return argv


def _without(argv: List[str], *flags: str) -> List[str]:
    """``argv`` minus each ``flag`` and its single value."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        else:
            out.append(a)
    return out


def _recipe_root(args: argparse.Namespace,
                 recipe: str,
                 run: int | str) -> Path:
    if getattr(args, "out_dir", None):
        return Path(args.out_dir).resolve()
    return (
        PROJECT_ROOT / "outputs" / "recipes" / f"{recipe}_run{run}"
    ).resolve()


def _execute(label: str, argv: List[str]) -> None:
    from pmtdeconv.cli import main

    print()
    print(f"══ {label} ══")
    print(f"pmtdeconv {' '.join(argv)}")
    code = main(argv)
    if code:
        raise SystemExit(code)


def _run_na22_timewalk(args: argparse.Namespace) -> int:
    if getattr(args, "dataset", "na22") != "na22":
        raise SystemExit("na22-timewalk requires --dataset na22.")
    runs = list(getattr(args, "runs", [65213]))
    if len(runs) != 1:
        raise SystemExit("na22-timewalk currently accepts exactly one run.")
    run_number = runs[0]
    root = _recipe_root(args, args.recipe, run_number)
    timing_dir = root / "raw-timing"
    area_dir = root / "raw-area"
    timewalk_dir = root / "timewalk"
    common = _base_common(args, dataset="na22", runs=runs)

    # Run 65213 is ~86% saturated on CH1, which the deconvolution drops, so
    # Δt comes from raw CFD-20 (1.56 vs 1.76 ns for STFT, 2026-10-21).
    timing_argv = ["raw-area", "mqana", *common, "--out-dir", str(timing_dir)]
    if args.cfd_frac is not None:
        timing_argv += ["--cfd-frac", str(args.cfd_frac)]
    _execute("raw timing", timing_argv)
    if args.raw_source == "drs":
        area_argv = ["raw-area", "drs", *_without(common, "--polarity", "--dt-ns"),
                     "--out-dir", str(area_dir)]
        if args.area_file:
            area_argv += ["--file", args.area_file]
        _execute("raw area", area_argv)
        area_csv = area_dir / f"area_drs_run{run_number}.csv"
    else:
        area_csv = timing_dir / f"area_mqana_run{run_number}.csv"
    timing_csv = timing_dir / f"raw_timing_run{run_number}.csv"

    # Guard against a raw step that ran yet produced no CSV, so the recipe
    # stops with a clear message instead of cascading into timewalk.
    missing = [p for p in (area_csv, timing_csv) if not p.exists()]
    if not getattr(args, "dry_run", False) and missing:
        raise SystemExit(
            f"raw-area produced no output for run {run_number} "
            f"(expected {', '.join(map(str, missing))}).\n"
            f"  The raw-area step likely couldn't find its input file — see the "
            f"error above, or point at your data with --data-root <dir> "
            f"(or $PMTDECONV_DATA_ROOT).")
    _execute(
        "timewalk",
        [
            "timewalk", "run", *common,
            "--area-csv", str(area_csv),
            "--hits-csv", str(timing_csv),
            "--time-col", "t_cfd",
            "--out-dir", str(timewalk_dir),
        ],
    )
    print()
    print(f"Results: {root}")
    return 0


def _run_na22_energy(args: argparse.Namespace) -> int:
    if getattr(args, "dataset", "na22") != "na22":
        raise SystemExit("na22-energy requires --dataset na22.")
    runs = list(getattr(args, "runs", [65212]))
    if len(runs) != 1:
        raise SystemExit("na22-energy currently accepts exactly one run.")
    root = _recipe_root(args, args.recipe, runs[0])
    common = _base_common(args, dataset="na22", runs=runs)
    argv = ["plot", "energy", *common, "--out-dir", str(root / "energy")]
    if args.proc_file:
        argv += ["--file", args.proc_file]
    _execute("energy comparison", argv)
    print()
    print(f"Results: {root}")
    return 0


def _run_build_template(args: argparse.Namespace) -> int:
    dataset = getattr(args, "dataset", "cd109")
    runs = list(getattr(args, "runs", []))
    root = _recipe_root(args, args.recipe, runs[0] if runs else dataset)
    argv = ["template", "build", "--dataset", dataset, "--out-dir", str(root)]
    for attribute, flag in (
        ("data_root", "--data-root"),
        ("entry_stop", "--entry-stop"),
        ("log_level", "--log-level"),
    ):
        if hasattr(args, attribute):
            argv += [flag, str(getattr(args, attribute))]
    hash_inputs = getattr(args, "hash_inputs", None)
    if hash_inputs is not None:
        argv.append("--hash-inputs" if hash_inputs else "--no-hash-inputs")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    _execute("SPE template build", argv)
    print()
    print(f"Results: {root}")
    return 0
