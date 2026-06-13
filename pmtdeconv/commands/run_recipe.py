"""``pmtdeconv run`` — execute a complete named analysis workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from pmtdeconv import PROJECT_ROOT


RECIPES = {
    "na22-timewalk": (
        "Run raw-area extraction, hit reconstruction, and the timewalk scan "
        "for Na22 Run 65213 using the stable timewalk preset."
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
        "--area-root",
        default=None,
        help="Explicit DRSprocessing ROOT file for na22-timewalk.",
    )
    parser.add_argument(
        "--proc-root",
        default=None,
        help="Explicit DRSprocessing ROOT file for na22-energy.",
    )


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
        ("sat_thresh", "--sat-thresh"),
        ("out_suffix", "--out-suffix"),
        ("log_level", "--log-level"),
    )
    for attribute, flag in scalar_flags:
        if hasattr(args, attribute):
            argv += [flag, str(getattr(args, attribute))]

    if hasattr(args, "branches"):
        argv += ["--branches", *map(str, args.branches)]
    if getattr(args, "hash_inputs", False):
        argv.append("--hash-inputs")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    return argv


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
    area_dir = root / "raw-area"
    hits_dir = root / "hits"
    timewalk_dir = root / "timewalk"
    common = _base_common(
        args,
        dataset="na22",
        runs=runs,
        preset="na22-timewalk",
    )

    area_argv = [
        "raw-area", args.raw_source, *common,
        "--out-dir", str(area_dir),
    ]
    if args.raw_source == "drs" and args.area_root:
        area_argv += ["--root", args.area_root]
    _execute("raw area", area_argv)

    _execute(
        "hit reconstruction",
        ["hits", *common, "--out-dir", str(hits_dir)],
    )

    area_csv = area_dir / f"area_{args.raw_source}_run{run_number}.csv"
    pulse_csv = hits_dir / "deconv_pulse.csv"
    _execute(
        "timewalk",
        [
            "timewalk", "run", *common,
            "--area-csv", str(area_csv),
            "--pulse-csv", str(pulse_csv),
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
    if args.proc_root:
        argv += ["--proc-root", args.proc_root]
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
    if getattr(args, "hash_inputs", False):
        argv.append("--hash-inputs")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    _execute("SPE template build", argv)
    print()
    print(f"Results: {root}")
    return 0
