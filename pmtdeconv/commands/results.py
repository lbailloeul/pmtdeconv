"""``pmtdeconv results`` — list, inspect, and locate manifest-backed outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.artifacts import build_artifact_index

# Manifests written before 2026-09-21 stored the pre-rename command names
# (the verbs were renamed earlier, but the stored value lagged). New manifests
# store the real names. Both are normalized to the display name on read, so
# `results list --command hits` finds old AND new runs.
DISPLAY_COMMANDS = {
    "pulse": "hits",
    "waveform": "area",
    "raw_area": "raw-area",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from pmtdeconv.cli import add_common_args

    sub = parser.add_subparsers(dest="results_action", required=True)

    list_parser = sub.add_parser("list", help="List recent manifest-backed results")
    _add_outputs_dir(list_parser)
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--command", default=None)
    add_common_args(list_parser)
    list_parser.set_defaults(_results_fn=_list)

    show = sub.add_parser("show", help="Show one result manifest")
    show.add_argument("result_id")
    show.add_argument("--json", action="store_true")
    _add_outputs_dir(show)
    add_common_args(show)
    show.set_defaults(_results_fn=_show)

    index = sub.add_parser(
        "index",
        help="MAINTAIN the output index: write artifact_index.json, audit "
             "orphan dirs and invalid manifests. (Was `info artifacts` "
             "until 2026-09-14.) To FIND outputs day-to-day use "
             "`results list` — same underlying index.")
    _add_outputs_dir(index)
    index.add_argument("--write", nargs="?", const="", default=None,
                       help="Write JSON index. With no value, writes "
                            "<outputs-dir>/artifact_index.json.")
    index.add_argument("--no-orphans", action="store_true",
                       help="Only include directories that contain manifests.")
    index.add_argument("--limit", type=int, default=15,
                       help="Rows to print (default 15).")
    add_common_args(index)
    index.set_defaults(_results_fn=_index)

    path = sub.add_parser("path", help="Print one result directory")
    path.add_argument("result_id")
    _add_outputs_dir(path)
    add_common_args(path)
    path.set_defaults(_results_fn=_path)


def _add_outputs_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--outputs-dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Outputs directory to scan.",
    )


def run(args: argparse.Namespace) -> int:
    return int(args._results_fn(args) or 0)


def _entries(args: argparse.Namespace) -> tuple[Path, List[Dict[str, Any]]]:
    root = Path(args.outputs_dir).resolve()
    index = build_artifact_index(root, include_orphans=False)
    entries = [
        entry for entry in index["artifacts"]
        if entry.get("kind") == "manifest"
    ]
    return root, entries


def _result_id(entry: Dict[str, Any], root: Path) -> str:
    out_dir = Path(str(entry["out_dir"])).resolve()
    try:
        return out_dir.relative_to(root).as_posix()
    except ValueError:
        return out_dir.name


def _filtered(args: argparse.Namespace,
              root: Path,
              entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = getattr(args, "dataset", None)
    preset = getattr(args, "preset", None)
    command = getattr(args, "command", None)
    if dataset:
        entries = [entry for entry in entries if entry.get("dataset") == dataset]
    if preset:
        entries = [entry for entry in entries if entry.get("config") == preset]
    if command:
        # Compare in DISPLAY space: an old manifest says "pulse", a new one
        # "hits", and the user may type either — all three must match.
        wanted = _display_command(command)
        entries = [
            entry for entry in entries
            if _display_command(entry.get("subcommand")).startswith(wanted)
        ]
    return entries


def _list(args: argparse.Namespace) -> int:
    root, entries = _entries(args)
    entries = _filtered(args, root, entries)
    rows = entries[:max(args.limit, 0)]
    if not rows:
        print(f"No matching manifest-backed results under {root}.")
        return 0

    print(f"{'result_id':<45} {'command':<18} {'dataset':<12} {'runs'}")
    print("-" * 100)
    for entry in rows:
        runs = entry.get("runs") or []
        runs_text = ",".join(str(run) for run in runs) if isinstance(runs, list) else str(runs)
        command = _display_command(entry.get("subcommand"))
        print(
            f"{_result_id(entry, root):<45} "
            f"{command:<18} "
            f"{str(entry.get('dataset') or ''):<12} "
            f"{runs_text}"
        )
    return 0


def _select(result_id: str,
            root: Path,
            entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact = [
        entry for entry in entries
        if result_id in {
            _result_id(entry, root),
            Path(str(entry["out_dir"])).name,
            str(entry["manifest_path"]),
        }
    ]
    if len(exact) == 1:
        return exact[0]

    prefix = [
        entry for entry in entries
        if _result_id(entry, root).startswith(result_id)
    ]
    matches = exact or prefix
    if not matches:
        raise SystemExit(
            f"No result matching {result_id!r} under {root}. "
            "Use `pmtdeconv results list`."
        )
    if len(matches) > 1:
        ids = ", ".join(_result_id(entry, root) for entry in matches[:8])
        raise SystemExit(f"Ambiguous result {result_id!r}. Matches: {ids}")
    return matches[0]


def _show(args: argparse.Namespace) -> int:
    root, entries = _entries(args)
    entry = _select(args.result_id, root, entries)
    manifest_path = Path(str(entry["manifest_path"]))
    manifest = json.loads(manifest_path.read_text())
    if args.json:
        print(json.dumps(manifest, indent=2))
        return 0

    print(f"result_id : {_result_id(entry, root)}")
    print(f"path      : {entry['out_dir']}")
    print(f"command   : {_display_command(entry.get('subcommand'))}")
    print(f"dataset   : {entry.get('dataset')}")
    print(f"preset    : {entry.get('config')}")
    print(f"runs      : {entry.get('runs')}")
    print(f"created   : {entry.get('timestamp_utc')}")
    outputs = manifest.get("outputs") or []
    if outputs:
        print("outputs:")
        for output in outputs:
            print(f"  {output.get('role', 'output'):<24} {output.get('path')}")
    return 0


def _path(args: argparse.Namespace) -> int:
    root, entries = _entries(args)
    entry = _select(args.result_id, root, entries)
    print(entry["out_dir"])
    return 0


def _display_command(command: Any) -> str:
    value = str(command or "")
    head, separator, tail = value.partition(" ")
    replacement = DISPLAY_COMMANDS.get(head, head)
    return replacement + (separator + tail if separator else "")


def _index(args: argparse.Namespace) -> int:
    from pmtdeconv.io.artifacts import (
        DEFAULT_INDEX_NAME, build_artifact_index, write_artifact_index,
    )

    outputs_dir = Path(args.outputs_dir)
    index = build_artifact_index(
        outputs_dir,
        include_orphans=not args.no_orphans,
    )
    summary = index["summary"]

    print(f"{outputs_dir}")
    print(f"  manifests         : {summary['n_manifests']}")
    print(f"  invalid manifests : {summary['n_invalid_manifests']}")
    print(f"  orphan dirs       : {summary['n_orphan_dirs']}")
    print(f"  artifacts indexed : {summary['n_artifacts']}")
    if summary["output_role_counts"]:
        role_bits = [
            f"{role}={count}"
            for role, count in sorted(summary["output_role_counts"].items())
        ]
        print(f"  output roles      : {', '.join(role_bits)}")
    print()

    rows = index["artifacts"][:max(int(args.limit), 0)]
    if rows:
        print(f"{'kind':<14} {'subcommand':<18} {'dataset':<10} "
              f"{'config':<16} {'runs':<16} out_dir")
        print("-" * 120)
        for ent in rows:
            runs = ent.get("runs")
            if isinstance(runs, list):
                runs_s = ",".join(str(r) for r in runs[:4])
                if len(runs) > 4:
                    runs_s += ",..."
            elif runs is None:
                runs_s = ""
            else:
                runs_s = str(runs)
            print(f"{str(ent.get('kind') or ''):<14} "
                  f"{str(ent.get('subcommand') or ''):<18} "
                  f"{str(ent.get('dataset') or ''):<10} "
                  f"{str(ent.get('config') or ''):<16} "
                  f"{runs_s:<16} "
                  f"{Path(str(ent.get('out_dir') or '')).name}")

    if args.write is not None:
        target = (outputs_dir / DEFAULT_INDEX_NAME
                  if args.write == "" else Path(args.write))
        write_artifact_index(index, target)
        print()
        print(f"Wrote: {target}")
    return 0
