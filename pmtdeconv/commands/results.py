"""``pmtdeconv results`` — list, inspect, and locate manifest-backed outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.artifacts import build_artifact_index

DISPLAY_COMMANDS = {
    "pulse": "hits",
    "waveform": "area",
    "raw_area": "raw-area",
}
STORED_COMMANDS = {display: stored for stored, display in DISPLAY_COMMANDS.items()}


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
        stored_command = STORED_COMMANDS.get(command, command)
        entries = [
            entry for entry in entries
            if str(entry.get("subcommand") or "").startswith(stored_command)
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
