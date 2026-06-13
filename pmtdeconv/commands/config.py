"""``pmtdeconv config`` — inspect and resolve CLI configuration."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from pmtdeconv.presets import resolve_settings
from pmtdeconv.presets.configs import CONFIGS
from pmtdeconv.presets.datasets import DATASETS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from pmtdeconv.cli import add_common_args

    sub = parser.add_subparsers(dest="config_action", required=True)

    show = sub.add_parser("show", help="Show a dataset or named preset")
    show.add_argument("name", help="Dataset or preset name")
    show.add_argument("--json", action="store_true")
    add_common_args(show)
    show.set_defaults(_config_fn=_show)

    resolve = sub.add_parser(
        "resolve",
        help="Print final settings and where every value came from",
    )
    resolve.add_argument(
        "name",
        nargs="?",
        help="Optional preset name; equivalent to --preset NAME",
    )
    resolve.add_argument("--json", action="store_true")
    add_common_args(resolve)
    resolve.set_defaults(_config_fn=_resolve)


def run(args: argparse.Namespace) -> int:
    return int(args._config_fn(args) or 0)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _show(args: argparse.Namespace) -> int:
    if args.name in CONFIGS:
        cfg = CONFIGS[args.name]
        payload = {
            "kind": "preset",
            "name": cfg.name,
            "base_dataset": cfg.base_dataset,
            "description": cfg.description,
            "references": list(cfg.references),
            "overrides": {k: _json_value(v) for k, v in cfg.overrides.items()},
        }
    elif args.name in DATASETS:
        ds = DATASETS[args.name]
        payload = {
            "kind": "dataset",
            **{
                field.name: _json_value(getattr(ds, field.name))
                for field in dataclasses.fields(ds)
            },
        }
    else:
        known = sorted(set(CONFIGS) | set(DATASETS))
        raise KeyError(
            f"Unknown dataset or preset {args.name!r}. Known: {', '.join(known)}"
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{payload['kind']}: {payload['name']}")
    for key, value in payload.items():
        if key in {"kind", "name"}:
            continue
        if key == "overrides":
            print("overrides:")
            for field_name, field_value in value.items():
                print(f"  {field_name:<24} {field_value!r}")
        else:
            print(f"{key:<16} {value}")
    return 0


def _resolve(args: argparse.Namespace) -> int:
    positional = args.name
    flagged = getattr(args, "preset", None)
    if positional and flagged and positional != flagged:
        raise SystemExit(
            f"Preset supplied twice with different values: {positional!r} "
            f"and {flagged!r}."
        )
    if positional:
        args.preset = positional

    settings, provenance = resolve_settings(args, require_data_root=False)
    values = {
        field.name: _json_value(getattr(settings, field.name))
        for field in dataclasses.fields(settings)
    }
    payload = {
        "settings": values,
        "provenance": {
            field_name: provenance.get(field_name, "default")
            for field_name in values
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{'setting':<28} {'source':<9} value")
    print("-" * 90)
    for field_name, value in values.items():
        print(
            f"{field_name:<28} "
            f"{payload['provenance'][field_name]:<9} "
            f"{value!r}"
        )
    return 0
