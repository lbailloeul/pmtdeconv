"""``pmtdeconv dataset lint <card>`` — validate a YAML dataset card.

Loads the card (resolving ``base:``), checks the detector fields and the
``settings:`` block, and reports errors/warnings. Exit 0 if clean, 1 on errors.
Warnings (e.g. a not-yet-present data_dir) do not fail.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("card", help="Path to a card file or a directory "
                                     "containing dataset.yaml.")


def _settings_fields() -> set:
    from pmtdeconv.presets import Settings
    return {f.name for f in dataclasses.fields(Settings)}


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.presets.cards import load_card, read_card_settings

    try:
        ds = load_card(args.card)
        settings = read_card_settings(args.card)
    except Exception as exc:                       # noqa: BLE001 - report cleanly
        print(f"FAIL: card did not load: {exc}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    if ds.format not in ("flat", "object"):
        errors.append(f"format {ds.format!r} must be 'flat' or 'object'")
    if ds.polarity not in (1, -1):
        errors.append(f"polarity {ds.polarity!r} must be +1 or -1")
    if ds.dt_ns != "auto" and not (
            isinstance(ds.dt_ns, (int, float)) and ds.dt_ns > 0):
        errors.append(f"dt_ns {ds.dt_ns!r} must be a positive number or 'auto'")
    if ds.branch_pattern != "auto" and (
            not isinstance(ds.branch_pattern, list) or not ds.branch_pattern):
        errors.append(
            f"branches {ds.branch_pattern!r} must be a non-empty list or 'auto'")
    if not (len(ds.spe_band_mv) == 2 and ds.spe_band_mv[0] < ds.spe_band_mv[1]):
        errors.append(f"spe_band {ds.spe_band_mv} must be [lo, hi] with lo < hi")
    if ds.cosine_cutoff_mhz <= 0:
        errors.append(f"cosine_cutoff_mhz {ds.cosine_cutoff_mhz} must be > 0")

    # Path existence is a WARNING — the data may live elsewhere at run time
    # (overridable with --data-root).
    if not Path(ds.data_dir).exists():
        warnings.append(f"data_dir {ds.data_dir} does not exist (yet)")
    if ds.default_template_csv and not Path(ds.default_template_csv).exists():
        warnings.append(f"template {ds.default_template_csv} does not exist")

    valid = _settings_fields()
    for key in settings:
        if key not in valid:
            errors.append(f"settings: unknown Settings field {key!r}")

    if ds.fom is not None:
        from pmtdeconv.fom import FOMS
        if ds.fom not in FOMS:
            errors.append(
                f"fom {ds.fom!r} is not a known FoM ({', '.join(FOMS)})")

    name = ds.name
    for w in warnings:
        print(f"WARN  [{name}] {w}")
    for e in errors:
        print(f"ERROR [{name}] {e}")
    if errors:
        print(f"FAIL: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"OK: card {name!r} is valid ({len(warnings)} warning(s))")
    return 0
