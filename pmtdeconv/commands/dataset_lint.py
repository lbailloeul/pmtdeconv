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
    from pmtdeconv.presets.cards import (
        _read_card_yaml, load_card, read_card_settings,
    )

    try:
        ds = load_card(args.card)
        settings = read_card_settings(args.card)
        raw, _card_dir = _read_card_yaml(args.card)
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

    # Unknown TOP-LEVEL keys: a warning, not an error — a card written for a
    # newer pmtdeconv must still lint on an older one. But it is reported,
    # because a silently-ignored `polarty: -1` is a setting the expert believes
    # is in effect and is not.
    from pmtdeconv.presets.cards import suggest_card_key, unknown_card_keys
    for key in unknown_card_keys(raw):
        hint = suggest_card_key(key)
        warnings.append(
            f"unknown key {key!r} is ignored"
            + (f" — did you mean {hint!r}?" if hint
               else " — free-form provenance belongs under `meta:`"))

    valid = _settings_fields()
    for key in settings:
        if key not in valid:
            errors.append(f"settings: unknown Settings field {key!r}")

    if ds.fom is not None:
        from pmtdeconv.fom import FOMS
        if ds.fom not in FOMS:
            errors.append(
                f"fom {ds.fom!r} is not a known FoM ({', '.join(FOMS)})")

    if ds.spe_charge_pvs is not None:
        for br, v in ds.spe_charge_pvs.items():
            if not (isinstance(v, (int, float)) and 0 < float(v) < 1e7):
                errors.append(
                    f"spe_charge[{br}] {v!r} must be a positive 1-PE charge "
                    f"in pVs")

    if not (len(ds.pe_band) == 2 and float(ds.pe_band[0]) < float(ds.pe_band[1])
            and float(ds.pe_band[0]) >= 0):
        errors.append(f"pe_band {ds.pe_band!r} must be [lo, hi] with "
                      f"0 <= lo < hi (nPE)")

    if ds.template_csv_by_channel:
        for br, p in ds.template_csv_by_channel.items():
            if not Path(p).exists():
                warnings.append(
                    f"per-channel template for {br}: {p} does not exist")

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
