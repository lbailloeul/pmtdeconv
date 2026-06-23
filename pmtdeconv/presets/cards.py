"""Dataset cards — the file-backed (YAML) twin of :class:`DatasetPreset`.

A *card* describes a dataset the same way an in-code ``DatasetPreset`` does, but
lives in a YAML file so a new detector needs no Python edit. ``--dataset
./path/to/card.yaml`` (or a directory containing ``dataset.yaml``) loads it via
:func:`load_card`. ``presets/datasets.py`` stays the in-code *seed* registry;
cards layer on top — S1 wires the path form, S2 adds name-based discovery +
seed shadowing.

Field mapping (YAML key → ``DatasetPreset`` field), with defaults for omitted
keys so a minimal card just needs the detector specifics:

    name              → name (default: card file stem)
    data_dir          → data_dir (default: the card's own directory;
                          relative paths resolve against the card file)
    file_glob         → file_glob (default "flattened_Run*.root")
    exclude_glob      → exclude_glob (default None)
    format            → format (default "flat")
    tree              → tree (default "waveforms")
    branches          → branch_pattern (list or "auto"; default "auto")
    dt_ns             → dt_ns (float or "auto"; default "auto")
    polarity          → polarity (default +1)
    cosine_cutoff_mhz → cosine_cutoff_mhz (default 100.0)
    template          → default_template_csv (path vs card, or null)
    template_na22fmt  → default_template_csv_na22fmt
    spe_band          → spe_band_mv ([lo, hi]; default (5, 30))
    pmt_type / hv_volts / notes / offline_data_dir / offline_file_glob → same
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from pmtdeconv.presets.datasets import DatasetPreset

_DEFAULTS = dict(
    file_glob="flattened_Run*.root", exclude_glob=None, format="flat",
    tree="waveforms", branch_pattern="auto", dt_ns="auto", polarity=1,
    cosine_cutoff_mhz=100.0, spe_band_mv=(5.0, 30.0),
)


def is_card_reference(value) -> bool:
    """True if a ``--dataset`` value looks like a card PATH (vs a registry
    name): ends in .yaml/.yml, contains a path separator, or is an existing
    file/dir. A bare registry name like ``na22`` is not a card reference."""
    if value is None:
        return False
    s = str(value)
    if s.endswith((".yaml", ".yml")):
        return True
    if "/" in s or os.sep in s:
        return True
    return Path(s).expanduser().exists()


def load_card(path) -> DatasetPreset:
    """Load a YAML dataset card into a :class:`DatasetPreset`.

    ``path`` may be the card file, or a directory holding ``dataset.yaml`` /
    ``dataset.yml``. Relative ``data_dir`` / ``template`` paths resolve against
    the card file's directory; ``data_dir`` defaults to that directory.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        for cand in ("dataset.yaml", "dataset.yml"):
            if (path / cand).exists():
                path = path / cand
                break
        else:
            raise FileNotFoundError(
                f"no dataset.yaml/.yml in directory {path}")
    if not path.exists():
        raise FileNotFoundError(f"dataset card not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"dataset card {path} must be a YAML mapping, got {type(raw).__name__}")

    card_dir = path.parent

    def _resolve(p) -> Optional[Path]:
        if p in (None, ""):
            return None
        pp = Path(str(p)).expanduser()
        return pp if pp.is_absolute() else (card_dir / pp)

    dt_ns = raw.get("dt_ns", _DEFAULTS["dt_ns"])
    spe = raw.get("spe_band", _DEFAULTS["spe_band_mv"])

    return DatasetPreset(
        name=str(raw.get("name") or path.stem),
        data_dir=Path(_resolve(raw.get("data_dir")) or card_dir),
        file_glob=raw.get("file_glob", _DEFAULTS["file_glob"]),
        exclude_glob=raw.get("exclude_glob", _DEFAULTS["exclude_glob"]),
        format=raw.get("format", _DEFAULTS["format"]),
        tree=raw.get("tree", _DEFAULTS["tree"]),
        branch_pattern=raw.get("branches", _DEFAULTS["branch_pattern"]),
        dt_ns=(dt_ns if dt_ns == "auto" else float(dt_ns)),
        polarity=int(raw.get("polarity", _DEFAULTS["polarity"])),
        cosine_cutoff_mhz=float(
            raw.get("cosine_cutoff_mhz", _DEFAULTS["cosine_cutoff_mhz"])),
        default_template_csv=_resolve(raw.get("template")),
        spe_band_mv=(float(spe[0]), float(spe[1])),
        notes=raw.get("notes", ""),
        default_template_csv_na22fmt=_resolve(raw.get("template_na22fmt")),
        offline_data_dir=_resolve(raw.get("offline_data_dir")),
        offline_file_glob=raw.get("offline_file_glob"),
        pmt_type=raw.get("pmt_type"),
        hv_volts=raw.get("hv_volts"),
    )
