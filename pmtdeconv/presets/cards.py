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


# Every top-level key a card may set. ``base`` and ``settings`` are handled
# outside _field_overrides; ``meta`` is free-form provenance. Anything else is
# almost certainly a typo — silently ignoring it (the behavior until
# 2026-09-18) meant a card could declare `polarty: -1`, lint clean, and run
# with the wrong polarity and no indication why.
CARD_KEYS = frozenset({
    "name", "base", "settings", "meta", "notes",
    "data_dir", "file_glob", "exclude_glob", "format", "tree", "branches",
    "dt_ns", "polarity", "cosine_cutoff_mhz",
    "template", "template_na22fmt", "templates",
    "spe_band", "spe_charge", "pe_band", "fom",
    "offline_data_dir", "offline_file_glob",
    "pmt_type", "hv_volts",
})


def unknown_card_keys(raw: dict) -> list:
    """Top-level keys a card sets that pmtdeconv does not understand."""
    return sorted(k for k in (raw or {}) if k not in CARD_KEYS)


def suggest_card_key(key: str) -> Optional[str]:
    """Closest real key name, for a 'did you mean' hint."""
    import difflib
    match = difflib.get_close_matches(key, CARD_KEYS, n=1, cutoff=0.7)
    return match[0] if match else None


# Cards are meant to be SHARED, so a path in one must not pin it to the author's
# filesystem. ``${PMTDECONV_DATA_ROOT}`` expands to the same data root the
# in-code seeds resolve (env var -> nearest marker-bearing data/ dir -> legacy
# fallback), so a card written on one machine resolves correctly on another
# with no edit. Other ${VARS} expand from the environment as usual.
DATA_ROOT_TOKEN = "${PMTDECONV_DATA_ROOT}"


def expand_card_path(raw: str) -> str:
    """Expand ``${PMTDECONV_DATA_ROOT}`` and any other environment variables."""
    if DATA_ROOT_TOKEN in raw:
        from pmtdeconv.presets.datasets import _resolve_data_root
        raw = raw.replace(DATA_ROOT_TOKEN, str(_resolve_data_root()))
    return os.path.expandvars(raw)


def _field_overrides(raw: dict, card_dir: Path) -> dict:
    """Map the card keys PRESENT in ``raw`` to ``DatasetPreset`` field kwargs.

    Only keys the card actually sets are returned, so a card with ``base:`` can
    be applied as a sparse override (via :func:`dataclasses.replace`) and inherit
    everything else from the base. ``data_dir`` / template paths resolve against
    the card directory.
    """
    def _resolve(p) -> Optional[Path]:
        if p in (None, ""):
            return None
        pp = Path(expand_card_path(str(p))).expanduser()
        pp = pp if pp.is_absolute() else (card_dir / pp)
        # Normalize: a card may legitimately spell a path with ".." (the
        # bundled templates sit one directory over from the shipped cards).
        # Resolving here keeps manifests recording real absolute paths rather
        # than "…/datasets/../templates/…" relative to whatever the CWD was.
        # abspath (not resolve): normalizes and absolutizes without
        # following symlinks, so a symlinked data root stays as written.
        return Path(os.path.abspath(pp))

    ov: dict = {}
    if "name" in raw:
        ov["name"] = str(raw["name"])
    if "data_dir" in raw:
        ov["data_dir"] = Path(_resolve(raw["data_dir"]))
    if "file_glob" in raw:
        ov["file_glob"] = raw["file_glob"]
    if "exclude_glob" in raw:
        ov["exclude_glob"] = raw["exclude_glob"]
    if "format" in raw:
        ov["format"] = raw["format"]
    if "tree" in raw:
        ov["tree"] = raw["tree"]
    if "branches" in raw:
        ov["branch_pattern"] = raw["branches"]
    if "dt_ns" in raw:
        ov["dt_ns"] = (raw["dt_ns"] if raw["dt_ns"] == "auto"
                       else float(raw["dt_ns"]))
    if "polarity" in raw:
        ov["polarity"] = int(raw["polarity"])
    if "cosine_cutoff_mhz" in raw:
        ov["cosine_cutoff_mhz"] = float(raw["cosine_cutoff_mhz"])
    if "template" in raw:
        ov["default_template_csv"] = _resolve(raw["template"])
    if "template_na22fmt" in raw:
        ov["default_template_csv_na22fmt"] = _resolve(raw["template_na22fmt"])
    if "spe_band" in raw:
        s = raw["spe_band"]
        ov["spe_band_mv"] = (float(s[0]), float(s[1]))
    if "notes" in raw:
        ov["notes"] = raw["notes"]
    if "offline_data_dir" in raw:
        ov["offline_data_dir"] = _resolve(raw["offline_data_dir"])
    if "offline_file_glob" in raw:
        ov["offline_file_glob"] = raw["offline_file_glob"]
    if "pmt_type" in raw:
        ov["pmt_type"] = raw["pmt_type"]
    if "hv_volts" in raw:
        ov["hv_volts"] = raw["hv_volts"]
    if "fom" in raw:
        ov["fom"] = raw["fom"]
    if "templates" in raw and raw["templates"]:
        # per-channel SPE CSVs: {branch: path}, each resolved vs the card dir
        ov["template_csv_by_channel"] = {
            str(br): _resolve(p) for br, p in raw["templates"].items()}
    if "spe_charge" in raw and raw["spe_charge"] is not None:
        # per-branch 1-PE charge anchor in pVs: {branch: value}
        ov["spe_charge_pvs"] = {
            str(br): float(v) for br, v in raw["spe_charge"].items()}
    if "meta" in raw and raw["meta"] is not None:
        # Free-form: carried verbatim onto Settings and into every manifest.
        ov["meta"] = dict(raw["meta"])
    if "pe_band" in raw and raw["pe_band"] is not None:
        b = raw["pe_band"]
        ov["pe_band"] = (float(b[0]), float(b[1]))
    return ov


def _resolve_base(base_ref, card_dir: Path) -> DatasetPreset:
    """Resolve a card's ``base:`` to a DatasetPreset. The base may be a registry
    name, a registry card name, or a card path (relative paths resolve against
    the referring card). Recurses through :func:`resolve_dataset`, so a base card
    can itself have a base."""
    from pmtdeconv.presets.datasets import resolve_dataset
    s = str(base_ref)
    if (not Path(s).is_absolute()
            and (s.endswith((".yaml", ".yml")) or "/" in s or os.sep in s)):
        s = str(card_dir / s)
    return resolve_dataset(s)


def _resolve_card_path(path) -> Path:
    """Resolve ``path`` to a card file: a file directly, or ``dataset.yaml`` /
    ``dataset.yml`` inside a directory. Raises if not found."""
    path = Path(path).expanduser()
    if path.is_dir():
        for cand in ("dataset.yaml", "dataset.yml"):
            if (path / cand).exists():
                return path / cand
        raise FileNotFoundError(f"no dataset.yaml/.yml in directory {path}")
    if not path.exists():
        raise FileNotFoundError(f"dataset card not found: {path}")
    return path


def _read_card_yaml(path) -> tuple[dict, Path]:
    """Return ``(raw_mapping, card_dir)`` for a card path (file or dir)."""
    p = _resolve_card_path(path)
    with open(p) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"dataset card {p} must be a YAML mapping, got {type(raw).__name__}")
    return raw, p.parent


def read_card_settings(path) -> dict:
    """Return a card's ``settings:`` block — analysis-knob overrides (Settings
    fields like ``denoiser`` / ``wthresh_k`` / ``no_vetoes``) that let a card
    subsume what a config preset did. Empty dict if absent."""
    raw, _ = _read_card_yaml(path)
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"card {path}: `settings:` must be a mapping")
    return dict(settings)


def load_card(path) -> DatasetPreset:
    """Load a YAML dataset card into a :class:`DatasetPreset`.

    ``path`` may be the card file, or a directory holding ``dataset.yaml`` /
    ``dataset.yml``. With ``base: <name|card>`` the card INHERITS that base's
    fields and overrides only the keys it sets (the inheritance primitive);
    without ``base:`` it builds from the per-field defaults. Relative
    ``data_dir`` / ``template`` paths resolve against the card's directory.

    The card's ``settings:`` block (analysis overrides) is read separately by
    :func:`read_card_settings`; it does not affect the returned DatasetPreset
    (a detector-only object).
    """
    raw, card_dir = _read_card_yaml(path)
    path = _resolve_card_path(path)

    # Loud at USE time, not just at lint time: a stray key is a setting the
    # expert believes is in effect and is not. Warn (never fail) so a card
    # written for a newer pmtdeconv still loads on an older one.
    for key in unknown_card_keys(raw):
        hint = suggest_card_key(key)
        print(f"[card] {path.name}: ignoring unknown key {key!r}"
              + (f" — did you mean {hint!r}?" if hint else
                 " — put free-form provenance under `meta:`"))

    overrides = _field_overrides(raw, card_dir)

    if "base" in raw and raw["base"] is not None:
        base = _resolve_base(raw["base"], card_dir)
        # The card's name defaults to its own file stem (not the base's) so a
        # derived card is identifiable; data_dir + everything else inherit
        # unless the card overrides them.
        overrides.setdefault("name", path.stem)
        import dataclasses
        return dataclasses.replace(base, **overrides)

    # No base → full preset from defaults + the card's fields.
    full = dict(
        name=path.stem, data_dir=card_dir,
        file_glob=_DEFAULTS["file_glob"], exclude_glob=_DEFAULTS["exclude_glob"],
        format=_DEFAULTS["format"], tree=_DEFAULTS["tree"],
        branch_pattern=_DEFAULTS["branch_pattern"], dt_ns=_DEFAULTS["dt_ns"],
        polarity=_DEFAULTS["polarity"],
        cosine_cutoff_mhz=_DEFAULTS["cosine_cutoff_mhz"],
        default_template_csv=None, spe_band_mv=_DEFAULTS["spe_band_mv"],
    )
    full.update(overrides)
    return DatasetPreset(**full)
