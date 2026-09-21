"""Gain catalog — ``pmt_type`` -> per-branch 1-PE charge (pVs).

The PE-scale anchor of the 0–20 PE re-scope (2026-07-28): a per-tube,
transferable calibration mapping each branch to its single-photoelectron
charge in pVs (mV·ns), so deconvolved charge divides into nPE. Mirrors the
template catalog (:mod:`pmtdeconv.presets.template_catalog`) keyed on
``pmt_type``, with the veto-profile file-backing pattern: a YAML file under
``<data_root>/calibrations/gain/<pmt_type>.yaml`` SHADOWS the in-code seed.

The in-code seed list is EMPTY on purpose — a 1-PE charge is a measured
quantity; ``pmtdeconv gain calibrate`` fits it from a real charge spectrum and
(with ``--write-calibration``) persists it here. The tool never invents a gain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pmtdeconv.pipeline.calibration import Catalog


@dataclass(frozen=True)
class GainCalibration:
    """Per-tube 1-PE charge calibration.

    ``spe_charge_pvs`` maps branch -> 1-PE charge in pVs. ``fit`` optionally
    carries the per-branch fit details (mu/sigma/n/ok) for provenance.
    """
    pmt_type: str
    spe_charge_pvs: Dict[str, float]
    note: str = ""
    version: str = "0"
    fit: Dict[str, Dict[str, float]] = field(default_factory=dict)


class GainCatalog(Catalog):
    """``pmt_type`` -> :class:`GainCalibration` (read-through + in-memory save)."""
    family = "gain"

    def __init__(self, entries: List[GainCalibration]):
        self._by_pmt: Dict[str, GainCalibration] = {
            e.pmt_type: e for e in entries}

    def lookup(self, *, pmt_type: Optional[str] = None, **_: Any
               ) -> Optional[GainCalibration]:
        if not pmt_type:
            return None
        return self._by_pmt.get(pmt_type)

    def pmt_types(self) -> List[str]:
        return sorted(self._by_pmt)

    def save(self, name: str, artifact: GainCalibration, provenance=None) -> None:
        self._by_pmt[name] = artifact


# Deliberately empty: gains are measured, never shipped as code defaults.
GAIN_CATALOG = GainCatalog([])


# ─── file-backed, versioned gain catalog (mirrors veto_profiles) ─────────

def gain_dir(data_root) -> Path:
    """``<data_root>/calibrations/gain`` — the on-disk gain catalog."""
    return Path(data_root) / "calibrations" / "gain"


def load_gain_yaml(path) -> GainCalibration:
    """Load one YAML gain calibration (strict: unknown keys are an error)."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"gain calibration {path} must be a YAML mapping")
    known = set(GainCalibration.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"gain calibration {path}: unknown field(s) {sorted(unknown)}")
    raw["spe_charge_pvs"] = {
        str(k): float(v) for k, v in (raw.get("spe_charge_pvs") or {}).items()}
    return GainCalibration(**raw)


def save_gain_yaml(cal: GainCalibration, data_root) -> Path:
    """Persist a gain calibration to the on-disk catalog; returns the path."""
    import yaml
    out_dir = gain_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cal.pmt_type}.yaml"
    out.write_text(yaml.safe_dump(asdict(cal), sort_keys=False))
    return out


def discover_gains(data_root) -> Dict[str, GainCalibration]:
    """Every file-backed gain calibration under the data root's catalog dir."""
    d = gain_dir(data_root)
    found: Dict[str, GainCalibration] = {}
    if d.is_dir():
        for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            cal = load_gain_yaml(p)
            found[cal.pmt_type] = cal
    return found


def resolve_gain(pmt_type: str, data_root=None):
    """Resolve a ``pmt_type`` to ``(GainCalibration, source)`` or (None, None).
    A file under ``<data_root>/calibrations/gain/`` SHADOWS the in-code seed."""
    if data_root is not None:
        file_cals = discover_gains(data_root)
        if pmt_type in file_cals:
            return file_cals[pmt_type], "file"
    cal = GAIN_CATALOG.lookup(pmt_type=pmt_type)
    if cal is not None:
        return cal, "builtin"
    return None, None
