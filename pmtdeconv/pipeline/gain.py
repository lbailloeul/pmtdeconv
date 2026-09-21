"""PE-scale resolution — per-branch 1-PE charge (pVs) + provenance.

:meth:`Gain.from_settings` is the single resolution chain for the charge→PE
anchor, mirroring :meth:`pmtdeconv.pipeline.template.Template.from_settings`:

1. explicit ``--spe-charge BR=VAL`` / card ``spe_charge:`` / dataset preset
   (``settings.spe_charge_pvs``; the provenance layer distinguishes cli/card/
   dataset)                                     -> provenance per layer
2. gain catalog by ``pmt_type`` (file-backed under
   ``<data_root>/calibrations/gain/`` shadows in-code)  -> ``catalog``
3. nothing                                       -> ``unresolved`` (LEGAL)

Unresolved is a first-class state: nPE columns stay blank and only
``--in-regime-only`` refuses to run (pointing at ``gain calibrate``) — the
tool never silently invents a charge scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from pmtdeconv.pipeline.calibration import ProvenanceRecord
from pmtdeconv.presets.gain_catalog import resolve_gain


def raw_window_fraction(r_calib, dt_ns: float, window_ns: float = 35.0
                        ) -> float:
    """Fraction of a unit-NET-area kernel's area inside the mqana ±window/2
    around its peak.

    Converts the deconvolved 1-PE anchor (``spe_charge`` = the kernel's NET
    area, undershoot included) into the RAW windowed 1-PE charge the
    ``npe_raw`` estimator actually integrates: ``raw_1pe ≈ spe_charge ×
    this``. ≈0.6–1.0 for biexp kernels; ≫1 for AC-coupled kernels whose net
    area is the small difference of main peak and undershoot (e.g. the
    Tang-2024 R5912 model, where it is ~20).
    """
    r = np.asarray(r_calib, dtype=float)
    if r.size == 0:
        return 1.0
    pk = int(np.argmax(r))
    half = max(int(round(0.5 * window_ns / dt_ns)), 1)
    lo = max(pk - half, 0)
    hi = min(pk + half + 1, r.size)
    frac = float(np.sum(r[lo:hi]) * dt_ns)
    return frac if np.isfinite(frac) and frac > 0 else 1.0


def estimate_npe_raw(wave, dt_ns: float, spe_charge: float, *,
                     n_pre: int = 50, window_ns: float = 35.0) -> float:
    """Raw (pre-deconvolution) nPE estimate for one waveform.

    MQAna-style narrow-window raw charge — pedestal = mean of the first
    ``n_pre`` samples, integrate ±``window_ns``/2 around the argmax, divide by
    the 1-PE charge (pVs). Deconvolution-INDEPENDENT by design, so the regime
    gate never depends on the thing being validated. ``wave`` must be
    polarity-corrected (positive-going), as the event source guarantees.
    """
    if spe_charge is None or not np.isfinite(spe_charge) or spe_charge <= 0:
        return float("nan")
    w = np.asarray(wave, dtype=float)
    if w.size == 0 or not np.isfinite(w).any():
        return float("nan")          # empty / all-NaN (DRS dead cells)
    ped_slice = w[:n_pre] if w.size >= n_pre else w
    ped = (float(np.nanmean(ped_slice)) if np.isfinite(ped_slice).any()
           else float(np.nanmean(w)))
    sig = w - ped
    pk = int(np.nanargmax(sig))
    half = max(int(round(0.5 * window_ns / dt_ns)), 1)
    lo = max(pk - half, 0)
    hi = min(pk + half + 1, w.size)
    return float(np.nansum(sig[lo:hi]) * dt_ns) / float(spe_charge)


@dataclass
class Gain:
    """Resolved per-branch 1-PE charge in pVs (or unresolved: ``per_branch is
    None``)."""
    per_branch: Optional[Dict[str, float]]
    provenance: ProvenanceRecord
    pmt_type: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return bool(self.per_branch)

    def spe_charge(self, branch: str) -> Optional[float]:
        """1-PE charge for ``branch`` (None when uncalibrated for it)."""
        if not self.per_branch:
            return None
        return self.per_branch.get(branch)

    @classmethod
    def from_settings(cls, settings, provenance: Optional[Dict[str, str]] = None,
                      *, command: str = "hits") -> "Gain":
        prov = provenance or {}
        layer = prov.get("spe_charge_pvs")           # "cli" | "card" | "dataset" | None
        per_branch = getattr(settings, "spe_charge_pvs", None)
        if per_branch:
            source = {"cli": "explicit"}.get(layer, layer or "dataset")
            return cls(
                per_branch=dict(per_branch),
                provenance=ProvenanceRecord(
                    source=source, family="gain", command=command,
                    dataset=getattr(settings, "dataset", None),
                    detail=f"spe_charge from {layer or 'dataset'} layer"),
                pmt_type=getattr(settings, "pmt_type", None),
            )

        pmt_type = getattr(settings, "pmt_type", None)
        if pmt_type:
            cal, cat_source = resolve_gain(
                pmt_type, data_root=getattr(settings, "data_root", None))
            if cal is not None:
                return cls(
                    per_branch=dict(cal.spe_charge_pvs),
                    provenance=ProvenanceRecord(
                        source="catalog", family="gain", command=command,
                        dataset=getattr(settings, "dataset", None),
                        detail=f"gain catalog[{pmt_type}] ({cat_source}, "
                               f"version {cal.version})"),
                    pmt_type=pmt_type,
                )

        return cls(
            per_branch=None,
            provenance=ProvenanceRecord(
                source="unresolved", family="gain", command=command,
                dataset=getattr(settings, "dataset", None),
                detail="no spe_charge configured — run `pmtdeconv gain "
                       "calibrate` to fit the 1-PE quantization peak"),
            pmt_type=getattr(settings, "pmt_type", None),
        )
