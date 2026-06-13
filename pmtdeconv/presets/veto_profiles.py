"""Veto-profile catalog — the per-(mode, denoiser) threshold sets (step 5).

The hits pipeline's veto + baseline-gate thresholds previously lived as hardcoded
literals in ``process_hits.WAVELET_PROFILE`` / ``STFT_PROFILE``, with the STFT
path silently relaxing them (``afterpulse_height_ratio`` 5→100, ``min_charge_
purity`` 0.5→0, the loose ``0.50/10/20/900/10`` baseline gate, ``no_trigger_amp``
15→1). Step 5 lifts those literals into a named :class:`VetoProfile` catalog —
``process_hits`` now *builds* its ``HitsProfile``s from these entries, so this is
the single source of truth (the byte-identical hits goldens guard the values).

``Detector.from_settings`` records which profile a run resolved to (provenance =
catalog), and ``vetoes calibrate`` derives a data-driven profile and writes it
here via ``--write-profile``. The chain Level A: explicit CLI thresholds →
catalog profile by (mode, denoiser) → generic default.

The ``allow_*`` behavior switches and ``denoiser_kind`` are NOT veto thresholds
(they are path behavior) and stay on ``HitsProfile``; a :class:`VetoProfile`
carries only the tunable threshold + baseline-gate fields, whose names match
``HitsProfile`` exactly so it can be splatted in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from pmtdeconv.pipeline.calibration import Catalog


@dataclass(frozen=True)
class VetoProfile:
    """A named set of veto + baseline-gate thresholds (field names mirror
    :class:`pmtdeconv.pipeline.process_hits.HitsProfile`)."""
    name: str
    # baseline quality gate
    flat_frac_min: float
    flat_k: float
    spike_k: float
    max_baseline_adc: float
    peak_height_adc: float
    # vetoes / windows
    no_trigger_amp: float
    raw_snr_mult: float
    t_peak_min_ns: float
    t_peak_max_ns: float
    afterpulse_window_ns: float
    afterpulse_height_ratio: float
    post_sat_window_ns: float
    post_sat_sigma_mult: float
    post_sat_offset_ns: float
    min_charge_purity: float
    baseline_search_end_ns: float = 600.0
    pad_ns: float = 18.0

    def hits_fields(self) -> Dict[str, Any]:
        """The threshold kwargs for building a ``HitsProfile`` (drops ``name``)."""
        d = asdict(self)
        d.pop("name")
        return d


# Neutral default = the strict Na22 wavelet-hits thresholds.
WAVELET_HITS_PROFILE = VetoProfile(
    name="wavelet_hits",
    flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
    max_baseline_adc=500.0, peak_height_adc=50.0,
    no_trigger_amp=15.0, raw_snr_mult=10.0,
    t_peak_min_ns=20.0, t_peak_max_ns=1400.0,
    afterpulse_window_ns=40.0, afterpulse_height_ratio=5.0,
    post_sat_window_ns=50.0, post_sat_sigma_mult=3.0, post_sat_offset_ns=20.0,
    min_charge_purity=0.5,
)

# Relaxed STFT-hits thresholds (the former silent override table).
STFT_HITS_PROFILE = VetoProfile(
    name="stft_hits",
    flat_frac_min=0.50, flat_k=10.0, spike_k=20.0,
    max_baseline_adc=900.0, peak_height_adc=10.0,
    no_trigger_amp=1.0, raw_snr_mult=1.0,
    t_peak_min_ns=1.0, t_peak_max_ns=1400.0,
    afterpulse_window_ns=40.0, afterpulse_height_ratio=100.0,
    post_sat_window_ns=50.0, post_sat_sigma_mult=100.0, post_sat_offset_ns=20.0,
    min_charge_purity=0.0,
)


class VetoProfileCatalog(Catalog):
    """``name`` / (mode, denoiser) -> :class:`VetoProfile`. Read-through +
    in-memory ``save`` for ``vetoes calibrate --write-profile`` (session-scoped;
    a persistent backing store is a step-8 follow-up)."""
    family = "veto"

    # denoiser -> default profile name (the per-path table, now named)
    _BY_DENOISER = {"wavelet": "wavelet_hits", "none": "wavelet_hits",
                    "stft": "stft_hits"}

    def __init__(self, profiles: List[VetoProfile]):
        self._by_name: Dict[str, VetoProfile] = {p.name: p for p in profiles}

    def lookup(self, *, name: Optional[str] = None,
               denoiser: Optional[str] = None, **_: Any
               ) -> Optional[VetoProfile]:
        if name is not None:
            return self._by_name.get(name)
        if denoiser is not None:
            return self._by_name.get(self._BY_DENOISER.get(denoiser, "wavelet_hits"))
        return None

    def resolve_name(self, denoiser: Optional[str]) -> str:
        return self._BY_DENOISER.get(denoiser or "wavelet", "wavelet_hits")

    def names(self) -> List[str]:
        return sorted(self._by_name)

    def save(self, name: str, artifact: VetoProfile, provenance=None) -> None:
        self._by_name[name] = artifact


VETO_PROFILES = VetoProfileCatalog([WAVELET_HITS_PROFILE, STFT_HITS_PROFILE])
