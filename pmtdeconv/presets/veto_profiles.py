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
    min_charge_purity: float
    baseline_search_end_ns: float = 600.0
    pad_ns: float = 18.0
    # 0–20 PE re-scope (2026-07-28): multi-hit / benchmark knobs, all
    # defaulting to the historical hardcoded behavior (byte-identical).
    # `neighbor_min_ratio` — a neighbouring peak must reach this fraction of
    # the current peak's height to cap its integration window (the invisible
    # multi-hit charge-sharing parameter, now named).
    neighbor_min_ratio: float = 0.1
    # `window_extend_frac` — integration windows extend by this fraction of
    # the pulse width on each side beyond the troughs (1.0 = legacy).
    window_extend_frac: float = 1.0
    # `baseline_sigma_scale` — multiplier on the absolute log-σ baseline pass
    # threshold (7 mV @200 ns → 1.5 mV @64 ns anchors). >1 admits noisier
    # detectors (e.g. the σ=12 mV `noisy` regime) into veto-on processing.
    baseline_sigma_scale: float = 1.0
    # S8: catalog version stamp (recorded in provenance; "0" = the built-in
    # defaults). A file-backed profile written by `vetoes calibrate` carries the
    # date it was derived so a run records exactly which calibration it used.
    version: str = "0"

    def hits_fields(self) -> Dict[str, Any]:
        """The threshold kwargs for building a ``HitsProfile`` (drops the
        bookkeeping ``name`` + ``version``, which are not HitsProfile fields)."""
        d = asdict(self)
        d.pop("name")
        d.pop("version", None)
        return d


# Neutral default = the strict Na22 wavelet-hits thresholds.
WAVELET_HITS_PROFILE = VetoProfile(
    name="wavelet_hits",
    flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
    max_baseline_adc=500.0, peak_height_adc=50.0,
    no_trigger_amp=15.0, raw_snr_mult=10.0,
    t_peak_min_ns=20.0, t_peak_max_ns=1400.0,
    afterpulse_window_ns=40.0, afterpulse_height_ratio=5.0,
    min_charge_purity=0.5,
)

# Neutral benchmark profile (0–20 PE re-scope, user-approved 2026-07-28):
# wavelet_hits base with ONLY the multi-hit-hostile members relaxed, so
# wavelet and STFT arms run under IDENTICAL thresholds and real small
# companions of multi-hit bunches survive:
#   no_trigger_amp 15→5 mV        (admit ~0.3 PE events at a 15 mV SPE)
#   afterpulse_height_ratio 5→20  (stop deleting genuine 1:5 companions;
#                                  still kills true afterpulses)
#   min_charge_purity 0.5→0.25    (companions straddling deconv undershoot)
# Everything else stays wavelet-strict (incl. the baseline gate — identical
# for both denoisers = fair, even where lossy).
BENCHMARK_HITS_PROFILE = VetoProfile(
    name="benchmark_hits",
    flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
    max_baseline_adc=500.0, peak_height_adc=50.0,
    no_trigger_amp=5.0, raw_snr_mult=10.0,
    t_peak_min_ns=20.0, t_peak_max_ns=1400.0,
    afterpulse_window_ns=40.0, afterpulse_height_ratio=20.0,
    min_charge_purity=0.25,
    version="1",
)

# Relaxed STFT-hits thresholds (the former silent override table).
STFT_HITS_PROFILE = VetoProfile(
    name="stft_hits",
    flat_frac_min=0.50, flat_k=10.0, spike_k=20.0,
    max_baseline_adc=900.0, peak_height_adc=10.0,
    no_trigger_amp=1.0, raw_snr_mult=1.0,
    t_peak_min_ns=1.0, t_peak_max_ns=1400.0,
    afterpulse_window_ns=40.0, afterpulse_height_ratio=100.0,
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


VETO_PROFILES = VetoProfileCatalog([WAVELET_HITS_PROFILE, STFT_HITS_PROFILE,
                                    BENCHMARK_HITS_PROFILE])


# ─── S8: file-backed, versioned veto-profile catalog ────────────────────
# A YAML profile under <data_root>/calibrations/veto_profiles/<name>.yaml
# SHADOWS the in-code catalog (same shadowing rule as dataset cards), so
# `vetoes calibrate --write-profile` persists a derived profile that later runs
# resolve by name — the "persistent backing store" promised above.
from pathlib import Path as _Path  # noqa: E402


def veto_profiles_dir(data_root) -> _Path:
    """``<data_root>/calibrations/veto_profiles`` — the on-disk veto catalog."""
    return _Path(data_root) / "calibrations" / "veto_profiles"


# Removed 2026-10-23 with the post_sat_shift veto; profiles saved before then
# still carry them, and are read without them.
RETIRED_FIELDS = ("post_sat_window_ns", "post_sat_sigma_mult", "post_sat_offset_ns")


def load_veto_profile_yaml(path) -> VetoProfile:
    """Load one YAML veto profile into a :class:`VetoProfile` (strict: unknown
    keys are an error, so a typo'd threshold never silently does nothing)."""
    import yaml
    raw = yaml.safe_load(_Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"veto profile {path} must be a YAML mapping")
    for retired in RETIRED_FIELDS:
        raw.pop(retired, None)
    known = set(VetoProfile.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"veto profile {path}: unknown field(s) {sorted(unknown)}")
    return VetoProfile(**raw)


def save_veto_profile_yaml(profile: VetoProfile, data_root) -> _Path:
    """Persist a profile to the on-disk catalog; returns the written path."""
    import yaml
    out_dir = veto_profiles_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{profile.name}.yaml"
    out.write_text(yaml.safe_dump(asdict(profile), sort_keys=False))
    return out


def discover_veto_profiles(data_root) -> Dict[str, VetoProfile]:
    """Every file-backed profile under the data root's veto catalog dir."""
    d = veto_profiles_dir(data_root)
    found: Dict[str, VetoProfile] = {}
    if d.is_dir():
        for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            prof = load_veto_profile_yaml(p)
            found[prof.name] = prof
    return found


def resolve_veto_profile(name: str, data_root=None):
    """Resolve a veto-profile NAME to ``(VetoProfile, source)``. A file under
    ``<data_root>/calibrations/veto_profiles/`` SHADOWS the in-code catalog;
    ``source`` is ``"file"`` or ``"builtin"``."""
    if data_root is not None:
        file_profs = discover_veto_profiles(data_root)
        if name in file_profs:
            return file_profs[name], "file"
    prof = VETO_PROFILES.lookup(name=name)
    if prof is not None:
        return prof, "builtin"
    raise KeyError(f"unknown veto profile {name!r}")
