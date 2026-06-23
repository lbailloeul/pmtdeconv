"""Regime registry for parameterized synthetic fixtures (PR 10).

A *regime* is a named (amp range, noise σ, clip threshold) tuple that
``make_synth_root.py --regime <name>`` consumes to build a fixture
matching that detector regime. Five regimes are defined to span the
useful range for denoiser/cosine-filter tuning:

- ``level1``: low-SNR / Compton continuum
- ``level2``: clean small pulses
- ``level3``: current default (Na22 511 keV-ish; unflagged fixture)
- ``level4``: bright, near rail
- ``level5``: saturated (clipped at 950 mV)

`apply_regime(rng, n_evt, regime)` returns the synthetic pulse shape
parameters and the per-event channel waveforms. Used by both the
fixture builder and the truth-CSV generator.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow `pmtdeconv` to import when this fixture builder is run standalone
# from tests/fixtures (the package isn't pip-installed in lanl-env; it's
# a local package at the repo root, three levels up from this file).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pmtdeconv.utils import MILLIQAN_TEMPLATE, build_spe_template  # noqa: E402


def _spe_kernel(n_samp: int, dt_ns: float, branch: str) -> np.ndarray:
    """Unit-area SPE response used to BUILD the synthetic waveform.

    Matched to what the deconvolution will divide by: load the na22
    dataset's default template (the Cd109 stacked biexp) per channel so
    ``box ⊛ SPE`` deconvolves cleanly back to the box. Falls back to the
    MilliQan biexp when the template CSV isn't on disk (fresh checkout),
    leaving a small, realistic kernel mismatch.
    """
    params = {**MILLIQAN_TEMPLATE, "t0": 0.0}
    try:
        from pmtdeconv.io.templates import load_template
        from pmtdeconv.presets.datasets import get_dataset
        tmpls = load_template(get_dataset("na22").default_template_csv,
                              strict=False, verbose=False)
        if branch in tmpls:
            params = {**tmpls[branch], "t0": 0.0}
    except Exception:
        pass
    return build_spe_template(n_samp, dt_ns, params)


@dataclass(frozen=True)
class Regime:
    """One (amp range, noise, clip) point in the regime grid.

    The optional pile-up fields (``n_pulses`` > 1) turn an event into a
    burst of ``n_pulses`` overlapping ``box ⊛ SPE`` pulses separated by a
    uniform ``U(sep_lo_ns, sep_hi_ns)`` gap — the only regime with
    multi-pulse ground truth (one truth row per pulse). Single-pulse
    regimes keep ``n_pulses = 1`` and the legacy one-row-per-(event,
    branch) truth.
    """
    name: str
    amp_ch1_lo: float    # mV — uniform lower bound
    amp_ch1_hi: float    # mV — uniform upper bound
    noise_sigma_mv: float
    clip_mv: Optional[float]   # None = no clipping
    description: str
    n_pulses: int = 1          # >1 → pile-up (multi-pulse truth)
    sep_lo_ns: float = 0.0     # min inter-pulse gap (pile-up only)
    sep_hi_ns: float = 0.0     # max inter-pulse gap (pile-up only)


REGIMES: Dict[str, Regime] = {
    "level1": Regime(
        name="level1",
        amp_ch1_lo=10.0, amp_ch1_hi=30.0,
        noise_sigma_mv=1.2, clip_mv=None,
        description="low-SNR / Compton continuum",
    ),
    "level2": Regime(
        name="level2",
        amp_ch1_lo=30.0, amp_ch1_hi=80.0,
        noise_sigma_mv=1.2, clip_mv=None,
        description="clean small pulses",
    ),
    "level3": Regime(
        name="level3",
        amp_ch1_lo=80.0, amp_ch1_hi=200.0,
        noise_sigma_mv=1.2, clip_mv=None,
        description="current default — Na22 511 keV-ish",
    ),
    "level4": Regime(
        name="level4",
        amp_ch1_lo=200.0, amp_ch1_hi=500.0,
        noise_sigma_mv=1.5, clip_mv=None,
        description="bright, near rail",
    ),
    "level5": Regime(
        name="level5",
        amp_ch1_lo=500.0, amp_ch1_hi=1100.0,
        noise_sigma_mv=2.0, clip_mv=950.0,
        description="saturated (clipped at 950 mV)",
    ),
    # ── realism regimes (S0): the denoiser + pile-up stress cases the
    #    clean level1-5 grid never exercised. ──────────────────────────
    "noisy": Regime(
        name="noisy",
        amp_ch1_lo=80.0, amp_ch1_hi=200.0,
        noise_sigma_mv=12.0, clip_mv=None,
        description="Na22-amplitude pulses at 10x noise — denoiser stress "
                    "(SNR ~7-17, vs ~67-167 on level3)",
    ),
    "pileup": Regime(
        name="pileup",
        amp_ch1_lo=80.0, amp_ch1_hi=200.0,
        noise_sigma_mv=1.2, clip_mv=None,
        n_pulses=3, sep_lo_ns=5.0, sep_hi_ns=30.0,
        description="2-3 overlapping pulses/event, 5-30 ns apart — "
                    "multi-hit separation + per-pulse charge truth",
    ),
}


# Pulse shape constants held fixed across regimes — SNR + saturation
# are the variables we sweep against.
SIGMA_SAMP = 25.0          # samples — legacy Gaussian width (still used by the
                           #   byte-identical golden fixture's truth helper)
PEDESTAL_MV = 5.0          # DC offset
BOX_WIDTH_SAMP = 12        # PE-burst (scintillation) box width in samples
                           #   ≈6 ns at na22 dt — what the deconv must recover


def get_regime(name: str) -> Regime:
    if name not in REGIMES:
        raise KeyError(f"Unknown regime {name!r}. "
                       f"Known: {', '.join(REGIMES)}.")
    return REGIMES[name]


def list_regimes() -> Tuple[str, ...]:
    return tuple(REGIMES.keys())


def make_waveforms(rng: np.random.Generator,
                   n_evt: int, n_samp: int, dt_ns: float,
                   regime: Regime,
                   ) -> Tuple[np.ndarray, ...]:
    """Build (n_evt, n_samp) ch1 + ch2 waveforms for one regime.

    Physically-correct synthetic (2026-05-30): each event injects a
    **box of photoelectrons** (a short scintillation burst, width
    ``BOX_WIDTH_SAMP``) and **convolves it with the SPE kernel**, then
    adds Gaussian noise + pedestal + optional saturation clip:

        waveform = (PE box) ⊛ SPE_template + N(0, σ) + pedestal

    The SPE kernel is the one the deconvolution divides by (matched), so
    deconvolving recovers the box: its integral is the charge and its
    centre is the arrival time. This replaces the old bare-Gaussian
    injection, which was a kernel mismatch (a Gaussian is not what the
    biexp deconvolution expects, so it produced a meaningless bipolar
    artifact).

    Returns ``(w1, w2, amps_ch1, amps_ch2, centers_ch1, centers_ch2,
    charges_ch1, charges_ch2)`` — ``centers`` are the box centres (the
    timing truth, in samples) and ``charges`` the clean-signal integrals
    in pVs (the charge truth the deconv should recover).
    """
    amps_ch1 = rng.uniform(regime.amp_ch1_lo, regime.amp_ch1_hi, n_evt)
    # ch2 coincident, smaller (matches the legacy default 0.4–0.8× scaling)
    amps_ch2 = amps_ch1 * rng.uniform(0.4, 0.8, n_evt)
    centers_ch1 = rng.uniform(350.0, 600.0, n_evt)   # box CENTRE [samples]
    # ch2 leads/lags ch1 by ~2±1.5 samples (synthetic timewalk)
    centers_ch2 = centers_ch1 + rng.normal(2.0, 1.5, n_evt)

    W = BOX_WIDTH_SAMP
    out: List[np.ndarray] = []
    out_charges: List[np.ndarray] = []
    for branch, amps, centers in [("d0_ch1", amps_ch1, centers_ch1),
                                  ("d0_ch2", amps_ch2, centers_ch2)]:
        spe = _spe_kernel(n_samp, dt_ns, branch)
        w = np.zeros((n_evt, n_samp), dtype=np.float64)
        charges = np.zeros(n_evt, dtype=np.float64)
        for i in range(n_evt):
            lo = int(round(centers[i] - W / 2.0))
            box = np.zeros(n_samp, dtype=np.float64)
            box[max(lo, 0):min(lo + W, n_samp)] = 1.0
            clean = np.convolve(box, spe)[:n_samp]      # box ⊛ SPE (linear)
            pk = clean.max()
            if pk > 0:
                clean *= amps[i] / pk                    # scale to target peak mV
            clean_rec = (np.minimum(clean, regime.clip_mv)
                         if regime.clip_mv is not None else clean)
            charges[i] = float(clean_rec.sum()) * dt_ns  # ∫ clean × dt = pVs
            wave = clean + rng.normal(0.0, regime.noise_sigma_mv, n_samp) + PEDESTAL_MV
            if regime.clip_mv is not None:
                np.minimum(wave, regime.clip_mv, out=wave)
            w[i] = wave
        out.append(w.astype(np.float32))
        out_charges.append(charges)
    return (out[0], out[1], amps_ch1, amps_ch2,
            centers_ch1, centers_ch2, out_charges[0], out_charges[1])


def make_waveforms_pileup(rng: np.random.Generator,
                          n_evt: int, n_samp: int, dt_ns: float,
                          regime: Regime,
                          ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Build pile-up waveforms: ``n_pulses`` overlapping box⊛SPE pulses/event.

    Each event injects ``U(2, regime.n_pulses)`` pulses on each channel,
    consecutive pulses separated by ``U(sep_lo_ns, sep_hi_ns)``; CH2 mirrors
    CH1 with a per-pulse timewalk offset and 0.4-0.8× amplitude. Charge is
    linear in the convolution, so each pulse's truth charge is its OWN clean
    integral regardless of overlap.

    Returns ``(w1, w2, truth_rows)`` where ``truth_rows`` is a list of dicts
    (one per pulse) carrying an extra ``pulse_id`` column on top of the
    standard truth schema — the multi-pulse ground truth.
    """
    spe_by_branch = {b: _spe_kernel(n_samp, dt_ns, b)
                     for b in ("d0_ch1", "d0_ch2")}
    W = BOX_WIDTH_SAMP
    box_width_ns = W * dt_ns
    w1 = np.zeros((n_evt, n_samp), dtype=np.float64)
    w2 = np.zeros((n_evt, n_samp), dtype=np.float64)
    waves = {"d0_ch1": w1, "d0_ch2": w2}
    truth_rows: List[dict] = []

    for ev in range(n_evt):
        npul = int(rng.integers(2, regime.n_pulses + 1))  # 2..n_pulses
        c0 = rng.uniform(350.0, 450.0)
        centers1 = [c0]
        for _ in range(npul - 1):
            sep_ns = rng.uniform(regime.sep_lo_ns, regime.sep_hi_ns)
            centers1.append(centers1[-1] + sep_ns / dt_ns)
        amps1 = rng.uniform(regime.amp_ch1_lo, regime.amp_ch1_hi, npul)
        tw = rng.normal(2.0, 1.5, npul)
        centers2 = [centers1[k] + tw[k] for k in range(npul)]
        amps2 = amps1 * rng.uniform(0.4, 0.8, npul)

        for branch, centers, amps in [("d0_ch1", centers1, amps1),
                                      ("d0_ch2", centers2, amps2)]:
            spe = spe_by_branch[branch]
            total = np.zeros(n_samp, dtype=np.float64)
            for pid in range(npul):
                lo = int(round(centers[pid] - W / 2.0))
                box = np.zeros(n_samp, dtype=np.float64)
                box[max(lo, 0):min(lo + W, n_samp)] = 1.0
                clean = np.convolve(box, spe)[:n_samp]
                pk = clean.max()
                if pk > 0:
                    clean *= amps[pid] / pk
                charge = float(clean.sum()) * dt_ns
                total += clean
                truth_rows.append({
                    "event": ev, "branch": branch, "pulse_id": pid,
                    "true_amp_mv": float(amps[pid]),
                    "true_center_ns": float(centers[pid] * dt_ns),
                    "true_charge_pVs": charge,
                    "true_sigma_ns": float(box_width_ns),
                    "saturated": bool(regime.clip_mv is not None
                                      and amps[pid] > regime.clip_mv),
                })
            wave = (total + rng.normal(0.0, regime.noise_sigma_mv, n_samp)
                    + PEDESTAL_MV)
            if regime.clip_mv is not None:
                np.minimum(wave, regime.clip_mv, out=wave)
            waves[branch][ev] = wave

    return w1.astype(np.float32), w2.astype(np.float32), truth_rows
