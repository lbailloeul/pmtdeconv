#!/usr/bin/env python3
"""Build a deterministic synthetic Na22-shaped ROOT file for tests.

Two modes:

1. **Unflagged (default)** — produces the legacy single-regime fixture at
   ``tests/fixtures/flattened_Run0.root``. The byte sequence here is
   pinned by the three flat-format golden CSVs in
   ``tests/fixtures/golden/``. Do not modify the unflagged code path
   without also regenerating those goldens.

2. **--regime level<N>** — produces a regime-tagged fixture under
   ``tests/fixtures/regime/level<N>/flattened_Run0.root`` plus a
   ``flattened_Run0.truth.csv`` describing the synthetic pulses. Used
   by ``pmtdeconv tune {cosine,wavelet,stft}`` to score sweeps against
   ground truth.

To re-generate the golden CSVs after a deliberate schema change:

    python tests/fixtures/make_synth_root.py
    python -m pmtdeconv waveform --data-root tests/fixtures \\
        --dataset na22 --runs 0 \\
        --out-dir tests/fixtures/golden/deconv_area
    cp tests/fixtures/golden/deconv_area/areas_run0.csv \\
       tests/fixtures/golden/deconv_area_run0.csv
    # (similarly for raw_area mqana, pulse)

The RNG seed is fixed (42) so the file bytes are reproducible.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uproot

from regimes import (BOX_WIDTH_SAMP, PEDESTAL_MV, SIGMA_SAMP, get_regime,
                     list_regimes, make_waveforms)


N_EVENTS = 100
N_SAMPLES = 1024
DT_NS = 0.5051
SEED = 42


# ─── legacy unflagged path (byte-identical to PR 9) ──────────────────
def _make_dataset(rng: np.random.Generator, n_evt: int, n_samp: int):
    """Return (wave_ch1, wave_ch2) — both shape (n_evt, n_samp), float.

    Two-channel Na22-ish synthetic pulses: amplitude ~ uniform, peak
    centered in the middle third of the buffer, ~25 ns FWHM. This is
    the legacy fixture path; bytes are pinned by the three flat
    goldens. **Do not change** without regenerating those.
    """
    samp = np.arange(n_samp)
    amps_ch1 = rng.uniform(60.0, 200.0, n_evt)
    amps_ch2 = amps_ch1 * rng.uniform(0.4, 0.8, n_evt)
    centers_ch1 = rng.uniform(350.0, 600.0, n_evt)
    centers_ch2 = centers_ch1 + rng.normal(2.0, 1.5, n_evt)
    sigma_samp = 25.0
    waves = []
    for amps, centers in [(amps_ch1, centers_ch1), (amps_ch2, centers_ch2)]:
        w = np.zeros((n_evt, n_samp), dtype=np.float64)
        for i in range(n_evt):
            w[i] = amps[i] * np.exp(
                -0.5 * ((samp - centers[i]) / sigma_samp) ** 2)
        w += rng.normal(0.0, 1.2, (n_evt, n_samp))
        w += 5.0
        waves.append(w.astype(np.float32))
    return waves[0], waves[1]


# ─── truth-CSV writer (regime path only) ─────────────────────────────
def _build_truth_rows(amps_ch1, amps_ch2, centers_ch1, centers_ch2,
                      charges_ch1, charges_ch2,
                      dt_ns: float, clip_mv: Optional[float]):
    """Return one truth row per (event, branch).

    With the box⊛SPE model the charge is the integral of the clean
    convolved signal (computed clip-aware in ``regimes.make_waveforms``) —
    that's what the deconvolution recovers. ``true_center_ns`` is the
    PE-box centre (arrival time); ``true_sigma_ns`` carries the box width
    (PE-burst duration) in ns.
    """
    box_width_ns = BOX_WIDTH_SAMP * dt_ns
    rows = []
    for event_i, (a1, a2, c1, c2, q1, q2) in enumerate(
            zip(amps_ch1, amps_ch2, centers_ch1, centers_ch2,
                charges_ch1, charges_ch2)):
        for branch, amp, center, charge in [("d0_ch1", a1, c1, q1),
                                            ("d0_ch2", a2, c2, q2)]:
            saturated = bool(clip_mv is not None and amp > clip_mv)
            rows.append({
                "event": event_i,
                "branch": branch,
                "true_amp_mv": float(amp),
                "true_center_ns": float(center * dt_ns),
                "true_charge_pVs": float(charge),
                "true_sigma_ns": float(box_width_ns),
                "saturated": saturated,
            })
    return rows


def _write_root(out_path: Path, w1: np.ndarray, w2: np.ndarray) -> None:
    """Write a flat ``waveforms`` tree compatible with the na22 dataset
    loader. Suppresses uproot's RNTuple FutureWarning so we explicitly
    keep the TTree layout the goldens were generated against.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with uproot.recreate(out_path) as f:
            f["waveforms"] = {"d0_ch1": w1, "d0_ch2": w2}


# ─── entry point ─────────────────────────────────────────────────────
def main(regime: Optional[str] = None,
         n_events: int = N_EVENTS,
         write_truth: bool = True,
         dt_ns: float = DT_NS) -> Path:
    """Build the fixture. Returns the ROOT path.

    Args:
        regime: None for the legacy single-regime path (byte-identical
            to PR 9). Pass ``level1``..``level5`` to produce a regime-
            tagged fixture under ``tests/fixtures/regime/<regime>/``.
        n_events: Override the event count. Only respected when
            ``regime`` is set (the legacy path is locked to N_EVENTS=100
            to preserve byte-identity).
        write_truth: When ``regime`` is set, write a sibling truth CSV.
        dt_ns: Sample spacing for the regime path (Q6). Defaults to
            DT_NS (0.5051 = na22). Cd109 uses 1.432; LANL uses 1.25.
            Truth CSV's ``true_center_ns`` and ``true_sigma_ns`` scale
            with this. Ignored on the legacy path (byte-id locked).
    """
    fixtures_dir = Path(__file__).resolve().parent

    if regime is None:
        # Legacy path. Name matches the na22 dataset's file glob.
        out_path = fixtures_dir / "flattened_Run0.root"
        rng = np.random.default_rng(SEED)
        w1, w2 = _make_dataset(rng, N_EVENTS, N_SAMPLES)
        _write_root(out_path, w1, w2)
        print(f"wrote {out_path}  ({N_EVENTS} events, "
              f"{N_SAMPLES} samples, seed={SEED})")
        print(f"  dt_ns = {DT_NS}  (legacy single-regime, σ_samp=25)")
        return out_path

    reg = get_regime(regime)
    out_dir = fixtures_dir / "regime" / regime
    out_path = out_dir / "flattened_Run0.root"
    rng = np.random.default_rng(SEED)
    (w1, w2, amps_ch1, amps_ch2, centers_ch1, centers_ch2,
     charges_ch1, charges_ch2) = make_waveforms(
        rng, n_events, N_SAMPLES, dt_ns, reg)
    _write_root(out_path, w1, w2)
    print(f"wrote {out_path}  ({n_events} events, "
          f"{N_SAMPLES} samples, seed={SEED}, regime={regime})")
    print(f"  dt_ns={dt_ns}  amp range: U({reg.amp_ch1_lo}, "
          f"{reg.amp_ch1_hi}) mV  "
          f"noise σ={reg.noise_sigma_mv} mV  "
          f"clip={reg.clip_mv or 'none'}")

    if write_truth:
        truth_path = out_dir / "flattened_Run0.truth.csv"
        rows = _build_truth_rows(amps_ch1, amps_ch2,
                                 centers_ch1, centers_ch2,
                                 charges_ch1, charges_ch2,
                                 dt_ns, reg.clip_mv)
        pd.DataFrame(rows).to_csv(truth_path, index=False)
        # Record dt so callers can detect "rebuild needed if dt changed".
        (out_dir / ".dt_ns").write_text(f"{dt_ns}\n")
        n_sat = sum(1 for r in rows if r["saturated"])
        print(f"  truth CSV: {truth_path}  "
              f"({len(rows)} rows, {n_sat} saturated)")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regime", choices=list_regimes(), default=None,
                        help="Build a regime-tagged fixture under "
                             "regime/<regime>/. Default (unset) writes the "
                             "byte-identical legacy fixture.")
    parser.add_argument("--all-regimes", action="store_true",
                        help="Build ALL 5 regime fixtures sequentially. "
                             "Ignores --regime.")
    parser.add_argument("--n-events", type=int, default=N_EVENTS,
                        help="Override event count (regime path only; the "
                             "legacy path is locked to preserve byte-id).")
    parser.add_argument("--no-truth", action="store_true",
                        help="Skip writing the sibling truth CSV.")
    parser.add_argument("--dt-ns", type=float, default=DT_NS,
                        help=f"Sample spacing in ns for the regime path "
                             f"(default {DT_NS} = na22). Ignored on the "
                             f"legacy --regime-unset path.")
    ns = parser.parse_args()

    if ns.all_regimes:
        for r in list_regimes():
            main(regime=r, n_events=ns.n_events,
                 write_truth=not ns.no_truth, dt_ns=ns.dt_ns)
    else:
        main(regime=ns.regime,
             n_events=ns.n_events if ns.regime else N_EVENTS,
             write_truth=not ns.no_truth,
             dt_ns=ns.dt_ns)
