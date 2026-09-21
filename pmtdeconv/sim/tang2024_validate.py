"""Validate the Tang-2024 MC reproduction against the paper's numbers.

The reproduction is accepted not by matching the (unrecoverable) exact
equation forms but by reproducing the PUBLISHED results
(:data:`pmtdeconv.sim.tang2024.TANG_PUBLISHED`):

- **fig 14** — per-waveform reconstructed/true charge ratio: mean ≈ 1.004,
  σ ≈ 0.076 (FFT+STFT) / 0.080 (FFT), residual nonlinearity ≈ 1% over
  0–20 PE  → :func:`per_event_charge_ratio`, :func:`nonlinearity_profile`.
- **fig 12** — two-hit recognition efficiency vs separation: exactly-2 peaks
  found AND both within ±2 ns of truth; FFT+STFT recovers 3–6 ns separations
  plain FFT misses (and is ~3% worse at 7–10 ns)
  → :func:`recognition_efficiency`.

The paper's STFT threshold (0.02 A.U. below 50 MHz) is in THEIR amplitude
normalization; :func:`paper_stft_fixed_threshold` transports it into scipy's
STFT units by measuring the T-F amplitude of a clean deconvolved 1-PE spike
and taking the same relative level (0.02 of their ~0.4 1-PE spike ⇒ 5%).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pmtdeconv.sim.tang2024 import (DT_NS, N_SAMPLES, TRIGGER_NS,
                                    central_kernel)

# paper: threshold 0.02 on a ~0.4-amplitude 1-PE spike (fig 10) ⇒ 5% relative
PAPER_THRESHOLD_REL = 0.05
PAPER_STFT_BAND_MHZ = 50.0
PAPER_CUTOFF_MHZ = 450.0


def paper_stft_fixed_threshold(dt_ns: float = DT_NS,
                             cutoff_mhz: float = PAPER_CUTOFF_MHZ,
                             nperseg_ns: float = 20.0) -> float:
    """The paper's 0.02-A.U. STFT threshold transported to scipy units.

    Deconvolves a clean (noise-free) central-kernel 1-PE waveform with the
    450 MHz cosine filter, takes the max |STFT| in the <50 MHz band of the
    resulting spike, and returns ``PAPER_THRESHOLD_REL`` × that. Deterministic
    (no RNG) — the absolute threshold then means the same thing the paper's
    0.02 meant relative to their 1-PE spike.
    """
    from scipy.signal import stft
    from pmtdeconv.core.dsp import deconvolve_hits
    from pmtdeconv.core.filters import cosine_lowpass_filter
    from pmtdeconv.core.kernels import build_spe_template

    kern = central_kernel(dt_ns)
    wave = np.zeros(N_SAMPLES)
    i0 = int(TRIGGER_NS / dt_ns)
    n_add = min(len(kern), N_SAMPLES - i0)
    wave[i0:i0 + n_add] = kern[:n_add]

    n_fft = 1
    while n_fft < 2 * N_SAMPLES:
        n_fft <<= 1
    tmpl = dict(kind="sampled", t_ns=np.arange(len(kern)) * dt_ns, y=kern)
    r = build_spe_template(n_fft, dt_ns, tmpl)
    filt = cosine_lowpass_filter(n_fft, dt_ns, cutoff_mhz)
    y = np.zeros(n_fft)
    y[:N_SAMPLES] = wave
    u = deconvolve_hits(y, r, filt)[:N_SAMPLES]

    fs = 1.0 / (dt_ns * 1e-9)
    nperseg = int(nperseg_ns / dt_ns)
    f, _, Z = stft(u, fs=fs, window=("gaussian", nperseg / 6),
                   nperseg=nperseg, noverlap=nperseg - 1)
    band = np.abs(Z[f <= PAPER_STFT_BAND_MHZ * 1e6, :])
    return float(PAPER_THRESHOLD_REL * band.max())


def per_event_charge_ratio(hits_csv: Path | str, truth_csv: Path | str
                           ) -> Dict[str, float]:
    """fig-14 observable: per-waveform Σ(successful rec charge)/Σ(true)."""
    rec = pd.read_csv(hits_csv)
    if "success" in rec.columns:
        rec = rec[rec["success"] == True]              # noqa: E712
    truth = pd.read_csv(truth_csv)
    tot_rec = rec.groupby("entry")["charge"].sum()
    tot_true = truth.groupby("event")["true_charge_pVs"].sum()
    m = pd.concat([tot_rec, tot_true], axis=1, keys=["rec", "true"])
    m["rec"] = m["rec"].fillna(0.0)
    m = m.dropna(subset=["true"])
    m = m[m["true"] > 0]
    r = (m["rec"] / m["true"]).to_numpy()
    return {"mean": float(np.mean(r)) if r.size else float("nan"),
            "std": float(np.std(r)) if r.size else float("nan"),
            "n": int(r.size)}


def nonlinearity_profile(hits_csv: Path | str, truth_csv: Path | str,
                         n_bins: int = 20) -> pd.DataFrame:
    """fig-14(e): mean rec/true ratio binned by true total nPE (1..20).
    Nonlinearity = max |bin mean − overall mean| / overall mean."""
    rec = pd.read_csv(hits_csv)
    if "success" in rec.columns:
        rec = rec[rec["success"] == True]              # noqa: E712
    truth = pd.read_csv(truth_csv)
    tot_rec = rec.groupby("entry")["charge"].sum()
    per_ev = truth.groupby("event").agg(
        true=("true_charge_pVs", "sum"), npe=("n_pe", "sum"))
    per_ev["rec"] = tot_rec.reindex(per_ev.index).fillna(0.0)
    per_ev = per_ev[per_ev["true"] > 0]
    per_ev["ratio"] = per_ev["rec"] / per_ev["true"]
    per_ev["npe_bin"] = np.clip(np.round(per_ev["npe"]), 1, n_bins)
    prof = per_ev.groupby("npe_bin").agg(
        ratio_mean=("ratio", "mean"), ratio_std=("ratio", "std"),
        n=("ratio", "size")).reset_index()
    return prof


def nonlinearity_frac(profile: pd.DataFrame, min_bin_n: int = 20) -> float:
    """Max fractional deviation of the per-nPE-bin mean ratio from the
    overall (n-weighted) mean, over bins with ≥ ``min_bin_n`` entries."""
    p = profile[profile["n"] >= min_bin_n]
    if not len(p):
        return float("nan")
    overall = float(np.average(p["ratio_mean"], weights=p["n"]))
    if overall == 0:
        return float("nan")
    return float(np.max(np.abs(p["ratio_mean"] - overall)) / abs(overall))


def recognition_efficiency(hits_csv: Path | str, truth_csv: Path | str,
                           tol_ns: float = 2.0) -> float:
    """fig-12 criterion on a two-hit scan run: fraction of events with
    EXACTLY two found peaks, both matching truth within ±``tol_ns``."""
    from pmtdeconv.pipeline.scoring import match_pulses
    rec = pd.read_csv(hits_csv)
    if "success" in rec.columns:
        rec = rec[rec["success"] == True]              # noqa: E712
    truth = pd.read_csv(truth_csv)
    n_events = truth["event"].nunique()
    n_ok = 0
    rec_by = dict(tuple(rec.groupby("entry")))
    for ev, tg in truth.groupby("event"):
        fg = rec_by.get(ev)
        ftimes = fg["t_peak"].values if fg is not None else np.array([])
        m = match_pulses(tg["true_center_ns"].values, ftimes,
                         tol_ns=tol_ns, exactly_n=2)
        n_ok += int(m["recognized"])
    return n_ok / n_events if n_events else float("nan")
