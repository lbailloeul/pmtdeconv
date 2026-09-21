"""MQAna-style narrow-window raw area (NEW in pmtdeconv).

The legacy ``deconvolution/raw_area_Na22.py`` integrates a wide fixed
window (samples 50-950 ≈ 25-475 ns), which does NOT match David's
MQAna definition. The ELOG E41473 reference and the ``reference_ucsb_elog``
memory both describe the MQAna area as a *narrow* window centered on
the per-event sample-max (typical width ~35 ns total, i.e. ±~17.5 ns
about the peak).

This module implements the narrow-window definition. Units are mV·ns.
Sign convention: the baseline-subtracted waveform is multiplied by
``polarity`` (±1) so pulses are positive-going, then
``area = Σ(polarity·(s_i - baseline))·dt`` — real signals give positive
area for either raw polarity.

Reverting to the legacy wide-window behavior is exposed via
``window_ns=None`` (the caller can also pass ``signal_start``/``signal_end``
sample indices directly).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def baseline_subtract(
    wave2d: np.ndarray,
    n_pre: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a per-event baseline (mean of the first ``n_pre`` samples).

    Returns ``(signal, baseline)`` where ``signal`` is the same shape as
    ``wave2d`` and ``baseline`` is shape (N_events, 1).
    """
    bl = wave2d[:, :n_pre].mean(axis=1, keepdims=True)
    return wave2d - bl, bl


def mqana_area(
    wave2d: np.ndarray,
    dt_ns: float,
    *,
    n_pre: int = 50,
    window_ns: float = 35.0,
    polarity: int = +1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MQAna-style narrow-window raw area for every event in a batch.

    Parameters
    ----------
    wave2d : (N_events, N_samples) float ndarray
        Raw waveforms (mV).
    dt_ns : float
        Sample spacing (ns).
    n_pre : int
        Number of pre-trigger samples for baseline estimation.
    window_ns : float
        Total integration window centered on per-event sample-max.
        E.g. 35.0 → ±17.5 ns around the peak.
    polarity : int
        +1 for positive-going pulses (Na22, Cd109, MilliQan),
        -1 for negative-going (LANL).

    Returns
    -------
    area : (N_events,) ndarray
        Integrated area in mV·ns (positive for real signals).
    peak_height : (N_events,) ndarray
        Per-event peak amplitude above baseline (mV, positive).
    """
    if polarity not in (+1, -1):
        raise ValueError(f"polarity must be ±1, got {polarity}")
    signal, _ = baseline_subtract(wave2d, n_pre=n_pre)
    # Flip polarity so peaks are positive (LANL is negative-going)
    signal = polarity * signal

    N_evt, N_samp = signal.shape
    half_bins = max(int(round(0.5 * window_ns / dt_ns)), 1)

    # Per-event sample-max (the "peak" in MQAna's sense)
    peak_idx = np.argmax(signal, axis=1)
    peak_height = signal[np.arange(N_evt), peak_idx]

    # Build per-event integration windows, clipped to the buffer.
    lo = np.clip(peak_idx - half_bins, 0, N_samp - 1)
    hi = np.clip(peak_idx + half_bins, 0, N_samp - 1) + 1  # exclusive

    # Vectorize the area integration via a per-event mask
    cols = np.arange(N_samp)[None, :]
    in_window = (cols >= lo[:, None]) & (cols < hi[:, None])
    area = (signal * in_window).sum(axis=1) * dt_ns
    return area, peak_height


def fixed_window_area(
    wave2d: np.ndarray,
    dt_ns: float,
    *,
    n_pre: int = 50,
    signal_start: int = 50,
    signal_end: int = 950,
    polarity: int = +1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wide-window raw area (``raw_area_Na22.compute_areas`` math).

    Same units / sign / polarity convention as :func:`mqana_area`.
    """
    if polarity not in (+1, -1):
        raise ValueError(f"polarity must be ±1, got {polarity}")
    signal, _ = baseline_subtract(wave2d, n_pre=n_pre)
    signal = polarity * signal
    area = signal[:, signal_start:signal_end].sum(axis=1) * dt_ns
    # peak in the fixed window
    peak_height = signal[:, signal_start:signal_end].max(axis=1)
    return area, peak_height
