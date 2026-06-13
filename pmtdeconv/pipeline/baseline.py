"""Baseline estimation — wraps utils.baseline_noise_bestwindow.

In ``--no-vetoes`` mode, a RuntimeError from the upstream algorithm
falls back to the first N pre-trigger samples instead of rejecting the
event. This matches the explicit ``baseline_failed_flag`` path added to
``wavelets_cosine_deconvolution_Na22.py`` during the 2026-05 session.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

# utils.py lives at the project root one level above the pmtdeconv pkg
from pmtdeconv.utils import baseline_noise_bestwindow            # noqa: E402

DEFAULT_FALLBACK_SAMPLES = 100


class BaselineResult(NamedTuple):
    ped: float
    noise: float
    b0_ns: float
    b1_ns: float
    wbins: int
    passed: bool
    failed: bool      # True if the upstream algorithm raised RuntimeError


def estimate(
    t: np.ndarray,
    wave: np.ndarray,
    dt_ns: float,
    *,
    no_vetoes: bool = False,
    search_end_ns: float = 600.0,
    fallback_samples: int = DEFAULT_FALLBACK_SAMPLES,
    # The remaining knobs mirror the upstream defaults; expose them only
    # for future per-detector tuning.
    window_ns: float = 200.0,
    step_ns: float = 25.0,
    min_sigma: float = 1e-1,
    flat_frac_min: float = 0.95,
    flat_k: float = 4.0,
    spike_k: float = 5.0,
    max_baseline_adc: float = 500.0,
    peak_height_adc: float = 50.0,
) -> BaselineResult:
    """Adaptive baseline + noise estimate with a no-vetoes safety net.

    Returns a NamedTuple with ``ped``, ``noise``, the chosen window
    ``[b0_ns, b1_ns]`` (sample bins ``wbins``), the upstream
    ``passed`` flag, and a ``failed`` flag set when the upstream
    algorithm raised RuntimeError. In ``no_vetoes`` mode the latter two
    still appear in the result so the caller can record them but the
    event is not dropped.
    """
    try:
        ped, noise, (b0, b1), wbins, passed = baseline_noise_bestwindow(
            t, wave, dt_ns,
            window_ns=window_ns, step_ns=step_ns, min_sigma=min_sigma,
            flat_frac_min=flat_frac_min, flat_k=flat_k, spike_k=spike_k,
            search_end_ns=search_end_ns,
            max_baseline_adc=max_baseline_adc,
            peak_height_adc=peak_height_adc,
        )
        return BaselineResult(float(ped), float(noise),
                              float(b0), float(b1), int(wbins),
                              bool(passed), False)
    except RuntimeError:
        if not no_vetoes:
            raise
        fb = wave[:fallback_samples]
        return BaselineResult(
            ped=float(np.median(fb)),
            noise=float(np.std(fb)) or 1.0,
            b0_ns=0.0,
            b1_ns=float(fallback_samples * dt_ns),
            wbins=int(fallback_samples),
            passed=False,
            failed=True,
        )
