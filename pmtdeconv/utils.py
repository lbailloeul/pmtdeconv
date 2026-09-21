#!/usr/bin/env python3
"""Compatibility shim — re-exports :mod:`pmtdeconv.core` under the old name.

The implementations were split by domain into ``pmtdeconv/core/`` (2026-05-31),
which is the single source of truth:

  - :mod:`pmtdeconv.core.kernels`  — SPE template models + parameters
  - :mod:`pmtdeconv.core.filters`  — frequency-domain low-pass filters
  - :mod:`pmtdeconv.core.dsp`      — FFT deconvolution, STFT/wavelet denoise, baseline
  - :mod:`pmtdeconv.core.peaks`    — integration windows, FWHM, per-peak vetoes

**Nothing inside ``pmtdeconv`` imports this module any more** (migrated
2026-09-04). It exists so the local, gitignored ``legacy/`` scripts — which
predate the split and are not part of the shipped surface — keep working
unchanged. Import from ``pmtdeconv.core.*`` in all new code; this shim may be
removed once ``legacy/`` is retired.
"""
# Re-exports are explicit (not ``import *``) so the underscore trough-walk
# helpers and the module constants are guaranteed to come across.
from pmtdeconv.core.kernels import (
    DTYPE,
    DT_NS,
    LANL_TEMPLATE,
    MILLIQAN_TEMPLATE,
    gaussian_smeared_biexponential,
    build_spe_template,
    spe_overshoot,
)
from pmtdeconv.core.filters import (
    gaussian_lowpass_filter,
    cosine_lowpass_filter,
)
from pmtdeconv.core.dsp import (
    deconvolve_hits,
    stft_denoise,
    baseline_noise_bestwindow,
    sliding_local_mean,
    wavelet_denoise,
)
from pmtdeconv.core.peaks import (
    _walk_to_trough_right,
    _walk_to_trough_left,
    find_integration_windows,
    cfd50_time,
    fwhm_ns,
    apply_peak_vetoes,
)

__all__ = [
    "DTYPE",
    "DT_NS",
    "LANL_TEMPLATE",
    "MILLIQAN_TEMPLATE",
    "gaussian_smeared_biexponential",
    "build_spe_template",
    "spe_overshoot",
    "gaussian_lowpass_filter",
    "cosine_lowpass_filter",
    "deconvolve_hits",
    "stft_denoise",
    "baseline_noise_bestwindow",
    "sliding_local_mean",
    "wavelet_denoise",
    "_walk_to_trough_right",
    "_walk_to_trough_left",
    "find_integration_windows",
    "cfd50_time",
    "fwhm_ns",
    "apply_peak_vetoes",
]
