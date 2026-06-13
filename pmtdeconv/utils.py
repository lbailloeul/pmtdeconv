#!/usr/bin/env python3
"""
Shared utilities for deconvolution and fitting pipelines.

**Facade (2026-05-31).** The implementations were split by domain into
``pmtdeconv/core/`` — see :mod:`pmtdeconv.core`:

  - :mod:`pmtdeconv.core.kernels`  — SPE template models + parameters
  - :mod:`pmtdeconv.core.filters`  — frequency-domain low-pass filters
  - :mod:`pmtdeconv.core.dsp`      — FFT deconvolution, STFT/wavelet denoise, baseline
  - :mod:`pmtdeconv.core.peaks`    — integration windows, FWHM, per-peak vetoes

This module re-exports every name (public functions, the ``_walk_to_trough_*``
helpers, and the ``DTYPE`` / ``DT_NS`` / template-parameter constants) so the
package modules and the ~35 legacy scripts keep importing from
``pmtdeconv.utils`` unchanged. Behaviour is byte-for-byte identical — the
golden fixtures are the guarantee. Prefer importing from the ``core`` submodule
in new code.
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
    "fwhm_ns",
    "apply_peak_vetoes",
]
