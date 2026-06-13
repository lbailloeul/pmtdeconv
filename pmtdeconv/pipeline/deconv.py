"""FFT deconvolution helpers — wraps utils.deconvolve_hits.

Provides a small kernel-cache so a long run over thousands of waveforms
doesn't rebuild ``FFT(SPE) / cosine_filter`` per event when N+dt+kernel
are constant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pmtdeconv.utils import (                                      # noqa: E402
    build_spe_template,
    cosine_lowpass_filter,
    deconvolve_hits,
)


def build_kernel(template_params: Dict[str, float],
                 N: int, dt_ns: float) -> np.ndarray:
    """Build a length-N, unit-area SPE template. Thin wrapper."""
    return build_spe_template(N, dt_ns, template_params)


def build_filter(N: int, dt_ns: float, cutoff_mhz: float) -> np.ndarray:
    """Build the cosine low-pass FFT filter. Thin wrapper."""
    return cosine_lowpass_filter(N, dt_ns, cutoff_mhz=cutoff_mhz)


def deconvolve(m_t: np.ndarray, kernel: np.ndarray,
               filt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-event deconvolution: IFFT(FFT(m) * filt / FFT(kernel))."""
    return deconvolve_hits(m_t, kernel, filt, eps=eps)
