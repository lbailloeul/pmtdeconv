#!/usr/bin/env python3
"""Frequency-domain low-pass filters (regularize the FFT deconvolution).

Moved verbatim from ``pmtdeconv/utils.py`` (2026-05-31 domain split).
"""
import numpy as np
from numpy import fft


def gaussian_lowpass_filter(N, dt_ns, flat_mhz=15.0, sigma_mhz=10.0, kill_mhz=50.0):
    """Gaussian-rolloff low-pass filter in frequency domain.

    Not used by the production pipeline (cosine filter only). Kept: this
    flat-top + Gaussian-rolloff shape is the "custom-defined low-pass filter"
    of Tang et al. 2411.07620 (figs. 3/7c) and Daya Bay 1707.03699, so the
    denoiser benchmark can trial it as a filter arm.
    """
    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)
    freqs_mhz = np.abs(freqs_hz) / 1e6
    filt = np.ones(N, dtype=np.float64)
    rolloff = (freqs_mhz >= flat_mhz) & (freqs_mhz <= kill_mhz)
    filt[rolloff] = np.exp(-0.5 * ((freqs_mhz[rolloff] - flat_mhz) / sigma_mhz) ** 2)
    filt[freqs_mhz > kill_mhz] = 0.0
    return filt


def cosine_lowpass_filter(N, dt_ns, cutoff_mhz=50.0):
    """Cosine low-pass: gain goes from 1 at DC to 0 at cutoff_mhz.

    The response has a discontinuous first derivative at ``cutoff_mhz``
    (cos(πf/2f_c) arrives with slope −π/2f_c, then hard 0 above), which rings
    in the time domain — the documented ~7% pre-pulse side-lobe that
    ``FIRST_PULSE_MIN_FRAC`` / ``--peak-order first`` exist to dodge. A smooth
    taper (raised-cosine/Tukey) is a tracked follow-up; changing the shape
    would change every output byte and invalidate every tuned cutoff.
    """
    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)
    freqs_mhz = np.abs(freqs_hz) / 1e6
    filt = np.zeros(N, dtype=np.float64)
    mask = freqs_mhz <= cutoff_mhz
    filt[mask] = np.cos(np.pi * freqs_mhz[mask] / (2 * cutoff_mhz))
    return filt
