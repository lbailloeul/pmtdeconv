#!/usr/bin/env python3
"""`plot energy` denoiser-consistency (architecture restructure, step 6).

Before step 6, ``plotting.energy`` denoised with hardcoded module constants
(STFT 24/80/3, wavelet sym8/5/hard/1) that diverged from the hits pipeline —
so the plot's "wavelet"/"STFT" curves were NOT the same transform as
``hits --denoiser <x>``. These guard the fix: ``_process_channel`` now routes
through ``pipeline.denoise.denoise`` with the threaded config, so the curves are
provably identical, and the knobs actually change the output.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from pmtdeconv.plotting.energy import _process_channel, _build_kernel
from pmtdeconv.pipeline.denoise import denoise
from pmtdeconv.utils import deconvolve_hits, cosine_lowpass_filter

N = 1024
DT = 0.5051


def _synthetic_channel(n_evt=5):
    rng = np.random.default_rng(1)
    v = rng.normal(500.0, 2.0, (n_evt, N))
    for k in range(n_evt):
        v[k, 300:312] += 80.0
    base = np.full(n_evt, 500.0)
    d = {"mask": np.ones(n_evt, bool), "base": base, "v": v}
    row = pd.Series({"tau_rise_ns": 5.0, "tau_fall_ns": 20.0, "sigma_ns": 2.0})
    kern = _build_kernel(row, N, DT)
    filt = cosine_lowpass_filter(N, DT, cutoff_mhz=100.0)
    return d, kern, filt


_DN = {"wavelet": "sym8", "wlevel": 5, "wmode": "hard", "wthresh_k": 1.0,
       "stft_nperseg_ns": 20.0, "stft_freq_cutoff_mhz": 175.0,
       "stft_amp_threshold": 10.0}


def test_wavelet_curve_is_the_denoise_wrapper():
    d, kern, filt = _synthetic_channel()
    c_lo, c_hi = int(20 / DT), N - int(20 / DT)
    idx, whole, qual, _ = _process_channel(d, kern, filt, c_lo, c_hi,
                                           int(50 / DT), DT, N, _DN)
    # independently recompute event 0's wavelet whole-charge via the wrapper
    y0 = d["v"][0].astype(np.float64) - d["base"][0]
    u_fft0 = deconvolve_hits(y0, kern, filt)
    u_wav0 = denoise(u_fft0, denoiser="wavelet", wavelet="sym8", wlevel=5,
                     wmode="hard", wthresh_k=1.0)
    assert whole["wavelet"][0] == float(np.sum(u_wav0[c_lo:c_hi]))


def test_stft_curve_uses_threaded_knobs():
    d, kern, filt = _synthetic_channel()
    c_lo, c_hi = int(20 / DT), N - int(20 / DT)
    _, whole, _, _ = _process_channel(d, kern, filt, c_lo, c_hi,
                                      int(50 / DT), DT, N, _DN)
    y0 = d["v"][0].astype(np.float64) - d["base"][0]
    u_fft0 = deconvolve_hits(y0, kern, filt)
    u_stft0 = denoise(u_fft0, denoiser="stft", dt_ns=DT, stft_nperseg_ns=20.0,
                      stft_freq_cutoff_hz=175.0 * 1e6, stft_amp_threshold=10.0)
    assert whole["stft"][0] == float(np.sum(u_stft0[c_lo:c_hi]))


def test_threaded_stft_differs_from_old_hardcoded():
    # The pre-step-6 module hardcoded STFT 24/80/3; the threaded config is the
    # hits default 20/175/10. They must produce a different curve — that is the
    # bug (the plot's STFT no longer secretly diverges from `hits --denoiser stft`).
    d, kern, filt = _synthetic_channel()
    c_lo, c_hi = int(20 / DT), N - int(20 / DT)
    _, whole_new, _, _ = _process_channel(d, kern, filt, c_lo, c_hi, int(50 / DT),
                                          DT, N, _DN)
    dn_old = {**_DN, "stft_nperseg_ns": 24.0, "stft_freq_cutoff_mhz": 80.0,
              "stft_amp_threshold": 3.0}
    _, whole_old, _, _ = _process_channel(d, kern, filt, c_lo, c_hi, int(50 / DT),
                                          DT, N, dn_old)
    assert not np.allclose(whole_new["stft"], whole_old["stft"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
