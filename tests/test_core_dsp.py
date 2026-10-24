#!/usr/bin/env python3
"""Properties of the deconvolution core: the cosine filter and the FFT divide.

Why this module exists: mutation testing (2026-09-18) shifted the cosine
cutoff by 10% and separately removed the divide-by-zero guard from
``deconvolve_hits``. The whole suite caught both — but ONLY through the
byte-identity goldens. Goldens are re-blessed as a matter of routine, and a
re-bless bakes in whatever the code does at that moment, so a filter error
could be made permanent by a single ``--force``.

These tests pin the filter's frequency response and the guard's purpose as
PROPERTIES (gain at DC, gain at the cutoff, monotonicity, where the stop band
starts, finiteness on a null kernel) rather than as frozen numbers, so they
survive a deliberate re-bless and still fail on a real mistake.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmtdeconv.core.dsp import deconvolve_hits          # noqa: E402
from pmtdeconv.core.filters import cosine_lowpass_filter  # noqa: E402

N = 1024
DT_NS = 0.5051


def _freqs_mhz(n: int = N, dt_ns: float = DT_NS) -> np.ndarray:
    return np.abs(np.fft.fftfreq(n, d=dt_ns * 1e-9)) / 1e6


def test_filter_passes_dc_and_stops_at_the_cutoff():
    """Gain is exactly 1 at DC and 0 at (and above) the cutoff.

    A cutoff that drifts — the mutation that only the goldens caught — moves
    the frequency at which the response dies, which is what this measures.
    """
    cutoff = 100.0
    filt = cosine_lowpass_filter(N, DT_NS, cutoff)
    f = _freqs_mhz()

    assert filt[0] == pytest.approx(1.0)                  # DC
    assert np.all(filt[f > cutoff] == 0.0)                # hard stop band
    # The last passing bin sits just under the cutoff, where cos(pi/2) -> 0.
    last_pass = filt[f <= cutoff].min()
    assert 0.0 <= last_pass < 0.05


@pytest.mark.parametrize("cutoff", [50.0, 100.0, 450.0])
def test_filter_edge_tracks_the_requested_cutoff(cutoff):
    """The half-gain point must sit where the cutoff says it does.

    cos(pi f / 2 f_c) = 1/2 at f = (2/3) f_c, so the -6 dB frequency is a
    fixed fraction of the cutoff. Scaling the cutoff by 10% moves it by 10%,
    which is precisely the mutant the goldens alone used to catch.
    """
    filt = cosine_lowpass_filter(N, DT_NS, cutoff)
    f = _freqs_mhz()
    order = np.argsort(f)
    f_sorted, filt_sorted = f[order], filt[order]
    half = np.interp(0.5, filt_sorted[::-1], f_sorted[::-1])
    assert half == pytest.approx((2.0 / 3.0) * cutoff, rel=0.02)


def test_filter_is_monotonic_across_the_pass_band():
    """Gain never rises with frequency — a low-pass that is not monotonic
    would amplify a band it is meant to suppress."""
    filt = cosine_lowpass_filter(N, DT_NS, 100.0)
    f = _freqs_mhz()
    order = np.argsort(f)
    gains = filt[order]
    assert np.all(np.diff(gains) <= 1e-12)


def test_deconvolution_stays_finite_when_the_kernel_has_a_null():
    """The ``eps`` in ``M * filt / (R + eps)`` is a divide-by-zero GUARD.

    Removing it was the second goldens-only mutant. A kernel whose spectrum
    hits exactly zero (here a boxcar, which has periodic nulls) divides by
    zero without it: the result must stay finite.
    """
    kernel = np.zeros(N)
    kernel[:8] = 1.0 / 8.0                      # boxcar -> exact spectral nulls
    spectrum = np.fft.fft(kernel)
    assert np.isclose(np.abs(spectrum), 0.0, atol=1e-12).any(), \
        "fixture must actually contain a spectral null"

    signal = np.zeros(N)
    signal[100:140] = 1.0
    filt = cosine_lowpass_filter(N, DT_NS, 100.0)

    out = deconvolve_hits(signal, kernel, filt)
    assert np.all(np.isfinite(out)), "eps guard must keep the output finite"

    # And the guard must not be doing the filtering's job: with eps alone and
    # no filter the result is still finite, but the filter is what bounds it.
    assert np.all(np.isfinite(deconvolve_hits(signal, kernel, np.ones(N))))


def test_deconvolution_inverts_a_convolution():
    """Round trip: deconvolving (signal * kernel) by the same kernel returns
    the signal. Pins that the divide is the inverse of the convolution — a
    sign, conjugate or ordering error would break this while leaving the
    output finite."""
    kernel = np.zeros(N)
    t = np.arange(60)
    kernel[:60] = np.exp(-t / 12.0) - np.exp(-t / 2.0)
    kernel /= kernel.sum()

    signal = np.zeros(N)
    signal[[150, 400, 700]] = [1.0, 0.6, 0.3]

    measured = np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel)))
    flat = np.ones(N)                       # no filtering: exact inversion
    recovered = deconvolve_hits(measured, kernel, flat)
    assert np.allclose(recovered, signal, atol=1e-6)


@pytest.mark.parametrize("sigma", [0.3, 0.5, 2.0])
def test_sample_noise_sigma_survives_integer_adc_quantization(sigma):
    from pmtdeconv.core.dsp import sample_noise_sigma
    x = np.random.default_rng(0).normal(0.0, sigma, 2000)
    assert abs(sample_noise_sigma(x) - sigma) < 0.1 * sigma
    quantized = sample_noise_sigma(np.round(x))
    assert 0.2 < quantized < 2.0 * max(sigma, 0.5), quantized
