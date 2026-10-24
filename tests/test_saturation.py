#!/usr/bin/env python3
"""Saturation detection: the flat clipped-top plateau.

``core.dsp.detect_saturation_plateau`` flags a run of >=K samples within a
sub-noise epsilon of vmax. Since 2026-10-23 a flagged waveform is dropped
before deconvolution (``hits``/``area`` reason ``saturated``; contract tested
in ``test_spe_scope.py``). These are unit tests of the detector itself.

Detector parameters (validated on real na22 Run 65213, hard 950 mV clip):
ε = 0.5σ, K = 10. K=10 sits between the worst rounded-peak false positive (the
legacy bare-Gaussian fixture, 7-sample near-max run) and real clips (level5
25-41 samples; na22 150+), so it rejects broad unclipped tops AND catches every
real clip — with zero rail knowledge.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from pmtdeconv.core.dsp import detect_saturation_plateau


# ─── unit tests: the universal plateau detector (no rail needed) ────

def test_plateau_detects_hard_clip():
    rng = np.random.default_rng(0)
    w = rng.normal(5.0, 1.0, 1024)          # baseline σ ≈ 1 mV
    w[400:430] = 950.0                       # 30-sample hard clip at the rail
    sat, run = detect_saturation_plateau(w, noise_sigma=1.0)
    assert sat
    assert run >= 30


def test_plateau_ignores_rounded_peak():
    """A tall but UNCLIPPED rounded peak (the level4 case) must NOT flag."""
    x = np.arange(1024)
    w = 5.0 + 500.0 * np.exp(-0.5 * ((x - 400) / 8.0) ** 2)
    w = w + np.random.default_rng(1).normal(0.0, 1.0, 1024)
    sat, _run = detect_saturation_plateau(w, noise_sigma=1.0)
    assert not sat


def test_plateau_subnoise_eps_required():
    """eps ≈ σ over-flags; the sub-noise default (0.5σ) does not fire on a
    noisy rounded top."""
    x = np.arange(1024)
    w = 5.0 + 300.0 * np.exp(-0.5 * ((x - 400) / 20.0) ** 2)  # broad top
    w = w + np.random.default_rng(2).normal(0.0, 2.0, 1024)   # σ = 2
    assert not detect_saturation_plateau(w, 2.0, eps_sigma_frac=0.5)[0]


def test_plateau_ignores_broad_small_pulse():
    """Real Cd109 CH2: a broad ~9 mV single-PE pulse whose sample jitter is
    ~0.4 mV, while the pre-trigger σ the caller passes is 3.7 mV (slow
    baseline wander). Its rounded top stays within 0.5σ of the max for >10
    samples, yet it is not clipped and must not be dropped as saturated
    (2026-10-23 regression: 27% of Cd109 CH2 waveforms were)."""
    x = np.arange(1024)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(50):
        w = 16.0 + 9.0 * np.exp(-0.5 * ((x - 300) / 18.0) ** 2) + rng.normal(0.0, 0.4, 1024)
        hits += detect_saturation_plateau(w, 3.7)[0]
    assert hits == 0


def test_plateau_below_k_not_saturated():
    w = np.full(200, 5.0)
    w[100:102] = 950.0                       # only 2 samples at the rail
    sat, run = detect_saturation_plateau(w, 1.0, k_samples=4)
    assert not sat
    assert run == 2


def test_plateau_handles_nan_cells():
    """A NaN (DRS dead cell) in the plateau must not crash; it conservatively
    splits the run but a long clip still passes."""
    w = np.full(200, 5.0)
    w[100:130] = 950.0
    w[115] = np.nan
    sat, _run = detect_saturation_plateau(w, 1.0)
    assert sat


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
