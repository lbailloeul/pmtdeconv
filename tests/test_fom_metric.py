#!/usr/bin/env python3
"""Figure-of-merit primitive — the Gaussian sigma/mu fit must be correct.

The real-data FoM (S6/S7) reports a photopeak's resolution sigma/mu and a
timing resolution sigma, both via a Gaussian fit of a histogram. This locks
the fit primitive (``pipeline.timewalk._fit_gauss`` — the one the timewalk and
the future FoM metric module share): given a histogram sampled from a known
N(mu, sigma), it must recover mu, sigma, and hence sigma/mu.

When S6 factors the photopeak/timing fits into a dedicated metric module,
extend this file to assert the metric module returns the same sigma/mu on a
synthetic charge spectrum with a known photopeak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.pipeline.timewalk import _fit_gauss  # noqa: E402


def _hist_from_gaussian(mu, sigma, n=20000, lo=None, hi=None, nbins=80, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(mu, sigma, n)
    lo = mu - 5 * sigma if lo is None else lo
    hi = mu + 5 * sigma if hi is None else hi
    counts, edges = np.histogram(samples, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(float)


def test_fit_gauss_recovers_mu_sigma():
    centers, counts = _hist_from_gaussian(mu=100.0, sigma=10.0)
    mu, sig, _A, _mu_err, _sig_err, ok = _fit_gauss(centers, counts)
    assert ok
    assert abs(mu - 100.0) < 0.5, mu
    assert abs(sig - 10.0) < 0.5, sig


def test_fit_gauss_recovers_sigma_over_mu():
    """sigma/mu is the photopeak FoM the tune/validate path optimizes."""
    centers, counts = _hist_from_gaussian(mu=511.0, sigma=35.0, seed=3)
    mu, sig, *_rest, ok = _fit_gauss(centers, counts)
    assert ok
    fom = sig / mu
    assert abs(fom - 35.0 / 511.0) < 0.01, fom


def test_fit_gauss_narrower_peak_lower_fom():
    """A genuinely better resolution (narrower peak) must yield a lower FoM —
    the monotonicity the cutoff/denoiser tuning relies on."""
    c1, n1 = _hist_from_gaussian(mu=511.0, sigma=40.0, seed=1)
    c2, n2 = _hist_from_gaussian(mu=511.0, sigma=20.0, seed=2)
    mu1, s1, *_a, ok1 = _fit_gauss(c1, n1)
    mu2, s2, *_b, ok2 = _fit_gauss(c2, n2)
    assert ok1 and ok2
    assert (s2 / mu2) < (s1 / mu1)


def test_fit_gauss_too_few_counts_returns_not_ok():
    centers = np.linspace(-5, 5, 40)
    counts = np.zeros_like(centers)
    counts[20] = 5  # < 20 total
    *_vals, ok = _fit_gauss(centers, counts)
    assert not ok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
