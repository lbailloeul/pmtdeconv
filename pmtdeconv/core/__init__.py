"""Core DSP primitives for the PMT-deconvolution toolkit.

``utils.py`` was split by domain (PR: 2026-05-31) into:
  - :mod:`pmtdeconv.core.kernels`  — SPE template models + parameters
  - :mod:`pmtdeconv.core.filters`  — frequency-domain low-pass filters
  - :mod:`pmtdeconv.core.dsp`      — deconvolution, denoising, baseline
  - :mod:`pmtdeconv.core.peaks`    — integration windows, FWHM, vetoes

**This package is the single source of truth for the math.** Every module
inside ``pmtdeconv`` imports these names directly from ``core`` (2026-09-04);
:mod:`pmtdeconv.utils` is a compatibility shim that re-exports them for the
local, gitignored ``legacy/`` scripts only. New code imports from ``core``.
"""
