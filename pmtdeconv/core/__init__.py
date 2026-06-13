"""Core DSP primitives for the PMT-deconvolution toolkit.

``utils.py`` was split by domain (PR: 2026-05-31) into:
  - :mod:`pmtdeconv.core.kernels`  — SPE template models + parameters
  - :mod:`pmtdeconv.core.filters`  — frequency-domain low-pass filters
  - :mod:`pmtdeconv.core.dsp`      — deconvolution, denoising, baseline
  - :mod:`pmtdeconv.core.peaks`    — integration windows, FWHM, vetoes

``pmtdeconv.utils`` remains a thin facade re-exporting all of these, so the
package modules and the ~35 legacy scripts keep importing from the same place
and the byte-identical golden fixtures are unaffected.
"""
