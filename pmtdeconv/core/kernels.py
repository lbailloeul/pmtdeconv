#!/usr/bin/env python3
"""SPE template models and parameters.

Moved verbatim from ``pmtdeconv/utils.py`` (2026-05-31 domain split). Owns the
shared ``DTYPE`` / ``DT_NS`` constants that the rest of ``core`` re-uses.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

DTYPE = np.float64
# Legacy global default sampling period [ns]. Matches NO current dataset
# (na22 0.5051, cd109 1.432, na22_3ch 0.50458) — kept only for legacy-script
# compatibility. Never rely on it implicitly; pipeline code must always pass
# the dataset's real dt_ns.
DT_NS = 1.25

# ═══════════════════════════════════════════════════════════════════
# SPE template parameters
# ═══════════════════════════════════════════════════════════════════
# NOTE on t0: a kernel t0 does NOT cancel in the FFT division — FFT(r) picks up
# e^{-iω·t0}, so a nonzero t0 time-shifts the deconvolved output by −t0. The
# loaders pin t0=0 (io.templates.DEFAULT_T0, pipeline.template.FALLBACK_TEMPLATE);
# never feed these dicts' raw t0 into a deconvolution kernel.

LANL_TEMPLATE = dict(
    A=300.0,
    tau_rise=1.06069764323418,
    tau_fall=1.48020379040983,
    sigma=1.1079849521775,
    t0=2.0,
)

MILLIQAN_TEMPLATE = dict(
    A=400.0,
    tau_rise=12.3709373015379,
    tau_fall=18.0852574335868,
    sigma=2.50780690507711,
    t0=16.0,
)

# ═══════════════════════════════════════════════════════════════════
# SPE template generation
# ═══════════════════════════════════════════════════════════════════

def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    """Bi-exponential pulse convolved with a Gaussian (SPE model)."""
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)
    mask = t >= t0
    dt = t[mask] - t0
    tr = max(tau_rise, 1e-12)
    tf = max(tau_fall, 1e-12)
    output[mask] = A * (np.exp(-dt / tf) - np.exp(-dt / tr))
    dt_sample = t[1] - t[0]
    sigma_bins = max(sigma / dt_sample, 1e-6)
    return gaussian_filter1d(output, sigma_bins, mode="constant", axis=0)


def build_spe_template(N, dt_ns, template_params):
    """Build zero-padded, unit-area SPE response vector of length N.

    Accepts either the biexp parameter dict (A/tau_rise/tau_fall/t0/sigma) or
    a SAMPLED kernel dict (``kind="sampled"`` with ``t_ns``/``y`` arrays —
    io.templates schema #4): the samples are resampled onto the run grid and
    unit-area normalized under the same contract, so deconvolved charge stays
    pVs either way.
    """
    if isinstance(template_params, dict) and \
            template_params.get("kind") == "sampled":
        t = np.arange(N, dtype=DTYPE) * dt_ns
        r_calib = np.interp(t, np.asarray(template_params["t_ns"], dtype=DTYPE),
                            np.asarray(template_params["y"], dtype=DTYPE),
                            left=0.0, right=0.0).astype(np.float64)
        area = np.sum(r_calib) * dt_ns
        if not np.isfinite(area) or area <= 0:
            raise ValueError(
                f"sampled SPE template has non-positive net area ({area!r}) "
                f"on the N={N}, dt={dt_ns} grid — refusing to build a NaN "
                f"deconvolution kernel.")
        return r_calib / area

    t = np.arange(0, N * dt_ns, dt_ns, dtype=DTYPE)[:N]
    g = gaussian_smeared_biexponential(
        t,
        template_params["A"],
        template_params["tau_rise"],
        template_params["tau_fall"],
        t0=template_params["t0"],
        sigma=template_params["sigma"],
    )
    r_calib = np.zeros(N, dtype=np.float64)
    r_calib[:len(g)] = g
    area = np.sum(r_calib) * dt_ns
    if not np.isfinite(area) or area <= 0:
        raise ValueError(
            f"SPE template has non-positive/non-finite area ({area!r}); "
            f"degenerate parameters (e.g. tau_rise == tau_fall) yield an "
            f"all-zero kernel — refusing to build a NaN deconvolution kernel. "
            f"params={template_params!r}"
        )
    r_calib /= area
    return r_calib


def spe_overshoot(t_rel, U_os0, U_os1, tau_os, t0_os, sigma_os,
                  t_fermi=50.0, w_fermi=10.0):
    """Daya Bay baseline-restoration overshoot — their Eq. 4 (Section 2.4,
    "Baseline restoration") of the Daya Bay PMT-waveform-modeling paper.

    Returns a **non-negative** magnitude ``U_os(t) >= 0`` to be SUBTRACTED
    from a positive-going main pulse (our convention; Daya Bay's pulses are
    negative-going so they call it an *overshoot*, we see an *undershoot*
    below baseline as the AC-coupled baseline restores).

    ``t_rel`` is time measured from the pulse onset ``t0`` (i.e. ``t - t0``).
    Three pieces (see the paper):

      U_os(t) = [ U_os0 / (exp((t_fermi - t)/w_fermi) + 1) ] · exp(-t/tau_os)
                + U_os1 · exp(-½((t - t0_os)/sigma_os)²)

    - Fermi factor: smooth, delayed onset (~``t_fermi`` ns, width ``w_fermi``).
    - exp(-t/tau_os): slow baseline recovery (Daya Bay measured tau_os ≈ 150 ns).
    - Gaussian bump: a localized feature in the trough.

    Daya Bay reported U_os0 ≈ 0.4 mV ≈ 5% of the SPE peak height at gain
    2e7. For peak-normalized templates the amplitudes are therefore
    fractions of the peak (~0.05). ``t_fermi`` / ``w_fermi`` are Daya-Bay
    electronics constants; expose them so other detectors can free them.

    Not called by the reconstruction pipeline — retained as the Daya-Bay-style
    undershoot model for the Tang et al. 2411.07620 Monte-Carlo waveform
    simulation (``pmtdeconv.sim``).
    """
    t_rel = np.asarray(t_rel, dtype=DTYPE)
    fermi = 1.0 / (np.exp((t_fermi - t_rel) / max(w_fermi, 1e-6)) + 1.0)
    recovery = U_os0 * fermi * np.exp(-t_rel / max(tau_os, 1e-6))
    bump = U_os1 * np.exp(-0.5 * ((t_rel - t0_os) / max(sigma_os, 1e-6)) ** 2)
    return recovery + bump
