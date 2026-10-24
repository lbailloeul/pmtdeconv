"""Named figures of merit (FoM) — the catalog a card declares for ``validate``.

A FoM maps a per-event array (charges, or a Δt distribution) to a scalar score
plus details. The SAME FoM is computed on the deconvolved / raw / DRS area so
``pmtdeconv validate`` compares methods on a *physics-anchored* number the
detector expert chose — the tool never picks the FoM for them.

All current FoMs are resolution metrics where **lower is better** (see
``FOM_LOWER_IS_BETTER``). Grow the catalog by adding a function here and
registering it in ``FOMS``; ``dataset lint`` / ``validate`` pick it up by name.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def _gauss_fit_hist(values: np.ndarray, lo: float, hi: float,
                    nbins: int) -> Tuple[float, float, bool]:
    """Histogram ``values`` in [lo, hi] and Gaussian-fit it; return
    ``(mu, sigma, ok)``. Reuses the timewalk fitter so the FoM and the timewalk
    curve share one primitive (locked by test_fom_metric)."""
    from pmtdeconv.pipeline.timewalk import _fit_gauss
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(values, bins=edges)
    mu, sig, _A, _mu_err, _sig_err, ok = _fit_gauss(centers, counts.astype(float))
    return float(mu), float(sig), bool(ok)


def photopeak_sigma_over_mu(charges: Sequence[float], *,
                            window: Optional[Tuple[float, float]] = None,
                            nbins: int = 120) -> dict:
    """Energy resolution σ/μ of the photopeak (lower = better).

    Gaussian-fits the charge histogram, optionally restricted to ``window``
    (the photopeak region). ``window`` lets the expert isolate the 511 keV peak
    from Compton/pileup background instead of fitting the whole spectrum.
    """
    x = np.asarray(charges, dtype=float)
    x = x[np.isfinite(x)]
    if window is not None:
        x = x[(x >= window[0]) & (x <= window[1])]
    if x.size < 20:
        return {"value": float("nan"), "mu": float("nan"),
                "sigma": float("nan"), "n": int(x.size), "ok": False}
    lo, hi = (window if window is not None
              else tuple(np.percentile(x, [1.0, 99.0])))
    mu, sig, ok = _gauss_fit_hist(x, lo, hi, nbins)
    value = abs(sig / mu) if (ok and mu) else float("nan")
    return {"value": float(value), "mu": mu, "sigma": sig,
            "n": int(x.size), "ok": ok}


def fit_1pe_peak(charges: Sequence[float], *,
                 window: Optional[Tuple[float, float]] = None,
                 nbins: int = 120) -> dict:
    """Fit the single-PE charge-quantization peak of a charge spectrum.

    The PE-scale anchor fit (0–20 PE re-scope): the fitted ``mu`` is the
    1-PE charge in the spectrum's own units (pVs for deconvolved charge) —
    the value ``gain calibrate`` proposes as ``spe_charge``.

    With ``window`` the fit is a plain Gaussian fit in [lo, hi] (expert
    isolates the 1-PE peak). Without, the peak is auto-located: histogram the
    positive charges up to the 99th percentile, box-smooth, and take the most
    prominent local maximum away from the pedestal (mode > 5% of q99); the fit
    window is ±40% of the mode. ``resolved`` reports the amplitude-quantization
    criterion: the valley between pedestal and 1-PE peak dips below 1/3 of the
    peak count (vault 17/17b — a resolvable 1-PE peak is what certifies a
    single-PE population).
    """
    x = np.asarray(charges, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    out_nan = {"value": float("nan"), "mu": float("nan"), "sigma": float("nan"),
               "sigma_over_mu": float("nan"), "n": int(x.size), "ok": False,
               "resolved": False, "window": None}
    if x.size < 50:
        return out_nan

    if window is None:
        q99 = float(np.percentile(x, 99.0))
        counts, edges = np.histogram(x, bins=nbins, range=(0.0, q99))
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = np.convolve(counts.astype(float), np.ones(5) / 5.0,
                             mode="same")
        from scipy.signal import find_peaks as _fp
        pk_idx, props = _fp(smooth, prominence=0.05 * float(smooth.max()))
        pk_idx = pk_idx[centers[pk_idx] > 0.05 * q99]   # away from the pedestal
        if len(pk_idx) == 0:
            return out_nan
        mode_i = int(pk_idx[np.argmax(smooth[pk_idx])])
        mode = float(centers[mode_i])
        window = (0.6 * mode, 1.4 * mode)
        # amplitude-quantization check: valley between pedestal and the peak
        valley = float(smooth[:mode_i].min()) if mode_i > 0 else float("inf")
        resolved = valley < smooth[mode_i] / 3.0
    else:
        resolved = True   # expert-supplied window: quantization judged by eye

    mu, sig, ok = _gauss_fit_hist(x[(x >= window[0]) & (x <= window[1])],
                                  window[0], window[1], nbins)
    # A Gaussian whose fitted center escaped its own window is a runaway on a
    # monotone slope, not a peak — never propose it as a charge anchor.
    if ok and not (window[0] <= mu <= window[1]):
        ok = False
    som = abs(sig / mu) if (ok and mu) else float("nan")
    return {"value": float(mu), "mu": float(mu), "sigma": float(sig),
            "sigma_over_mu": float(som), "n": int(x.size), "ok": bool(ok),
            "resolved": bool(resolved),
            "window": (float(window[0]), float(window[1]))}


def timing_sigma(dt: Sequence[float], *, clip_ns: float = 40.0,
                 nbins: int = 100) -> dict:
    """Coincidence timing resolution σ of a Δt distribution (lower = better).

    Clips ``|Δt| > clip_ns`` noise-timed outliers (a no-veto tallest peak is
    occasionally a noise spike) before the Gaussian fit.
    """
    d = np.asarray(dt, dtype=float)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) <= clip_ns]
    if d.size < 20:
        return {"value": float("nan"), "mu": float("nan"),
                "sigma": float("nan"), "n": int(d.size), "ok": False}
    mu, sig, ok = _gauss_fit_hist(d, float(d.min()), float(d.max()), nbins)
    return {"value": float(abs(sig)) if ok else float("nan"),
            "mu": mu, "sigma": sig, "n": int(d.size), "ok": ok}


def spe_quantization_sigma_over_mu(charges: Sequence[float], *,
                                   window: Optional[Tuple[float, float]] = None,
                                   nbins: int = 120) -> dict:
    """σ/μ of the 1-PE charge-quantization peak (lower = better).

    The PROVEN in-regime (0–20 PE) observable — vault 17b's crossover figure:
    deconvolution's demonstrated charge win is resolving the 1-PE peak with a
    near-empty pedestal valley (+33%/+51% σ/μ vs raw on Cd109), NOT photopeak
    resolution. Wraps :func:`fit_1pe_peak`; ``value`` = sigma/mu so the
    registry direction convention (lower better) holds.
    """
    res = fit_1pe_peak(charges, window=window, nbins=nbins)
    res = dict(res)
    res["value"] = res["sigma_over_mu"]
    return res


FOMS: Dict[str, Callable[..., dict]] = {
    "photopeak_sigma_over_mu": photopeak_sigma_over_mu,
    "timing_sigma": timing_sigma,
    "spe_quantization_sigma_over_mu": spe_quantization_sigma_over_mu,
}

# All current FoMs are resolution metrics (smaller is better). Recorded so a
# `validate`/`tune` can mark the winner without hard-coding the direction.
FOM_LOWER_IS_BETTER: Dict[str, bool] = {
    "photopeak_sigma_over_mu": True,
    "timing_sigma": True,
    "spe_quantization_sigma_over_mu": True,
}


def get_fom(name: str) -> Callable[..., dict]:
    if name not in FOMS:
        raise KeyError(f"Unknown FoM {name!r}. Known: {', '.join(FOMS)}")
    return FOMS[name]

