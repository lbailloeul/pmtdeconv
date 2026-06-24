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


FOMS: Dict[str, Callable[..., dict]] = {
    "photopeak_sigma_over_mu": photopeak_sigma_over_mu,
    "timing_sigma": timing_sigma,
}

# All current FoMs are resolution metrics (smaller is better). Recorded so a
# `validate`/`tune` can mark the winner without hard-coding the direction.
FOM_LOWER_IS_BETTER: Dict[str, bool] = {
    "photopeak_sigma_over_mu": True,
    "timing_sigma": True,
}


def get_fom(name: str) -> Callable[..., dict]:
    if name not in FOMS:
        raise KeyError(f"Unknown FoM {name!r}. Known: {', '.join(FOMS)}")
    return FOMS[name]


def list_foms() -> list:
    return list(FOMS)
