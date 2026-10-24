"""Unit tests for the biexp-overshoot SPE model (Daya Bay Eq. 4).

Data-free: builds a synthetic peak-normalized stacked template with a
known undershoot, then checks (a) the overshoot term has the right shape,
(b) the fit recovers the undershoot DEPTH (the degeneracy-robust
observable) and beats a plain biexp fit, and (c) the plain biexp model
leaves the overshoot columns NaN.
"""
import numpy as np

from pmtdeconv.core.kernels import spe_overshoot
from pmtdeconv.pipeline.template_build_impl import (
    biexp_smeared, biexp_overshoot_smeared, fit_stacked, PK_CENTER, N_SAMP,
)

DT = 1.432  # Cd109-like sampling; wide enough record for the recovery tail


def test_spe_overshoot_shape():
    """Fermi onset → ~zero before t_fermi, positive after, non-negative."""
    t_rel = np.array([0.0, 10.0, 30.0, 60.0, 150.0, 400.0])
    u = spe_overshoot(t_rel, U_os0=0.05, U_os1=0.01, tau_os=150.0,
                      t0_os=65.0, sigma_os=28.0)
    assert np.all(u >= 0.0)
    assert u[0] < 0.01            # essentially off well before the 50 ns onset
    assert u[3] > u[0]            # turned on by ~60 ns
    assert np.isfinite(u).all()


def _synth_stacked(depth_params):
    t = np.arange(N_SAMP) * DT
    tpk = t[PK_CENTER]
    y = biexp_overshoot_smeared(
        t, 1.0, 4.0, 14.0, tpk - 4.0, 2.5, 0.0,
        depth_params["U0"], depth_params["U1"], depth_params["tos"],
        depth_params["t0o"], depth_params["sgo"], DT)
    y = y / y.max()
    rng = np.random.default_rng(0)
    return t, tpk, y + rng.normal(0.0, 0.004, size=y.shape)


def test_overshoot_fit_recovers_depth_and_beats_biexp():
    inj = dict(U0=0.06, U1=0.015, tos=140.0, t0o=65.0, sgo=28.0)
    t, tpk, stacked = _synth_stacked(inj)

    # True undershoot depth (fraction of peak), the physical observable.
    peak = biexp_smeared(t, 1.0, 4.0, 14.0, tpk - 4.0, 2.5, 0.0, DT)
    over = spe_overshoot(t - (tpk - 4.0), inj["U0"], inj["U1"], inj["tos"],
                         inj["t0o"], inj["sgo"])
    true_depth = over.max() / peak.max()

    fit = fit_stacked(stacked, DT, model="biexp-overshoot")
    assert fit["model"] == "biexp-overshoot"
    # Degeneracy-robust depth recovered to within ~1.5 percentage points.
    assert abs(fit["undershoot_depth_frac"] - true_depth) < 0.015
    assert fit["undershoot_charge_frac"] > 0.0
    assert 50.0 < fit["tau_os_ns"] < 400.0

    # The overshoot model fits the dip far better than a plain biexp.
    fit_b = fit_stacked(stacked, DT, model="biexp")
    assert fit["chi2_red"] < fit_b["chi2_red"]
    # Plain biexp carries the overshoot columns as NaN (schema stable).
    assert np.isnan(fit_b["U_os0"])
    assert np.isnan(fit_b["undershoot_depth_frac"])
    assert fit_b["model"] == "biexp"
