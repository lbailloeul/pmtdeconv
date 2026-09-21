#!/usr/bin/env python3
"""End-to-end ACCURACY floor — the correctness guard the golden tests lack.

The golden suite proves CSV bytes don't change; it cannot tell whether the
numbers are physically *right*. This suite runs the real pipeline (`area`,
`hits`) on regime fixtures with known ground truth and asserts the recovered
charge / timing land within tolerance — so a scientific regression (or a
deliberate change like the saturation rework) is caught by *accuracy*, not
only by byte drift.

Tolerances were measured on the current pipeline (2026-06, 80-event fixtures)
and set with ~2-3x margin so they catch gross regressions without flaking on
RNG. Measured baselines, for reference:

    regime   AREA charge scatter / eff      HITS primary-peak timing
    level3   0.04 / 1.00                    sigma 0.45 ns  (clean)
    noisy    0.14 / 1.00                    sigma ~100 ns  (SNR wall — untested)
    level1   0.06 / 0.94                    sigma ~93 ns   (SNR wall — untested)
    level5   0.05 / 1.00                    sigma 0.60 ns  (timing survives sat)
    pileup   —                              96% pulses found, sigma 0.71 ns

KEY scientific facts these lock in:
  * whole-waveform AREA charge is robust to noise AND saturation;
  * hits TIMING is excellent at high SNR and survives saturation (the basis of
    the timewalk analysis), but degrades on low-SNR single pulses;
  * the deconvolution genuinely RESOLVES overlapping pulses in time.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIXTURE_DIR))

from pmtdeconv.pipeline.scoring import score_run  # noqa: E402
from pmtdeconv.io.truth import load_truth  # noqa: E402


def _primary_timing_residuals(pulse_csv: Path, truth_csv: Path,
                              include_saturated: bool = False,
                              time_col: str = "t_peak") -> np.ndarray:
    """Per-(event, branch) residual <time_col>(primary) - true_center_ns.

    Robust timing is judged on these directly (fraction-within-tol + median),
    not on raw std: a handful of events where the tallest reconstructed peak
    is a noise spike give catastrophic outliers that inflate std but are
    exactly what a Gaussian-fit timewalk analysis ignores.
    """
    truth = load_truth(truth_csv)
    if not include_saturated and "saturated" in truth.columns:
        truth = truth[~truth["saturated"]]
    prod = pd.read_csv(pulse_csv)
    if "peak_id" in prod.columns:
        prod = prod[prod["peak_id"] == 1]
    prod = prod.rename(columns={"entry": "event"})
    merged = truth.merge(prod[["event", "branch", time_col]],
                         on=["event", "branch"], how="inner")
    return (merged[time_col].values - merged["true_center_ns"].values)


def _mad_sigma(x: np.ndarray) -> float:
    """Outlier-robust core scatter (1.4826·MAD) — what a Gaussian-fit
    timewalk analysis effectively measures."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))

DT_NS = 0.5051
N_EVENTS = 80
REGIMES_NEEDED = ["level3", "noisy", "level5", "pileup", "pileup_lowpe"]


@pytest.fixture(scope="module")
def regime_root(tmp_path_factory):
    """Build the regime fixtures once into a temp dir; return the `regime/` base.

    Uses the same builder `tune` uses, redirected to tmp via the module's
    ``__file__`` (so the repo's committed fixtures are never touched).
    """
    try:
        import make_synth_root as msr
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"cannot import fixture builder: {exc}")
    base = tmp_path_factory.mktemp("s0_regimes")
    orig = msr.__file__
    msr.__file__ = str(base / "make_synth_root.py")
    try:
        for reg in REGIMES_NEEDED:
            msr.main(regime=reg, n_events=N_EVENTS, write_truth=True,
                     dt_ns=DT_NS)
    finally:
        msr.__file__ = orig
    return base / "regime"


def _run(out_dir: Path, *args: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYBIN, "-m", "pmtdeconv", *args,
           "--out-dir", str(out_dir), "--log-level", "quiet"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"pmtdeconv {' '.join(args)} failed ({res.returncode}):\n"
            f"{res.stdout}\n{res.stderr}")


# ─── AREA charge floor (robust to noise + saturation) ───────────────

def test_area_charge_floor_clean(regime_root, tmp_path):
    rd = regime_root / "level3"
    _run(tmp_path / "a", "area", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0")
    m = score_run(tmp_path / "a" / "areas_run0.csv",
                  rd / "flattened_Run0.truth.csv", mode="area")
    assert m["efficiency"] >= 0.95, m
    assert m["charge_scatter"] < 0.12, m            # measured 0.04
    assert 0.5 < m["charge_ratio_median"] < 2.0, m  # scale sanity


def test_area_charge_degrades_gracefully_under_noise(regime_root, tmp_path):
    """10x noise must not destroy whole-waveform charge (the denoiser's job)."""
    rd = regime_root / "noisy"
    _run(tmp_path / "a", "area", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0")
    m = score_run(tmp_path / "a" / "areas_run0.csv",
                  rd / "flattened_Run0.truth.csv", mode="area")
    assert m["efficiency"] >= 0.90, m
    assert m["charge_scatter"] < 0.30, m            # measured 0.14


def test_area_charge_robust_to_saturation(regime_root, tmp_path):
    rd = regime_root / "level5"
    _run(tmp_path / "a", "area", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0")
    m = score_run(tmp_path / "a" / "areas_run0.csv",
                  rd / "flattened_Run0.truth.csv", mode="area")
    assert m["efficiency"] >= 0.90, m
    assert m["charge_scatter"] < 0.15, m            # measured 0.05


# ─── HITS timing floor (the timewalk basis) ─────────────────────────

def test_hits_timing_floor_clean(regime_root, tmp_path):
    """On clean pulses, the deconvolved primary-peak time tracks truth: the
    core is tight (median |resid| < 1.5 ns) and the vast majority of pulses
    land within 3 ns. (Raw std is not used — a few noise-peak-as-primary
    outliers inflate it, which the Gaussian-fit timewalk ignores.)"""
    rd = regime_root / "level3"
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0", "--no-vetoes")
    resid = _primary_timing_residuals(
        tmp_path / "h" / "hits.csv", rd / "flattened_Run0.truth.csv")
    assert len(resid) > 0
    frac_tight = float(np.mean(np.abs(resid) < 3.0))
    assert frac_tight >= 0.85, f"only {frac_tight:.2f} of pulses within 3 ns"
    assert np.median(np.abs(resid)) < 1.5, np.median(np.abs(resid))


def test_hits_timing_survives_saturation(regime_root, tmp_path):
    """Clipped pulses keep good edge timing — the premise of S5 (saturation =
    a charge defect, not a timing defect). Saturated truth rows are INCLUDED
    here on purpose."""
    rd = regime_root / "level5"
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0", "--no-vetoes")
    resid = _primary_timing_residuals(
        tmp_path / "h" / "hits.csv", rd / "flattened_Run0.truth.csv",
        include_saturated=True)
    assert len(resid) > 0
    frac_tight = float(np.mean(np.abs(resid) < 3.0))
    assert frac_tight >= 0.80, f"only {frac_tight:.2f} of pulses within 3 ns"
    assert np.median(np.abs(resid)) < 1.5, np.median(np.abs(resid))


def test_hits_thalf_low_jitter_and_saturation_robust(regime_root, tmp_path):
    """The t_half column (CFD-50 rising-edge crossing, MQAna `halftime`):

    * it sits ON THE RISING EDGE, so it is systematically EARLIER than the
      pulse center (negative median residual) with a tight core;
    * — the timewalk/S5 premise — its core jitter SURVIVES saturation: the
      rising edge is unclipped, so t_half stays tight where the apex-argmax
      t_peak degrades.

    Run with --peak-order first (the intended half-time pairing). Measured
    2026-07-01 (80-event fixtures): level3 median -3.6 ns / MAD-σ 0.77 ns;
    level5 (21 saturated) t_half MAD-σ 0.39 ns vs t_peak 1.20 ns.
    Tolerances set with ~2-3x margin, per this file's convention.
    """
    for regime, out_tag in (("level3", "h3"), ("level5", "h5")):
        rd = regime_root / regime
        _run(tmp_path / out_tag, "hits", "--dataset", "na22",
             "--data-root", str(rd), "--runs", "0", "--no-vetoes",
             "--peak-order", "first")

    # level3 (clean): rising-edge offset + tight core
    truth3 = regime_root / "level3" / "flattened_Run0.truth.csv"
    resid3 = _primary_timing_residuals(
        tmp_path / "h3" / "hits.csv", truth3, time_col="t_half")
    assert len(resid3) >= 0.9 * len(load_truth(truth3))
    assert -8.0 < np.median(resid3) < 0.0, np.median(resid3)
    assert _mad_sigma(resid3) < 2.0, _mad_sigma(resid3)   # measured 0.77

    # level5 (saturation included): the core must STAY tight
    truth5 = regime_root / "level5" / "flattened_Run0.truth.csv"
    resid5 = _primary_timing_residuals(
        tmp_path / "h5" / "hits.csv", truth5,
        include_saturated=True, time_col="t_half")
    assert len(resid5) > 0
    assert _mad_sigma(resid5) < 1.5, _mad_sigma(resid5)   # measured 0.39

    # sanity: the half-time crossing never comes after the apex
    prod = pd.read_csv(tmp_path / "h3" / "hits.csv")
    ok = np.isfinite(prod["t_half"]) & np.isfinite(prod["t_peak"])
    assert (prod.loc[ok, "t_half"] <= prod.loc[ok, "t_peak"]).all()


# ─── pile-up: the deconvolution's headline capability ───────────────

def test_pileup_separation(regime_root, tmp_path):
    """2-3 overlapping pulses (5-30 ns apart): the deconvolution must FIND
    most of them and time the matched ones tightly. Per-pulse charge in
    overlap is genuinely unreliable, so it is NOT asserted here.

    2026-07-28: rewired through the promoted library matcher
    (``scoring.match_pulses``) so this floor also guards the promotion —
    same greedy algorithm, same 15 ns tolerance, same floors."""
    from pmtdeconv.pipeline.scoring import match_pulses
    rd = regime_root / "pileup"
    truth = pd.read_csv(rd / "flattened_Run0.truth.csv")
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0", "--no-vetoes")
    found = pd.read_csv(tmp_path / "h" / "all_results.csv")
    if "success" in found.columns:
        found = found[found["success"] == True]  # noqa: E712

    n_true = 0
    matched = 0
    dt_resid: list[float] = []
    for (ev, br), tg in truth.groupby(["event", "branch"]):
        fg = found[(found["entry"] == ev) & (found["branch"] == br)]
        m = match_pulses(tg["true_center_ns"].values, fg["t_peak"].values,
                         tol_ns=15.0)
        n_true += m["n_true"]
        matched += m["n_matched"]
        dt_resid.extend(m["residuals_ns"].tolist())

    match_rate = matched / max(n_true, 1)
    assert match_rate > 0.85, f"only {match_rate:.2f} of pile-up pulses found"
    assert np.std(dt_resid) < 2.0, f"matched timing sigma {np.std(dt_resid):.2f} ns"


def test_pileup_lowpe_multi_scoring(regime_root, tmp_path):
    """The 0–20 PE target population: low-PE multi-hit bunches, scored with
    the per-pulse ``hits-multi`` mode (auto-selected on pulse_id truth).
    Floors measured 2026-07-28 then margined (loose by convention)."""
    from pmtdeconv.pipeline.scoring import score_run
    rd = regime_root / "pileup_lowpe"
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0", "--no-vetoes")
    s = score_run(tmp_path / "h" / "all_results.csv",
                  rd / "flattened_Run0.truth.csv", mode="hits")
    # auto-upgrade must have engaged (pulse_id in the truth)
    assert "split_rate" in s and np.isfinite(s["efficiency"])
    assert s["efficiency"] > 0.60, s
    assert s["timing_sigma_ns"] < 2.0, s
    assert s["merge_rate"] < 0.35, s
    # absolute per-pulse charge scale within 40% of unity (overlap-degraded,
    # loose floor — the benchmark reports the precise number)
    assert 0.6 < s["charge_ratio_mean"] < 1.4, s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
