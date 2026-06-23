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
                              include_saturated: bool = False) -> np.ndarray:
    """Per-(event, branch) residual t_peak(primary) - true_center_ns.

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
    merged = truth.merge(prod[["event", "branch", "t_peak"]],
                         on=["event", "branch"], how="inner")
    return (merged["t_peak"].values - merged["true_center_ns"].values)

DT_NS = 0.5051
N_EVENTS = 80
REGIMES_NEEDED = ["level3", "noisy", "level5", "pileup"]


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
        tmp_path / "h" / "deconv_pulse.csv", rd / "flattened_Run0.truth.csv")
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
        tmp_path / "h" / "deconv_pulse.csv", rd / "flattened_Run0.truth.csv",
        include_saturated=True)
    assert len(resid) > 0
    frac_tight = float(np.mean(np.abs(resid) < 3.0))
    assert frac_tight >= 0.80, f"only {frac_tight:.2f} of pulses within 3 ns"
    assert np.median(np.abs(resid)) < 1.5, np.median(np.abs(resid))


# ─── pile-up: the deconvolution's headline capability ───────────────

def test_pileup_separation(regime_root, tmp_path):
    """2-3 overlapping pulses (5-30 ns apart): the deconvolution must FIND
    most of them and time the matched ones tightly. Per-pulse charge in
    overlap is genuinely unreliable, so it is NOT asserted here."""
    rd = regime_root / "pileup"
    truth = pd.read_csv(rd / "flattened_Run0.truth.csv")
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(rd), "--runs", "0", "--no-vetoes")
    found = pd.read_csv(tmp_path / "h" / "all_results.csv")
    if "success" in found.columns:
        found = found[found["success"] == True]  # noqa: E712

    n_true = 0
    matched = 0
    dt_resid = []
    for (ev, br), tg in truth.groupby(["event", "branch"]):
        fg = found[(found["entry"] == ev) & (found["branch"] == br)]
        ftimes = fg["t_peak"].values
        n_true += len(tg)
        used: set[int] = set()
        for _, tr in tg.iterrows():
            if len(ftimes) == 0:
                continue
            d = np.abs(ftimes - tr["true_center_ns"])
            j = int(np.argmin(d))
            if d[j] < 15.0 and j not in used:
                used.add(j)
                matched += 1
                dt_resid.append(float(ftimes[j] - tr["true_center_ns"]))

    match_rate = matched / max(n_true, 1)
    assert match_rate > 0.85, f"only {match_rate:.2f} of pile-up pulses found"
    assert np.std(dt_resid) < 2.0, f"matched timing sigma {np.std(dt_resid):.2f} ns"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
