#!/usr/bin/env python3
"""Saturation = a data-quality FLAG, detected by the flat clipped-top plateau.

S5 replaced the fixed ``wave.max() >= sat_thresh`` gate (which needed each
detector's rail and DROPPED the event) with a universal plateau detector
(``core.dsp.detect_saturation_plateau``: a run of >=K samples within a sub-noise
epsilon of vmax). Saturated events are now KEPT and carry ``saturated=True``,
their timing recovered; only charge is unreliable (cut downstream via the flag).

Two test layers:
- unit tests of the detector (synthetic arrays — no fixtures);
- an end-to-end contract test that level5's clipped events are kept + flagged at
  DEFAULT (no --sat-thresh).

Detector parameters (validated on real na22 Run 65213, hard 950 mV clip):
ε = 0.5σ, K = 10. K=10 sits between the worst rounded-peak false positive (the
legacy bare-Gaussian fixture, 7-sample near-max run) and real clips (level5
25-41 samples; na22 150+), so it rejects broad unclipped tops AND catches every
real clip — with zero rail knowledge.
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

from pmtdeconv.core.dsp import detect_saturation_plateau  # noqa: E402

DT_NS = 0.5051


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


@pytest.fixture(scope="module")
def level5_root(tmp_path_factory):
    try:
        import make_synth_root as msr
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import fixture builder: {exc}")
    base = tmp_path_factory.mktemp("s0_sat")
    orig = msr.__file__
    msr.__file__ = str(base / "make_synth_root.py")
    try:
        msr.main(regime="level5", n_events=80, write_truth=True, dt_ns=DT_NS)
    finally:
        msr.__file__ = orig
    return base / "regime" / "level5"


def _run(out_dir: Path, *args: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYBIN, "-m", "pmtdeconv", *args,
           "--out-dir", str(out_dir), "--log-level", "quiet"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"pmtdeconv {' '.join(args)} failed ({res.returncode}):\n"
            f"{res.stdout}\n{res.stderr}")


def test_saturation_kept_and_flagged(level5_root, tmp_path):
    """S5 contract: clipped events are KEPT and FLAGGED, detected by the plateau
    at DEFAULT (no --sat-thresh, no rail knowledge), with timing recovered —
    never dropped."""
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(level5_root), "--runs", "0", "--no-vetoes")
    df = pd.read_csv(tmp_path / "h" / "all_results.csv")
    assert "saturated" in df.columns
    sat = df[df["saturated"] == True]  # noqa: E712
    # plateau flags level5's clipped events with no threshold supplied
    assert len(sat) > 0, "plateau should flag level5's clipped events at default"
    # saturation is no longer a DROP reason
    assert "saturated" not in set(df["reason"].dropna().astype(str))
    # kept (success) with recovered timing — saturation is a charge defect only
    sat_ok = sat[sat["success"] == True]  # noqa: E712
    assert len(sat_ok) > 0
    assert sat_ok["t_peak"].notna().any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
