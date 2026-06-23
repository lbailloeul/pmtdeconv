#!/usr/bin/env python3
"""Saturation-gate behavior — the contract S5 (saturation-as-flag) will flip.

TODAY the saturation gate fires unconditionally (pipeline.py: `wave.max() >=
sat_threshold` -> success=False, reason="saturated"), even under --no-vetoes.
This test LOCKS that current behavior so the S5 rework is a deliberate, visible
change rather than a silent one.

When S5 lands (saturation becomes a data-quality FLAG, not a drop), flip the
assertions here to: saturated events are KEPT (success=True), carry
``saturated=True``, and still produce a finite ``t_peak`` (timing recovered).

NOTE: the level5 fixture clips at 950 mV but na22's default sat_thresh is 960,
so the gate does NOT fire at the default — we pass ``--sat-thresh 900`` so the
clipped/bright events actually reach the threshold.

S5 will replace the fixed threshold with a flat-PLATEAU detector (saturation =
>=K consecutive samples within a sub-noise epsilon of vmax). Validated on the
regimes (2026-06): plateau ε=0.25σ, K=4 matches the correctly-set threshold
(23 TP / 1 FP / 0 FN on level5) with NO rail knowledge, and gives 0 false
positives on bright-but-unclipped level4 — because clipped runs are 25-41
samples flat to <0.5 mV while the flattest unclipped top is <=3 samples.
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


def test_saturation_gate_currently_drops_events(level5_root, tmp_path):
    """CURRENT behavior: events reaching sat_thresh are dropped with
    reason='saturated' — even under --no-vetoes. (S5 flips this.)"""
    _run(tmp_path / "h", "hits", "--dataset", "na22",
         "--data-root", str(level5_root), "--runs", "0",
         "--sat-thresh", "900", "--no-vetoes")
    df = pd.read_csv(tmp_path / "h" / "all_results.csv")
    assert "reason" in df.columns
    n_sat = int((df["reason"] == "saturated").sum())
    assert n_sat > 0, (
        "expected some events to hit the saturation gate at --sat-thresh 900; "
        f"reason counts: {df['reason'].value_counts().to_dict()}")
    # And today those saturated rows are NOT successes.
    sat_rows = df[df["reason"] == "saturated"]
    assert not sat_rows["success"].any(), \
        "saturated rows should currently be failures (pre-S5)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
