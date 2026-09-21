"""Regime flag + gate tests (0–20 PE re-scope step 4).

Covers: the raw nPE estimator math, blank-when-uncalibrated columns, the
`--in-regime-only` gate + `out_of_pe_range` accounting, and the uncalibrated
hard refusal.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
FIXTURE_DIR = THIS_DIR / "fixtures"
SYNTH_ROOT = FIXTURE_DIR / "flattened_Run0.root"
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.pipeline.gain import estimate_npe_raw       # noqa: E402


class TestEstimateNpeRaw:
    def test_box_pulse_scale(self):
        dt = 0.5
        wave = np.zeros(1024)
        # 20 mV × 10 ns box at sample 400 → raw charge = 200 pVs
        wave[400:400 + int(10 / dt)] = 20.0
        npe = estimate_npe_raw(wave, dt, spe_charge=100.0)
        assert abs(npe - 2.0) < 0.15, npe

    def test_window_excludes_far_charge(self):
        dt = 0.5
        wave = np.zeros(1024)
        wave[400:420] = 20.0            # main pulse
        wave[900:920] = 20.0            # far pulse, outside ±17.5 ns window
        npe_one = estimate_npe_raw(wave, dt, spe_charge=100.0)
        assert npe_one < 3.0            # only the argmax window counts

    def test_nan_when_uncalibrated(self):
        wave = np.zeros(64); wave[32] = 50.0
        assert np.isnan(estimate_npe_raw(wave, 1.0, spe_charge=None))
        assert np.isnan(estimate_npe_raw(wave, 1.0, spe_charge=0.0))


def _run_hits(out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "hits", "--dataset", "na22",
         "--data-root", str(FIXTURE_DIR), "--runs", "0",
         "--out-dir", str(out_dir), "--log-level", "quiet", *extra],
        cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.mark.skipif(not SYNTH_ROOT.exists(),
                    reason="synth fixture missing (make_synth_root.py)")
class TestRegimeGateCLI:
    def test_flag_only_populates_columns(self, tmp_path):
        res = _run_hits(tmp_path / "a", "--spe-charge",
                        "d0_ch1=800", "d0_ch2=800")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "a" / "all_results.csv")
        assert "npe_raw" in df.columns and "in_regime" in df.columns
        assert df["npe_raw"].notna().all()
        assert set(df["in_regime"].unique()) <= {True, False}
        # flag-only: nothing rejected for regime reasons
        assert not (df["reason"] == "out_of_pe_range").any()

    def test_uncalibrated_columns_blank(self, tmp_path):
        res = _run_hits(tmp_path / "b")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "b" / "all_results.csv")
        assert df["npe_raw"].isna().all()
        assert df["in_regime"].isna().all()
        assert df["npe"].isna().all()

    def test_gate_rejects_out_of_band(self, tmp_path):
        # Absurdly tight band → everything is out of regime.
        res = _run_hits(tmp_path / "c", "--spe-charge", "d0_ch1=800",
                        "d0_ch2=800", "--pe-band", "0", "0.0001",
                        "--in-regime-only")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "c" / "all_results.csv")
        rejected = df[df["reason"] == "out_of_pe_range"]
        assert len(rejected) == len(df), "tight band must reject every event"
        assert not rejected["success"].any()

    def test_gate_keeps_in_band(self, tmp_path):
        # Wide-open band → gate passes everything (matches flag-only rows).
        res = _run_hits(tmp_path / "d", "--spe-charge", "d0_ch1=800",
                        "d0_ch2=800", "--pe-band", "0", "1e9",
                        "--in-regime-only")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "d" / "all_results.csv")
        assert not (df["reason"] == "out_of_pe_range").any()

    def test_uncalibrated_gate_refuses(self, tmp_path):
        res = _run_hits(tmp_path / "e", "--in-regime-only")
        assert res.returncode != 0
        assert "gain calibrate" in (res.stderr + res.stdout)
