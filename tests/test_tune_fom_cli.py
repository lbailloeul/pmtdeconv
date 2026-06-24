"""S7: `tune fom` sweeps the cosine cutoff on REAL data, scores with the card
FoM, and reports anti-cheat columns (efficiency, charge drift). Uses the flat
synthetic fixture as stand-in real data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / "flattened_Run0.root").exists(),
    reason="flat synth fixture not built")


def _tune(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYBIN, "-m", "pmtdeconv", "tune", "fom", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_tune_fom_reports_curve_with_anticheat(tmp_path):
    out = tmp_path / "out"
    r = _tune("--dataset", "na22", "--data-root", str(FIXTURE_DIR), "--runs", "0",
              "--fom", "photopeak_sigma_over_mu",
              "--cutoff-range", "100", "300", "100", "--out-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out / "sweep_fom.csv")
    for col in ("branch", "cosine_cutoff_mhz", "fom", "efficiency",
                "charge_drift"):
        assert col in df.columns, col
    assert df["cosine_cutoff_mhz"].nunique() == 3
    assert list(out.glob("sweep_fom_*.png"))


def test_tune_fom_auto_select_writes_recipe(tmp_path):
    out = tmp_path / "out"
    r = _tune("--dataset", "na22", "--data-root", str(FIXTURE_DIR), "--runs", "0",
              "--fom", "photopeak_sigma_over_mu",
              "--cutoff-range", "100", "300", "100",
              "--auto-select", "--out-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "recipe.json").read_text())
    assert rec["fom"] == "photopeak_sigma_over_mu"
    assert rec["winners"]                          # >=1 channel selected


def test_tune_fom_warns_subbin_step(tmp_path):
    out = tmp_path / "out"
    r = _tune("--dataset", "na22", "--data-root", str(FIXTURE_DIR), "--runs", "0",
              "--fom", "photopeak_sigma_over_mu",
              "--cutoff-range", "100", "101", "0.5", "--out-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FFT bin" in r.stdout                    # Δf floor warning fired
