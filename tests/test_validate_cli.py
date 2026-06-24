"""S6: `validate` computes a card-declared FoM across area sources and reports
a per-channel comparison table + JSON, marking the better method.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable


def _area_csv(path: Path, mu: float, sigma: float, *, n: int = 8000,
              branch: str = "d0_ch1", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pd.DataFrame({"branch": branch,
                  "area": rng.normal(mu, sigma, n)}).to_csv(path, index=False)


def _validate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYBIN, "-m", "pmtdeconv", "validate", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_validate_reports_fom_and_winner(tmp_path):
    dec, raw = tmp_path / "deconv.csv", tmp_path / "raw.csv"
    _area_csv(dec, 5000, 200, seed=1)   # tight  -> σ/μ ≈ 0.04
    _area_csv(raw, 5000, 400, seed=2)   # loose  -> σ/μ ≈ 0.08
    out = tmp_path / "out"
    r = _validate("--deconv-csv", str(dec), "--raw-csv", str(raw),
                  "--fom", "photopeak_sigma_over_mu", "--out-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "best=deconv" in r.stdout
    rep = json.loads((out / "validate_report.json").read_text())
    assert rep["fom"] == "photopeak_sigma_over_mu"
    assert (rep["sources"]["deconv"]["d0_ch1"]["value"]
            < rep["sources"]["raw"]["d0_ch1"]["value"])


def test_validate_uses_card_declared_fom(tmp_path):
    card = tmp_path / "card.yaml"
    card.write_text(yaml.safe_dump(
        {"base": "na22", "fom": "photopeak_sigma_over_mu"}))
    dec, raw = tmp_path / "d.csv", tmp_path / "r.csv"
    _area_csv(dec, 5000, 200, seed=3)
    _area_csv(raw, 5000, 400, seed=4)
    r = _validate("--dataset", str(card), "--deconv-csv", str(dec),
                  "--raw-csv", str(raw), "--out-dir", str(tmp_path / "o"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "photopeak_sigma_over_mu" in r.stdout


def test_validate_no_fom_errors(tmp_path):
    dec, raw = tmp_path / "d.csv", tmp_path / "r.csv"
    _area_csv(dec, 5000, 200); _area_csv(raw, 5000, 400)
    r = _validate("--deconv-csv", str(dec), "--raw-csv", str(raw))
    assert r.returncode == 1
    assert "no FoM" in r.stdout
