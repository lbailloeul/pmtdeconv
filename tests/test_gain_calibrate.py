"""PE-scale foundation tests (0–20 PE re-scope step 3).

Covers: the 1-PE quantization-peak fitter (`fom.fit_1pe_peak`), the
file-backed gain catalog round-trip, the `Gain.from_settings` resolution
chain, and the `gain calibrate` CLI end-to-end (propose + write).
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
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.fom import fit_1pe_peak                       # noqa: E402
from pmtdeconv.pipeline.gain import Gain                     # noqa: E402
from pmtdeconv.presets import Settings                       # noqa: E402
from pmtdeconv.presets.gain_catalog import (                 # noqa: E402
    GainCalibration, discover_gains, resolve_gain, save_gain_yaml,
)


def _spe_spectrum(rng, n=4000, mu=800.0, som=0.30):
    """Pedestal + 1-PE + small 2-PE charge spectrum (pVs)."""
    ped = np.abs(rng.normal(0.0, 25.0, size=n // 2))
    pe1 = rng.normal(mu, som * mu, size=n)
    pe2 = rng.normal(2 * mu, som * mu * np.sqrt(2), size=n // 6)
    return np.concatenate([ped, pe1, pe2])


class TestFit1PePeak:
    def test_recovers_injected_scale(self):
        rng = np.random.default_rng(7)
        charges = _spe_spectrum(rng, mu=800.0)
        res = fit_1pe_peak(charges)
        assert res["ok"], res
        assert abs(res["mu"] - 800.0) / 800.0 < 0.10, res
        assert res["resolved"], "pedestal valley should be resolved"
        assert 0.15 < res["sigma_over_mu"] < 0.45, res

    def test_explicit_window(self):
        rng = np.random.default_rng(8)
        charges = _spe_spectrum(rng, mu=500.0)
        res = fit_1pe_peak(charges, window=(300.0, 700.0))
        assert res["ok"] and abs(res["mu"] - 500.0) / 500.0 < 0.10, res

    def test_too_few_charges_is_nan(self):
        res = fit_1pe_peak([10.0, 20.0, 30.0])
        assert not res["ok"] and np.isnan(res["mu"])


class TestGainCatalog:
    def test_yaml_roundtrip_and_shadow(self, tmp_path):
        cal = GainCalibration(pmt_type="test_tube",
                              spe_charge_pvs={"d0_ch1": 812.5, "d0_ch2": 790.0},
                              note="unit test", version="2026-07-28")
        p = save_gain_yaml(cal, tmp_path)
        assert p.exists()
        found = discover_gains(tmp_path)
        assert found["test_tube"].spe_charge_pvs["d0_ch1"] == 812.5
        resolved, source = resolve_gain("test_tube", data_root=tmp_path)
        assert resolved is not None and source == "file"
        missing, src = resolve_gain("no_such_tube", data_root=tmp_path)
        assert missing is None and src is None


class TestGainFromSettings:
    def test_explicit_settings_win(self):
        s = Settings(spe_charge_pvs={"d0_ch1": 700.0}, pmt_type="whatever")
        g = Gain.from_settings(s, {"spe_charge_pvs": "cli"})
        assert g.resolved and g.spe_charge("d0_ch1") == 700.0
        assert g.provenance.source == "explicit"

    def test_dataset_layer_provenance(self):
        s = Settings(spe_charge_pvs={"d0_ch1": 650.0})
        g = Gain.from_settings(s, {"spe_charge_pvs": "card"})
        assert g.resolved and g.provenance.source == "card"

    def test_catalog_by_pmt_type(self, tmp_path):
        save_gain_yaml(GainCalibration(
            pmt_type="tube_x", spe_charge_pvs={"d0_ch1": 555.0}), tmp_path)
        s = Settings(pmt_type="tube_x", data_root=tmp_path)
        g = Gain.from_settings(s, {})
        assert g.resolved and g.spe_charge("d0_ch1") == 555.0
        assert g.provenance.source == "catalog"

    def test_unresolved_is_legal(self):
        g = Gain.from_settings(Settings(), {})
        assert not g.resolved
        assert g.spe_charge("d0_ch1") is None
        assert g.provenance.source == "unresolved"


@pytest.fixture()
def charge_csv(tmp_path):
    rng = np.random.default_rng(11)
    rows = []
    for branch, mu in (("d0_ch1", 800.0), ("d0_ch2", 640.0)):
        for q in _spe_spectrum(rng, n=3000, mu=mu):
            rows.append({"branch": branch, "charge": q, "success": True})
    csv = tmp_path / "deconv_pulse.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYBIN, "-m", "pmtdeconv", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


class TestGainCalibrateCLI:
    def test_proposes_card_block(self, tmp_path, charge_csv):
        res = _run_cli("gain", "calibrate", "--dataset", "na22",
                       "--data-root", str(tmp_path), "--runs", "0",
                       "--hits-csv", str(charge_csv),
                       "--out-dir", str(tmp_path / "out"))
        assert res.returncode == 0, res.stderr
        assert "Proposed card block" in res.stdout
        assert "spe_charge:" in res.stdout
        assert (tmp_path / "out" / "manifest.json").exists()

    def test_write_calibration_lands_in_catalog(self, tmp_path, charge_csv):
        res = _run_cli("gain", "calibrate", "--dataset", "na22",
                       "--data-root", str(tmp_path), "--runs", "0",
                       "--hits-csv", str(charge_csv),
                       "--pmt-type", "cli_test_tube",
                       "--write-calibration",
                       "--out-dir", str(tmp_path / "out"))
        assert res.returncode == 0, res.stderr
        cal, source = resolve_gain("cli_test_tube", data_root=tmp_path)
        assert cal is not None and source == "file"
        assert abs(cal.spe_charge_pvs["d0_ch1"] - 800.0) / 800.0 < 0.10
        assert abs(cal.spe_charge_pvs["d0_ch2"] - 640.0) / 640.0 < 0.10
