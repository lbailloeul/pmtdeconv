"""Tang-2024 MC reproduction floors (0–20 PE re-scope step 7).

Small-N acceptance gates for what is SOLID in the reproduction:

- deterministic generation (same seed → same ROOT bytes);
- end-to-end pipeline consumption via the generated card (sampled template,
  tang2024_hits veto profile, exact spe_charge anchor → sane npe_raw);
- the paper's headline charge result: per-waveform rec/true ratio ≈ 1
  (published mean 1.004) with bounded scatter and nonlinearity, measured on
  the plain-FFT arm (200 MHz cosine ≈ their previous-method pulse width);
- the recognition-efficiency machinery (exactly-2 / ±2 ns criterion).

Exact fig-12 recognition-curve parity for the paper-STFT column-selection
arm is a documented Step-11 refinement, NOT a CI gate — the arm exists
(`--stft-threshold-mode absolute`) and the benchmark reports it.
"""
from __future__ import annotations

import filecmp
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

from pmtdeconv.sim import tang2024 as sim                      # noqa: E402
from pmtdeconv.sim.tang2024_validate import (                  # noqa: E402
    nonlinearity_frac, nonlinearity_profile, paper_stft_fixed_threshold,
    per_event_charge_ratio, recognition_efficiency,
)

N_MIX = 250
N_SCAN = 100


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("tang2024")
    rng = np.random.default_rng(42)
    tmpl = sim.write_template_csv(base)
    sim.write_dataset_card(base, tmpl)
    waves, truth = sim.make_mix(rng, N_MIX)
    sim.write_run(base, 0, waves, truth)
    scans = sim.make_two_hit_scan(rng, intervals_ns=(3, 8),
                                  per_interval=N_SCAN)
    for d, (w, t) in scans.items():
        sim.write_run(base, 1000 + d, w, t)
    return base


def _run_hits(sim_dir: Path, out: Path, run: int, *extra: str) -> None:
    res = subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "hits", "--dataset", str(sim_dir),
         "--runs", str(run), "--out-dir", str(out),
         "--log-level", "quiet", *extra],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]


class TestGeneration:
    def test_deterministic_content(self, tmp_path):
        # ROOT containers embed creation timestamps, so determinism is judged
        # on the DECODED waveforms + the truth CSV bytes, not container bytes.
        import uproot
        for sub in ("a", "b"):
            rng = np.random.default_rng(7)
            w, t = sim.make_mix(rng, 20)
            sim.write_run(tmp_path / sub, 0, w, t)
        wa = uproot.open(tmp_path / "a" / "flattened_Run0.root")[
            "waveforms"][sim.BRANCH].array(library="np")
        wb = uproot.open(tmp_path / "b" / "flattened_Run0.root")[
            "waveforms"][sim.BRANCH].array(library="np")
        assert np.array_equal(wa, wb)
        assert filecmp.cmp(tmp_path / "a" / "flattened_Run0.truth.csv",
                           tmp_path / "b" / "flattened_Run0.truth.csv",
                           shallow=False)

    def test_truth_carries_pulse_id_and_npe(self, sim_dir):
        truth = pd.read_csv(sim_dir / "flattened_Run0.truth.csv")
        assert {"pulse_id", "n_pe"}.issubset(truth.columns)
        per_ev = truth.groupby("event")["n_pe"].sum()
        assert per_ev.max() <= 20 * (1 + 5 * sim.SPE_CHARGE_SMEAR)

    def test_paper_threshold_deterministic(self):
        a, b = paper_stft_fixed_threshold(), paper_stft_fixed_threshold()
        assert a == b and a > 0


class TestPipelineConsumption:
    def test_npe_raw_tracks_truth(self, sim_dir, tmp_path):
        _run_hits(sim_dir, tmp_path / "h", 0)
        df = pd.read_csv(tmp_path / "h" / "all_results.csv")
        truth = pd.read_csv(sim_dir / "flattened_Run0.truth.csv")
        est = df.groupby("entry")["npe_raw"].first()
        true_npe = truth.groupby("event")["n_pe"].sum()
        m = pd.concat([est, true_npe], axis=1).dropna()
        m = m[m["n_pe"] > 0.5]
        corr = float(np.corrcoef(m["npe_raw"], m["n_pe"])[0, 1])
        assert corr > 0.90, f"npe_raw–truth correlation {corr:.2f}"
        # everything in this 0–20 PE sample must be flagged in-regime
        assert df["in_regime"].mean() > 0.95


class TestChargeReproduction:
    """The paper's fig-14 headline on the plain-FFT arm (their previous
    method ≈ 200 MHz cosine): mean rec/true ≈ 1.004. Floors margined from
    the 2026-07-29 measurement (400 events: mean 0.962, std 0.220,
    nonlinearity 0.139)."""

    @pytest.fixture(scope="class")
    def fft_csv(self, sim_dir, tmp_path_factory):
        out = tmp_path_factory.mktemp("fft_arm")
        _run_hits(sim_dir, out, 0, "--denoiser", "none",
                  "--cosine-cutoff-mhz", "200")
        return out / "all_results.csv"

    def test_ratio_mean_near_unity(self, fft_csv, sim_dir):
        r = per_event_charge_ratio(fft_csv,
                                   sim_dir / "flattened_Run0.truth.csv")
        assert 0.85 < r["mean"] < 1.10, r
        assert r["std"] < 0.40, r

    def test_nonlinearity_bounded(self, fft_csv, sim_dir):
        prof = nonlinearity_profile(fft_csv,
                                    sim_dir / "flattened_Run0.truth.csv")
        nl = nonlinearity_frac(prof, min_bin_n=8)
        assert nl < 0.30, f"nonlinearity {nl:.3f}"


class TestRecognitionMachinery:
    def test_efficiency_monotone_in_separation(self, sim_dir, tmp_path):
        effs = {}
        for d in (3, 8):
            _run_hits(sim_dir, tmp_path / f"r{d}", 1000 + d,
                      "--denoiser", "none", "--cosine-cutoff-mhz", "200")
            effs[d] = recognition_efficiency(
                tmp_path / f"r{d}" / "all_results.csv",
                sim_dir / f"flattened_Run{1000 + d}.truth.csv")
        assert 0.0 <= effs[3] <= effs[8] <= 1.0, effs
