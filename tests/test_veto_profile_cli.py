"""Fair-benchmark plumbing tests (0–20 PE re-scope step 5).

Covers: the `benchmark_hits` neutral profile values (user-approved
2026-07-28), `HitsProfile.from_veto_profile` behavior-switch policy,
cross-application of a named profile to either denoiser, file-backed profile
shadowing, and explicit-profile flag honoring on the STFT path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
FIXTURE_DIR = THIS_DIR / "fixtures"
SYNTH_ROOT = FIXTURE_DIR / "flattened_Run0.root"
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.pipeline.pipeline import (            # noqa: E402
    HitsProfile, STFT_PROFILE, WAVELET_PROFILE,
)
from pmtdeconv.presets.veto_profiles import (        # noqa: E402
    BENCHMARK_HITS_PROFILE, VETO_PROFILES, WAVELET_HITS_PROFILE,
    save_veto_profile_yaml,
)


class TestBenchmarkProfile:
    def test_approved_values(self):
        p = BENCHMARK_HITS_PROFILE
        assert p.no_trigger_amp == 5.0
        assert p.afterpulse_height_ratio == 20.0
        assert p.min_charge_purity == 0.25
        # everything else must equal the strict wavelet profile
        strict = asdict(WAVELET_HITS_PROFILE)
        bench = asdict(p)
        for k in strict:
            if k in ("name", "version", "no_trigger_amp",
                     "afterpulse_height_ratio", "min_charge_purity"):
                continue
            assert bench[k] == strict[k], f"{k} unexpectedly differs"

    def test_registered(self):
        assert "benchmark_hits" in VETO_PROFILES.names()


class TestFromVetoProfile:
    def test_default_switches_match_legacy(self):
        w = HitsProfile.from_veto_profile(WAVELET_HITS_PROFILE,
                                          denoiser_kind="wavelet")
        assert w == WAVELET_PROFILE
        s = HitsProfile.from_veto_profile(
            VETO_PROFILES.lookup(name="stft_hits"), denoiser_kind="stft")
        assert s == STFT_PROFILE



def _run_hits(out_dir: Path, *extra: str, data_root: Path = FIXTURE_DIR
              ) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "hits", "--dataset", "na22",
         "--data-root", str(data_root), "--runs", "0",
         "--out-dir", str(out_dir), "--log-level", "quiet", *extra],
        cwd=REPO_ROOT, capture_output=True, text=True)


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text())


@pytest.mark.skipif(not SYNTH_ROOT.exists(),
                    reason="synth fixture missing (make_synth_root.py)")
class TestVetoProfileCLI:
    def test_cross_application_wavelet_under_stft_profile(self, tmp_path):
        res = _run_hits(tmp_path / "x", "--denoiser", "wavelet",
                        "--veto-profile", "stft_hits")
        assert res.returncode == 0, res.stderr
        m = _manifest(tmp_path / "x")
        assert m["stats"]["veto_profile"] == "stft_hits"
        assert m["stats"]["veto_profile_source"] == "builtin"
        # relaxed thresholds admit more events than the strict default
        dflt = _run_hits(tmp_path / "y", "--denoiser", "wavelet")
        assert dflt.returncode == 0, dflt.stderr
        n_relaxed = len(pd.read_csv(tmp_path / "x" / "all_results.csv"))
        n_strict = len(pd.read_csv(tmp_path / "y" / "all_results.csv"))
        assert n_relaxed >= n_strict

    def test_benchmark_profile_runs_both_denoisers(self, tmp_path):
        for tag, extra in (("w", ()), ("s", ("--denoiser", "stft"))):
            res = _run_hits(tmp_path / tag, "--veto-profile",
                            "benchmark_hits", *extra)
            assert res.returncode == 0, res.stderr
            m = _manifest(tmp_path / tag)
            assert m["stats"]["veto_profile"] == "benchmark_hits"

    def test_stft_honors_no_vetoes_without_explicit_profile(self, tmp_path):
        # Until 2026-10 the STFT path silently ignored --no-vetoes (and
        # --peak-order) unless a profile was named. It is the default
        # denoiser now, so it must honor them on its own.
        res = _run_hits(tmp_path / "nv", "--denoiser", "stft", "--no-vetoes")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "nv" / "all_results.csv")
        assert (df["reason"] == "no_veto").any()

    def test_stft_honors_peak_order(self, tmp_path):
        res = _run_hits(tmp_path / "po", "--denoiser", "stft", "--no-vetoes",
                        "--peak-order", "tallest")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "po" / "all_results.csv")
        ok = df[df["success"] == True]                     # noqa: E712
        multi = ok.groupby(["entry", "branch"]).filter(lambda g: len(g) > 1)
        assert len(multi), "fixture must have multi-peak waveforms"
        for _, g in multi.groupby(["entry", "branch"]):
            heights = g.sort_values("peak_id")["height"].to_numpy()
            assert (heights[:-1] >= heights[1:]).all(), "peak_id 1 must be tallest"

    def test_stft_honors_no_vetoes_under_explicit_profile(self, tmp_path):
        # An explicitly named profile must honor it too — rows then carry
        # reason="no_veto".
        res = _run_hits(tmp_path / "nv", "--denoiser", "stft",
                        "--veto-profile", "benchmark_hits", "--no-vetoes")
        assert res.returncode == 0, res.stderr
        df = pd.read_csv(tmp_path / "nv" / "all_results.csv")
        assert (df["reason"] == "no_veto").any()

    def test_file_backed_profile_shadows(self, tmp_path):
        data_root = tmp_path / "root"
        data_root.mkdir()
        shutil.copy(SYNTH_ROOT, data_root / "flattened_Run0.root")
        custom = replace(WAVELET_HITS_PROFILE, name="custom_prof",
                         version="test-1", no_trigger_amp=3.0)
        save_veto_profile_yaml(custom, data_root)
        res = _run_hits(tmp_path / "o", "--veto-profile", "custom_prof",
                        data_root=data_root)
        assert res.returncode == 0, res.stderr
        m = _manifest(tmp_path / "o")
        assert m["stats"]["veto_profile"] == "custom_prof"
        assert m["stats"]["veto_profile_source"] == "file"
        assert m["stats"]["veto_profile_version"] == "test-1"

    def test_unknown_profile_errors_with_names(self, tmp_path):
        res = _run_hits(tmp_path / "u", "--veto-profile", "no_such_profile")
        assert res.returncode != 0
        assert "benchmark_hits" in (res.stderr + res.stdout)
