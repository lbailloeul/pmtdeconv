"""`benchmark denoisers` harness tests (0–20 PE re-scope step 8).

Covers: verdict.json schema/shape, and the anti-`tune denoiser` regression —
the harness must introduce NO private math: the control arm's recorded
metrics must equal a direct `score_run` on the arm's own CSV.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.pipeline.scoring import score_run          # noqa: E402


@pytest.fixture(scope="module")
def bench_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    res = subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "benchmark", "denoisers",
         "--suite", "regimes", "--regimes", "level3",
         "--n-events", "30", "--arms", "control", "wavelet",
         "--out-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    return out


class TestVerdictShape:
    def test_schema_and_structure(self, bench_out):
        v = json.loads((bench_out / "verdict.json").read_text())
        assert v["schema"] == "benchmark-denoisers-2.0"
        assert set(v["conditions"]["arms"]) == {"control", "wavelet"}
        assert v["conditions"]["fft_pad"] is True
        lvl = v["regimes"]["level3"]
        for arm in ("control", "wavelet"):
            m = lvl[arm]
            assert np.isfinite(m["efficiency"])
            assert "timing_sigma_ns" in m and "charge_scatter" in m
            assert Path(m["csv"]).exists()
        assert "head_to_head" in v
        assert (bench_out / "manifest.json").exists()

    def test_control_arm_no_private_math(self, bench_out):
        """The harness's recorded metrics must be exactly `score_run` on the
        arm's own CSV — the defect that killed `tune denoiser` (a private
        measurement loop) must be structurally impossible."""
        v = json.loads((bench_out / "verdict.json").read_text())
        m = v["regimes"]["level3"]["control"]
        csv = Path(m["csv"])
        truth = (REPO_ROOT / "tests" / "fixtures" / "regime" / "level3"
                 / "flattened_Run0.truth.csv")
        direct = score_run(csv, truth, mode="hits")
        for key in ("efficiency", "timing_sigma_ns", "charge_bias",
                    "charge_scatter", "charge_ratio_median"):
            a, b = m[key], direct[key]
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, rel=1e-9), key


# ─── every arm must produce flags `hits` actually accepts (2026-09-18) ──

@pytest.mark.parametrize("arm", ["control", "wavelet", "stft",
                                 "stft-fixed-tang2024"])
def test_every_arm_builds_flags_hits_accepts(arm):
    """Each arm's CLI fragment must parse under the real `hits` parser.

    The module fixture above only runs `control` and `wavelet`, so the two STFT
    arms were never exercised: renaming `--stft-abs-threshold` or the
    `absolute` mode value could break them with the suite still green. Parsing
    the fragment catches a stale flag name or an invalid choice without paying
    for a pipeline run.
    """
    import argparse
    from pmtdeconv.cli import add_common_args
    from pmtdeconv.commands import benchmark_denoisers as bd
    from pmtdeconv.commands import hits

    fragment = bd._arm_cli(arm, shared_cutoff=450.0, fixed_thr=0.02)
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    hits.add_arguments(parser)
    parsed = parser.parse_args(fragment)          # SystemExit on a bad flag
    if arm == "stft-fixed-tang2024":
        assert parsed.stft_threshold_mode == "fixed"
        assert parsed.stft_fixed_threshold == pytest.approx(0.02)


def test_stale_threshold_mode_is_rejected_not_silently_median():
    """`absolute` (the pre-rename spelling) must raise, not fall through.

    The median path is the `else` branch, so an unrecognized mode left in an
    old card or preset would otherwise silently run the wrong algorithm."""
    from pmtdeconv.core.dsp import stft_denoise
    u = np.zeros(1024)
    with pytest.raises(ValueError, match="renamed to 'fixed'"):
        stft_denoise(u, 0.5, threshold_mode="absolute", fixed_threshold=0.02)


# ─── the paper's own method as an arm (2026-09-18) ────────────────

@pytest.fixture(scope="module")
def tang_mc(tmp_path_factory):
    """A small Tang MC: a charge mix (run 0) and one two-hit scan (8 ns)."""
    d = tmp_path_factory.mktemp("tang")
    res = subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "sim", "tang2024", "--n-events", "80",
         "--two-hit-scan", "--per-interval", "60", "--intervals-ns", "8",
         "--seed", "7", "--out-dir", str(d), "--log-level", "quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    return d


@pytest.mark.parametrize("arm,method", [("tang2024", "enhanced"),
                                        ("tang2024-fft", "previous")])
def test_tang2024_arm_reproduces_the_papers_own_numbers(tang_mc, tmp_path,
                                                        arm, method):
    """The tang2024 arm, scored by the harness, must equal the paper module's
    own fig-14 charge figure and fig-12 recognition — exactly.

    The arm is an ADAPTER: the paper chain works in PE, the scorers compare
    pVs against ``true_charge_pVs = n_pe × a_central``. If the PE→pVs scaling
    or the per-hit split were wrong, the harness would silently report a
    different number for the paper's method than the paper module computes —
    and the head-to-head would be comparing against a distorted rival.
    Routed through ``_run_arm`` so the dispatch (not a `hits` call) is covered.
    """
    from pmtdeconv.commands import benchmark_denoisers as bd
    from pmtdeconv.sim.tang2024_paper import load_run as _load_run
    from pmtdeconv.sim.tang2024_paper import (PaperReconstructor,
                                              charge_figure,
                                              recognition_figure)
    from pmtdeconv.sim.tang2024_validate import (per_event_charge_ratio,
                                                 recognition_efficiency)
    recon = PaperReconstructor()

    # fig 14 — charge
    waves, truth = _load_run(tang_mc, 0)
    q_true = truth.groupby("event")["n_pe"].sum().to_dict()
    native = charge_figure(recon, waves, q_true, method)
    csv = bd._run_arm(str(tang_mc), 0, arm, tmp_path / "mix",
                      shared_cutoff=None, fixed_thr=None)
    scored = per_event_charge_ratio(csv, tang_mc / "flattened_Run0.truth.csv")
    assert scored["mean"] == pytest.approx(native["mean"], abs=1e-12)
    assert scored["std"] == pytest.approx(native["std"], abs=1e-12)

    # fig 12 — two-hit recognition at 8 ns
    waves, truth = _load_run(tang_mc, 1008)
    tt = {ev: g["true_center_ns"].to_numpy()
          for ev, g in truth.groupby("event")}
    native_eff = recognition_figure(recon, waves, tt, method)
    csv = bd._run_arm(str(tang_mc), 1008, arm, tmp_path / "scan",
                      shared_cutoff=None, fixed_thr=None)
    eff = recognition_efficiency(csv, tang_mc / "flattened_Run1008.truth.csv")
    eff = eff["efficiency"] if isinstance(eff, dict) else eff
    assert eff == pytest.approx(native_eff, abs=1e-12)


def test_tang2024_arm_is_not_run_outside_the_tang_suite():
    """The paper's reconstructor carries the Tang kernel; on regime fixtures
    or real data it would deconvolve with the wrong response. It must be
    registered as tang-only, and it must never be routed through `hits`."""
    from pmtdeconv.commands import benchmark_denoisers as bd
    for arm in ("tang2024", "tang2024-fft"):
        assert arm in bd.ALL_ARMS and arm in bd.TANG_ONLY_ARMS
        with pytest.raises(ValueError):
            bd._arm_cli(arm, shared_cutoff=None, fixed_thr=None)


def test_published_curve_is_data_keyed_by_the_arms_that_run_it():
    """The fig-12 published curve used to exist only as a code comment; it is
    now data the verdict reports beside the measured curve, keyed by arm."""
    from pmtdeconv.commands import benchmark_denoisers as bd
    from pmtdeconv.sim.tang2024 import TANG_PUBLISHED_RECOGNITION as pub
    assert set(pub) == set(bd.TANG_ONLY_ARMS)
    assert pub["tang2024"][3] == pytest.approx(0.58)   # the contested corner
    assert pub["tang2024-fft"][6] == pytest.approx(0.80)


def test_retired_paper_command_points_at_its_replacement():
    """`sim tang2024-paper` was folded into the benchmark. Its message must
    name the replacement AND the changed flag — the generic "same flags"
    pointer would have been false here."""
    res = subprocess.run([PYBIN, "-m", "pmtdeconv", "sim", "tang2024-paper",
                          "--sim-dir", "x"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert res.returncode != 0
    out = res.stdout + res.stderr
    assert "benchmark denoisers --suite tang2024" in out
    assert "--tang-dir" in out
    assert "Same flags" not in out
