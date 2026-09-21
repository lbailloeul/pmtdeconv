"""`select multihit` + `--entry-list` + restored `plot deconv-value` tests
(0–20 PE re-scope step 9)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYBIN, "-m", "pmtdeconv", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def pileup_hits(tmp_path_factory):
    """A pileup_lowpe fixture + one hits run over it."""
    sys.path.insert(0, str(THIS_DIR / "fixtures"))
    import make_synth_root as msr
    base = tmp_path_factory.mktemp("selmh")
    orig = msr.__file__
    msr.__file__ = str(base / "make_synth_root.py")
    try:
        msr.main(regime="pileup_lowpe", n_events=60, write_truth=True,
                 dt_ns=0.5051)
    finally:
        msr.__file__ = orig
    fixture_dir = base / "regime" / "pileup_lowpe"
    out = base / "hits"
    res = _cli("hits", "--dataset", "na22", "--data-root", str(fixture_dir),
               "--runs", "0", "--out-dir", str(out), "--log-level", "quiet")
    assert res.returncode == 0, res.stderr[-1500:]
    return fixture_dir, out


class TestSelector:
    def test_recall_on_true_multihit(self, pileup_hits, tmp_path):
        fixture_dir, hits_out = pileup_hits
        res = _cli("select", "multihit",
                   "--hits-csv", str(hits_out / "all_results.csv"),
                   "--out-dir", str(tmp_path / "sel"))
        assert res.returncode == 0, res.stderr[-1500:]
        sel = pd.read_csv(tmp_path / "sel" / "event_list.csv")
        truth = pd.read_csv(fixture_dir / "flattened_Run0.truth.csv")
        n_true_events = truth["event"].nunique()   # every event is multi-hit
        recall = sel["entry"].nunique() / n_true_events
        assert recall > 0.70, f"selector recall {recall:.2f}"
        assert (sel["n_pulses"] >= 2).all()
        # ch2 mirrors ch1 within a few ns → most entries should pair
        paired_frac = sel[sel["paired"]]["entry"].nunique() / max(
            sel["entry"].nunique(), 1)
        assert paired_frac > 0.5, f"paired fraction {paired_frac:.2f}"
        # clump bounds honour the 11e −30 ns pad
        good = pd.read_csv(hits_out / "all_results.csv")
        good = good[good["success"] == True]       # noqa: E712
        one = sel.iloc[0]
        gt = good[(good["entry"] == one["entry"])
                  & (good["branch"] == one["branch"])]["t_peak"]
        assert one["clump_lo_ns"] <= gt.min() - 29.9

    def test_entry_list_roundtrip(self, pileup_hits, tmp_path):
        fixture_dir, hits_out = pileup_hits
        res = _cli("select", "multihit",
                   "--hits-csv", str(hits_out / "all_results.csv"),
                   "--out-dir", str(tmp_path / "sel2"))
        assert res.returncode == 0, res.stderr[-1500:]
        elist = tmp_path / "sel2" / "event_list.csv"
        listed = set(pd.read_csv(elist)["entry"])
        res = _cli("hits", "--dataset", "na22",
                   "--data-root", str(fixture_dir), "--runs", "0",
                   "--entry-list", str(elist),
                   "--out-dir", str(tmp_path / "re"), "--log-level", "quiet")
        assert res.returncode == 0, res.stderr[-1500:]
        redone = pd.read_csv(tmp_path / "re" / "all_results.csv")
        assert set(redone["entry"]) == listed


class TestDeconvValuePlot:
    def test_restored_figures(self, pileup_hits, tmp_path):
        _fixture_dir, hits_out = pileup_hits
        res = _cli("plot", "deconv-value",
                   "--hits-csv", str(hits_out / "all_results.csv"),
                   "--out-dir", str(tmp_path / "dv"))
        assert res.returncode == 0, res.stderr[-1500:]
        assert (tmp_path / "dv" / "charge_vs_tpeak.png").exists()
        assert (tmp_path / "dv" / "npe_multihit.png").exists()
        assert (tmp_path / "dv" / "manifest.json").exists()
