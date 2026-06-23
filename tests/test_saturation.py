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
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIXTURE_DIR))

DT_NS = 0.5051


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
