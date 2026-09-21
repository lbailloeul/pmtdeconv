#!/usr/bin/env python3
"""``timewalk run`` smoke + content (step 8 test-gap fill).

The timewalk scan had no test. A byte golden is a poor fit here — each scan
bin is a scipy ``curve_fit`` Gaussian, so committed bytes would flap across
scipy versions. Instead we (a) inject a KNOWN timewalk (Δt that rises with
CH2 area) into hand-built area + pulse CSVs and assert ``scan_timewalk``
recovers the trend, and (b) smoke the ``timewalk run`` CLI end-to-end.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmtdeconv.io.schemas import TIMEWALK_SUMMARY_COLUMNS
from pmtdeconv.pipeline.timewalk import scan_timewalk

REPO_ROOT = Path(__file__).resolve().parent.parent
PYBIN = sys.executable

# injected timewalk: CH2-area bin center -> mean Δt (ns), strictly rising
_CENTERS = [25.0, 75.0, 125.0, 175.0]
_MUS = [-8.0, -5.0, -3.0, -2.0]
_PER_BIN = 60
_A_REF_LO, _A_REF_HI = 100.0, 110.0


def _write_csvs(dirpath: Path):
    """Hand-built area + pulse CSVs with a known timewalk. CH1 (ref) area is
    parked in the a_ref window; CH2 (scan) area spans the scan range; Δt =
    t_scan - t_ref follows _MUS(center) + Gaussian noise."""
    rng = np.random.default_rng(0)
    area_rows, pulse_rows = [], []
    ev = 0
    for c, mu in zip(_CENTERS, _MUS):
        for _ in range(_PER_BIN):
            a_scan = float(c + rng.uniform(-10.0, 10.0))
            a_ref = float(rng.uniform(_A_REF_LO + 1, _A_REF_HI - 1))
            t_ref = 0.0
            t_scan = float(mu + rng.normal(0.0, 1.0))
            for br, area, tpk in [("d0_ch1", a_ref, t_ref),
                                  ("d0_ch2", a_scan, t_scan)]:
                area_rows.append((0, 1, ev, br, 0, area, 10.0))
                pulse_rows.append((0, 1, ev, br, True, 1.0, tpk))
            ev += 1
    area_csv = dirpath / "areas_run0.csv"
    pulse_csv = dirpath / "deconv_pulse.csv"
    pd.DataFrame(area_rows, columns=["run", "cycle", "entry", "branch",
                                     "interval", "area", "peak_height"]
                 ).to_csv(area_csv, index=False)
    pd.DataFrame(pulse_rows, columns=["run", "cycle", "entry", "branch",
                                      "success", "peak_id", "t_peak"]
                 ).to_csv(pulse_csv, index=False)
    return area_csv, pulse_csv


def test_scan_recovers_injected_timewalk(tmp_path):
    area_csv, pulse_csv = _write_csvs(tmp_path)
    scan = scan_timewalk(
        area_csv=area_csv, pulse_csv=pulse_csv,
        a_ref=(_A_REF_LO, _A_REF_HI),
        a_scan_start=0.0, a_scan_stop=200.0, a_scan_step=50.0,
        ref_branch="d0_ch1", scan_branch="d0_ch2",
    )
    summ = scan.summary
    # schema + binning
    assert list(TIMEWALK_SUMMARY_COLUMNS)  # columns defined
    for col in ("ch2_center", "mean_dt", "sigma_dt", "N"):
        assert col in summ.columns
    assert len(summ) == 5            # arange(0,200,50) → 5 start bins
    # the 4 populated bins fit; recovered mean_dt rises with CH2 area
    fit = summ[summ["mean_dt"].notna()].sort_values("ch2_center")
    assert len(fit) >= 4
    means = fit["mean_dt"].to_numpy()
    assert np.all(np.diff(means) > 0), f"timewalk not monotonic: {means}"
    # recovered means land near the injected mus (within ~1 ns)
    assert means[0] < -6.0 and means[-1] > -3.5


def test_timewalk_run_cli_smoke(tmp_path):
    area_csv, pulse_csv = _write_csvs(tmp_path)
    out_dir = tmp_path / "tw"
    cmd = [PYBIN, "-m", "pmtdeconv", "timewalk", "run",
           "--dataset", "na22", "--runs", "0",
           "--area-csv", str(area_csv), "--hits-csv", str(pulse_csv),
           "--a-ref", str(_A_REF_LO), str(_A_REF_HI),
           "--a-scan-start", "0", "--a-scan-stop", "200", "--a-scan-step", "50",
           "--out-dir", str(out_dir), "--log-level", "quiet"]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"timewalk run failed:\n{r.stdout}\n{r.stderr}"
    summary = out_dir / "timewalk_summary_run0.csv"
    assert summary.exists()
    df = pd.read_csv(summary)
    assert "mean_dt" in df.columns and len(df) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
