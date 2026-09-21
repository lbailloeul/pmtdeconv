"""Validate the truth CSV schema produced by `make_synth_root.py --regime`.

Tiny test: build the level1 fixture into tmp_path, confirm the truth
CSV has the required columns + expected row count + plausible values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = THIS_DIR / "fixtures"


@pytest.fixture(scope="module")
def msr_module():
    """Import the fixture builder (lives outside the package)."""
    sys.path.insert(0, str(FIXTURES_DIR))
    try:
        import make_synth_root
        return make_synth_root
    finally:
        if str(FIXTURES_DIR) in sys.path:
            sys.path.remove(str(FIXTURES_DIR))


def test_truth_csv_has_required_columns(tmp_path, monkeypatch, msr_module):
    """Build level3 with 5 events into tmp; validate truth columns + count."""
    # Redirect the builder's output by patching its FIXTURES_DIR resolver
    monkeypatch.setattr(msr_module, "__file__",
                        str(tmp_path / "make_synth_root.py"))
    msr_module.main(regime="level3", n_events=5, write_truth=True)
    truth_path = tmp_path / "regime" / "level3" / "flattened_Run0.truth.csv"
    assert truth_path.exists(), f"truth CSV not at {truth_path}"
    df = pd.read_csv(truth_path)
    required = {"event", "branch", "true_amp_mv", "true_center_ns",
                "true_charge_pVs", "true_sigma_ns", "saturated"}
    assert required.issubset(df.columns)
    # 5 events × 2 branches = 10 rows
    assert len(df) == 10
    # level3 amp range U(80, 200) — sanity check
    ch1 = df[df.branch == "d0_ch1"]
    assert (ch1["true_amp_mv"] >= 80).all() and (ch1["true_amp_mv"] <= 200).all()
    # No saturation expected at level3
    assert not df["saturated"].any()


def test_level5_marks_saturated(tmp_path, monkeypatch, msr_module):
    """level5 (amp U(500, 1100), clip=950) should mark events whose
    amp exceeds 950 as saturated. With 20 events on ch1, expect ~5."""
    monkeypatch.setattr(msr_module, "__file__",
                        str(tmp_path / "make_synth_root.py"))
    msr_module.main(regime="level5", n_events=20, write_truth=True)
    truth_path = tmp_path / "regime" / "level5" / "flattened_Run0.truth.csv"
    df = pd.read_csv(truth_path)
    ch1 = df[df.branch == "d0_ch1"]
    # Some rows must be saturated; their amp must exceed 950
    sat = ch1[ch1["saturated"]]
    assert len(sat) > 0, "expected ≥1 saturated event on ch1 at level5"
    assert (sat["true_amp_mv"] > 950).all()


def test_dt_ns_propagates(tmp_path, monkeypatch, msr_module):
    """Truth CSV's true_sigma_ns scales with dt_ns (Q6 cross-dataset)."""
    monkeypatch.setattr(msr_module, "__file__",
                        str(tmp_path / "make_synth_root.py"))
    msr_module.main(regime="level3", n_events=3, write_truth=True,
                    dt_ns=1.25)
    truth_path = tmp_path / "regime" / "level3" / "flattened_Run0.truth.csv"
    import numpy as np
    from regimes import BOX_WIDTH_SAMP
    df = pd.read_csv(truth_path)
    # box⊛SPE model: true_sigma_ns carries the PE-box width in ns,
    # which scales with dt: BOX_WIDTH_SAMP * 1.25.
    assert np.allclose(df["true_sigma_ns"], BOX_WIDTH_SAMP * 1.25)
    # Sentinel file recorded the dt
    sentinel = tmp_path / "regime" / "level3" / ".dt_ns"
    assert sentinel.exists()
    assert float(sentinel.read_text().strip()) == 1.25


def test_pulse_id_roundtrip_via_save_truth(tmp_path):
    """v1.1: save_truth PRESERVES the optional pulse_id / n_pe columns
    (pre-1.1 it silently dropped them, making multi-pulse truth unscorable)."""
    from pmtdeconv.io.truth import (OPTIONAL_COLUMNS, REQUIRED_COLUMNS,
                                    TRUTH_SCHEMA_VERSION, load_truth,
                                    save_truth)
    assert TRUTH_SCHEMA_VERSION == "1.1"
    assert OPTIONAL_COLUMNS == ["pulse_id", "n_pe"]
    df = pd.DataFrame([
        {"event": 0, "branch": "d0_ch1", "true_amp_mv": 50.0,
         "true_center_ns": 200.0, "true_charge_pVs": 900.0,
         "true_sigma_ns": 6.0, "saturated": False, "pulse_id": 0, "n_pe": 3},
        {"event": 0, "branch": "d0_ch1", "true_amp_mv": 30.0,
         "true_center_ns": 212.0, "true_charge_pVs": 540.0,
         "true_sigma_ns": 6.0, "saturated": False, "pulse_id": 1, "n_pe": 2},
    ])
    p = save_truth(df, tmp_path / "t.csv")
    back = load_truth(p)
    assert "pulse_id" in back.columns and "n_pe" in back.columns
    assert list(back["pulse_id"]) == [0, 1]
    # required-only frames still round-trip (optional columns truly optional)
    p2 = save_truth(df[REQUIRED_COLUMNS], tmp_path / "t2.csv")
    assert "pulse_id" not in load_truth(p2).columns


def test_pileup_dense_regime_two_pulses(tmp_path, monkeypatch, msr_module):
    """pileup_dense: exactly two pulses per event, separations 2-10 ns —
    the resolving-power-scan input (0–20 PE re-scope)."""
    monkeypatch.setattr(msr_module, "__file__",
                        str(tmp_path / "make_synth_root.py"))
    msr_module.main(regime="pileup_dense", n_events=10, write_truth=True)
    truth_path = (tmp_path / "regime" / "pileup_dense"
                  / "flattened_Run0.truth.csv")
    df = pd.read_csv(truth_path)
    ch1 = df[df.branch == "d0_ch1"]
    per_event = ch1.groupby("event").size()
    assert (per_event == 2).all(), per_event.to_dict()
    seps = ch1.sort_values(["event", "pulse_id"]).groupby("event")[
        "true_center_ns"].diff().dropna()
    assert (seps >= 2.0 - 1e-9).all() and (seps <= 10.0 + 1e-9).all()
