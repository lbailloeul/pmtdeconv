#!/usr/bin/env python3
"""Veto profiles + high-veto-rate warning + calibrate (step 5).

- the named ``VetoProfile`` catalog carries the exact wavelet/STFT-hits
  thresholds, and ``process_hits`` builds its ``HitsProfile``s from them
  (the splat is wired correctly);
- ``Detector.from_settings`` resolves the expected profile per denoiser;
- the high-veto-rate warning fires above ``--veto-warn-frac`` with a correct
  per-reason breakdown, and is silent below it;
- ``vetoes calibrate`` derives a sane data-driven profile and round-trips it
  through the catalog.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from pmtdeconv.presets import resolve_settings
from pmtdeconv.presets.veto_profiles import VETO_PROFILES
from pmtdeconv.pipeline.detector import Detector
from pmtdeconv.pipeline.process_hits import WAVELET_PROFILE, STFT_PROFILE
from pmtdeconv.pipeline.veto_accounting import compute, format_warning
from pmtdeconv.commands.vetoes_calibrate import _derive, _derive_scored


# ── catalog values (hardcoded expectations: the byte-load-bearing literals) ──

def test_wavelet_hits_profile_values():
    vp = VETO_PROFILES.lookup(name="wavelet_hits")
    assert vp.flat_frac_min == 0.95
    assert vp.no_trigger_amp == 15.0
    assert vp.raw_snr_mult == 10.0
    assert vp.afterpulse_height_ratio == 5.0
    assert vp.min_charge_purity == 0.5
    assert vp.t_peak_min_ns == 20.0


def test_stft_hits_profile_values():
    vp = VETO_PROFILES.lookup(name="stft_hits")
    assert vp.flat_frac_min == 0.50
    assert vp.no_trigger_amp == 1.0
    assert vp.raw_snr_mult == 1.0
    assert vp.afterpulse_height_ratio == 100.0
    assert vp.min_charge_purity == 0.0
    assert vp.t_peak_min_ns == 1.0


def test_hits_profiles_built_from_catalog():
    # process_hits.WAVELET_PROFILE / STFT_PROFILE are splatted from the
    # catalog — guard that the wiring carries every threshold through.
    for name, hp in [("wavelet_hits", WAVELET_PROFILE), ("stft_hits", STFT_PROFILE)]:
        vp = VETO_PROFILES.lookup(name=name)
        for field, value in vp.hits_fields().items():
            assert getattr(hp, field) == value, f"{name}.{field}"


# ── Detector resolves the profile by denoiser ──

def _det(**kw) -> Detector:
    args = argparse.Namespace(**kw)
    settings, prov = resolve_settings(args, require_data_root=False)
    return Detector.from_settings(settings, prov)


def test_detector_resolves_veto_profile():
    # default denoiser is STFT since 2026-10 -> its profile
    assert _det(dataset="na22").veto_profile == "stft_hits"
    assert _det(dataset="na22", denoiser="wavelet").veto_profile == "wavelet_hits"
    assert _det(dataset="na22", denoiser="stft").veto_profile == "stft_hits"
    assert _det(dataset="na22", denoiser="none").veto_profile == "wavelet_hits"


# ── high-veto-rate warning ──

def test_warning_fires_with_breakdown():
    rows = pd.DataFrame(
        [{"success": True, "reason": "no_veto"}] * 10
        + [{"success": False, "reason": "saturated"}] * 80
        + [{"success": False, "reason": "negative_charge"}] * 10
    )
    acc = compute(rows)
    assert acc.total == 100
    assert acc.surviving == 10
    assert acc.vetoed_frac == 0.9
    assert acc.by_reason["saturated"] == 80
    msg = format_warning(acc, 0.5)
    assert msg is not None
    assert "saturated 80%" in msg
    assert "90%" in msg


def test_warning_silent_below_threshold():
    rows = pd.DataFrame(
        [{"success": True, "reason": "no_veto"}] * 90
        + [{"success": False, "reason": "saturated"}] * 10
    )
    acc = compute(rows)
    assert acc.vetoed_frac == 0.1
    assert format_warning(acc, 0.5) is None


def test_accounting_empty_is_safe():
    acc = compute(pd.DataFrame())
    assert acc.total == 0
    assert format_warning(acc, 0.5) is None


# ── vetoes calibrate derivation + round-trip ──

def test_calibrate_derives_and_roundtrips():
    base = VETO_PROFILES.lookup(name="wavelet_hits")
    rng = np.random.default_rng(0)

    def waves():
        for _ in range(60):
            w = rng.normal(0.0, 2.0, 1024)
            w[400:410] += 120.0          # a pulse near sample 400
            yield w, 0.5

    derived, stats = _derive(waves(), base, k_mad=5.0)
    assert stats["sample"] == 60
    assert derived.no_trigger_amp > 0.0
    assert derived.t_peak_min_ns <= derived.t_peak_max_ns
    # the pulse sits at sample ~400 → ~200 ns at dt=0.5
    assert 150.0 < derived.t_peak_min_ns < 250.0
    # inherited (not derived) thresholds are unchanged
    assert derived.afterpulse_height_ratio == base.afterpulse_height_ratio

    VETO_PROFILES.save("test-roundtrip", derived)
    got = VETO_PROFILES.lookup(name="test-roundtrip")
    assert got is not None
    assert got.no_trigger_amp == derived.no_trigger_amp


def test_calibrate_scored_with_truth():
    """The hybrid ground-truth branch: trigger floor retains target_eff of true
    pulses, time window from true centers, purity uses saturated rows as the
    negative class."""
    base = VETO_PROFILES.lookup(name="wavelet_hits")
    n_sig = 20
    # 20 non-saturated true pulses centered ~270-280 ns, raw peaks 150..380,
    # max_val below the rail; + 5 saturated rows (max_val at the rail).
    sig = pd.DataFrame({
        "event": list(range(n_sig)),
        "branch": ["d0_ch1"] * n_sig,
        "raw_peak": np.linspace(150.0, 380.0, n_sig),
        "t_raw_ns": [275.0] * n_sig,
        "max_val": [400.0] * n_sig,
        "true_amp_mv": np.linspace(160.0, 390.0, n_sig),
        "true_center_ns": np.linspace(270.0, 280.0, n_sig),
        "saturated": [False] * n_sig,
    })
    sat = pd.DataFrame({
        "event": list(range(n_sig, n_sig + 5)),
        "branch": ["d0_ch1"] * 5,
        "raw_peak": [950.0] * 5,
        "t_raw_ns": [275.0] * 5,
        "max_val": [960.0] * 5,
        "true_amp_mv": [800.0] * 5,
        "true_center_ns": [275.0] * 5,
        "saturated": [True] * 5,
    })
    full = pd.concat([sig, sat], ignore_index=True)
    stats_df = full[["event", "branch", "raw_peak", "t_raw_ns", "max_val"]]
    truth_df = full[["event", "branch", "true_amp_mv", "true_center_ns",
                     "saturated"]]

    derived, stats = _derive_scored(stats_df, truth_df, base,
                                    target_eff=0.95)
    assert derived is not None
    assert stats["criterion"] == "ground-truth-scored"
    assert stats["n_signal"] == n_sig          # saturated rows excluded
    assert derived.name.endswith("_scored")
    # window brackets the true centers
    assert derived.t_peak_min_ns <= 275.0 <= derived.t_peak_max_ns
    # ~95% of true pulses retained by the floor
    assert stats["achieved_eff"] >= 0.90
    # purity < 1 because the 5 saturated rows survive the floor in-window
    assert 0.0 < stats["purity"] < 1.0
    # inherited thresholds untouched
    assert derived.afterpulse_height_ratio == base.afterpulse_height_ratio

    # too-few-signal guard → (None, None) so run() falls back to data-driven
    tiny = full.head(2)
    d2, s2 = _derive_scored(
        tiny[["event", "branch", "raw_peak", "t_raw_ns", "max_val"]],
        tiny[["event", "branch", "true_amp_mv", "true_center_ns", "saturated"]],
        base, target_eff=0.95)
    assert d2 is None and s2 is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_profiles_saved_before_post_sat_removal_still_load(tmp_path):
    """Profiles written before 2026-10-23 carry the retired post_sat_* keys;
    they load without them, while any other unknown key is still an error."""
    import yaml
    from dataclasses import asdict
    from pmtdeconv.presets.veto_profiles import (BENCHMARK_HITS_PROFILE,
                                                 load_veto_profile_yaml)
    old = asdict(BENCHMARK_HITS_PROFILE)
    old.update(post_sat_window_ns=50.0, post_sat_sigma_mult=3.0, post_sat_offset_ns=20.0)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(old))
    assert load_veto_profile_yaml(path) == BENCHMARK_HITS_PROFILE
    old["post_sat_windw_ns"] = 1.0
    path.write_text(yaml.safe_dump(old))
    with pytest.raises(ValueError, match="unknown field"):
        load_veto_profile_yaml(path)
