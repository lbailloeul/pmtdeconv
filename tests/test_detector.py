#!/usr/bin/env python3
"""Detector resolution (architecture restructure, step 3).

``Detector.from_settings`` replaced the ``_build_legacy_args`` SimpleNamespace
round-trip. These are byte-safety guards: for each (dataset, denoiser) the
resolved scalar config must equal the legacy literals — the same values the
golden CSVs were produced with. The per-(mode, denoiser) veto/baseline constants
still live in ``process_hits.HitsProfile``; the literal guard at the bottom is
the ms-fast canary that step 5 (catalogued ``VetoProfile``) will repoint.
"""
from __future__ import annotations

import argparse
import sys

import pytest

from pmtdeconv.presets import resolve_settings
from pmtdeconv.presets.datasets import CD109_STACKED_UNLOCKED_NA22FMT_CSV
from pmtdeconv.pipeline.detector import Detector
from pmtdeconv.pipeline.process_hits import WAVELET_PROFILE, STFT_PROFILE


def _detector(**kw) -> Detector:
    """Resolve Settings from a minimal argparse Namespace, then build the
    Detector exactly as ``commands/pulse.py`` does."""
    args = argparse.Namespace(**kw)
    settings, prov = resolve_settings(args, require_data_root=False)
    return Detector.from_settings(settings, prov, mode="hits")


def test_na22_wavelet_detector_matches_legacy_literals():
    det = _detector(dataset="na22")
    assert det.cosine_cutoff_mhz == 450.0
    assert det.sat_thresh == 960.0
    assert det.polarity == 1
    assert det.denoiser == "wavelet"
    assert det.no_denoise is False
    assert det.no_vetoes is False
    assert det.peak_order == "auto"
    assert det.wavelet == "sym8"
    assert det.wlevel == 5
    assert det.wmode == "hard"
    assert det.wthresh_k == 1.0
    # dataset default template -> na22fmt schema swap (byte-identity path)
    assert det.fit_csv == str(CD109_STACKED_UNLOCKED_NA22FMT_CSV)
    assert det.template.provenance.source == "dataset"


def test_na22_stft_keeps_dataset_cutoff_not_100():
    # The STFT golden uses na22's 450 MHz cosine cutoff (NOT the STFT
    # god-module's hardcoded 100 MHz default) — regression guard.
    det = _detector(dataset="na22", denoiser="stft")
    assert det.denoiser == "stft"
    assert det.cosine_cutoff_mhz == 450.0
    assert det.sat_thresh == 960.0


def test_na22_none_denoiser():
    det = _detector(dataset="na22", denoiser="none")
    assert det.denoiser == "none"
    assert det.no_denoise is True


def test_na22_3ch_negative_polarity_and_cutoff():
    # na22_3ch has no default template; allow-generic lets the Detector build
    # so we can pin its scalar config (polarity/cutoff/sat).
    det = _detector(dataset="na22_3ch", allow_generic_fallback=True)
    assert det.cosine_cutoff_mhz == 100.0
    assert det.polarity == -1
    assert det.sat_thresh == 495.0
    assert det.template.provenance.source == "generic"


def test_lanl_object_format_carries_through():
    det = _detector(dataset="lanl_835", allow_generic_fallback=True)
    assert det.format == "object"
    assert det.polarity == -1
    assert det.tree == "Events"
    # object dt_ns=="auto" -> placeholder until the command resolves per-file
    assert isinstance(det.dt_ns, float)


def test_runs_default_preserved():
    det = _detector(dataset="na22")
    assert det.runs == [65213]


def test_profile_literals_unchanged_canary():
    # Step-5 canary: when veto/baseline constants move into catalogued
    # VetoProfiles, update this assertion deliberately.
    assert (WAVELET_PROFILE.flat_frac_min, WAVELET_PROFILE.no_trigger_amp,
            WAVELET_PROFILE.afterpulse_height_ratio,
            WAVELET_PROFILE.min_charge_purity) == (0.95, 15.0, 5.0, 0.5)
    assert (STFT_PROFILE.flat_frac_min, STFT_PROFILE.no_trigger_amp,
            STFT_PROFILE.afterpulse_height_ratio,
            STFT_PROFILE.min_charge_purity) == (0.50, 1.0, 100.0, 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
