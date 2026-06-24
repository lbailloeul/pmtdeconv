"""S8: file-backed, versioned veto-profile catalog.

A YAML profile under <data_root>/calibrations/veto_profiles/ persists across
runs and SHADOWS the in-code catalog by name (same rule as dataset cards),
carrying a version stamp — while the version never leaks into the golden hits
path (hits_fields drops it).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pmtdeconv.presets.veto_profiles import (  # noqa: E402
    WAVELET_HITS_PROFILE, discover_veto_profiles, load_veto_profile_yaml,
    resolve_veto_profile, save_veto_profile_yaml, veto_profiles_dir)


def test_save_load_roundtrip(tmp_path):
    prof = replace(WAVELET_HITS_PROFILE, name="myprof", version="2026-06-23")
    path = save_veto_profile_yaml(prof, tmp_path)
    assert path == veto_profiles_dir(tmp_path) / "myprof.yaml"
    assert load_veto_profile_yaml(path) == prof


def test_discover_finds_all(tmp_path):
    save_veto_profile_yaml(replace(WAVELET_HITS_PROFILE, name="a"), tmp_path)
    save_veto_profile_yaml(replace(WAVELET_HITS_PROFILE, name="b"), tmp_path)
    assert set(discover_veto_profiles(tmp_path)) == {"a", "b"}


def test_file_shadows_builtin(tmp_path):
    tweaked = replace(WAVELET_HITS_PROFILE, no_trigger_amp=99.0,
                      version="2026-06-23")
    save_veto_profile_yaml(tweaked, tmp_path)
    prof, src = resolve_veto_profile("wavelet_hits", data_root=tmp_path)
    assert src == "file"
    assert prof.no_trigger_amp == 99.0 and prof.version == "2026-06-23"
    # without a data root, the in-code catalog answers
    prof2, src2 = resolve_veto_profile("wavelet_hits")
    assert src2 == "builtin"
    assert prof2.no_trigger_amp == WAVELET_HITS_PROFILE.no_trigger_amp


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_veto_profile("does_not_exist")


def test_version_and_name_not_in_hits_fields():
    fields = WAVELET_HITS_PROFILE.hits_fields()
    assert "version" not in fields and "name" not in fields


def test_load_rejects_unknown_field(tmp_path):
    d = {f: getattr(WAVELET_HITS_PROFILE, f)
         for f in WAVELET_HITS_PROFILE.__dataclass_fields__}
    d["bogus_threshold"] = 1.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        load_veto_profile_yaml(bad)
