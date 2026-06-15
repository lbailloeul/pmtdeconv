#!/usr/bin/env python3
"""Portable data-root resolution (install/onboarding friction fix).

The old ``DATA_DIR = PROJECT_ROOT.parent.parent.parent / "data"`` only resolved
at one checkout depth — on other machines `info datasets` printed nonexistent
paths. Resolution is now: ``PMTDECONV_DATA_ROOT`` env → nearest marker-bearing
``data/`` walking up → legacy fallback. These pin that precedence.
"""
from __future__ import annotations

import sys

import pytest

from pmtdeconv.presets.datasets import _resolve_data_root, _DATA_MARKERS


def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PMTDECONV_DATA_ROOT", str(tmp_path / "mydata"))
    assert _resolve_data_root() == (tmp_path / "mydata")


def test_env_var_expanduser(monkeypatch):
    monkeypatch.setenv("PMTDECONV_DATA_ROOT", "~/somedata")
    got = _resolve_data_root()
    assert "~" not in str(got)
    assert str(got).endswith("somedata")


def test_discovery_finds_marker_bearing_data(tmp_path, monkeypatch):
    monkeypatch.delenv("PMTDECONV_DATA_ROOT", raising=False)
    # repo/pkg/start, with a real data dir two levels up
    real = tmp_path / "data"
    (real / _DATA_MARKERS[0]).mkdir(parents=True)
    start = tmp_path / "repo" / "pkg"
    start.mkdir(parents=True)
    assert _resolve_data_root(start=start) == real


def test_discovery_skips_stray_empty_data(tmp_path, monkeypatch):
    monkeypatch.delenv("PMTDECONV_DATA_ROOT", raising=False)
    # a stray empty data/ next to the package must NOT shadow the real one
    start = tmp_path / "a" / "b" / "pkg"
    start.mkdir(parents=True)
    stray = start.parent / "data"          # data/ with no marker subdir
    stray.mkdir()
    real = tmp_path / "data"               # marker-bearing, higher up
    (real / _DATA_MARKERS[1]).mkdir(parents=True)
    assert _resolve_data_root(start=start) == real


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
