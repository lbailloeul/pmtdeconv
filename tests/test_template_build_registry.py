#!/usr/bin/env python3
"""`template build` config now comes from the ONE dataset registry.

Guards the 2026-07-02 port: ``template_build_impl.config_from_preset``
adapts a DatasetPreset (or a YAML card resolved through the same registry)
into the build-config dict, replacing the impl's retired parallel 8-entry
registry + the command's hand-maintained alias map. Science-bearing fields
must map faithfully; only processing CAPS (max_events / max_files) live in
the impl's small BUILD_CAPS table.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.pipeline.template_build_impl import (  # noqa: E402
    BUILD_CAPS, DEFAULT_CAPS, config_from_preset, configs_from_registry,
)
from pmtdeconv.presets.datasets import DATASETS  # noqa: E402


def test_adapter_maps_science_fields_faithfully():
    """dir/glob/format/polarity/dt/channels/spe_band come 1:1 from the
    registry preset — no drift possible between `hits` and `template build`."""
    for name, ds in DATASETS.items():
        cfg = config_from_preset(ds)
        assert cfg["name"] == ds.name
        assert cfg["dir"] == str(ds.data_dir)
        assert cfg["glob"] == ds.file_glob
        assert cfg["format"] == ds.format
        assert cfg["polarity"] == ds.polarity
        assert cfg["dt"] == ds.dt_ns          # "auto" carried through as-is
        assert cfg["spe_band"] == tuple(ds.spe_band_mv)
        if isinstance(ds.branch_pattern, list):
            assert cfg["channels"] == list(ds.branch_pattern)
        else:
            assert cfg["channels"] == "auto"
        if ds.exclude_glob:
            assert cfg["exclude_glob"] == ds.exclude_glob


def test_adapter_applies_processing_caps():
    """max_events / max_files come from BUILD_CAPS (speed-only knobs)."""
    for name, ds in DATASETS.items():
        cfg = config_from_preset(ds)
        exp_events, exp_files = BUILD_CAPS.get(name, DEFAULT_CAPS)
        assert cfg["max_events"] == exp_events, name
        assert cfg.get("max_files") == exp_files or (
            exp_files is None and "max_files" not in cfg), name


def test_registry_sweep_covers_every_dataset():
    names = [c["name"] for c in configs_from_registry()]
    assert names == list(DATASETS)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
