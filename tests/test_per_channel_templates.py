"""S9: per-channel (per-tube) SPE template resolution.

A card can map a DIFFERENT SPE CSV to each channel so every PMT tube uses its
own transferable artifact; load_templates_by_channel merges them into the
{branch: template} dict the pipeline consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pmtdeconv.io.templates import load_templates_by_channel  # noqa: E402
from pmtdeconv.presets.datasets import resolve_dataset  # noqa: E402


def _write_tube_csv(path: Path, tau_fall: float, branch: str = "d0_ch1") -> None:
    pd.DataFrame([{"channel": branch, "tau_rise_ns": 2.0,
                   "tau_fall_ns": tau_fall, "sigma_ns": 1.0}]).to_csv(
        path, index=False)


def test_per_channel_picks_each_tube(tmp_path):
    a, b = tmp_path / "tubeA.csv", tmp_path / "tubeB.csv"
    _write_tube_csv(a, 20.0, "d0_ch1")     # internal key matches target
    _write_tube_csv(b, 50.0, "d0_ch9")     # single channel, different key
    out = load_templates_by_channel({"d0_ch1": a, "d0_ch2": b}, verbose=False)
    assert out["d0_ch1"]["tau_fall"] == 20.0          # matching key
    assert out["d0_ch2"]["tau_fall"] == 50.0          # single → reassigned


def test_ambiguous_multichannel_raises(tmp_path):
    m = tmp_path / "multi.csv"
    pd.DataFrame([
        {"channel": "d0_ch5", "tau_rise_ns": 2, "tau_fall_ns": 20, "sigma_ns": 1},
        {"channel": "d0_ch6", "tau_rise_ns": 2, "tau_fall_ns": 30, "sigma_ns": 1},
    ]).to_csv(m, index=False)
    with pytest.raises(ValueError):
        load_templates_by_channel({"d0_ch1": m}, verbose=False)


def test_card_maps_per_channel_templates(tmp_path):
    _write_tube_csv(tmp_path / "tubeA.csv", 20.0, "d0_ch1")
    card = tmp_path / "c.yaml"
    card.write_text(yaml.safe_dump(
        {"base": "na22", "templates": {"d0_ch1": "tubeA.csv"}}))
    ds = resolve_dataset(str(card))
    assert ds.template_csv_by_channel is not None
    assert Path(ds.template_csv_by_channel["d0_ch1"]) == tmp_path / "tubeA.csv"
