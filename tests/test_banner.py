"""S10: the run banner surfaces the resolved config + provenance to stdout."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pmtdeconv.io.banner import format_run_banner  # noqa: E402
from pmtdeconv.presets import resolve_settings  # noqa: E402


def test_banner_shows_dataset_and_seed_source():
    s, prov = resolve_settings(SimpleNamespace(dataset="na22"),
                               require_data_root=False)
    b = format_run_banner("area", s, prov)
    assert "pmtdeconv area" in b
    assert "na22" in b
    assert "[dataset]" in b          # in-code seed source tag
    assert "pmtdeconv" in b          # version/git tail line


def test_banner_flags_card_source_and_fom(tmp_path):
    card = tmp_path / "c.yaml"
    card.write_text(yaml.safe_dump(
        {"base": "na22", "cosine_cutoff_mhz": 123.0,
         "fom": "photopeak_sigma_over_mu"}))
    s, prov = resolve_settings(SimpleNamespace(dataset=str(card)),
                               require_data_root=False)
    b = format_run_banner("area", s, prov)
    assert "[card]" in b             # card-sourced run is flagged, never silent
    assert "123" in b                # the overridden cutoff is shown
    assert "photopeak_sigma_over_mu" in b   # declared FoM surfaced
