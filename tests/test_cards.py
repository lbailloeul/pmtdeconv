"""S1: a YAML dataset card resolves to a DatasetPreset and runs byte-identical
to the equivalent in-code seed.

The card is the file-backed twin of a DatasetPreset (presets/cards.py). This
locks the two acceptance criteria: (1) `resolve_dataset(<card path>)` yields a
preset matching the seed field-for-field, and (2) `area --dataset <card>`
produces the SAME bytes as `area --dataset na22 --data-root tests/fixtures`.
"""
from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable

sys.path.insert(0, str(REPO_ROOT))
from pmtdeconv.presets.cards import is_card_reference  # noqa: E402
from pmtdeconv.presets.datasets import get_dataset, resolve_dataset  # noqa: E402


def _write_na22_card(tmp_path: Path) -> Path:
    """A YAML card reproducing the na22 seed, pointed at the test fixtures."""
    seed = get_dataset("na22")
    card = {
        "name": "na22_card",
        "data_dir": str(FIXTURE_DIR),
        "file_glob": seed.file_glob,
        "format": seed.format,
        "tree": seed.tree,
        "branches": list(seed.branch_pattern),
        "dt_ns": seed.dt_ns,
        "polarity": seed.polarity,
        "cosine_cutoff_mhz": seed.cosine_cutoff_mhz,
        "template": str(seed.default_template_csv),
        "spe_band": list(seed.spe_band_mv),
        "pmt_type": seed.pmt_type,
    }
    p = tmp_path / "na22.yaml"
    p.write_text(yaml.safe_dump(card))
    return p


def test_is_card_reference():
    assert not is_card_reference("na22")          # bare registry name
    assert is_card_reference("./x.yaml")
    assert is_card_reference("a.yml")
    assert is_card_reference("some/dir/card")     # has a path separator


def test_card_resolves_to_matching_preset(tmp_path):
    ds = resolve_dataset(str(_write_na22_card(tmp_path)))
    seed = get_dataset("na22")
    for f in ("format", "tree", "branch_pattern", "dt_ns", "polarity",
              "cosine_cutoff_mhz", "spe_band_mv", "file_glob", "pmt_type"):
        assert getattr(ds, f) == getattr(seed, f), f
    assert Path(ds.default_template_csv) == Path(seed.default_template_csv)
    assert Path(ds.data_dir) == FIXTURE_DIR


def test_card_from_directory(tmp_path):
    """Pointing --dataset at a directory loads its dataset.yaml."""
    _write_na22_card(tmp_path)
    (tmp_path / "na22.yaml").rename(tmp_path / "dataset.yaml")
    ds = resolve_dataset(str(tmp_path))
    assert ds.name == "na22_card"
    assert ds.dt_ns == get_dataset("na22").dt_ns


def _run_area(out_dir: Path, *args: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYBIN, "-m", "pmtdeconv", "area", *args,
           "--out-dir", str(out_dir), "--log-level", "quiet"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"area failed:\n{res.stdout}\n{res.stderr}")


def test_card_area_byte_identical_to_seed(tmp_path):
    if not (FIXTURE_DIR / "flattened_Run0.root").exists():
        pytest.skip("synth fixture not built")
    card = _write_na22_card(tmp_path)
    _run_area(tmp_path / "card", "--dataset", str(card), "--runs", "0")
    _run_area(tmp_path / "seed", "--dataset", "na22",
              "--data-root", str(FIXTURE_DIR), "--runs", "0")
    a = tmp_path / "card" / "areas_run0.csv"
    b = tmp_path / "seed" / "areas_run0.csv"
    assert a.exists() and b.exists()
    assert filecmp.cmp(a, b, shallow=False), \
        "card-driven area CSV is not byte-identical to the seed's"


# ─── S2: name-based discovery + seed shadowing + provenance ─────────

def _write_registry_card(tmp_path: Path, name: str, cutoff: float) -> Path:
    """Drop <tmp>/calibrations/datasets/<name>.yaml shadowing a seed."""
    calib = tmp_path / "calibrations" / "datasets"
    calib.mkdir(parents=True, exist_ok=True)
    seed = get_dataset(name)
    card = {
        "name": name, "data_dir": str(FIXTURE_DIR), "format": seed.format,
        "tree": seed.tree, "branches": list(seed.branch_pattern),
        "dt_ns": seed.dt_ns, "polarity": seed.polarity,
        "cosine_cutoff_mhz": cutoff, "spe_band": list(seed.spe_band_mv),
    }
    p = calib / f"{name}.yaml"
    p.write_text(yaml.safe_dump(card))
    return p


def test_registry_card_shadows_seed(tmp_path, monkeypatch):
    from pmtdeconv.presets.datasets import resolve_dataset_with_source
    seed = get_dataset("na22")
    _write_registry_card(tmp_path, "na22", cutoff=123.0)  # != seed's 450
    monkeypatch.setenv("PMTDECONV_DATA_ROOT", str(tmp_path))
    ds, src = resolve_dataset_with_source("na22")
    assert src == "card"
    assert ds.cosine_cutoff_mhz == 123.0          # card value, not seed's
    # Without the calibrations dir, the in-code seed resolves normally.
    monkeypatch.delenv("PMTDECONV_DATA_ROOT")
    ds2, src2 = resolve_dataset_with_source("na22")
    assert src2 == "dataset"
    assert ds2.cosine_cutoff_mhz == seed.cosine_cutoff_mhz


def test_card_shadow_recorded_in_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pmtdeconv.presets import resolve_settings
    _write_registry_card(tmp_path, "na22", cutoff=123.0)
    monkeypatch.setenv("PMTDECONV_DATA_ROOT", str(tmp_path))
    settings, prov = resolve_settings(SimpleNamespace(dataset="na22"),
                                      require_data_root=False)
    assert prov["dataset"] == "card"
    assert settings.cosine_cutoff_mhz == 123.0
    assert prov["cosine_cutoff_mhz"] == "card"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
