"""S4: `dataset init` scaffolds a card; `dataset lint` validates it.

End-to-end through the CLI: a freshly-scaffolded card lints clean, an
inheriting card lints clean, init refuses to clobber, and lint flags a
broken card (bad polarity/format + unknown settings key).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYBIN, "-m", "pmtdeconv", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_init_then_lint_roundtrip(tmp_path):
    out = tmp_path / "mydet.yaml"
    r = _cli("dataset", "init", "mydet", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    r2 = _cli("dataset", "lint", str(out))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "is valid" in r2.stdout


def test_init_base_card_lints(tmp_path):
    out = tmp_path / "variant.yaml"
    r = _cli("dataset", "init", "variant", "--base", "na22", "--out", str(out))
    assert r.returncode == 0, r.stderr
    raw = yaml.safe_load(out.read_text())
    assert raw["base"] == "na22"
    r2 = _cli("dataset", "lint", str(out))
    assert r2.returncode == 0, r2.stdout + r2.stderr


def test_init_refuses_overwrite(tmp_path):
    out = tmp_path / "x.yaml"
    out.write_text("name: x\n")
    r = _cli("dataset", "init", "x", "--out", str(out))
    assert r.returncode == 1
    assert "refusing" in r.stdout


def test_lint_catches_bad_card(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(
        {"name": "bad", "polarity": 3, "format": "weird",
         "settings": {"not_a_real_field": 1}}))
    r = _cli("dataset", "lint", str(bad))
    assert r.returncode == 1
    assert "polarity" in r.stdout
    assert "unknown Settings field" in r.stdout
