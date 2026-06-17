#!/usr/bin/env python3
"""Layout-flexible run-file resolution + fail-fast on missing inputs.

Regression for the collaborator failure: `run na22-timewalk` cascaded into a
confusing `timewalk run: "Could not resolve --area-csv"` because the data file
existed but not in the expected layout, and the missing-file path silently
produced empty output. These tests pin: (a) the resolver finds a run's file in
each common layout; (b) the not-found error names what was searched + the
remedies; (c) the recipe now fails fast at the real step with that message.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pmtdeconv.io.data_paths import (
    not_found_message,
    resolve_run_file,
    resolve_runs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYBIN = sys.executable


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


# ── flat (flattened_Run<N>.root) ──────────────────────────────────

def test_flat_at_root(tmp_path):
    f = _touch(tmp_path / "flattened_Run5.root")
    assert resolve_run_file(tmp_path, 5, kind="flat") == f


def test_flat_in_subdir(tmp_path):
    f = _touch(tmp_path / "sub" / "flattened_Run5.root")
    assert resolve_run_file(tmp_path, 5, kind="flat") == f


# ── drs (Run<N>.root) ─────────────────────────────────────────────

def test_drs_under_processed(tmp_path):
    f = _touch(tmp_path / "processed" / "Run7.root")
    assert resolve_run_file(tmp_path, 7, kind="drs") == f


def test_drs_at_root(tmp_path):
    f = _touch(tmp_path / "Run7.root")
    assert resolve_run_file(tmp_path, 7, kind="drs") == f


def test_drs_by_glob(tmp_path):
    # file exists but with a non-canonical name — the *Run<N>*.root glob finds it
    f = _touch(tmp_path / "mqana_Run7_v2.root")
    assert resolve_run_file(tmp_path, 7, kind="drs") == f


def test_drs_glob_skips_flattened(tmp_path):
    # regression: the fuzzy *Run<N>*.root glob must NOT match the flat waveform
    # file (flattened_Run<N>.root) — it has no DRS area branches, so returning
    # it makes `raw-area drs` open the wrong file and choke on a missing tree.
    _touch(tmp_path / "flattened_Run7.root")
    assert resolve_run_file(tmp_path, 7, kind="drs") is None
    # but the flat resolver still finds it
    assert resolve_run_file(tmp_path, 7, kind="flat") == tmp_path / "flattened_Run7.root"


# ── explicit override + misses ────────────────────────────────────

def test_explicit_wins_when_it_exists(tmp_path):
    f = _touch(tmp_path / "elsewhere" / "custom.root")
    assert resolve_run_file(tmp_path, 9, kind="drs", explicit=str(f)) == f


def test_explicit_missing_returns_none(tmp_path):
    assert resolve_run_file(tmp_path, 9, kind="drs",
                            explicit=str(tmp_path / "nope.root")) is None


def test_missing_returns_none(tmp_path):
    assert resolve_run_file(tmp_path, 9, kind="flat") is None


# ── actionable error message ──────────────────────────────────────

def test_message_names_searched_paths_and_remedies(tmp_path):
    msg = not_found_message(tmp_path, 5, kind="flat")
    assert "flattened_Run5.root" in msg            # what it looked for
    assert "--data-root" in msg                    # the remedies
    assert "PMTDECONV_DATA_ROOT" in msg


def test_drs_message_mentions_root_flag(tmp_path):
    msg = not_found_message(tmp_path, 5, kind="drs")
    assert "--root" in msg and "Run5.root" in msg


# ── resolve_runs: all-missing raises, partial warns ───────────────

def test_resolve_runs_all_missing_raises(tmp_path):
    with pytest.raises(SystemExit) as exc:
        resolve_runs(tmp_path, [1, 2], kind="flat")
    assert "--data-root" in str(exc.value)


def test_resolve_runs_partial_warns_and_returns_found(tmp_path, capsys):
    _touch(tmp_path / "flattened_Run1.root")
    found = resolve_runs(tmp_path, [1, 2], kind="flat")
    assert [r for r, _ in found] == [1]
    assert "run 2" in capsys.readouterr().out


# ── recipe fails fast at the real step (not the downstream cascade) ─

def test_recipe_fails_fast_on_missing_data(tmp_path):
    """`run na22-timewalk` against an empty data root must abort at the
    raw-area step with the actionable message — NOT the misleading
    `timewalk run: "Could not resolve --area-csv"`."""
    r = subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "run", "na22-timewalk",
         "--dataset", "na22", "--runs", "0",
         "--data-root", str(tmp_path),
         "--out-dir", str(tmp_path / "out"), "--log-level", "quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "Could not resolve --area-csv" not in combined
    assert "--data-root" in combined or "not found" in combined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
