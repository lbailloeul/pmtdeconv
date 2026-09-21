"""Tune-CLI smoke tests (PR 10).

Each tune leaf is exercised on a tiny grid (1 regime, ≤4 cells) so the
CI budget stays under ~25 s. We don't assert the winning config bytes
— those depend on RNG noise on the small fixture — but we confirm that
the recipe.json + sweep_summary.csv + at least one sweep curve land
and have the right shape.

A separate ``test_truth_schema.py``-style test validates the truth CSV
columns produced by ``make_synth_root.py --regime level<N>``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYBIN = sys.executable


def _run_pmtdeconv(*args: str) -> str:
    """Invoke the CLI; return stdout. Raises on non-zero exit."""
    cmd = [PYBIN, "-m", "pmtdeconv", *args, "--log-level", "quiet"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"pmtdeconv {' '.join(args)} failed (exit {res.returncode}):\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return res.stdout


def test_tune_cosine_smoke(tmp_path):
    """``pmtdeconv tune cosine`` on level3 × 2 cutoffs emits the
    expected output set."""
    out_dir = tmp_path / "tune_cosine"
    _run_pmtdeconv(
        "tune", "cosine",
        "--dataset", "na22",
        "--regimes", "level3",
        "--cutoffs", "100", "450",
        "--denoiser", "none",
        "--n-events", "20",
        "--out-dir", str(out_dir),
    )
    recipe = json.loads((out_dir / "recipe.json").read_text())
    assert recipe["swept_axis"] == "cosine_cutoff_mhz"
    assert recipe["denoiser"] == "none"
    assert "level3" in recipe["regimes"]
    assert "cosine_cutoff_mhz" in recipe["regimes"]["level3"]["best_params"]
    # efficiency-first selection: max efficiency, drift band, then min width.
    assert recipe["selection_method"] == "effmax_driftband_minwidth"
    assert recipe["charge_tol"] == 0.02
    assert recipe["min_efficiency_frac"] == 1.0
    assert recipe["regimes"]["level3"]["selection"] in (
        "effmax_driftband_minwidth", "effmax_mindrift", "score_fallback")
    summary = pd.read_csv(out_dir / "sweep_summary.csv")
    assert len(summary) == 2
    # width_fwhm_ns is the secondary objective — must be measured + emitted.
    assert {"regime", "cosine_cutoff_mhz", "charge_scatter", "efficiency",
            "charge_bias", "score", "width_fwhm_ns"}.issubset(summary.columns)
    # single ranked-metrics figure (efficiency / drift / FWHM)
    assert (out_dir / "sweep_curves_level3.png").exists()


def test_tune_wavelet_smoke(tmp_path):
    """``pmtdeconv tune wavelet`` on level3 × 2-cell grid."""
    out_dir = tmp_path / "tune_wavelet"
    _run_pmtdeconv(
        "tune", "wavelet",
        "--dataset", "na22",
        "--regimes", "level3",
        "--wavelets", "sym8",
        "--wlevels", "5",
        "--wmodes", "hard",
        "--wks", "1.0", "2.0",
        "--n-events", "20",
        "--out-dir", str(out_dir),
    )
    recipe = json.loads((out_dir / "recipe.json").read_text())
    assert recipe["swept_axis"] == "wavelet_config"
    assert "level3" in recipe["regimes"]
    summary = pd.read_csv(out_dir / "sweep_summary.csv")
    assert len(summary) == 2


def test_tune_stft_smoke(tmp_path):
    """``pmtdeconv tune stft`` on level3 × 1-cell grid."""
    out_dir = tmp_path / "tune_stft"
    _run_pmtdeconv(
        "tune", "stft",
        "--dataset", "na22",
        "--regimes", "level3",
        "--nperseg-ns", "20",
        "--freq-cutoff-mhz", "175",
        "--amp-thresholds", "10",
        "--n-events", "20",
        "--out-dir", str(out_dir),
    )
    recipe = json.loads((out_dir / "recipe.json").read_text())
    assert recipe["swept_axis"] == "stft_config"


def test_tune_denoiser_is_removed(tmp_path):
    """``tune denoiser`` was removed 2026-09-14 (superseded by
    ``benchmark denoisers``). It must fail with the migration instruction,
    not a bare argparse "invalid choice" — difflib cannot bridge the rename.
    """
    import subprocess
    res = subprocess.run(
        [PYBIN, "-m", "pmtdeconv", "tune", "denoiser", "--dataset", "na22"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "benchmark denoisers" in combined, (
        f"removal message must name the replacement; got:\n{combined}")


def test_tune_dry_run(tmp_path):
    """All leaves respect --dry-run (no ROOT I/O, manifest only)."""
    for leaf in ("cosine", "wavelet", "stft"):
        out_dir = tmp_path / f"dry_{leaf}"
        _run_pmtdeconv("tune", leaf,
                       "--dataset", "na22",
                       "--out-dir", str(out_dir),
                       "--dry-run")
        assert (out_dir / "manifest.json").exists()
        # No sweep_summary because no cells ran
        assert not (out_dir / "sweep_summary.csv").exists()


def test_write_presets_idempotent(tmp_path):
    """Running tune --write-presets twice on the same date produces a
    file with exactly one entry per (regime, date) — no duplicates."""
    out_dir = tmp_path / "writepre"
    # First run
    _run_pmtdeconv(
        "tune", "cosine",
        "--dataset", "na22",
        "--regimes", "level3",
        "--cutoffs", "100", "450",
        "--denoiser", "none",
        "--n-events", "20",
        "--out-dir", str(out_dir),
        "--write-presets",
    )
    tuned_path = REPO_ROOT / "pmtdeconv" / "presets" / "tuned_configs.py"
    src_after_1 = tuned_path.read_text()
    # Count occurrences of the would-be preset name
    import re
    name_pattern = r"na22-cosine-tuned-\d{4}-\d{2}-\d{2}-level3"
    matches_1 = re.findall(name_pattern, src_after_1)
    # Re-run — must NOT duplicate
    _run_pmtdeconv(
        "tune", "cosine",
        "--dataset", "na22",
        "--regimes", "level3",
        "--cutoffs", "100", "450",
        "--denoiser", "none",
        "--n-events", "20",
        "--out-dir", str(out_dir),
        "--write-presets",
    )
    src_after_2 = tuned_path.read_text()
    matches_2 = re.findall(name_pattern, src_after_2)
    # Same count → idempotent
    assert len(matches_2) == len(matches_1)


# ─── _collapse_tuned retention policy (2026-07-02) ───────────────────
# Dated preset names are immutable provenance anchors: manifests record
# them, so a re-run that reproduces the current winner must be a NO-OP.
# (The pre-2026-07-02 "newest wins" retention re-stamped the entry with
# the re-run's date — the old name vanished and every manifest written
# against it stopped resolving.)

def _entry(name, cutoff, base="na22"):
    return {"name": name, "base_dataset": base,
            "overrides": {"cosine_cutoff_mhz": cutoff},
            "description": f"Auto-tuned ({name})."}


def test_collapse_rerun_same_winner_is_noop():
    """Reproducing the current winner on a later date keeps the ORIGINAL
    dated name (first appearance) and leaves the alias pointing at it."""
    from pmtdeconv.commands._tune_shared import _collapse_tuned
    merged = {
        "na22-cosine-tuned-2026-06-14-level3": _entry(
            "na22-cosine-tuned-2026-06-14-level3", 100.0),
        "na22-cosine-tuned-2026-07-01-level3": _entry(
            "na22-cosine-tuned-2026-07-01-level3", 100.0),
    }
    out = _collapse_tuned(merged)
    assert "na22-cosine-tuned-2026-06-14-level3" in out
    assert "na22-cosine-tuned-2026-07-01-level3" not in out
    alias = out["na22-cosine-tuned-latest-level3"]
    assert "na22-cosine-tuned-2026-06-14-level3" in alias["description"]


def test_collapse_new_winner_appends_and_moves_alias():
    """A genuinely different result gets its own dated entry; the alias
    moves to it; the older dated entry SURVIVES (audit trail)."""
    from pmtdeconv.commands._tune_shared import _collapse_tuned
    merged = {
        "na22-cosine-tuned-2026-06-14-level3": _entry(
            "na22-cosine-tuned-2026-06-14-level3", 100.0),
        "na22-cosine-tuned-2026-07-01-level3": _entry(
            "na22-cosine-tuned-2026-07-01-level3", 450.0),
    }
    out = _collapse_tuned(merged)
    assert "na22-cosine-tuned-2026-06-14-level3" in out
    assert "na22-cosine-tuned-2026-07-01-level3" in out
    alias = out["na22-cosine-tuned-latest-level3"]
    assert alias["overrides"] == {"cosine_cutoff_mhz": 450.0}
    assert "na22-cosine-tuned-2026-07-01-level3" in alias["description"]


def test_collapse_alias_never_references_collapsed_name():
    """When the newest dated name collapses into an older twin, the alias
    must reference the RETAINED name, never the deleted one."""
    from pmtdeconv.commands._tune_shared import _collapse_tuned
    merged = {
        "na22-cosine-tuned-2026-05-31-level3": _entry(
            "na22-cosine-tuned-2026-05-31-level3", 450.0),
        "na22-cosine-tuned-2026-06-14-level3": _entry(
            "na22-cosine-tuned-2026-06-14-level3", 100.0),
        "na22-cosine-tuned-2026-07-01-level3": _entry(
            "na22-cosine-tuned-2026-07-01-level3", 100.0),
    }
    out = _collapse_tuned(merged)
    dated_kept = {n for n in out if "-latest-" not in n}
    alias = out["na22-cosine-tuned-latest-level3"]
    referenced = [n for n in dated_kept if n in alias["description"]]
    assert referenced, "alias must reference a retained dated preset"
