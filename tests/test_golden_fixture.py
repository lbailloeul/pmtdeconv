#!/usr/bin/env python3
"""Regression tests against deterministic synthetic ROOT fixtures.

Guards against accidental byte drift in CSV outputs across refactors. Each
fixture is built from a fixed RNG seed by a ``tests/fixtures/make_synth_*.py``
script; the expected outputs live under ``tests/fixtures/golden/``.

**Neither the ROOT fixtures nor the goldens are committed** — both are
gitignored. The ROOT files are large and regenerable; the golden bytes depend
on the local numpy/scipy build, so a shared reference set would fail on other
machines for reasons that have nothing to do with the code. Each checkout
blesses its own set once, and byte-identity then guards refactors *within*
that checkout.

On a fresh checkout, before running this suite::

    python tests/fixtures/bless_goldens.py

That builds any missing ROOT fixtures and regenerates all 12 goldens. Every
recipe lives in :mod:`tests.fixtures.golden_spec` — the single source of
truth, kept honest by ``test_golden_spec_matches_tests`` below.

A failure here means *something changed the bytes of a CSV output*. Either:

1. You introduced a real regression — fix the code, do NOT re-bless.
2. The change was deliberate (schema bump, defaults shift) — re-bless with
   ``python tests/fixtures/bless_goldens.py --force``, diff the result to
   confirm every changed byte is intended, bump the matching schema version in
   ``pmtdeconv/io/schemas.py``, and add an entry to
   ``tests/fixtures/GOLDEN_PROVENANCE.md`` (tracked; the bytes are not).

Coverage (12 goldens over four dataset formats):

- flat (``flattened_Run0.root``) — ``raw-area mqana``, ``area`` (padded and
  unpadded), ``hits`` (preset / vetoes-on / STFT)
- object (``MilliQan_Run835.99_triple.root``) — ``raw-area mqana``, ``area``,
  ``hits``
- DRSprocessing (``processed/Run0.root``) — ``raw-area drs``
- offline pulse list (``MilliQan_Run99.synth_triple_offline.root``) —
  ``raw-area offline`` (area + timing CSVs)
"""
from __future__ import annotations

import filecmp
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
GOLDEN_DIR = FIXTURE_DIR / "golden"
SYNTH_ROOT = FIXTURE_DIR / "flattened_Run0.root"
REPO_ROOT = THIS_DIR.parent

PYBIN = sys.executable    # the interpreter pytest itself runs under

# Conftest already adds repo root to sys.path
sys.path.insert(0, str(REPO_ROOT))


def _run_pmtdeconv(*args: str, out_dir: Path) -> None:
    """Invoke `python -m pmtdeconv <args>` and write into `out_dir`."""
    cmd = [PYBIN, "-m", "pmtdeconv", *args,
           "--out-dir", str(out_dir),
           "--log-level", "quiet"]
    result = subprocess.run(cmd, cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"pmtdeconv {' '.join(args)} failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


@pytest.fixture(scope="module")
def synth_root_exists():
    """Confirm the fixture ROOT file is present. Skips the test if not."""
    if not SYNTH_ROOT.exists():
        pytest.skip(
            f"{SYNTH_ROOT.name} not found — run "
            f"`python tests/fixtures/make_synth_root.py` to build it.")
    return SYNTH_ROOT


def _diff_csv_byte_identical(produced: Path, golden: Path) -> None:
    """Raise AssertionError with a helpful preview on byte-mismatch."""
    if not produced.exists():
        raise AssertionError(f"produced CSV does not exist: {produced}")
    if not golden.exists():
        raise AssertionError(
            f"golden CSV does not exist: {golden}\n"
            f"Goldens are gitignored and blessed per checkout — run:\n"
            f"    python tests/fixtures/bless_goldens.py")
    if filecmp.cmp(produced, golden, shallow=False):
        return
    # Show the first diverging lines for debugging
    with open(produced) as f1, open(golden) as f2:
        prod_lines = f1.readlines()
        gold_lines = f2.readlines()
    n = min(len(prod_lines), len(gold_lines))
    diffs = []
    for i in range(n):
        if prod_lines[i] != gold_lines[i]:
            diffs.append(
                f"  line {i+1}:\n"
                f"    produced: {prod_lines[i].rstrip()}\n"
                f"    golden  : {gold_lines[i].rstrip()}")
            if len(diffs) >= 3:
                break
    raise AssertionError(
        f"CSV byte-mismatch:\n  {produced}\n  vs\n  {golden}\n"
        f"first differing lines:\n" + "\n".join(diffs)
        + "\n\n(If the change is intentional, re-bless with "
        "`python tests/fixtures/bless_goldens.py --force`, then diff the "
        "result to confirm every changed byte is intended.)")


def test_raw_area_mqana_golden(tmp_path, synth_root_exists):
    """`pmtdeconv raw_area mqana` against the synth fixture must match the
    committed golden CSV byte-for-byte."""
    out_dir = tmp_path / "raw_area_mqana"
    _run_pmtdeconv("raw-area", "mqana",
                   "--dataset", "na22",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "area_mqana_run0.csv"
    golden = GOLDEN_DIR / "area_mqana_run0.csv"
    _diff_csv_byte_identical(produced, golden)


def test_waveform_golden(tmp_path, synth_root_exists):
    """`pmtdeconv waveform` against the synth fixture must match the
    committed golden CSV byte-for-byte."""
    out_dir = tmp_path / "waveform"
    _run_pmtdeconv("area",
                   "--dataset", "na22", "--cosine-cutoff-mhz", "450",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "areas_run0.csv"
    golden = GOLDEN_DIR / "deconv_area_run0.csv"
    _diff_csv_byte_identical(produced, golden)


def test_pulse_golden(tmp_path, synth_root_exists):
    """`pmtdeconv pulse` against the synth fixture must match the
    committed golden CSV byte-for-byte. Added in PR 2 / decision 9
    after the data_root + out_dir parameterization unblocked this path.

    NOTE: this runs the ``na22-timewalk-winning-2026-05`` preset, which
    has ``no_vetoes=True`` — so it does NOT exercise the per-peak veto
    chain. ``test_pulse_vetoon_golden`` covers that.
    """
    out_dir = tmp_path / "pulse"
    _run_pmtdeconv("hits",
                   "--dataset", "na22",
                   "--preset", "na22-timewalk-winning-2026-05",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "hits.csv"
    golden = GOLDEN_DIR / "deconv_pulses_run0.csv"
    _diff_csv_byte_identical(produced, golden)
    # The byte-identical `deconv_pulses.csv` copy was dropped 2026-09-14.
    assert not (out_dir / "deconv_pulses.csv").exists()


def test_pulse_vetoon_golden(tmp_path, synth_root_exists):
    """`pmtdeconv pulse --vetoes` against the synth fixture must match the
    committed golden byte-for-byte.

    Step-0 characterization (architecture restructure): the veto-OFF
    ``test_pulse_golden`` left the per-peak veto chain byte-untested on
    the flat path — exactly the code the EventSource/Pipeline collapse
    rewrites. The golden is ``all_results.csv`` (every peak + its veto
    ``reason``), so byte-identity here locks down every veto DECISION,
    not just the survivors. On this fixture the chain fires
    no_raw_peak / negative_charge / out_of_time (101 of 598 peaks
    survive).
    """
    out_dir = tmp_path / "pulse_vetoon"
    # --denoiser wavelet pinned since 2026-10 (STFT became the default):
    # this golden locks the WAVELET veto path; the STFT one is its own golden.
    _run_pmtdeconv("hits", "--denoiser", "wavelet",
                   "--dataset", "na22", "--cosine-cutoff-mhz", "450",
                   "--vetoes",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "all_results.csv"
    golden = GOLDEN_DIR / "deconv_pulses_vetoon_run0.csv"
    _diff_csv_byte_identical(produced, golden)


def test_pulse_stft_golden(tmp_path, synth_root_exists):
    """`pmtdeconv pulse --denoiser stft` against the synth fixture must
    match the committed golden byte-for-byte.

    Step-2 characterization (architecture restructure): the STFT path
    (``process_hits_stft.py``) had NO golden, so it could not be folded
    into the unified ``_process_event`` safely. This golden is
    ``all_results.csv`` and locks the STFT-specific behavior — relaxed
    veto constants (afterpulse_height_ratio=100, min_charge_purity=0),
    the loose baseline gate, and the 1.0×-noise raw-peak cut — so the
    fold (and the deletion of ``process_hits_stft.py``) is provably
    byte-identical.
    """
    out_dir = tmp_path / "pulse_stft"
    # --stft-amp-threshold 10 pinned: the golden was blessed at the pre-2026-10
    # default; the new default (5) is covered by the object-STFT golden.
    _run_pmtdeconv("hits", "--denoiser", "stft", "--stft-amp-threshold", "10",
                   "--dataset", "na22", "--cosine-cutoff-mhz", "450",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "all_results.csv"
    golden = GOLDEN_DIR / "deconv_pulses_stft_run0.csv"
    _diff_csv_byte_identical(produced, golden)


def test_waveform_circular_golden(tmp_path, synth_root_exists):
    """`pmtdeconv area --no-fft-pad` reproduces the pre-2026-10-21 circular
    area bytes; the padded default is locked by ``deconv_area_run0``."""
    out_dir = tmp_path / "waveform_circular"
    _run_pmtdeconv("area", "--no-fft-pad",
                   "--dataset", "na22", "--cosine-cutoff-mhz", "450",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "areas_run0.csv"
    golden = GOLDEN_DIR / "deconv_area_circular_run0.csv"
    _diff_csv_byte_identical(produced, golden)


# ─── Object-format goldens (PR 5) ─────────────────────────────────

OBJECT_ROOT = FIXTURE_DIR / "MilliQan_Run835.99_triple.root"


@pytest.fixture(scope="module")
def synth_object_exists():
    """Confirm the object-format fixture is present. Skips if not.

    To (re)build: ``python tests/fixtures/make_synth_object_root.py``.
    The ROOT file is ~24 MB so it's NOT auto-built — make sure to run
    that script after a fresh checkout if you want this test to run.
    """
    if not OBJECT_ROOT.exists():
        pytest.skip(
            f"{OBJECT_ROOT.name} not found — run "
            f"`python tests/fixtures/make_synth_object_root.py` to build it.")
    return OBJECT_ROOT


def test_raw_area_mqana_object_golden(tmp_path, synth_object_exists):
    """`pmtdeconv raw_area mqana --dataset lanl_835` against the synthetic
    object-format fixture must match the committed golden byte-for-byte.
    Locks down the PR 4 object-format reader.
    """
    out_dir = tmp_path / "object_raw_area_mqana"
    _run_pmtdeconv("raw-area", "mqana",
                   "--dataset", "lanl_835",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "area_mqana_run99.csv"
    golden = GOLDEN_DIR / "object_area_mqana_run99.csv"
    _diff_csv_byte_identical(produced, golden)


def test_waveform_object_golden(tmp_path, synth_object_exists):
    """`pmtdeconv waveform --dataset lanl_835` against the synthetic
    object-format fixture must match the committed golden byte-for-byte.
    """
    out_dir = tmp_path / "object_waveform"
    _run_pmtdeconv("area",
                   "--dataset", "lanl_835",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "areas_run99.csv"
    golden = GOLDEN_DIR / "object_deconv_area_run99.csv"
    _diff_csv_byte_identical(produced, golden)


def test_pulse_object_golden(tmp_path, synth_object_exists):
    """`pmtdeconv pulse --dataset lanl_835 --no-vetoes` against
    the synthetic object-format fixture must match the committed
    golden byte-for-byte. Locks down the PR 5 process_run_object path.

    Step 3 (architecture restructure): ``lanl_835`` has no per-channel
    template CSV, so it rides the previously-silent generic fallback — the
    one branch the template resolver now gates. ``--allow-generic-fallback``
    restores that exact behavior (fit_csv=None -> FALLBACK_TEMPLATE per
    branch), keeping this golden byte-identical while the resolver errors
    loudly for *new* untemplated runs.
    """
    out_dir = tmp_path / "object_pulse"
    _run_pmtdeconv("hits", "--denoiser", "wavelet",    # pinned, see vetoon
                   "--dataset", "lanl_835",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "99",
                   "--no-vetoes",
                   "--allow-generic-fallback",
                   out_dir=out_dir)
    produced = out_dir / "hits.csv"
    golden = GOLDEN_DIR / "object_deconv_pulses_run99.csv"
    _diff_csv_byte_identical(produced, golden)
    # The byte-identical `deconv_pulses.csv` copy was dropped 2026-09-14.
    assert not (out_dir / "deconv_pulses.csv").exists()


def test_pulse_object_stft_golden(tmp_path, synth_object_exists):
    """Object-format (LANL) files through the STFT denoiser.

    Until 2026-10 STFT was flat-only: an object-format dataset with
    ``--denoiser stft`` fell into the flat reader. STFT became the default
    then, so object files now run it through ``process_run_object``; this
    golden locks that path.
    """
    out_dir = tmp_path / "object_pulse_stft"
    _run_pmtdeconv("hits", "--denoiser", "stft",
                   "--dataset", "lanl_835",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "99",
                   "--no-vetoes",
                   "--allow-generic-fallback",
                   out_dir=out_dir)
    _diff_csv_byte_identical(out_dir / "hits.csv",
                             GOLDEN_DIR / "object_deconv_pulses_stft_run99.csv")


# ─── DRSprocessing-format golden (step 8 test-gap fill) ──────────

DRS_ROOT = FIXTURE_DIR / "processed" / "Run0.root"


@pytest.fixture(scope="module")
def synth_drs_exists():
    """Confirm the synthetic DRSprocessing fixture is present (skip if not).

    To (re)build: ``python tests/fixtures/make_synth_drs_root.py``. It writes
    a flat ``Events`` tree with precomputed ``area_<board>_<ch>`` /
    ``vMax_<board>_<ch>`` branches — the format ``raw-area drs`` consumes.
    """
    if not DRS_ROOT.exists():
        pytest.skip(
            f"{DRS_ROOT} not found — run "
            f"`python tests/fixtures/make_synth_drs_root.py` to build it.")
    return DRS_ROOT


def test_raw_area_drs_golden(tmp_path, synth_drs_exists):
    """`pmtdeconv raw-area drs` against the synth DRSprocessing fixture must
    match the committed golden byte-for-byte. Closes the long-deferred
    coverage gap for the precomputed-area reader."""
    out_dir = tmp_path / "raw_area_drs"
    _run_pmtdeconv("raw-area", "drs",
                   "--dataset", "na22",
                   "--file", str(DRS_ROOT),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "area_drs_run0.csv"
    golden = GOLDEN_DIR / "area_drs_run0.csv"
    _diff_csv_byte_identical(produced, golden)


# ─── Offline-format goldens (PR 8) ───────────────────────────────

OFFLINE_ROOT = FIXTURE_DIR / "MilliQan_Run99.synth_triple_offline.root"


@pytest.fixture(scope="module")
def synth_offline_exists():
    """Confirm the offline-format fixture is present. Skips if not.

    To (re)build: ``python tests/fixtures/make_synth_offline_root.py``.
    Synth file is ~108 KB so the fresh-checkout cost is small.
    """
    if not OFFLINE_ROOT.exists():
        pytest.skip(
            f"{OFFLINE_ROOT.name} not found — run "
            f"`python tests/fixtures/make_synth_offline_root.py` to build it.")
    return OFFLINE_ROOT


def test_raw_area_offline_area_golden(tmp_path, synth_offline_exists):
    """`pmtdeconv raw_area offline` against the synth offline fixture
    must produce a byte-identical area CSV."""
    out_dir = tmp_path / "raw_area_offline"
    _run_pmtdeconv("raw-area", "offline",
                   "--dataset", "lanl_835",
                   "--file", str(OFFLINE_ROOT),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "area_offline_run99.csv"
    golden = GOLDEN_DIR / "area_offline_run99.csv"
    _diff_csv_byte_identical(produced, golden)


def test_raw_area_offline_hit_golden(tmp_path, synth_offline_exists):
    """Same run must also produce a byte-identical pulse CSV
    (primary-pulse rows, peak_id=1, t_peak from default time field)."""
    out_dir = tmp_path / "raw_area_offline_hits"
    _run_pmtdeconv("raw-area", "offline",
                   "--dataset", "lanl_835",
                   "--file", str(OFFLINE_ROOT),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "offline_hit_results_run99.csv"
    golden = GOLDEN_DIR / "offline_hit_results_run99.csv"
    _diff_csv_byte_identical(produced, golden)


# ─── anti-rot: the bless recipe must cover exactly these goldens ───

def test_golden_spec_matches_tests():
    """``tests/fixtures/golden_spec.py`` must register a regeneration recipe
    for exactly the goldens this module compares against.

    Why this exists: between 2026-05 and 2026-09 the regeneration recipe
    (then a prose block in this module's docstring) covered 4 of 12 goldens,
    and claimed the goldens were committed when they are gitignored. A fresh
    checkout therefore could not reproduce the suite. Keeping the recipe as
    DATA and asserting it matches the tests makes that drift impossible: add
    a golden test without a recipe and this fails.
    """
    sys.path.insert(0, str(FIXTURE_DIR))
    from golden_spec import CASES, golden_names

    source = Path(__file__).read_text()
    # Only .csv names, so the regex literal in this test doesn't match itself.
    referenced = {m for m in re.findall(r'GOLDEN_DIR / "([^"]+)"', source)
                  if m.endswith(".csv")}
    registered = golden_names()

    assert referenced, "no GOLDEN_DIR references found — did the tests move?"
    missing = referenced - registered
    assert not missing, (
        "golden(s) compared by a test but with no recipe in golden_spec.py "
        f"(so `bless_goldens.py` cannot regenerate them): {sorted(missing)}")
    extra = registered - referenced
    assert not extra, (
        "golden_spec.py registers recipe(s) no test compares against — "
        f"stale entries: {sorted(extra)}")
    assert len(CASES) == len(registered), \
        "golden_spec.CASES has duplicate golden filenames"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ─── removed verb aliases (2026-09-14) ────────────────────────────

def test_removed_aliases_name_their_replacement(tmp_path, synth_root_exists):
    """`pulse` / `waveform` / `raw_area` were removed after their deprecation
    cycle. Each must fail with the migration instruction rather than argparse's
    bare "invalid choice" — difflib cannot suggest `hits` for `pulse`.
    """
    for alias, canonical in (("pulse", "hits"),
                             ("waveform", "area"),
                             ("raw_area", "raw-area")):
        res = subprocess.run(
            [PYBIN, "-m", "pmtdeconv", alias, "--dataset", "na22"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode != 0, f"`{alias}` must not run"
        combined = res.stdout + res.stderr
        assert canonical in combined, (
            f"`{alias}` must point at `{canonical}`; got:\n{combined}")


def test_canonical_verbs_still_work(tmp_path, synth_root_exists):
    """The replacements accept the same flags (--dry-run: no ROOT I/O)."""
    for verb in ("hits", "area"):
        out_dir = tmp_path / verb
        _run_pmtdeconv(verb, "--dataset", "na22",
                       "--data-root", str(FIXTURE_DIR), "--runs", "0",
                       "--dry-run", out_dir=out_dir)
        assert (out_dir / "manifest.json").exists()
