#!/usr/bin/env python3
"""Regression tests against a deterministic synthetic ROOT fixture.

Guards against accidental byte drift in CSV outputs across refactors.
The fixture is built by ``tests/fixtures/make_synth_root.py`` from a
fixed RNG seed; the expected outputs are committed under
``tests/fixtures/golden/``.

A failure here means *something changed the bytes of a CSV output*.
Either:

1. You introduced a real regression — fix the code.
2. The change was deliberate (schema bump, defaults shift) — regenerate
   the goldens by re-running the make commands at the bottom of this
   file's docstring AND bump the matching schema version in
   ``pmtdeconv/io/schemas.py``.

Coverage today (PR 1 + PR 2 / decision 9):
- ``pmtdeconv raw_area mqana`` against synth fixture → ``area_mqana_run0.csv``
- ``pmtdeconv waveform`` against synth fixture → ``deconv_area_run0.csv``
- ``pmtdeconv pulse`` against synth fixture → ``deconv_pulse.csv``
  (added after the PR 2 / decision 9 extraction; the parameterized
  data_root + out_dir unblocked it.)

NOT covered yet:
- ``pmtdeconv raw_area drs`` — needs a DRSprocessing-format fixture (object
  tree with area_<board>_<ch> branches); deferred.

Regenerating goldens after a deliberate change::

    python tests/fixtures/make_synth_root.py
    python -m pmtdeconv raw_area mqana --dataset na22 --data-root tests/fixtures \\
        --runs 0 --out-dir tests/fixtures/golden/area_mqana --log-level quiet
    python -m pmtdeconv waveform --dataset na22 --data-root tests/fixtures \\
        --runs 0 --out-dir tests/fixtures/golden/deconv_area --log-level quiet
    python -m pmtdeconv pulse --dataset na22 \\
        --preset na22-timewalk-winning-2026-05 --data-root tests/fixtures \\
        --runs 0 --out-dir tests/fixtures/golden/deconv_pulses --log-level quiet
    python -m pmtdeconv pulse --dataset na22 --vetoes --data-root tests/fixtures \\
        --runs 0 --out-dir tests/fixtures/golden/deconv_pulses_vetoon --log-level quiet
    cp tests/fixtures/golden/area_mqana/area_mqana_run0.csv \\
       tests/fixtures/golden/area_mqana_run0.csv
    cp tests/fixtures/golden/deconv_area/areas_run0.csv \\
       tests/fixtures/golden/deconv_area_run0.csv
    cp tests/fixtures/golden/deconv_pulses/deconv_pulse.csv \\
       tests/fixtures/golden/deconv_pulses_run0.csv
    cp tests/fixtures/golden/deconv_pulses_vetoon/all_results.csv \\
       tests/fixtures/golden/deconv_pulses_vetoon_run0.csv
"""
from __future__ import annotations

import filecmp
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
            f"golden CSV does not exist: {golden} — regenerate the goldens "
            f"and commit them (see test_golden_fixture.py docstring).")
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
        + "\n\n(If the change is intentional, regenerate the goldens — "
        "see the test file's docstring.)")


def test_raw_area_mqana_golden(tmp_path, synth_root_exists):
    """`pmtdeconv raw_area mqana` against the synth fixture must match the
    committed golden CSV byte-for-byte."""
    out_dir = tmp_path / "raw_area_mqana"
    _run_pmtdeconv("raw_area", "mqana",
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
    _run_pmtdeconv("waveform",
                   "--dataset", "na22",
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
    _run_pmtdeconv("pulse",
                   "--dataset", "na22",
                   "--preset", "na22-timewalk-winning-2026-05",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "deconv_pulse.csv"
    golden = GOLDEN_DIR / "deconv_pulses_run0.csv"
    _diff_csv_byte_identical(produced, golden)
    assert (out_dir / "deconv_pulses.csv").exists()


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
    _run_pmtdeconv("pulse",
                   "--dataset", "na22",
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
    _run_pmtdeconv("pulse", "--denoiser", "stft",
                   "--dataset", "na22",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "all_results.csv"
    golden = GOLDEN_DIR / "deconv_pulses_stft_run0.csv"
    _diff_csv_byte_identical(produced, golden)


def test_waveform_fftpad_golden(tmp_path, synth_root_exists):
    """`pmtdeconv waveform --fft-pad` against the synth fixture must match the
    committed golden byte-for-byte.

    Step-4 characterization (architecture restructure): the area path's
    optional FFT zero-padding (``fft_pad`` -> next_pow2(2N) kernel/filter
    variant) was byte-untested — the ``deconv_area_run0`` golden runs with
    ``fft_pad=False``. This locks the padded path before the AreaMode rewrite
    touches it. Output differs from ``deconv_area_run0.csv`` (padding changes
    the deconvolution).
    """
    out_dir = tmp_path / "waveform_fftpad"
    _run_pmtdeconv("waveform", "--fft-pad",
                   "--dataset", "na22",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "0",
                   out_dir=out_dir)
    produced = out_dir / "areas_run0.csv"
    golden = GOLDEN_DIR / "deconv_area_fftpad_run0.csv"
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
    _run_pmtdeconv("raw_area", "mqana",
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
    _run_pmtdeconv("waveform",
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
    _run_pmtdeconv("pulse",
                   "--dataset", "lanl_835",
                   "--data-root", str(FIXTURE_DIR),
                   "--runs", "99",
                   "--no-vetoes",
                   "--allow-generic-fallback",
                   out_dir=out_dir)
    produced = out_dir / "deconv_pulse.csv"
    golden = GOLDEN_DIR / "object_deconv_pulses_run99.csv"
    _diff_csv_byte_identical(produced, golden)
    assert (out_dir / "deconv_pulses.csv").exists()


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
                   "--root", str(DRS_ROOT),
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
    _run_pmtdeconv("raw_area", "offline",
                   "--dataset", "lanl_835",
                   "--root", str(OFFLINE_ROOT),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "area_offline_run99.csv"
    golden = GOLDEN_DIR / "area_offline_run99.csv"
    _diff_csv_byte_identical(produced, golden)


def test_raw_area_offline_hit_golden(tmp_path, synth_offline_exists):
    """Same run must also produce a byte-identical pulse CSV
    (primary-pulse rows, peak_id=1, t_peak from default time field)."""
    out_dir = tmp_path / "raw_area_offline_hits"
    _run_pmtdeconv("raw_area", "offline",
                   "--dataset", "lanl_835",
                   "--root", str(OFFLINE_ROOT),
                   "--runs", "99",
                   out_dir=out_dir)
    produced = out_dir / "offline_hit_results_run99.csv"
    golden = GOLDEN_DIR / "offline_hit_results_run99.csv"
    _diff_csv_byte_identical(produced, golden)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
