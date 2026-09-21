#!/usr/bin/env python3
"""The byte-identity golden catalogue — one entry per golden CSV.

This is the SINGLE source of truth for "which commands produce which
goldens". Two consumers read it:

- ``tests/fixtures/bless_goldens.py`` regenerates every golden from it.
- ``tests/test_golden_fixture.py`` asserts (in
  ``test_golden_spec_matches_tests``) that this catalogue covers exactly the
  goldens the test bodies actually compare against.

That second check is the anti-rot guarantee: a new golden test whose recipe
is not registered here fails the suite, so the regeneration instructions can
never silently fall behind the tests again (they did between 2026-05 and
2026-09, when the docstring recipe covered 4 of 12).

Goldens are NOT committed — ``tests/fixtures/golden/`` is gitignored because
the reference bytes are environment-specific (numpy/scipy version, platform).
Each checkout blesses its own, then byte-identity guards refactors *within*
that checkout. See the README "Tests" section.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# Which synthetic ROOT fixture each case needs, and the script that builds it.
FIXTURE_BUILDERS = {
    "flat":    ("flattened_Run0.root",                     "make_synth_root.py"),
    "object":  ("MilliQan_Run835.99_triple.root",          "make_synth_object_root.py"),
    "drs":     ("processed/Run0.root",                     "make_synth_drs_root.py"),
    "offline": ("MilliQan_Run99.synth_triple_offline.root", "make_synth_offline_root.py"),
}


@dataclass(frozen=True)
class GoldenCase:
    """One golden CSV: the command that makes it and where it lands."""
    golden: str            # filename under tests/fixtures/golden/
    fixture: str           # key into FIXTURE_BUILDERS
    argv: List[str]        # args to `python -m pmtdeconv` (minus --out-dir)
    produced: str          # filename the command writes into --out-dir
    note: str = ""         # why this case exists (mirrors the test docstring)

    # ``argv`` carries placeholders the runner interpolates:
    #   {FIXTURES}     -> tests/fixtures
    #   {DRS_ROOT}     -> tests/fixtures/processed/Run0.root
    #   {OFFLINE_ROOT} -> tests/fixtures/MilliQan_Run99.synth_triple_offline.root


CASES: List[GoldenCase] = [
    # ── flat format (flattened_Run0.root) ──────────────────────────────
    GoldenCase(
        golden="area_mqana_run0.csv", fixture="flat",
        argv=["raw-area", "mqana", "--dataset", "na22",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="area_mqana_run0.csv",
        note="MQAna narrow-window raw area.",
    ),
    GoldenCase(
        golden="deconv_area_run0.csv", fixture="flat",
        argv=["area", "--dataset", "na22",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="areas_run0.csv",
        note="Whole-waveform deconvolved area (unpadded / circular).",
    ),
    GoldenCase(
        golden="deconv_area_fftpad_run0.csv", fixture="flat",
        argv=["area", "--fft-pad", "--dataset", "na22",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="areas_run0.csv",
        note="Area path with linear (zero-padded) deconvolution.",
    ),
    GoldenCase(
        golden="deconv_pulses_run0.csv", fixture="flat",
        argv=["hits", "--dataset", "na22",
              "--preset", "na22-timewalk-winning-2026-05",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="hits.csv",
        note="Hits under the settled preset (no_vetoes=True — survivors only).",
    ),
    GoldenCase(
        golden="deconv_pulses_vetoon_run0.csv", fixture="flat",
        argv=["hits", "--dataset", "na22", "--vetoes",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="all_results.csv",
        note="Every peak + its veto reason — locks each veto DECISION.",
    ),
    GoldenCase(
        golden="deconv_pulses_stft_run0.csv", fixture="flat",
        argv=["hits", "--denoiser", "stft", "--dataset", "na22",
              "--data-root", "{FIXTURES}", "--runs", "0"],
        produced="all_results.csv",
        note="STFT profile: relaxed veto constants + loose baseline gate.",
    ),
    # ── object format (MilliQan_Run835.99_triple.root) ─────────────────
    GoldenCase(
        golden="object_area_mqana_run99.csv", fixture="object",
        argv=["raw-area", "mqana", "--dataset", "lanl_835",
              "--data-root", "{FIXTURES}", "--runs", "99"],
        produced="area_mqana_run99.csv",
        note="Object-format reader through the mqana path.",
    ),
    GoldenCase(
        golden="object_deconv_area_run99.csv", fixture="object",
        argv=["area", "--dataset", "lanl_835",
              "--data-root", "{FIXTURES}", "--runs", "99"],
        produced="areas_run99.csv",
        note="Object-format reader through the area path.",
    ),
    GoldenCase(
        golden="object_deconv_pulses_run99.csv", fixture="object",
        argv=["hits", "--dataset", "lanl_835",
              "--data-root", "{FIXTURES}", "--runs", "99",
              "--no-vetoes", "--allow-generic-fallback"],
        produced="hits.csv",
        note="Object hits path on the generic fallback template.",
    ),
    # ── DRSprocessing format (processed/Run0.root) ─────────────────────
    GoldenCase(
        golden="area_drs_run0.csv", fixture="drs",
        argv=["raw-area", "drs", "--dataset", "na22",
              "--file", "{DRS_ROOT}", "--runs", "0"],
        produced="area_drs_run0.csv",
        note="Precomputed area_<board>_<ch> reader.",
    ),
    # ── offline pulse list (MilliQan_Run99.synth_triple_offline.root) ──
    GoldenCase(
        golden="area_offline_run99.csv", fixture="offline",
        argv=["raw-area", "offline", "--dataset", "lanl_835",
              "--file", "{OFFLINE_ROOT}", "--runs", "99"],
        produced="area_offline_run99.csv",
        note="LANL offline pulse list → area CSV.",
    ),
    GoldenCase(
        golden="offline_hit_results_run99.csv", fixture="offline",
        argv=["raw-area", "offline", "--dataset", "lanl_835",
              "--file", "{OFFLINE_ROOT}", "--runs", "99"],
        produced="offline_hit_results_run99.csv",
        note="Same run → primary-pulse timing CSV.",
    ),
]


def golden_names() -> set:
    """The set of golden filenames this catalogue knows how to regenerate."""
    return {c.golden for c in CASES}
