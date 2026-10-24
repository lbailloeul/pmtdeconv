#!/usr/bin/env python3
"""Polarity is applied in exactly ONE place — the event source.

Regression guard for the 2026-06-10 bug: the flat ``process_run`` path
negated the waveform only ``if polarity < 0`` while the object path
always multiplied, and an earlier flat path forgot the flip entirely.
After the EventSource collapse (architecture restructure, step 1) every
format flows through ``_apply_polarity`` once, so polarity ``-1`` on a
dataset is exactly the element-wise negation of polarity ``+1``.

These tests pin that property at the iterator level (no negated fixture
needed): read the SAME fixture at +1 and at -1 and assert the yielded
waves are exact negations, in the same (branch, entry) order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_DIR / "fixtures"
FLAT_ROOT = FIXTURE_DIR / "flattened_Run0.root"
OBJECT_ROOT = FIXTURE_DIR / "MilliQan_Run835.99_triple.root"

from pmtdeconv.pipeline.event_source import (   # noqa: E402
    _apply_polarity, iter_flat_run, iter_object_run_events,
)
from pmtdeconv.core.kernels import DTYPE         # noqa: E402


def test_apply_polarity_is_the_single_flip_point():
    rng = np.random.default_rng(0)
    w = rng.normal(size=256).astype(DTYPE)
    # +1 is a value-preserving no-op; -1 negates; -1 of the negated twin
    # recovers the original — the equivalence the flat path used to break.
    assert np.array_equal(_apply_polarity(w, 1), w)
    assert np.array_equal(_apply_polarity(w, -1), -w)
    assert np.array_equal(_apply_polarity(-w, -1), w)
    assert _apply_polarity(w, -1).dtype == DTYPE


def _pairs(records):
    return [(r.branch, r.entry, r.wave) for r in records]


@pytest.mark.skipif(not FLAT_ROOT.exists(),
                    reason="flat fixture missing — run make_synth_root.py")
def test_flat_polarity_negation_equivalence():
    pos = _pairs(iter_flat_run(str(FLAT_ROOT), 0, dt_ns=0.5051, polarity=1))
    neg = _pairs(iter_flat_run(str(FLAT_ROOT), 0, dt_ns=0.5051, polarity=-1))
    assert len(pos) == len(neg) > 0
    for (b1, e1, wp), (b2, e2, wn) in zip(pos, neg):
        assert (b1, e1) == (b2, e2)          # identical iteration order
        assert np.array_equal(wn, -wp)        # single flip point


@pytest.mark.skipif(not OBJECT_ROOT.exists(),
                    reason="object fixture missing — run make_synth_object_root.py")
def test_object_polarity_negation_equivalence():
    from pmtdeconv.io.object_format import detect_populated_channels
    branches = detect_populated_channels(OBJECT_ROOT, tree_name="Events")
    pos = _pairs(iter_object_run_events(
        OBJECT_ROOT, 99, dt_ns=1.0, polarity=1, branches=branches))
    neg = _pairs(iter_object_run_events(
        OBJECT_ROOT, 99, dt_ns=1.0, polarity=-1, branches=branches))
    assert len(pos) == len(neg) > 0
    for (b1, e1, wp), (b2, e2, wn) in zip(pos, neg):
        assert (b1, e1) == (b2, e2)
        assert np.array_equal(wn, -wp)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ─── the CLI hop (2026-09-14) ─────────────────────────────────────

def test_polarity_flag_exists_and_reaches_settings():
    """``--polarity`` must exist on the CLI *and* land in Settings.

    Regression guard for the inverse of the usual flag-plumbing bug: the
    ``CLI_TO_SETTINGS`` entry, the ``Settings`` field, the ``Detector`` field
    and ``_apply_polarity`` all existed, but **no command defined the argparse
    flag** — so negative-going data was reachable only through a dataset preset
    or a YAML card. Asserting both halves keeps either from rotting alone.
    """
    import argparse
    from pmtdeconv.cli import add_common_args
    from pmtdeconv.presets import CLI_TO_SETTINGS, resolve_settings

    assert CLI_TO_SETTINGS.get("polarity") == "polarity", \
        "polarity lost its CLI_TO_SETTINGS mapping"

    parser = argparse.ArgumentParser()
    add_common_args(parser)

    # argparse must accept the negative value (it parses `-1` as a value only
    # because no option string looks like a negative number — pin that).
    args = parser.parse_args(["--data-root", ".", "--polarity", "-1"])
    assert args.polarity == -1

    settings, provenance = resolve_settings(args)
    assert settings.polarity == -1
    assert provenance["polarity"] == "cli"


def test_polarity_flag_is_absent_unless_passed():
    """Omitting the flag must leave the dataset/card value untouched —
    the flag uses ``argparse.SUPPRESS``, so goldens stay byte-identical."""
    import argparse
    from pmtdeconv.cli import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--data-root", "."])
    assert not hasattr(args, "polarity"), \
        "--polarity must default to SUPPRESS so it cannot override a card"
