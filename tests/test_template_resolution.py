#!/usr/bin/env python3
"""Template resolution chain (architecture restructure, step 3).

Pins :meth:`pmtdeconv.pipeline.template.Template.from_settings`:

1. explicit ``--template`` CSV wins (provenance ``explicit``).
2. dataset default CSV wins, with the na22fmt schema swap (``dataset``) —
   the byte-safe path every golden rides.
3. a known ``pmt_type`` with no CSV resolves to the catalog template
   (``catalog``).
4. an unknown ``pmt_type``, non-interactive, errors — unless
   ``--allow-generic-fallback`` restores the generic MilliQan template
   (``generic``).
5. ``--dry-run`` never hard-fails (``unresolved``).
6. the resolved provenance is exposed for the manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pmtdeconv.presets import Settings
from pmtdeconv.presets.datasets import (
    CD109_STACKED_UNLOCKED_CSV,
    CD109_STACKED_UNLOCKED_NA22FMT_CSV,
)
from pmtdeconv.pipeline.template import Template, TemplateResolutionError


def _no_tty(monkeypatch):
    """Force the resolver's interactive branch off (deterministic batch mode)."""
    fake = type("F", (), {"isatty": staticmethod(lambda: False)})()
    monkeypatch.setattr(sys, "stdin", fake)
    monkeypatch.setattr(sys, "stdout", fake)


def test_explicit_template_wins():
    s = Settings(template_csv=Path("/tmp/explicit_spe.csv"))
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.csv_path == Path("/tmp/explicit_spe.csv")
    assert t.fit_csv == "/tmp/explicit_spe.csv"
    assert t.provenance.source == "explicit"


def test_dataset_default_applies_na22fmt_swap():
    # Emulates the resolved na22 settings: the dataset default IS template_csv,
    # so the resolver swaps to the na22fmt schema CSV (byte-identity path).
    s = Settings(
        dataset="na22",
        template_csv=CD109_STACKED_UNLOCKED_CSV,
        default_template_csv=CD109_STACKED_UNLOCKED_CSV,
        default_template_csv_na22fmt=CD109_STACKED_UNLOCKED_NA22FMT_CSV,
    )
    t = Template.from_settings(s, {"template_csv": "dataset"})
    assert t.csv_path == CD109_STACKED_UNLOCKED_NA22FMT_CSV
    assert t.provenance.source == "dataset"


def test_explicit_override_does_not_swap():
    # User passed --template (different path): no na22fmt swap even though the
    # dataset offers one.
    s = Settings(
        dataset="na22",
        template_csv=Path("/tmp/my_override.csv"),
        default_template_csv=CD109_STACKED_UNLOCKED_CSV,
        default_template_csv_na22fmt=CD109_STACKED_UNLOCKED_NA22FMT_CSV,
    )
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.csv_path == Path("/tmp/my_override.csv")
    assert t.provenance.source == "explicit"


def test_known_pmt_type_resolves_via_catalog():
    s = Settings(dataset="newrun", template_csv=None,
                 default_template_csv=None, pmt_type="milliqan_lab")
    t = Template.from_settings(s, {})
    assert t.csv_path == CD109_STACKED_UNLOCKED_CSV
    assert t.provenance.source == "catalog"
    assert t.pmt_type == "milliqan_lab"


def test_unknown_pmt_type_noninteractive_errors(monkeypatch):
    _no_tty(monkeypatch)
    s = Settings(dataset="newrun", template_csv=None,
                 default_template_csv=None, pmt_type="totally_unknown",
                 dry_run=False)
    with pytest.raises(TemplateResolutionError):
        Template.from_settings(s, {})


def test_allow_generic_fallback_restores_generic():
    s = Settings(dataset="newrun", template_csv=None, default_template_csv=None,
                 pmt_type=None, allow_generic_fallback=True)
    t = Template.from_settings(s, {})
    assert t.csv_path is None          # -> god-module FALLBACK_TEMPLATE
    assert t.fit_csv is None
    assert t.provenance.source == "generic"


def test_allow_generic_beats_catalog():
    # The explicit opt-in is unconditional (restores today's generic behavior),
    # even when a catalog entry would otherwise match.
    s = Settings(dataset="newrun", template_csv=None, default_template_csv=None,
                 pmt_type="milliqan_lab", allow_generic_fallback=True)
    t = Template.from_settings(s, {})
    assert t.csv_path is None
    assert t.provenance.source == "generic"


def test_dry_run_never_hard_fails(monkeypatch):
    _no_tty(monkeypatch)
    s = Settings(dataset=None, template_csv=None, default_template_csv=None,
                 pmt_type=None, dry_run=True)
    t = Template.from_settings(s, {})
    assert t.csv_path is None
    assert t.provenance.source == "unresolved"


def test_manifest_block_records_provenance():
    s = Settings(dataset="newrun", template_csv=None, default_template_csv=None,
                 pmt_type="milliqan_lab")
    block = Template.from_settings(s, {}).manifest_block()
    assert block["source"] == "catalog"
    assert block["family"] == "template"
    assert block["pmt_type"] == "milliqan_lab"
    assert block["csv"] == str(CD109_STACKED_UNLOCKED_CSV)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
