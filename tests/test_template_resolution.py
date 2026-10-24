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
from pmtdeconv.pipeline.template import (
    Template, TemplateResolutionError, configured_template_csv,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _no_tty(monkeypatch):
    """Force the resolver's interactive branch off (deterministic batch mode)."""
    fake = type("F", (), {"isatty": staticmethod(lambda: False)})()
    monkeypatch.setattr(sys, "stdin", fake)
    monkeypatch.setattr(sys, "stdout", fake)


def test_explicit_template_wins(tmp_path):
    # The CSV must EXIST: since 2026-09-18 a configured-but-missing template is
    # a hard error rather than a silent generic-kernel substitution.
    csv = tmp_path / "explicit_spe.csv"
    csv.write_text("det,channel\n")
    s = Settings(template_csv=csv)
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.csv_path == csv
    assert t.fit_csv == str(csv)
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


def test_explicit_override_does_not_swap(tmp_path):
    # User passed --template (different path): no na22fmt swap even though the
    # dataset offers one.
    override = tmp_path / "my_override.csv"
    override.write_text("det,channel\n")
    s = Settings(
        dataset="na22",
        template_csv=override,
        default_template_csv=CD109_STACKED_UNLOCKED_CSV,
        default_template_csv_na22fmt=CD109_STACKED_UNLOCKED_NA22FMT_CSV,
    )
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.csv_path == override
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


# ─── missing-template safety (2026-09-18) ─────────────────────────

def test_missing_template_is_a_hard_error(tmp_path):
    """A configured template that does not exist must STOP the run.

    Regression guard for the worst failure mode this tool had: a typo'd
    `--template`, a moved CSV, or (before the templates were bundled) any fresh
    clone, produced a complete run with plausible charges computed from the
    generic kernel — while the manifest recorded
    ``source="explicit", csv=<the missing path>``. Wrong physics with a
    provenance record that named a file nobody read.
    """
    missing = tmp_path / "nope.csv"
    s = Settings(dataset="na22", template_csv=missing)
    with pytest.raises(TemplateResolutionError) as exc:
        Template.from_settings(s, {"template_csv": "cli"})
    message = str(exc.value)
    assert str(missing) in message
    assert "template build" in message          # names the way out
    assert "--allow-generic-fallback" in message


def test_missing_template_allowed_with_explicit_opt_in(tmp_path):
    """--allow-generic-fallback still works, and the manifest then says
    GENERIC — it must not keep claiming the CSV that was never read."""
    missing = tmp_path / "nope.csv"
    s = Settings(dataset="na22", template_csv=missing,
                 allow_generic_fallback=True)
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.csv_path is None
    assert t.provenance.source == "generic"
    assert str(missing) in (t.provenance.detail or "")


def test_dry_run_does_not_hard_fail_on_missing_template(tmp_path):
    """--dry-run inspects configuration; it must not require the CSV to exist."""
    s = Settings(dataset="na22", template_csv=tmp_path / "nope.csv",
                 dry_run=True)
    t = Template.from_settings(s, {"template_csv": "cli"})
    assert t.provenance.source == "explicit"


def test_generic_path_loads_no_per_branch_templates():
    """When resolution chose the generic kernel, the reader must load NOTHING.

    It used to fall back to ``FIT_CSV_DEFAULT`` — a hardcoded path into the
    gitignored ``outputs/`` tree — so "generic" meant the author's Na22 biexp
    fit on the author's machine and a real generic kernel everywhere else,
    with the manifest saying "generic" either way.
    """
    from types import SimpleNamespace
    from pmtdeconv.pipeline.process_hits import _templates_for_run
    assert _templates_for_run(SimpleNamespace(fit_csv=None)) == {}


# ── the same rule for commands that load the CSV themselves ─────────────
# `area` and the `tune` sweeps read settings.template_csv directly instead of
# going through Template.from_settings. Until 2026-09-23 each called
# load_template(strict=False): one warning, then the generic kernel for every
# branch — a complete area CSV, or a declared tune winner, from the wrong
# response function, with a manifest naming a file nobody read.

def test_configured_template_csv_rule(tmp_path):
    present = tmp_path / "spe.csv"
    present.write_text("placeholder")
    missing = tmp_path / "nope.csv"

    assert configured_template_csv(Settings(dataset="na22")) is None
    assert configured_template_csv(
        Settings(dataset="na22", template_csv=present)) == present
    with pytest.raises(TemplateResolutionError, match="nope.csv"):
        configured_template_csv(
            Settings(dataset="na22", template_csv=missing), {"template_csv": "cli"})
    assert configured_template_csv(
        Settings(dataset="na22", template_csv=missing,
                 allow_generic_fallback=True)) is None
    assert configured_template_csv(
        Settings(dataset="na22", template_csv=missing, dry_run=True)) == missing


def _card_with_missing_template(tmp_path) -> Path:
    """`tune fom` has no --template flag, so its template comes from a card."""
    card = tmp_path / "card.yaml"
    card.write_text(
        f"name: missing_template\n"
        f"data_dir: {FIXTURES}\n"
        f"file_glob: flattened_Run*.root\n"
        f"format: flat\n"
        f"tree: waveforms\n"
        f"dt_ns: 0.5051\n"
        f"polarity: 1\n"
        f"template: nope.csv\n")
    return card


@pytest.mark.parametrize("command", [
    ["area", "--dataset", "na22", "--data-root", str(FIXTURES), "--runs", "0"],
    ["tune", "cosine", "--dataset", "na22"],
    ["tune", "wavelet", "--dataset", "na22"],
    ["tune", "stft", "--dataset", "na22"],
    None,  # tune fom: template from a card
], ids=["area", "tune-cosine", "tune-wavelet", "tune-stft", "tune-fom"])
def test_missing_template_stops_area_and_tune(command, tmp_path):
    from pmtdeconv import cli
    out = tmp_path / "out"
    if command is None:
        argv = ["tune", "fom", "--dataset", str(_card_with_missing_template(tmp_path)),
                "--runs", "0"]
    else:
        argv = command + ["--template", str(tmp_path / "nope.csv")]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv + ["--out-dir", str(out)])
    assert "SPE template not found" in str(exc.value.code)
    assert "nope.csv" in str(exc.value.code)
    assert not out.exists(), "must stop before writing any output"


def test_area_generic_fallback_is_an_explicit_opt_in(tmp_path):
    """--allow-generic-fallback lets `area` run on the generic kernel, and the
    manifest does not claim the missing CSV as an input."""
    import json
    from pmtdeconv import cli
    out = tmp_path / "out"
    rc = cli.main(["area", "--dataset", "na22", "--data-root", str(FIXTURES),
                   "--runs", "0", "--template", str(tmp_path / "nope.csv"),
                   "--allow-generic-fallback", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "areas_run0.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert not [i for i in manifest["inputs"] if i["role"] == "template_csv"]

