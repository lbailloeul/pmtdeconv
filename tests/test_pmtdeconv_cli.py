#!/usr/bin/env python3
"""CLI wiring + preset resolution tests for the pmtdeconv package.

Synthetic dummies only — no ROOT reads, no real data, so this suite
runs in well under 5 s. The deconv math is already covered by
``tests/test_vetoes.py``; here we test only the surrounding plumbing
(argparse dispatch, preset layering, manifest correctness, template
loader, MQAna area math).

Run:  python -m pytest tests/test_pmtdeconv_cli.py -v
  or: python tests/test_pmtdeconv_cli.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmtdeconv import __version__, cli
from pmtdeconv.io.artifacts import build_artifact_index, write_artifact_index
from pmtdeconv.io.manifest import write_manifest, file_sha256
from pmtdeconv.io.templates import load_template
from pmtdeconv.pipeline.deconv_pulse_results import (
    write_deconv_pulse_result_tables,
)
from pmtdeconv.pipeline.area_mqana import mqana_area, fixed_window_area
from pmtdeconv.presets import resolve_settings, Settings
from pmtdeconv.presets.configs import CONFIGS
from pmtdeconv.presets.datasets import list_datasets


# ─── helpers ──────────────────────────────────────────────────────

def _make_args(**overrides) -> argparse.Namespace:
    """Build a Namespace with every common arg defaulted to None and
    overrides applied. Mirrors what argparse would produce.
    """
    defaults = dict(
        dataset=None, preset=None, runs=None, entry_stop=None,
        branches=None, data_root=None, tree=None, dt_ns=None,
        out_dir=None, out_suffix=None,
        manifest=None, hash_inputs=None, dry_run=True,
        log_level=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ─── CLI dispatch ─────────────────────────────────────────────────

def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["template", "--help"],
    ["template", "build", "--help"],
    ["template", "aggregate", "--help"],
    ["run", "--help"],
    ["hits", "--help"],
    ["area", "--help"],
    ["raw-area", "--help"],
    ["raw-area", "mqana", "--help"],
    ["timewalk", "--help"],
    ["timewalk", "run", "--help"],
    ["timewalk", "gif", "--help"],
    ["plot", "--help"],
    ["plot", "marginals", "--help"],
    ["plot", "energy", "--help"],
    ["plot", "analyze", "--help"],
    ["info", "--help"],
    ["results", "index", "--help"],
    ["config", "--help"],
    ["config", "resolve", "--help"],
    ["results", "--help"],
    ["results", "list", "--help"],
])
def test_cli_help_runs(argv):
    """Every --help path exits 0 (= argparse SystemExit code 0)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0


def test_deconv_command_group_is_not_public():
    with pytest.raises(SystemExit) as exc:
        cli.main(["deconv", "pulse", "--help"])
    assert exc.value.code == 2


def test_area_is_a_leaf_not_a_raw_area_group():
    with pytest.raises(SystemExit) as exc:
        cli.main(["area", "mqana"])
    assert exc.value.code == 2


# ─── dataset preset resolution ────────────────────────────────────

def test_list_datasets_has_seven():
    names = list_datasets()
    assert set(names) == {"na22", "na22_3ch", "cd109", "milliqan",
                          "lanl_806", "lanl_835", "lanl_movedBar"}


@pytest.mark.parametrize("name,expected", [
    ("na22",         dict(dt_ns=0.5051, polarity=+1,
                          cosine_cutoff_mhz=75.0)),
    ("cd109",        dict(dt_ns=1.432, polarity=+1,
                          cosine_cutoff_mhz=50.0)),
    ("milliqan",     dict(dt_ns=1.25, polarity=+1,
                          cosine_cutoff_mhz=100.0)),
    ("lanl_806",     dict(dt_ns=1.25, polarity=-1,
                          cosine_cutoff_mhz=100.0)),
])
def test_dataset_preset_resolution(name, expected):
    args = _make_args(dataset=name)
    settings, prov = resolve_settings(args)
    for k, v in expected.items():
        assert getattr(settings, k) == v, f"{name}.{k}: got {getattr(settings, k)} expected {v}"
        assert prov[k] == "dataset", f"{name}.{k} provenance wrong: {prov[k]!r}"


# ─── named config preset overlay ─────────────────────────────────

def test_configs_include_winning():
    assert "na22-timewalk-winning-2026-05" in CONFIGS


def test_config_preset_overrides_dataset():
    args = _make_args(dataset="na22", preset="na22-timewalk-winning-2026-05")
    settings, prov = resolve_settings(args)
    # The preset pins 450 over the dataset's 75 and claims provenance.
    assert settings.cosine_cutoff_mhz == 450.0
    assert prov["cosine_cutoff_mhz"] == "config"
    # Wavelet thresholding k=1.0 is preset-only (dataset doesn't set it)
    assert settings.wthresh_k == 1.0
    assert prov["wthresh_k"] == "config"
    # The "tallest-first when no vetoes" flag is preset-on
    assert settings.no_vetoes is True
    assert prov["no_vetoes"] == "config"


def test_preset_without_explicit_dataset_uses_base():
    args = _make_args(preset="na22-timewalk-winning-2026-05")
    settings, prov = resolve_settings(args)
    assert settings.dataset == "na22"
    assert settings.cosine_cutoff_mhz == 450.0    # pinned by the preset
    assert settings.wthresh_k == 1.0           # from preset


# ─── CLI flag overrides ───────────────────────────────────────────

def test_cli_flag_overrides_dataset():
    args = _make_args(dataset="na22", dt_ns="1.0")
    settings, prov = resolve_settings(args)
    assert settings.dt_ns == 1.0
    assert prov["dt_ns"] == "cli"
    # Other dataset fields untouched
    assert settings.polarity == +1
    assert prov["polarity"] == "dataset"


def test_cli_flag_overrides_preset():
    args = _make_args(dataset="na22", preset="na22-timewalk-winning-2026-05")
    args.cosine_cutoff_mhz = 200.0
    settings, prov = resolve_settings(args)
    assert settings.cosine_cutoff_mhz == 200.0
    assert prov["cosine_cutoff_mhz"] == "cli"


def test_dt_ns_auto_string_passes_through():
    args = _make_args(dataset="lanl_835", dt_ns="auto")
    settings, _ = resolve_settings(args)
    assert settings.dt_ns == "auto"


# ─── data-root requirement ────────────────────────────────────────

def test_missing_dataset_raises_without_dry_run():
    args = _make_args(dry_run=False)
    with pytest.raises(SystemExit):
        resolve_settings(args)


def test_missing_dataset_ok_with_dry_run():
    args = _make_args(dry_run=True)
    settings, _ = resolve_settings(args)
    assert settings.dataset is None


def test_require_data_root_false_skips_check():
    args = _make_args(dry_run=False)
    settings, _ = resolve_settings(args, require_data_root=False)
    assert settings.dataset is None


# ─── manifest schema ──────────────────────────────────────────────

def test_manifest_schema_required_keys(tmp_path):
    args = _make_args(dataset="na22")
    settings, prov = resolve_settings(args)
    out = tmp_path / "manifest.json"
    write_manifest(
        subcommand="raw_area mqana",
        settings=settings,
        provenance=prov,
        inputs=[{"role": "root_data", "path": str(tmp_path / "fake.root")}],
        outputs=[{"role": "area_csv", "path": str(tmp_path / "fake.csv")}],
        stats={"n_triggers": 100},
        out_path=out,
        duration_seconds=0.42,
    )
    m = json.loads(out.read_text())
    for key in ("schema_version", "pmtdeconv_version", "subcommand",
                "timestamp_utc", "settings", "settings_provenance",
                "inputs", "outputs", "stats", "git", "environment", "preset"):
        assert key in m, f"manifest missing {key!r}"
    assert m["subcommand"] == "raw_area mqana"
    assert m["pmtdeconv_version"] == __version__
    assert m["preset"]["dataset"] == "na22"


def test_manifest_provenance_field(tmp_path):
    args = _make_args(dataset="na22", preset="na22-timewalk-winning-2026-05")
    settings, prov = resolve_settings(args)
    out = tmp_path / "manifest.json"
    write_manifest(subcommand="pulse", settings=settings, provenance=prov,
                   inputs=[], outputs=[], out_path=out)
    m = json.loads(out.read_text())
    sp = m["settings_provenance"]
    assert sp["cosine_cutoff_mhz"] == "config"
    assert sp["polarity"] == "dataset"


def test_file_sha256_empty_for_missing(tmp_path):
    assert file_sha256(tmp_path / "does-not-exist") == ""


def test_direct_granularity_commands_write_canonical_manifests(tmp_path):
    """`hits` / `area` write a manifest on --dry-run, recording their REAL
    command names. Until 2026-09-21 they stored the pre-rename "pulse" /
    "waveform"; `results` still normalizes those when reading old runs."""
    hits_out = tmp_path / "hits"
    assert cli.main([
        "hits", "--out-dir", str(hits_out), "--dry-run",
        "--log-level", "quiet",
    ]) == 0
    assert json.loads((hits_out / "manifest.json").read_text())["subcommand"] == "hits"

    area_out = tmp_path / "area"
    assert cli.main([
        "area", "--out-dir", str(area_out), "--dry-run",
        "--log-level", "quiet",
    ]) == 0
    assert json.loads((area_out / "manifest.json").read_text())["subcommand"] == "area"


def test_stft_pulse_dry_run_writes_canonical_manifest(tmp_path):
    out_dir = tmp_path / "pulse_stft"
    assert cli.main([
        "hits", "--denoiser", "stft", "--out-dir", str(out_dir),
        "--dry-run", "--log-level", "quiet",
    ]) == 0
    m = json.loads((out_dir / "manifest.json").read_text())
    assert m["subcommand"] == "hits"
    assert m["settings"]["denoiser"] == "stft"


def test_real_cli_parser_does_not_override_preset_defaults(tmp_path):
    # The 2026-05 snapshot sets every wavelet knob; argparse defaults must
    # not clobber any of them.
    out_dir = tmp_path / "hits"
    assert cli.main([
        "hits", "--preset", "na22-timewalk-winning-2026-05", "--runs", "65213",
        "--out-dir", str(out_dir), "--dry-run", "--log-level", "quiet",
    ]) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    provenance = manifest["settings_provenance"]
    assert provenance["denoiser"] == "config"
    assert provenance["wavelet"] == "config"
    assert provenance["wlevel"] == "config"
    assert provenance["wmode"] == "config"
    assert provenance["wthresh_k"] == "config"
    assert provenance["no_vetoes"] == "config"
    assert provenance["peak_order"] == "config"
    assert "stft_nperseg_ns" not in provenance


def test_explicit_boolean_flag_can_override_true_preset(tmp_path):
    out_dir = tmp_path / "hits"
    assert cli.main([
        "hits", "--preset", "na22-timewalk", "--vetoes",
        "--out-dir", str(out_dir), "--dry-run", "--log-level", "quiet",
    ]) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["settings"]["no_vetoes"] is False
    assert manifest["settings_provenance"]["no_vetoes"] == "cli"


def test_run_recipe_dry_run_and_results_commands(tmp_path, capsys):
    workflow = tmp_path / "workflow"
    assert cli.main([
        "run", "na22-timewalk", "--out-dir", str(workflow),
        "--dry-run", "--log-level", "quiet",
    ]) == 0
    for step in ("raw-timing", "raw-area", "timewalk"):
        assert (workflow / step / "manifest.json").exists()

    assert cli.main([
        "results", "list", "--outputs-dir", str(workflow),
        "--log-level", "quiet",
    ]) == 0
    output = capsys.readouterr().out
    assert "raw-area" in output
    assert "timewalk" in output

    assert cli.main([
        "results", "list", "--outputs-dir", str(workflow),
        "--command", "timewalk", "--log-level", "quiet",
    ]) == 0
    filtered = capsys.readouterr().out
    assert "timewalk" in filtered
    assert "raw-area" not in filtered

    assert cli.main([
        "results", "path", "timewalk", "--outputs-dir", str(workflow),
        "--log-level", "quiet",
    ]) == 0
    assert str(workflow / "timewalk") in capsys.readouterr().out


def test_na22_recipe_rejects_other_dataset():
    with pytest.raises(SystemExit, match="requires --dataset na22"):
        cli.main([
            "run", "na22-timewalk", "--dataset", "cd109", "--dry-run",
        ])


def test_config_resolve_reports_preset_provenance(capsys):
    assert cli.main([
        "config", "resolve", "na22-timewalk", "--runs", "65213",
        "--json", "--log-level", "quiet",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["dataset"] == "na22"
    assert payload["provenance"]["denoiser"] == "config"
    assert payload["provenance"]["peak_order"] == "config"


def test_artifact_index_reads_manifest_and_orphan_dir(tmp_path):
    args = _make_args(dataset="na22", runs=[65213])
    settings, prov = resolve_settings(args)
    outputs_dir = tmp_path / "outputs"
    out_dir = outputs_dir / "deconv_pulse_na22"
    out_dir.mkdir(parents=True)
    write_manifest(
        subcommand="pulse",
        settings=settings,
        provenance=prov,
        inputs=[],
        outputs=[{"role": "deconv_pulse", "path": str(out_dir / "deconv_pulse.csv")}],
        stats={"runs": [65213]},
        out_path=out_dir / "manifest.json",
        duration_seconds=0.1,
    )
    orphan = outputs_dir / "old_plot_without_manifest"
    orphan.mkdir()
    (orphan / "plot.pdf").write_text("dummy")

    index = build_artifact_index(outputs_dir)
    summary = index["summary"]
    assert summary["n_manifests"] == 1
    assert summary["n_orphan_dirs"] == 1
    assert summary["output_role_counts"]["deconv_pulse"] == 1

    manifest_entries = [a for a in index["artifacts"] if a["kind"] == "manifest"]
    assert manifest_entries[0]["subcommand"] == "pulse"
    assert manifest_entries[0]["dataset"] == "na22"
    assert manifest_entries[0]["runs"] == [65213]

    target = write_artifact_index(index, tmp_path / "artifact_index.json")
    assert json.loads(target.read_text())["summary"]["n_artifacts"] == 2


def test_deconv_pulse_result_tables_write_canonical_only(tmp_path):
    rows = [
        {"success": True, "reason": "", "charge": 1.0},
        {"success": False, "reason": "negative_charge", "charge": -0.2},
    ]
    tables = write_deconv_pulse_result_tables(rows, tmp_path)

    # The byte-identical `deconv_pulses.csv` copy was dropped 2026-09-14 —
    # every run had been writing the whole table twice for a 2026-06 rename.
    compat = tmp_path / "deconv_pulses.csv"
    assert tables.deconv_pulse_csv.name == "hits.csv"
    assert tables.deconv_pulse_csv.exists()
    assert not compat.exists()
    assert tables.all_results_csv.exists()
    assert tables.negative_charge_csv is not None
    assert tables.negative_charge_csv.exists()


# ─── template CSV loader: both schemas ────────────────────────────

def _write_csv(path, content):
    path.write_text(textwrap.dedent(content).strip() + "\n")


def test_template_loader_stacked_schema(tmp_path):
    csv = tmp_path / "stacked.csv"
    _write_csv(csv, """
        det,channel,tau_rise_ns,tau_fall_ns,sigma_ns
        d0,1,35.6,49.4,9.05
        d0,2,24.9,46.2,5.26
    """)
    templates = load_template(csv)
    assert set(templates) == {"d0_ch1", "d0_ch2"}
    assert templates["d0_ch1"]["tau_rise"] == pytest.approx(35.6)
    assert templates["d0_ch2"]["sigma"] == pytest.approx(5.26)


def test_template_loader_per_event_branch_schema(tmp_path):
    csv = tmp_path / "per_event_branch.csv"
    _write_csv(csv, """
        branch,success,tau_rise,tau_fall,sigma
        d0_ch1,True,30.0,50.0,8.0
        d0_ch1,True,32.0,52.0,9.0
        d0_ch2,True,20.0,45.0,5.0
        d0_ch1,False,9999,9999,9999
    """)
    templates = load_template(csv)
    assert set(templates) == {"d0_ch1", "d0_ch2"}
    # Median of (30, 32) = 31.0; the failed row was filtered out
    assert templates["d0_ch1"]["tau_rise"] == pytest.approx(31.0)
    assert templates["d0_ch2"]["tau_rise"] == pytest.approx(20.0)


def test_template_loader_per_event_det_chan_schema(tmp_path):
    csv = tmp_path / "per_event_detchan.csv"
    _write_csv(csv, """
        det,chan,success,tau_rise,tau_fall,sigma
        d0,1,True,25.0,48.0,7.0
        d0,1,True,27.0,50.0,8.0
        d0,2,True,18.0,44.0,4.5
    """)
    templates = load_template(csv)
    assert set(templates) == {"d0_ch1", "d0_ch2"}
    assert templates["d0_ch1"]["tau_rise"] == pytest.approx(26.0)


def test_template_loader_missing_none_returns_empty():
    """None path means 'no template configured' — fall back to built-in."""
    assert load_template(None) == {}


def test_template_loader_missing_path_hard_fails(tmp_path):
    """Decision 6: a non-existent CLI-provided path must hard-fail
    (no silent MilliQan fallback) with a pointer to template build."""
    from pmtdeconv.io.templates import TemplateNotFoundError
    with pytest.raises(TemplateNotFoundError) as exc:
        load_template(tmp_path / "does-not-exist.csv")
    assert "pmtdeconv template build" in str(exc.value)


def test_template_loader_missing_strict_false_falls_back(tmp_path):
    """`strict=False` opt-in preserves the old silent-fallback semantics
    for callers that know they want it."""
    assert load_template(tmp_path / "nope.csv", strict=False, verbose=False) == {}


# ─── MQAna area math ──────────────────────────────────────────────

def test_mqana_area_csv_schema_order():
    """The CSV column order is (run, cycle, entry, branch, interval,
    area, peak_height) and area is in mV·ns."""
    # The MQAna area function returns the numbers; the column ordering
    # is enforced by area_raw.run. Here we just check the function's
    # outputs are coherent (correct sign, positive area for a positive
    # bump).
    N, dt = 1024, 0.5051
    wave = np.zeros((1, N))
    # Insert a Gaussian peak at sample 500, amplitude 100 mV
    samp = np.arange(N)
    wave[0] = 100.0 * np.exp(-0.5 * ((samp - 500) / 25.0) ** 2)
    area, ph = mqana_area(wave, dt, n_pre=50, window_ns=35.0, polarity=+1)
    assert area.shape == (1,)
    assert area[0] > 0, "positive pulse should yield positive area"
    assert ph[0] == pytest.approx(100.0, abs=0.1)


def test_mqana_polarity_invariance():
    """Negative-going LANL pulses should give the same magnitude as
    positive-going pulses after the polarity flip."""
    N, dt = 1024, 0.5051
    wave_pos = np.zeros((2, N))
    samp = np.arange(N)
    wave_pos[0] = 80.0 * np.exp(-0.5 * ((samp - 300) / 30.0) ** 2)
    wave_pos[1] = 80.0 * np.exp(-0.5 * ((samp - 700) / 30.0) ** 2)
    a_pos, ph_pos = mqana_area(wave_pos, dt, polarity=+1)
    a_neg, ph_neg = mqana_area(-wave_pos, dt, polarity=-1)
    np.testing.assert_allclose(a_pos, a_neg, rtol=1e-12)
    np.testing.assert_allclose(ph_pos, ph_neg, rtol=1e-12)


def test_mqana_invalid_polarity():
    with pytest.raises(ValueError):
        mqana_area(np.zeros((1, 1024)), 0.5051, polarity=0)


def test_fixed_window_matches_legacy_math():
    """`fixed_window_area` should match the formula
    `sum(signal[s_start:s_end]) * dt` after baseline subtraction."""
    N, dt = 1024, 1.25
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 1.0, (3, N)) + 50.0
    # Add a known integrated bump in [200, 250]
    wave[:, 200:250] += 10.0
    area, _ = fixed_window_area(wave, dt, n_pre=50,
                                signal_start=200, signal_end=250,
                                polarity=+1)
    expected = (10.0 * 50) * dt
    np.testing.assert_allclose(area.mean(), expected, rtol=0.05)


# ─── stand-alone execution ────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ─── introspection reshuffle (2026-09-14) ─────────────────────────

def test_moved_introspection_leaves_kept_their_capability():
    """`info presets` -> `config show` (no name); `info artifacts` ->
    `results index`.

    Both were MOVES, not deletions: `config show` required a name so it could
    not list, and `results list` had no index-writing or orphan audit. This
    pins that each verb now owns one domain — info = detectors and files,
    config = settings, results = outputs — and that neither capability was
    dropped on the way.
    """
    from pmtdeconv.commands import info as info_mod
    from pmtdeconv.commands import config as config_mod
    from pmtdeconv.commands import results as results_mod

    # the moved leaves are gone from `info` ...
    parser = argparse.ArgumentParser()
    info_mod.add_arguments(parser)
    info_leaves = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            info_leaves = set(action.choices)
    assert info_leaves == {"datasets", "template", "offline-file"}, info_leaves

    # ... and `config show` now lists presets with no name given
    assert hasattr(config_mod, "_list_presets")
    parser = argparse.ArgumentParser()
    config_mod.add_arguments(parser)
    args = parser.parse_args(["show"])
    assert args.name is None, "`config show` must accept a missing name"

    # ... and `results index` carries the artifact-index maintenance flags
    parser = argparse.ArgumentParser()
    results_mod.add_arguments(parser)
    args = parser.parse_args(["index", "--write", "--no-orphans"])
    assert args.results_action == "index"
    assert args.no_orphans is True


# ─── --dataset / --preset separation (2026-09-18) ─────────────────

def _resolve_argv(argv):
    parser = argparse.ArgumentParser()
    cli.add_common_args(parser)
    return resolve_settings(parser.parse_args(argv + ["--dry-run"]))


def test_dataset_rejects_a_preset_name():
    """`--dataset` names a DETECTOR, `--preset` names a RECIPE.

    Until 2026-09-18 `--dataset` also took a preset name, giving three
    spellings for one run. The real defect was `--dataset P1 --preset P2`:
    both presets' overrides stacked while only P2's name reached the manifest
    and the output directory. Refusing a preset here makes that
    unrepresentable rather than merely discouraged.
    """
    preset = next(iter(CONFIGS))
    with pytest.raises(SystemExit) as exc:
        _resolve_argv(["--dataset", preset])
    message = str(exc.value)
    assert f"--preset {preset}" in message, message
    assert "names a config preset" in message


def test_preset_still_composes_onto_any_dataset():
    """The one thing keeping both flags buys: a recipe from one detector
    applied to another. `--preset` must override the dataset's own values."""
    settings, provenance = _resolve_argv(
        ["--dataset", "lanl_806", "--preset", "na22-timewalk"])
    assert settings.dataset == "lanl_806"
    assert settings.config_preset_name == "na22-timewalk"
    # lanl_806's own cutoff is 100 MHz; the preset's 450 must win.
    assert settings.cosine_cutoff_mhz == 450.0
    assert provenance["cosine_cutoff_mhz"] == "config"
    assert provenance["polarity"] == "dataset"      # detector value survives
    assert settings.polarity == -1


def test_preset_alone_still_fills_in_its_base_dataset():
    """`--preset X` with no --dataset keeps resolving X's base detector."""
    settings, _ = _resolve_argv(["--preset", "na22-timewalk"])
    assert settings.dataset == "na22"
    assert settings.config_preset_name == "na22-timewalk"


def test_a_real_dataset_name_wins_over_a_same_named_preset(monkeypatch):
    """A name that is BOTH a dataset and a preset resolves as the dataset —
    `--dataset` should never be refused for a detector that genuinely exists."""
    from pmtdeconv.presets import configs as configs_mod
    monkeypatch.setitem(CONFIGS, "na22", configs_mod.ConfigPreset(
        name="na22", base_dataset="na22", description="collision fixture",
        overrides={}))
    settings, _ = _resolve_argv(["--dataset", "na22"])
    assert settings.dataset == "na22"


# ─── run.log console transcript (2026-09-18) ──────────────────────

def test_run_log_written_beside_manifest(tmp_path):
    """Every manifest-writing run drops a run.log transcript next to it.

    The manifest is the structured record; run.log is what the run actually
    SAID — resolved paths, template lines, the veto-rate banner. Before this,
    that stream vanished with the terminal, so a collaborator reporting "weird
    results" had no way to show what happened.
    """
    out_dir = tmp_path / "hits"
    assert cli.main([
        "hits", "--dataset", "na22", "--runs", "0", "--dry-run",
        "--out-dir", str(out_dir), "--log-level", "quiet",
    ]) == 0

    log = out_dir / "run.log"
    assert log.exists(), "run.log must sit beside manifest.json"
    text = log.read_text()
    assert "pmtdeconv hits" in text          # the provenance banner
    assert "dataset" in text

    manifest = json.loads((out_dir / "manifest.json").read_text())
    roles = {o["role"] for o in manifest["outputs"]}
    assert "run_log" in roles, "run.log must be recorded as an output"


def test_run_log_capture_is_restored_after_main():
    """cli.main() must always restore sys.stdout — an in-process caller
    (this test suite) must not be left writing into a dangling tee."""
    import sys as _sys
    from pmtdeconv.io import run_log
    before_out, before_err = _sys.stdout, _sys.stderr
    with pytest.raises(SystemExit):
        cli.main(["hits", "--dataset", "definitely-not-a-dataset"])
    assert _sys.stdout is before_out
    assert _sys.stderr is before_err
    assert run_log.captured_text() == ""


# ─── --root -> --file rename (2026-09-21) ─────────────────────────

@pytest.mark.parametrize("module,old,new,dest,extra", [
    ("raw_area_drs",     "--root",      "--file",      "file",      []),
    ("raw_area_offline", "--root",      "--file",      "file",      []),
    ("plot_energy",      "--proc-root", "--file",      "file",      []),
    ("run_recipe",       "--area-root", "--area-file", "area_file", []),
    ("run_recipe",       "--proc-root", "--proc-file", "proc_file", []),
])
def test_renamed_file_flags(module, old, new, dest, extra, capsys):
    """`--root` named the file FORMAT and sat beside `--data-root`, a
    DIRECTORY — easy to swap. The new spelling must work; the old one must
    keep working until 1.0, point at the new name, and stay out of --help so
    it is not advertised."""
    import importlib
    mod = importlib.import_module(f"pmtdeconv.commands.{module}")

    def parse(argv):
        parser = argparse.ArgumentParser()
        mod.add_arguments(parser)
        return parser, parser.parse_args(argv)

    parser, args = parse([*extra, new, "/x/Run7.root"])
    value = getattr(args, dest)
    assert (value == ["/x/Run7.root"]) or (value == "/x/Run7.root")

    _, args = parse([*extra, old, "/x/Run7.root"])
    value = getattr(args, dest)
    assert (value == ["/x/Run7.root"]) or (value == "/x/Run7.root")
    assert new in capsys.readouterr().err, "old spelling must name its replacement"

    assert old not in parser.format_help(), f"{old} must be hidden from --help"
    assert new in parser.format_help()


# ─── 2026-09-21 batch: hashing default, dirty banner, vocabulary ──

def test_input_hashing_is_on_by_default():
    """Hashing is what lets two manifests prove they used the same card,
    template and data. Its old opt-in rationale ("slow") was measured false."""
    for argv, expect in (([], True), (["--no-hash-inputs"], False),
                         (["--hash-inputs"], True)):
        parser = argparse.ArgumentParser()
        cli.add_common_args(parser)
        settings, _ = resolve_settings(parser.parse_args(argv + ["--dry-run"]),
                                       require_data_root=False)
        assert settings.hash_inputs is expect, argv


@pytest.mark.parametrize("dirty", [True, False])
def test_banner_warns_loudly_only_when_code_is_dirty(dirty, monkeypatch):
    """A run from modified code cannot be reproduced from its recorded SHA —
    say so up front, and stay silent when the code is committed (a warning
    that always fires gets ignored)."""
    from pmtdeconv.io import banner
    monkeypatch.setattr(banner, "get_git_info", lambda: {
        "sha": "abcdef1234", "branch": "b", "dirty": dirty,
        "dirty_count": 3 if dirty else 0,
        "dirty_files": ["pmtdeconv/cli.py"] if dirty else []})
    text = banner.format_run_banner("hits", Settings(dataset="na22"), {})
    assert ("NOT REPRODUCIBLE FROM GIT" in text) is dirty
    if dirty:
        assert "3 uncommitted change(s)" in text


def test_results_filter_matches_old_and_new_manifest_names():
    """Old manifests store "pulse"; new ones store "hits". `--command hits`
    (or the legacy "pulse") must find both — the previous stored-name
    translation would have silently missed every new run."""
    from types import SimpleNamespace
    from pmtdeconv.commands import results
    entries = [{"subcommand": "pulse"}, {"subcommand": "hits"},
               {"subcommand": "raw_area mqana"}, {"subcommand": "raw-area drs"},
               {"subcommand": "area"}]
    for query in ("hits", "pulse"):
        got = results._filtered(
            SimpleNamespace(dataset=None, preset=None, command=query),
            Path("."), entries)
        assert {e["subcommand"] for e in got} == {"pulse", "hits"}, query
    got = results._filtered(
        SimpleNamespace(dataset=None, preset=None, command="raw-area"),
        Path("."), entries)
    assert {e["subcommand"] for e in got} == {"raw_area mqana", "raw-area drs"}


def test_hits_csv_is_written_and_legacy_names_still_resolve(tmp_path):
    """`hits` writes hits.csv; output dirs from before the rename (named
    deconv_pulse.csv or deconv_pulses.csv) must still be found."""
    import pandas as pd
    from pmtdeconv.io.deconv_pulse import (resolve_deconv_pulse_csv,
                                           write_deconv_pulse_csv)
    assert write_deconv_pulse_csv(pd.DataFrame({"a": [1]}), tmp_path / "new"
                                  ).name == "hits.csv"
    for legacy in ("deconv_pulse.csv", "deconv_pulses.csv"):
        d = tmp_path / legacy.replace(".csv", "")
        d.mkdir()
        (d / legacy).write_text("a\n1\n")
        assert resolve_deconv_pulse_csv(d).name == legacy
    assert resolve_deconv_pulse_csv(tmp_path / "empty") is None


@pytest.mark.parametrize("module", ["gain_calibrate", "plot_deconv_value",
                                    "select_multihit", "timewalk_gif",
                                    "timewalk_run"])
def test_pulse_csv_flag_renamed_to_hits_csv(module, capsys):
    import importlib
    mod = importlib.import_module(f"pmtdeconv.commands.{module}")
    parser = argparse.ArgumentParser()
    mod.add_arguments(parser)
    assert parser.parse_args(["--hits-csv", "x.csv"]).pulse_csv == "x.csv"
    assert parser.parse_args(["--pulse-csv", "x.csv"]).pulse_csv == "x.csv"
    assert "--hits-csv" in capsys.readouterr().err
    assert "--pulse-csv" not in parser.format_help()


def test_hits_default_out_dir_holds_csv_and_manifest(tmp_path, monkeypatch):
    """Without --out-dir, hits.csv and manifest.json land in ONE directory, and
    the manifest records the ROOT file that was actually read.

    Until 2026-09-23 the flat wavelet reader picked its own hardcoded
    ``deconv_pulse_Na22_lab_cosine_wavelet`` folder whenever --out-dir was
    omitted, while the manifest went to the dataset-named folder: a manifest
    listing a hits.csv that was not beside it. Its data inputs were looked up
    under data/Na22_lab regardless of --data-root.
    """
    from pmtdeconv.commands import hits as hits_cmd
    fixtures = Path(__file__).resolve().parent / "fixtures"
    auto = tmp_path / "auto"
    monkeypatch.setattr(hits_cmd, "hits_output_dir", lambda settings: auto)

    assert cli.main(["hits", "--dataset", "na22", "--data-root", str(fixtures),
                     "--runs", "0"]) == 0

    assert (auto / "hits.csv").exists()
    manifest = json.loads((auto / "manifest.json").read_text())
    data_inputs = [Path(i["path"]).resolve() for i in manifest["inputs"]
                   if i["role"] == "root_data"]
    assert data_inputs == [(fixtures / "flattened_Run0.root").resolve()]


def test_plot_analyze_renders_from_a_hits_csv(tmp_path):
    """`plot analyze --hits-csv` draws its plot set from any hits CSV (its old
    presets pointed at output folders that exist on one machine only)."""
    rng = np.random.default_rng(0)
    n = 200
    import pandas as pd
    pd.DataFrame({
        "run": 0, "cycle": 1, "entry": np.repeat(np.arange(n // 2), 2),
        "chan": np.tile([1, 2], n // 2), "branch": np.tile(["d0_ch1", "d0_ch2"], n // 2),
        "success": True, "peak_id": 1.0,
        "t_peak": rng.uniform(100, 400, n), "height": rng.uniform(5, 50, n),
        "charge": rng.uniform(1e3, 5e4, n),
        "win_lo_ns": 90.0, "win_hi_ns": 450.0,
    }).to_csv(tmp_path / "hits.csv", index=False)

    out = tmp_path / "plots"
    assert cli.main(["plot", "analyze", "--hits-csv", str(tmp_path / "hits.csv"),
                     "--out-dir", str(out)]) == 0
    pdfs = sorted(p.name for p in out.glob("*.pdf"))
    assert any(n.startswith("charge_histogram") for n in pdfs), pdfs
    assert any(n.startswith("ch1_vs_ch2") for n in pdfs), pdfs


def test_plot_analyze_requires_a_csv():
    with pytest.raises(SystemExit) as exc:
        cli.main(["plot", "analyze"])
    assert "--hits-csv" in str(exc.value.code)


# ─── 2026-10 decisions ───────────────────────────────────────────────

def test_default_denoiser_is_stft_and_wavelet_is_flagged(tmp_path, capsys):
    """STFT is the `hits` default since the 2026-10 denoiser study; the
    wavelet still runs but says it is experimental."""
    fixtures = Path(__file__).resolve().parent / "fixtures"
    common = ["hits", "--dataset", "na22", "--data-root", str(fixtures),
              "--runs", "0", "--entry-stop", "20"]
    assert cli.main(common + ["--out-dir", str(tmp_path / "d")]) == 0
    manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert manifest["settings"]["denoiser"] == "stft"
    assert manifest["settings"]["stft_amp_threshold"] == 5.0
    assert "[experimental]" not in capsys.readouterr().out

    assert cli.main(common + ["--denoiser", "wavelet",
                              "--out-dir", str(tmp_path / "w")]) == 0
    assert "[experimental] wavelet denoiser" in capsys.readouterr().out


def test_timewalk_presets_name_their_denoiser():
    """The 2026-05 snapshot was validated with the wavelet and must keep it;
    the stable alias follows the 2026-10 STFT snapshot."""
    from pmtdeconv.presets.configs import CONFIGS
    assert CONFIGS["na22-timewalk-winning-2026-05"].overrides["denoiser"] == "wavelet"
    assert CONFIGS["na22-timewalk-2026-10"].overrides["denoiser"] == "stft"
    assert (CONFIGS["na22-timewalk"].overrides
            == CONFIGS["na22-timewalk-2026-10"].overrides)


def test_flag_abbreviations_are_rejected():
    """`hits --all` used to be read as --allow-generic-fallback."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["hits", "--dataset", "na22", "--runs", "0", "--all",
                  "--dry-run"])
    assert exc.value.code == 2


def test_template_build_aggregates_unless_told_not_to(tmp_path, monkeypatch):
    """`template build` runs the per-channel aggregation itself (2026-10);
    --no-aggregate stops after the per-file step."""
    from pmtdeconv.commands import template_aggregate
    from pmtdeconv.pipeline import template_build_impl
    calls = []
    monkeypatch.setattr(template_build_impl, "main",
                        lambda configs, model, out_dir: None)
    monkeypatch.setattr(template_aggregate, "aggregate",
                        lambda out_dir: calls.append(out_dir) or [])
    base = ["template", "build", "--dataset", "cd109", "--data-root", str(tmp_path)]
    assert cli.main(base + ["--out-dir", str(tmp_path / "a")]) == 0
    assert calls == [(tmp_path / "a").resolve()]
    assert cli.main(base + ["--out-dir", str(tmp_path / "b"), "--no-aggregate"]) == 0
    assert len(calls) == 1



def test_timewalk_fails_loudly_when_the_reference_window_is_empty(tmp_path):
    """An --a-ref window outside the area source's scale used to write an
    empty curve and exit 0 (`run na22-timewalk --raw-source mqana`)."""
    fixtures = Path(__file__).resolve().parent / "fixtures"
    common = ["--dataset", "na22", "--data-root", str(fixtures), "--runs", "0"]
    assert cli.main(["raw-area", "mqana", *common, "--out-dir", str(tmp_path / "a")]) == 0
    assert cli.main(["hits", *common, "--out-dir", str(tmp_path / "h")]) == 0
    with pytest.raises(SystemExit) as exc:
        cli.main(["timewalk", "run", *common,
                  "--area-csv", str(tmp_path / "a" / "area_mqana_run0.csv"),
                  "--hits-csv", str(tmp_path / "h" / "hits.csv"),
                  "--a-ref", "1e12", "2e12", "--out-dir", str(tmp_path / "t")])
    assert "No events in the reference window" in str(exc.value.code)
    assert "areas span" in str(exc.value.code)


def test_na22_timewalk_recipe_uses_raw_cfd_timing(monkeypatch):
    """Run 65213 is mostly saturated, so the recipe times on raw CFD
    (raw_timing CSV, t_cfd) and never runs the deconvolution."""
    from pmtdeconv.commands import run_recipe
    steps = []
    monkeypatch.setattr(run_recipe, "_execute", lambda label, argv: steps.append(argv))
    assert cli.main(["run", "na22-timewalk", "--dataset", "na22",
                     "--runs", "65213", "--dry-run"]) == 0
    timewalk = next(a for a in steps if a[:2] == ["timewalk", "run"])
    assert timewalk[timewalk.index("--time-col") + 1] == "t_cfd"
    assert timewalk[timewalk.index("--hits-csv") + 1].endswith("raw_timing_run65213.csv")
    assert not [a for a in steps if a[0] == "hits"]
