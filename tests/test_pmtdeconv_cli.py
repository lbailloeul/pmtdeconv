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
import tempfile
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
from pmtdeconv.presets.configs import CONFIGS, list_configs
from pmtdeconv.presets.datasets import DATASETS, list_datasets


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
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
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
    ["pulse", "--help"],
    ["waveform", "--help"],
    ["raw_area", "--help"],
    ["raw_area", "mqana", "--help"],
    ["raw_area", "drs", "--help"],
    ["raw_area", "offline", "--help"],
    ["timewalk", "--help"],
    ["timewalk", "run", "--help"],
    ["timewalk", "gif", "--help"],
    ["plot", "--help"],
    ["plot", "marginals", "--help"],
    ["plot", "energy", "--help"],
    ["plot", "analyze", "--help"],
    ["info", "--help"],
    ["info", "artifacts", "--help"],
    ["config", "--help"],
    ["config", "resolve", "--help"],
    ["results", "--help"],
    ["results", "list", "--help"],
])
def test_cli_help_runs(argv):
    """Every --help path exits 0 (= argparse SystemExit code 0)."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 0


def test_deconv_command_group_is_not_public():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["deconv", "pulse", "--help"])
    assert exc.value.code == 2


def test_area_is_a_leaf_not_a_raw_area_group():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["area", "mqana"])
    assert exc.value.code == 2


# ─── dataset preset resolution ────────────────────────────────────

def test_list_datasets_has_seven():
    names = list_datasets()
    assert set(names) == {"na22", "na22_3ch", "cd109", "milliqan",
                          "lanl_806", "lanl_835", "lanl_movedBar"}


@pytest.mark.parametrize("name,expected", [
    ("na22",         dict(dt_ns=0.5051, polarity=+1,
                          cosine_cutoff_mhz=450.0)),
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

def test_list_configs_includes_winning():
    assert "na22-timewalk-winning-2026-05" in list_configs()


def test_config_preset_overrides_dataset():
    args = _make_args(dataset="na22", preset="na22-timewalk-winning-2026-05")
    settings, prov = resolve_settings(args)
    # Cosine cutoff is the same as the dataset (both 450) — but the
    # preset still claims provenance because it explicitly set it.
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
    assert settings.cosine_cutoff_mhz == 450.0    # from dataset
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
    pulse_out = tmp_path / "pulse"
    assert cli.main([
        "pulse", "--out-dir", str(pulse_out), "--dry-run",
        "--log-level", "quiet",
    ]) == 0
    assert json.loads((pulse_out / "manifest.json").read_text())["subcommand"] == "pulse"

    waveform_out = tmp_path / "waveform"
    assert cli.main([
        "waveform", "--out-dir", str(waveform_out), "--dry-run",
        "--log-level", "quiet",
    ]) == 0
    assert json.loads((waveform_out / "manifest.json").read_text())["subcommand"] == "waveform"


def test_stft_pulse_dry_run_writes_canonical_manifest(tmp_path):
    out_dir = tmp_path / "pulse_stft"
    assert cli.main([
        "pulse", "--denoiser", "stft", "--out-dir", str(out_dir),
        "--dry-run", "--log-level", "quiet",
    ]) == 0
    m = json.loads((out_dir / "manifest.json").read_text())
    assert m["subcommand"] == "pulse"
    assert m["settings"]["denoiser"] == "stft"


def test_real_cli_parser_does_not_override_preset_defaults(tmp_path):
    out_dir = tmp_path / "hits"
    assert cli.main([
        "hits", "--preset", "na22-timewalk", "--runs", "65213",
        "--out-dir", str(out_dir), "--dry-run", "--log-level", "quiet",
    ]) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    provenance = manifest["settings_provenance"]
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
    for step in ("raw-area", "hits", "timewalk"):
        assert (workflow / step / "manifest.json").exists()

    assert cli.main([
        "results", "list", "--outputs-dir", str(workflow),
        "--log-level", "quiet",
    ]) == 0
    output = capsys.readouterr().out
    assert "raw-area" in output
    assert "hits" in output
    assert "timewalk" in output

    assert cli.main([
        "results", "list", "--outputs-dir", str(workflow),
        "--command", "hits", "--log-level", "quiet",
    ]) == 0
    filtered = capsys.readouterr().out
    assert "hits" in filtered
    assert "raw-area" not in filtered

    assert cli.main([
        "results", "path", "hits", "--outputs-dir", str(workflow),
        "--log-level", "quiet",
    ]) == 0
    assert str(workflow / "hits") in capsys.readouterr().out


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
    assert payload["provenance"]["wthresh_k"] == "config"


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


def test_deconv_pulse_result_tables_write_canonical_and_compat(tmp_path):
    rows = [
        {"success": True, "reason": "", "charge": 1.0},
        {"success": False, "reason": "negative_charge", "charge": -0.2},
    ]
    tables = write_deconv_pulse_result_tables(rows, tmp_path)

    compat = tmp_path / "deconv_pulses.csv"
    assert tables.deconv_pulse_csv.name == "deconv_pulse.csv"
    assert tables.deconv_pulse_csv.exists()
    assert compat.exists()
    assert tables.deconv_pulse_csv.read_bytes() == compat.read_bytes()
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
