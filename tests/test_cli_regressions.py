"""Regressions found in the 2026-10-21 simplicity review, driven through the CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from pmtdeconv import cli

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FLAT = ["--dataset", "na22", "--data-root", str(FIXTURES), "--runs", "0"]


def test_timewalk_auto_resolves_default_hits_output(tmp_path, monkeypatch):
    from pmtdeconv.io import output_paths
    from pmtdeconv.presets import resolve_settings
    monkeypatch.setattr(output_paths, "PROJECT_ROOT", tmp_path)

    assert cli.main(["hits", *FLAT, "--entry-stop", "20"]) == 0

    args = argparse.Namespace(dataset="na22", runs=[0], data_root=str(FIXTURES))
    settings, _ = resolve_settings(args)
    found = output_paths.resolve_pulse_csv(settings)
    assert found is not None and found.name == "hits.csv"
    assert tmp_path in found.parents


@pytest.mark.parametrize("command", ["hits", "area"])
def test_missing_runs_is_an_error_not_run_65213(command, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--dataset", "cd109", "--data-root", str(FIXTURES),
                  "--out-dir", str(tmp_path / "out")])
    assert "--runs" in str(exc.value.code)


def test_area_and_hits_agree_on_npe_raw(tmp_path):
    common = [*FLAT, "--entry-stop", "30", "--spe-charge", "d0_ch1=500",
              "d0_ch2=500"]
    assert cli.main(["hits", *common, "--no-vetoes",
                     "--out-dir", str(tmp_path / "hits")]) == 0
    assert cli.main(["area", *common,
                     "--out-dir", str(tmp_path / "area")]) == 0

    hits = pd.read_csv(tmp_path / "hits" / "all_results.csv")
    area = pd.read_csv(tmp_path / "area" / "areas_run0.csv")
    key = ["entry", "branch"]
    merged = (hits[key + ["npe_raw"]].dropna().drop_duplicates(key)
              .merge(area[key + ["npe_raw"]].dropna(), on=key,
                     suffixes=("_hits", "_area")))
    assert len(merged) > 0
    pd.testing.assert_series_equal(merged["npe_raw_hits"], merged["npe_raw_area"],
                                   check_names=False, rtol=1e-6)


def _flat_card_without_dt(tmp_path) -> Path:
    card = tmp_path / "dataset.yaml"
    card.write_text(f"name: mydet\ndata_dir: {FIXTURES}\nformat: flat\n")
    return card


def test_flat_card_without_dt_ns_is_refused(tmp_path):
    card = _flat_card_without_dt(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["hits", "--dataset", str(card), "--runs", "0",
                  "--allow-generic-fallback", "--out-dir", str(tmp_path / "o")])
    assert "dt_ns" in str(exc.value.code)
    assert not (tmp_path / "o").exists()


def test_flat_dt_ns_auto_from_cli_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["hits", *FLAT, "--dt-ns", "auto",
                  "--out-dir", str(tmp_path / "o")])
    assert "dt_ns" in str(exc.value.code)


def test_lint_flags_flat_card_without_dt_ns(tmp_path, capsys):
    assert cli.main(["dataset", "lint", str(_flat_card_without_dt(tmp_path))]) != 0
    assert "dt_ns" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--hash-inputs", "--no-hash-inputs"])
def test_recipe_forwards_input_hashing_choice(flag, tmp_path, capsys):
    assert cli.main(["run", "na22-timewalk", "--runs", "65213", flag,
                     "--dry-run", "--out-dir", str(tmp_path)]) == 0
    steps = [line.split() for line in capsys.readouterr().out.splitlines()
             if line.startswith("pmtdeconv ")]
    assert steps and all(flag in argv for argv in steps)


@pytest.mark.parametrize("flag", ["--scatter-tol", "--width-tol"])
def test_tune_cosine_has_no_dead_tolerance_flags(flag):
    with pytest.raises(SystemExit) as exc:
        cli.main(["tune", "cosine", "--dataset", "na22", flag, "0.1", "--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("denoiser", ["stft", "wavelet"])
def test_hits_branches_flag_restricts_flat_reader(denoiser, tmp_path):
    out = tmp_path / denoiser
    assert cli.main(["hits", *FLAT, "--entry-stop", "10", "--denoiser", denoiser,
                     "--branches", "d0_ch2", "--out-dir", str(out)]) == 0
    assert set(pd.read_csv(out / "all_results.csv")["branch"]) == {"d0_ch2"}


@pytest.mark.parametrize("command", ["hits", "area"])
def test_branch_missing_from_template_warns(command, tmp_path, capsys):
    import numpy as np
    t = np.arange(0, 100, 0.5)
    tmpl = tmp_path / "ch2_only.csv"
    pd.DataFrame({"branch": "d0_ch2", "t_ns": t,
                  "r": np.exp(-t / 10) * (1 - np.exp(-t / 2))}).to_csv(tmpl, index=False)
    assert cli.main([command, *FLAT, "--entry-stop", "5", "--template", str(tmpl),
                     "--out-dir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "no SPE template for branch 'd0_ch1'" in out
    assert "branch 'd0_ch2'" not in out


def test_run_lookup_matches_whole_run_numbers(tmp_path):
    from pmtdeconv.io.data_paths import find_run_files, resolve_run_file
    for name in ("MilliQan_Run835.1_triple.root", "MilliQan_Run835.10_triple.root",
                 "Run835.root"):
        (tmp_path / name).touch()
    glob = "MilliQan_Run835.*_triple.root"
    assert [p.name for p in find_run_files(tmp_path, glob, 1)] == [
        "MilliQan_Run835.1_triple.root"]
    assert find_run_files(tmp_path, glob, 0) == []
    assert len(find_run_files(tmp_path, glob, 835)) == 2
    assert resolve_run_file(tmp_path, 83, kind="drs") is None
    assert resolve_run_file(tmp_path, 835, kind="drs").name == "Run835.root"


@pytest.mark.parametrize("command,produced", [("hits", "all_results.csv"),
                                              ("area", "areas_run0.csv")])
def test_flat_readers_honour_tree(command, produced, tmp_path):
    import numpy as np
    import uproot
    with uproot.open(FIXTURES / "flattened_Run0.root") as f:
        waves = {b: np.asarray(f["waveforms"][b].array(library="np", entry_stop=5)
                               .tolist(), dtype=np.float32) for b in ("d0_ch1", "d0_ch2")}
    data = tmp_path / "data"
    data.mkdir()
    with uproot.recreate(data / "flattened_Run0.root") as f:
        f.mktree("pulses", {b: np.dtype((np.float32, w.shape[1])) for b, w in waves.items()})
        f["pulses"].extend(waves)
    assert cli.main([command, "--dataset", "na22", "--data-root", str(data), "--runs", "0",
                     "--tree", "pulses", "--out-dir", str(tmp_path / "out")]) == 0
    assert len(pd.read_csv(tmp_path / "out" / produced)) > 0


def _write_object_run(path, waves, dt_ns):
    import numpy as np
    import uproot
    w = np.zeros((len(waves), 2, 16, waves.shape[1]), dtype=np.float32)
    w[:, 0, 0] = waves
    with uproot.recreate(path) as f:
        f.mktree("Events", {
            "event/digitizers/digitizers.waveform": np.dtype((np.float32, w.shape[1:])),
            "event/digitizers/digitizers.nanosecondsPerSample": np.dtype((np.float64, (2,))),
        })
        f["Events"].extend({
            "event/digitizers/digitizers.waveform": w,
            "event/digitizers/digitizers.nanosecondsPerSample":
                np.full((len(waves), 2), dt_ns),
        })


def test_object_hits_reads_dt_from_each_run_file(tmp_path):
    import numpy as np
    s = np.arange(1024)
    rng = np.random.default_rng(0)
    centers = np.array([400, 500, 600])
    waves = np.array([rng.normal(0, 0.05, 1024) - 120 * np.exp(-0.5 * ((s - c) / 20) ** 2)
                      for c in centers])
    _write_object_run(tmp_path / "MilliQan_Run835.1_triple.root", waves, 1.25)
    _write_object_run(tmp_path / "MilliQan_Run835.2_triple.root", waves, 2.5)
    out = tmp_path / "out"
    assert cli.main(["hits", "--dataset", "lanl_835", "--data-root", str(tmp_path),
                     "--runs", "1", "2", "--branches", "d0_ch00", "--no-vetoes",
                     "--allow-generic-fallback", "--out-dir", str(out)]) == 0
    first = (pd.read_csv(out / "hits.csv").query("peak_id == 1")
             .pivot(index="entry", columns="run", values="t_peak"))
    # The kernel delay (true time minus t_peak) is the same in ns at either
    # sample spacing: a first-run dt, or a kernel cached at it, shifts run 2.
    delay = {dt: centers * dt - first[run].to_numpy()
             for run, dt in ((1, 1.25), (2, 2.5))}
    assert np.allclose(delay[2.5], delay[1.25], atol=5.0), delay


def test_timewalk_recipe_forwards_polarity_and_cfd_frac(tmp_path, capsys):
    assert cli.main(["run", "na22-timewalk", "--runs", "65213", "--polarity", "-1",
                     "--cfd-frac", "0.3", "--dry-run", "--out-dir", str(tmp_path)]) == 0
    steps = [line.split() for line in capsys.readouterr().out.splitlines()
             if line.startswith("pmtdeconv ")]
    assert [a[1] for a in steps] == ["raw-area", "raw-area", "timewalk"]
    for argv in (steps[0], steps[2]):
        assert argv[argv.index("--polarity") + 1] == "-1", argv
    assert "--polarity" not in steps[1], "precomputed DRS areas have no polarity"
    assert [("--cfd-frac" in a) for a in steps] == [True, False, False]
    assert steps[0][steps[0].index("--cfd-frac") + 1] == "0.3"


@pytest.mark.parametrize("flags,denoiser,source", [
    ([], "wavelet", "default (area path has no STFT)"),
    (["--denoiser", "none"], "none", "cli"),
])
def test_tune_fom_manifest_records_the_denoiser_it_runs(flags, denoiser, source, tmp_path):
    import json
    assert cli.main(["tune", "fom", *FLAT, "--preset", "na22-timewalk", *flags,
                     "--dry-run", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["settings"]["denoiser"] == denoiser
    assert manifest["settings_provenance"]["denoiser"] == source


def test_area_without_any_template_does_not_warn_per_branch(tmp_path, capsys):
    card = tmp_path / "dataset.yaml"
    card.write_text(f"name: bare\ndata_dir: {FIXTURES}\nformat: flat\ndt_ns: 0.5051\n"
                    "branches: [d0_ch1, d0_ch2]\n")
    assert cli.main(["area", "--dataset", str(card), "--runs", "0", "--entry-stop", "5",
                     "--allow-generic-fallback", "--out-dir", str(tmp_path / "out")]) == 0
    assert "no SPE template for branch" not in capsys.readouterr().out


DRS = ["raw-area", "drs", "--dataset", "na22", "--file",
       str(FIXTURES / "processed" / "Run0.root"), "--runs", "0"]


def test_raw_area_drs_honours_branches_and_entry_stop(tmp_path):
    assert cli.main([*DRS, "--branches", "d0_ch2", "--entry-stop", "5",
                     "--out-dir", str(tmp_path)]) == 0
    df = pd.read_csv(tmp_path / "area_drs_run0.csv")
    assert set(df.branch) == {"d0_ch2"} and sorted(df.entry) == list(range(5))


@pytest.mark.parametrize("flag", [["--polarity", "-1"], ["--dt-ns", "0.5"]])
def test_raw_area_drs_refuses_waveform_flags(flag, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([*DRS, *flag, "--out-dir", str(tmp_path)])
    assert flag[0] in str(exc.value.code)
