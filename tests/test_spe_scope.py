"""SPE-level scope (2026-10-23): saturated waveforms never reach the
deconvolution, and saturated-era timewalk uses raw CFD timing instead."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmtdeconv import cli

THIS_DIR = Path(__file__).resolve().parent
FIXTURES = THIS_DIR / "fixtures"
sys.path.insert(0, str(FIXTURES))


@pytest.fixture(scope="module")
def level5(tmp_path_factory):
    import make_synth_root as msr
    base = tmp_path_factory.mktemp("spe_scope")
    orig = msr.__file__
    msr.__file__ = str(base / "make_synth_root.py")
    try:
        msr.main(regime="level5", n_events=40, write_truth=True, dt_ns=0.5051)
    finally:
        msr.__file__ = orig
    return base / "regime" / "level5"


def _common(data_root):
    return ["--dataset", "na22", "--data-root", str(data_root), "--runs", "0",
            "--log-level", "quiet"]


def test_hits_drops_saturated_waveforms_even_without_vetoes(level5, tmp_path):
    assert cli.main(["hits", *_common(level5), "--no-vetoes",
                     "--out-dir", str(tmp_path)]) == 0
    rows = pd.read_csv(tmp_path / "all_results.csv")
    sat = rows[rows.reason == "saturated"]
    assert len(sat) > 0, "level5 clips some waveforms at 950 mV"
    assert not sat.success.any()
    kept = pd.read_csv(tmp_path / "hits.csv")
    assert len(kept) > 0, "level5 also has unclipped waveforms"
    assert not set(map(tuple, kept[["entry", "branch"]].values)) & set(
        map(tuple, sat[["entry", "branch"]].values))


def test_area_drops_saturated_waveforms(level5, tmp_path):
    assert cli.main(["area", *_common(level5), "--out-dir", str(tmp_path)]) == 0
    areas = pd.read_csv(tmp_path / "areas_run0.csv")
    hits_dir = tmp_path / "h"
    cli.main(["hits", *_common(level5), "--no-vetoes", "--out-dir", str(hits_dir)])
    sat = pd.read_csv(hits_dir / "all_results.csv").query("reason == 'saturated'")
    clipped = set(map(tuple, sat[["entry", "branch"]].values))
    assert clipped
    assert not set(map(tuple, areas[["entry", "branch"]].values)) & clipped
    import json
    stats = json.loads((tmp_path / "manifest.json").read_text())["stats"]
    assert stats["dropped"]["saturated"] == len(clipped)


def test_raw_cfd_timing_covers_saturated_waveforms(level5, tmp_path):
    for frac in ("0.2", "0.5"):
        assert cli.main(["raw-area", "mqana", *_common(level5), "--cfd-frac", frac,
                         "--out-dir", str(tmp_path / frac)]) == 0
    early = pd.read_csv(tmp_path / "0.2" / "raw_timing_run0.csv")
    late = pd.read_csv(tmp_path / "0.5" / "raw_timing_run0.csv")
    area = pd.read_csv(tmp_path / "0.2" / "area_mqana_run0.csv")
    assert len(early) == len(area)
    assert early.success.all() and np.isfinite(early.t_cfd).all()
    assert area.area.isna().any(), "saturated rows exist and still get a time"
    assert (early.t_cfd.values <= late.t_cfd.values).all()
    assert (early.t_cfd.values < late.t_cfd.values).mean() > 0.9


def test_timewalk_runs_on_raw_cfd_timing(level5, tmp_path):
    raw = tmp_path / "raw"
    assert cli.main(["raw-area", "mqana", *_common(level5), "--out-dir", str(raw)]) == 0
    assert cli.main(["timewalk", "run", *_common(level5),
                     "--area-csv", str(raw / "area_mqana_run0.csv"),
                     "--hits-csv", str(raw / "raw_timing_run0.csv"),
                     "--time-col", "t_cfd", "--a-ref", "0", "1e9",
                     "--out-dir", str(tmp_path / "tw")]) == 0
    assert (tmp_path / "tw" / "timewalk_summary_run0.csv").exists()


def test_cfd_time_fraction():
    from pmtdeconv.core.peaks import cfd_time
    u = np.zeros(300)
    u[90:101] = np.linspace(0.0, 100.0, 11)
    t = np.arange(300) * 2.0
    assert abs(cfd_time(u, 100, t, frac=0.2) - 184.0) < 1e-9
    assert abs(cfd_time(u, 100, t) - 190.0) < 1e-9


@pytest.fixture(scope="module")
def two_photon(tmp_path_factory):
    from pmtdeconv.sim import tang2024 as T
    out = tmp_path_factory.mktemp("two_photon")
    rng = np.random.default_rng(7)
    waves = np.zeros((40, T.N_SAMPLES), dtype=np.float32)
    truth = []
    for ev in range(40):
        w, rows = T.make_event(rng, 2, hit_times_ns=np.array([150.0, 156.0]))
        waves[ev] = w
        for r in rows:
            r["event"] = ev
        truth.extend(rows)
    T.write_run(out, 0, waves, truth)
    T.write_dataset_card(out, T.write_template_csv(out))
    return out


def test_hits_keeps_the_earliest_photon_of_a_pair(two_photon, tmp_path):
    """The raw-amplitude veto must look where the photon's RAW pulse is (one
    template peak delay after its deconvolved arrival), or the earlier of two
    close photons is vetoed with its charge."""
    assert cli.main(["hits", "--dataset", str(two_photon), "--runs", "0",
                     "--log-level", "quiet", "--out-dir", str(tmp_path)]) == 0
    hits = pd.read_csv(tmp_path / "hits.csv")
    first = hits.groupby("entry").t_peak.min()
    assert (np.abs(first - 150.0) <= 2.0).mean() >= 0.9
    truth = pd.read_csv(two_photon / "flattened_Run0.truth.csv").groupby("event").true_charge_pVs.sum()
    ratio = hits.groupby("entry").charge.sum().reindex(truth.index, fill_value=0) / truth
    assert abs(np.median(ratio) - 1.0) < 0.15


@pytest.fixture(scope="module")
def wandering_baseline(tmp_path_factory):
    """Cd109-like: slow pre-trigger baseline structure (tails / recovery of
    earlier pulses) on top of 0.42 mV sample noise. Runs 1 = a 1-PE pulse,
    2 = no pulse at all."""
    from pmtdeconv.sim import tang2024 as T
    out = tmp_path_factory.mktemp("wander")
    rng = np.random.default_rng(11)
    t = np.arange(T.N_SAMPLES)
    kern = T.central_kernel()
    kern = kern / kern.max()
    for run, amp in ((1, 15.0), (2, 0.0)):
        waves = np.zeros((30, T.N_SAMPLES), dtype=np.float32)
        truth = []
        for ev in range(30):
            wander = 4.0 * np.sin(2 * np.pi * t / 700.0 + rng.uniform(0, 2 * np.pi))
            w = wander + rng.normal(0.0, 0.42, T.N_SAMPLES)
            w[300:300 + len(kern)] += amp * kern[:T.N_SAMPLES - 300]
            truth.append({"event": ev, "branch": T.BRANCH, "pulse_id": 0, "true_amp_mv": amp,
                          "true_center_ns": 300.0, "true_charge_pVs": amp / 15.0, "true_sigma_ns": 1.0,
                          "saturated": False, "n_pe": amp / 15.0})
            waves[ev] = w
        T.write_run(out, run, waves, truth)
    T.write_dataset_card(out, T.write_template_csv(out))
    return out


def test_raw_snr_gate_ignores_slow_baseline_structure(wandering_baseline, tmp_path):
    """The gate's noise must be the sample-to-sample noise: slow pre-trigger
    structure inflated the level σ 5× on real Cd109 and the 10σ gate then
    dropped 72% of CH2's 1-PE waveforms (2026-10-23)."""
    assert cli.main(["hits", "--dataset", str(wandering_baseline), "--runs", "1", "2",
                     "--veto-profile", "benchmark_hits", "--log-level", "quiet",
                     "--out-dir", str(tmp_path)]) == 0
    a = pd.read_csv(tmp_path / "all_results.csv").drop_duplicates(["run", "entry"])
    pulse = a[a.run == 1]
    assert (pulse.reason == "no_raw_pulse").mean() < 0.1
    flat = a[a.run == 2]
    assert flat.reason.isin(["no_raw_pulse", "no_trigger", "baseline_quality"]).mean() > 0.5


@pytest.fixture(scope="module")
def pulse_in_dip(tmp_path_factory):
    """A 40-PE pulse whose own AC-coupled undershoot (part of the R5912
    model) sags the baseline, with a 1-PE pulse inside the sag."""
    from pmtdeconv.sim import tang2024 as T
    out = tmp_path_factory.mktemp("dip")
    rng = np.random.default_rng(13)
    waves = np.zeros((30, T.N_SAMPLES), dtype=np.float32)
    truth = []
    for ev in range(30):
        w = np.zeros(T.N_SAMPLES)
        for pid, (tc, q) in enumerate(((150.0, 40.0), (260.0, 1.0))):
            k = T.tang_spe_kernel(T.SpeParams())
            i0 = int(tc)
            w[i0:i0 + len(k)] += q * k[:T.N_SAMPLES - i0]
            truth.append({"event": ev, "branch": T.BRANCH, "pulse_id": pid, "true_amp_mv": q * T.MAIN_PEAK_MV,
                          "true_center_ns": tc, "true_charge_pVs": q, "true_sigma_ns": 1.0,
                          "saturated": False, "n_pe": q})
        waves[ev] = w + rng.normal(0.0, T.NOISE_MV, T.N_SAMPLES)
    T.write_run(out, 0, waves, truth)
    T.write_dataset_card(out, T.write_template_csv(out))
    return out


def test_no_raw_peak_measures_against_a_sagged_baseline(pulse_in_dip, tmp_path):
    """A 1-PE pulse inside the post-pulse baseline sag barely rises above the
    event pedestal; the raw check must measure it from the sagged local
    baseline or it is vetoed as no_raw_peak."""
    assert cli.main(["hits", "--dataset", str(pulse_in_dip), "--runs", "0",
                     "--log-level", "quiet", "--out-dir", str(tmp_path)]) == 0
    hits = pd.read_csv(tmp_path / "hits.csv")
    late = hits[(hits.t_peak > 255) & (hits.t_peak < 265)]
    assert late.entry.nunique() >= 25


def test_plot_energy_raw_check_follows_the_kernel_delay():
    from pmtdeconv.plotting.energy import _qualifying_charge
    n, dt, delay = 1024, 1.0, 40
    t = np.arange(n) * dt
    u = 50.0 * np.exp(-0.5 * ((t - 300) / 3.0) ** 2)
    raw = 30.0 * np.exp(-0.5 * ((t - 300 - delay) / 8.0) ** 2)
    kept = _qualifying_charge(u, raw, 0.0, 1.0, 5, dt, delay, n)[1]
    unshifted = _qualifying_charge(u, raw, 0.0, 1.0, 5, dt, 0, n)[1]
    assert (kept, unshifted) == (1, 0)


def test_hits_wavelet_adds_no_edge_peaks_before_a_late_pulse(tmp_path):
    import uproot
    from pmtdeconv.core.kernels import build_spe_template
    from pmtdeconv.io.templates import load_template
    from pmtdeconv.presets.datasets import CD109_STACKED_UNLOCKED_CSV
    n, dt = 1024, 0.5051
    k = build_spe_template(n, dt, load_template(CD109_STACKED_UNLOCKED_CSV)["d0_ch1"])
    k /= k.max()
    rng = np.random.default_rng(3)
    t = np.arange(n) * dt
    waves = []
    for _ in range(40):
        late = rng.uniform(380, 480)
        pulse = np.roll(k, int(late / dt) - int(np.argmax(k))) * (t >= late - 60)
        waves.append((rng.uniform(40, 120) * pulse + rng.normal(0, 1.0, n)).astype(np.float32))
    with uproot.recreate(tmp_path / "flattened_Run0.root") as f:
        f.mktree("waveforms", {"d0_ch1": np.dtype((np.float32, n))})
        f["waveforms"].extend({"d0_ch1": np.array(waves)})
    out = tmp_path / "out"
    assert cli.main(["hits", "--dataset", "na22", "--data-root", str(tmp_path), "--runs", "0",
                     "--branches", "d0_ch1", "--denoiser", "wavelet", "--no-vetoes",
                     "--out-dir", str(out)]) == 0
    found = pd.read_csv(out / "all_results.csv")
    early = found[(found.success == True) & (found.t_peak < 30)].entry.nunique()  # noqa: E712
    assert early <= 4, f"{early}/40 events have a peak in the first 30 ns (no pulse there)"
