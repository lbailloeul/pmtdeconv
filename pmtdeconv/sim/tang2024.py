"""Tang et al. 2411.07620 Monte-Carlo waveform simulation.

Faithful-as-recoverable reimplementation of the paper's Section 2 simulation
(Daya Bay R5912 SPE model per Jetter et al. / Huang et al., Table 1
parameters), used to (a) validate our pipeline against the paper's published
numbers and (b) benchmark denoisers on the exact population that established
the 0–20 PE applicability domain.

Model (positive-going, our pipeline convention):

- **Main peak**: log-normal ``U0 · exp(-½ (ln(t/τ)/σ)²)`` with U0 = 42 mV,
  τ = 8.4 ± 0.3 ns (Gaussian-fluctuated per hit), σ = 0.28 ± 0.02.
- **Undershoot** (their eq 2.2, approximated from the paper text): two decay
  exponentials (τ_fast = 45 ns, amp 0.11 ± 0.02; τ_slow = 290 ns, amp
  0.03 ± 0.005 — relative to the main peak) with a Fermi-function onset
  (delay t_d1 = 20 ns, width τ_fermi = 10 ns), plus a Gaussian trough bump
  (amp 0.4×U_fast, σ = 20 ns). Composed from
  :func:`pmtdeconv.core.kernels.spe_overshoot` + the slow exponential.
- **Reflections**: 12 copies of the main-peak shape at 7 ns spacing, relative
  amplitudes 0.24, 0.18, 0.14, 0.10, 0.08, 0.05, 0.02, 0.02, 0.01, 0.008,
  0.008, 0.008 (the first three Gaussian-fluctuated ±0.02/±0.02/±0.015).
- **Events**: n_hits ~ U{0..20} (the mix adds a 1-PE-only sample for low-PE
  statistics, paper ratio 195k:5k); hit times = trigger + the liquid-
  scintillator fluorescence mix (τ = 7 ns @ 80.5%, τ = 31 ns @ 19.5%);
  per-hit charge ~ N(1, 0.30) PE (truncated > 0.05); white noise σ = 0.7 mV;
  1000 samples @ 1 GHz (dt = 1 ns); ADC quantization at 2.2 ADC/mV.

Exact equation forms for the undershoot are not fully recoverable from the
paper; the model is validated INSTEAD against the paper's published numbers
(:data:`TANG_PUBLISHED`) — charge-ratio mean/σ, ≤1% nonlinearity over
0–20 PE, and the two-hit recognition-efficiency crossover.

Everything is driven by an explicit ``numpy.random.Generator`` — same seed,
same bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pmtdeconv.core.kernels import spe_overshoot

# ─── published reference numbers (paper §4.3, fig 12/14) ─────────────────
TANG_PUBLISHED: Dict[str, float] = {
    "charge_ratio_mean": 1.004,        # rec/true per waveform, both methods
    "ratio_sigma_fft_stft": 0.076,     # std of the ratio, FFT+STFT
    "ratio_sigma_fft": 0.080,          # std of the ratio, plain FFT
    "nonlinearity_frac": 0.01,         # |mean ratio − 1| across 0–20 PE bins
    "single_hit_width_ns": 3.0,        # reconstructed width @ 450 MHz cosine
    # recognition-efficiency behavior (fig 12): FFT+STFT recovers 3–6 ns
    # separations that plain FFT misses; ~3% WORSE at 7–10 ns.
}

# Fig 12 two-hit recognition efficiency vs separation [ns], READ OFF the
# paper's arXiv source figure (rec_efficiency.pdf) — approximate, and the
# curve the paper-method recognition constants were CALIBRATED against, so
# agreement with it is a consistency check, not independent validation.
# Keyed by the benchmark arm that runs each chain. (Previously this lived only
# in a code comment in sim/tang2024_paper.py.)
TANG_PUBLISHED_RECOGNITION: Dict[str, Dict[int, float]] = {
    "tang2024":     {3: 0.58, 4: 0.72, 5: 0.84, 6: 0.92, 7: 0.95, 8: 0.96},
    "tang2024-fft": {3: 0.00, 4: 0.00, 5: 0.14, 6: 0.80, 7: 0.99, 8: 0.99},
}

# ─── Table 1 (central values ± Gaussian fluctuations) ────────────────────
MAIN_PEAK_MV = 42.0          # U0 — 1-PE main-peak amplitude
TAU_NS = (8.4, 0.3)          # log-normal time scale (fluctuated)
SHAPE_SIGMA = (0.28, 0.02)   # log-normal width (fluctuated)
U_FAST = (0.11, 0.02)        # fast undershoot amp, rel. to peak (fluctuated)
U_SLOW = (0.03, 0.005)       # slow undershoot amp (fluctuated)
TAU_FERMI_NS = 10.0
TAU_FAST_NS = 45.0
TAU_SLOW_NS = 290.0
T_D1_NS = 20.0               # undershoot onset delay
T_D2_NS = 5.0                # secondary delay constant
U_GAUS_FRAC = 0.4            # Gaussian trough bump = 0.4 × U_fast
SIGMA_US_NS = 20.0
REFL_SPACING_NS = 7.0
REFL_AMPS = (0.24, 0.18, 0.14, 0.10, 0.08, 0.05,
             0.02, 0.02, 0.01, 0.008, 0.008, 0.008)
REFL_SIGMAS = (0.02, 0.02, 0.015)      # first three fluctuate

# ─── digitization / event constants (paper §2 steps 4–5) ─────────────────
DT_NS = 1.0
N_SAMPLES = 1000
NOISE_MV = 0.7
ADC_PER_MV = 2.2
TRIGGER_NS = 150.0           # hits start here (≥150 quiet pre-samples)
LS_TAU_FAST = (7.0, 0.805)   # fluorescence decay mix (τ ns, weight)
LS_TAU_SLOW = (31.0, 0.195)
SPE_CHARGE_SMEAR = 0.30      # per-hit N(1, 0.30) PE
KERNEL_LEN_NS = 400.0        # sampled-kernel length (covers refl + undershoot)

BRANCH = "d0_ch1"

# ─── alternative kernel family: the MilliQan lab biexp ────────────────────
# Our own detectors' SPE shape (pmtdeconv MILLIQAN_TEMPLATE): a simple
# gaussian-smeared biexponential with NO undershoot and NO reflections —
# τ_rise 12.37 ns, τ_fall 18.09 ns, σ 2.51 ns. Peak scaled to the same
# 42 mV as the R5912 model so ONLY the pulse shape changes (controlled
# comparison; real lab SPEs are ~15–26 mV, i.e. lower SNR than simulated
# here). Per-hit shape fluctuation: 4% Gaussian scales on τ_rise/τ_fall/σ —
# the same few-percent kernel-mismatch realism as the R5912 Table-1 rows.
MQ_TAU_RISE_NS = 12.3709373015379
MQ_TAU_FALL_NS = 18.0852574335868
MQ_SIGMA_NS = 2.50780690507711
MQ_FLUCT = 0.04

# "milliqan_gs" = the MilliQan DETECTOR template: timewalksim's POPT_GS
# (repo-root constants.py), the gaussian_smeared_biexponential fit to the
# pulse-injection averaged waveform (dataset812_averaged_waveform.root).
# Much slower than our lab biexp — τ_fall 39.2 ns, σ 7.5 ns smearing. The
# fitted amplitude is 65.6 mV, but the peak is rescaled to the same 42 mV
# here so only the SHAPE changes across kernel families. t0 within the
# kernel grid is 30 ns (4σ of leading smear); it cancels in deconvolution.
GS_TAU_RISE_NS = 7.64452628
GS_TAU_FALL_NS = 39.19087399
GS_SIGMA_NS = 7.52914099

# (tau_rise, tau_fall, sigma, kernel-grid t0) per biexp-family kernel
BIEXP_PARAMS = {
    "milliqan_biexp": (MQ_TAU_RISE_NS, MQ_TAU_FALL_NS, MQ_SIGMA_NS, 10.0),
    "milliqan_gs": (GS_TAU_RISE_NS, GS_TAU_FALL_NS, GS_SIGMA_NS, 30.0),
}


def biexp_spe_kernel(rng: "Optional[np.random.Generator]" = None, *,
                     dt_ns: float = DT_NS,
                     length_ns: float = KERNEL_LEN_NS,
                     kernel: str = "milliqan_biexp") -> np.ndarray:
    """MilliQan-style biexp 1-PE waveform in mV (peak 42 mV). ``rng=None``
    gives the central (calibrated-template) shape; with an rng the three
    shape parameters fluctuate by ``MQ_FLUCT`` per hit."""
    from pmtdeconv.core.kernels import gaussian_smeared_biexponential
    tau_r, tau_f, sig, t0 = BIEXP_PARAMS[kernel]
    if rng is None:
        sr = sf = ss = 1.0
    else:
        sr, sf, ss = (max(float(rng.normal(1.0, MQ_FLUCT)), 0.5)
                      for _ in range(3))
    t = np.arange(0.0, length_ns, dt_ns)
    g = gaussian_smeared_biexponential(
        t, 1.0, tau_r * sr, tau_f * sf, t0=t0, sigma=sig * ss)
    m = float(g.max())
    return MAIN_PEAK_MV * g / m if m > 0 else g


@dataclass(frozen=True)
class SpeParams:
    """One hit's (possibly fluctuated) SPE shape parameters."""
    tau: float = TAU_NS[0]
    sigma: float = SHAPE_SIGMA[0]
    u_fast: float = U_FAST[0]
    u_slow: float = U_SLOW[0]
    refl: Tuple[float, ...] = REFL_AMPS


def sample_spe_params(rng: np.random.Generator) -> SpeParams:
    """Per-hit shape fluctuation (Table 1 'Gaussian' rows) — the built-in
    kernel mismatch."""
    refl = list(REFL_AMPS)
    for i, s in enumerate(REFL_SIGMAS):
        refl[i] = max(float(rng.normal(REFL_AMPS[i], s)), 0.0)
    return SpeParams(
        tau=max(float(rng.normal(*TAU_NS)), 1.0),
        sigma=max(float(rng.normal(*SHAPE_SIGMA)), 0.05),
        u_fast=max(float(rng.normal(*U_FAST)), 0.0),
        u_slow=max(float(rng.normal(*U_SLOW)), 0.0),
        refl=tuple(refl),
    )


def _main_peak(t: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """Log-normal main peak, unit amplitude at its maximum."""
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.exp(-0.5 * (np.log(t[pos] / tau) / sigma) ** 2)
    return out


def tang_spe_kernel(params: SpeParams = SpeParams(), *,
                    dt_ns: float = DT_NS,
                    length_ns: float = KERNEL_LEN_NS,
                    undershoot: bool = True) -> np.ndarray:
    """The 1-PE waveform r(t) in mV on a ``dt_ns`` grid (positive-going,
    peak = 42 mV × the fluctuated shape), including undershoot + 12
    reflections."""
    t = np.arange(0.0, length_ns, dt_ns)
    main = _main_peak(t, params.tau, params.sigma)
    wave = main.copy()
    for k, amp in enumerate(params.refl, start=1):
        wave += amp * _main_peak(t - k * REFL_SPACING_NS, params.tau,
                                 params.sigma)
    # undershoot: fast exponential + Gaussian bump via the Daya Bay helper,
    # slow exponential with the same Fermi onset added explicitly.
    t_rel = t - T_D2_NS
    under = spe_overshoot(t_rel, U_os0=params.u_fast,
                          U_os1=U_GAUS_FRAC * params.u_fast,
                          tau_os=TAU_FAST_NS,
                          t0_os=T_D1_NS, sigma_os=SIGMA_US_NS,
                          t_fermi=T_D1_NS, w_fermi=TAU_FERMI_NS)
    fermi = 1.0 / (np.exp((T_D1_NS - t_rel) / TAU_FERMI_NS) + 1.0)
    under = under + params.u_slow * fermi * np.exp(
        np.clip(-t_rel / TAU_SLOW_NS, -700, 0))
    # `undershoot=False`: the AC-coupling undershoot is an R5912/FADC-circuit
    # feature; undershoot-free waveforms are closer to our lab PMT+DRS4
    # response (biexp-like) and remove the undershoot-amplitude fluctuation
    # as a kernel-mismatch source.
    if undershoot:
        wave = wave - under
    return MAIN_PEAK_MV * wave


def central_kernel(dt_ns: float = DT_NS, undershoot: bool = True,
                   kernel: str = "r5912") -> np.ndarray:
    """The calibrated template r_calib (central parameters) for the chosen
    kernel family: "r5912" (the paper's model), "milliqan_biexp" (our lab
    fit) or "milliqan_gs" (the detector's POPT_GS template)."""
    if kernel in BIEXP_PARAMS:
        return biexp_spe_kernel(None, dt_ns=dt_ns, kernel=kernel)
    return tang_spe_kernel(SpeParams(), dt_ns=dt_ns, undershoot=undershoot)


def central_spe_charge_pvs(dt_ns: float = DT_NS, undershoot: bool = True,
                           kernel: str = "r5912") -> float:
    """Net area of the central 1-PE waveform in mV·ns = pVs — the EXACT
    ``spe_charge`` anchor for this simulation (deconvolved charge of a
    q-PE hit ≈ q × this)."""
    return float(central_kernel(dt_ns, undershoot, kernel).sum() * dt_ns)


def sample_hit_times(rng: np.random.Generator, n: int) -> np.ndarray:
    """Hit times (ns): trigger + the LS fluorescence two-exponential mix."""
    fast = rng.random(n) < LS_TAU_FAST[1]
    tau = np.where(fast, LS_TAU_FAST[0], LS_TAU_SLOW[0])
    return TRIGGER_NS + rng.exponential(tau)


def make_event(rng: np.random.Generator, n_hits: int, *,
               dt_ns: float = DT_NS, n_samp: int = N_SAMPLES,
               adc_quantize: bool = True,
               hit_times_ns: Optional[np.ndarray] = None,
               undershoot: bool = True,
               kernel: str = "r5912",
               ) -> Tuple[np.ndarray, List[dict]]:
    """One simulated waveform + its per-hit truth rows.

    Each hit: charge q ~ N(1, 0.30) PE (truncated > 0.05), time from the LS
    profile (or ``hit_times_ns`` for controlled scans, e.g. the two-hit Δt
    scan), waveform contribution q × (per-hit FLUCTUATED kernel). Truth
    ``true_charge_pVs`` = q × central-kernel area (the ideal linear answer;
    kernel fluctuation is part of the measured resolution, as in the paper).
    """
    wave = np.zeros(n_samp, dtype=np.float64)
    a_central = central_spe_charge_pvs(dt_ns, undershoot, kernel)
    if hit_times_ns is None:
        hit_times_ns = sample_hit_times(rng, n_hits)
    truth: List[dict] = []
    for pid in range(n_hits):
        q = float(rng.normal(1.0, SPE_CHARGE_SMEAR))
        while q <= 0.05:
            q = float(rng.normal(1.0, SPE_CHARGE_SMEAR))
        if kernel in BIEXP_PARAMS:
            kern = biexp_spe_kernel(rng, dt_ns=dt_ns, kernel=kernel)
        else:
            params = sample_spe_params(rng)
            kern = tang_spe_kernel(params, dt_ns=dt_ns,
                                   undershoot=undershoot)
        i0 = int(round(hit_times_ns[pid] / dt_ns))
        n_add = min(len(kern), n_samp - i0)
        if n_add > 0:
            wave[i0:i0 + n_add] += q * kern[:n_add]
        truth.append({
            "event": -1, "branch": BRANCH, "pulse_id": pid,
            "true_amp_mv": q * MAIN_PEAK_MV,
            "true_center_ns": float(hit_times_ns[pid]),
            "true_charge_pVs": q * a_central,
            "true_sigma_ns": 1.0,          # their u(t) box width
            "saturated": False,
            "n_pe": q,
        })
    wave += rng.normal(0.0, NOISE_MV, n_samp)
    if adc_quantize:
        lsb = 1.0 / ADC_PER_MV
        wave = np.round(wave / lsb) * lsb
    return wave.astype(np.float32), truth


def make_mix(rng: np.random.Generator, n_events: int, *,
             one_pe_frac: float = 5000.0 / 195000.0,
             adc_quantize: bool = True,
             undershoot: bool = True,
             kernel: str = "r5912",
             ) -> Tuple[np.ndarray, List[dict]]:
    """The paper's charge-reconstruction sample: ``n_events`` waveforms with
    n_hits ~ U{0..20}, plus ``one_pe_frac × n_events`` exactly-1-PE waveforms
    (their 195k:5k low-PE statistics booster). Returns (waves2d, truth_rows)
    with truth ``event`` indices filled in."""
    n_one = int(round(one_pe_frac * n_events))
    waves = np.zeros((n_events + n_one, N_SAMPLES), dtype=np.float32)
    truth: List[dict] = []
    for ev in range(n_events):
        n_hits = int(rng.integers(0, 21))
        w, rows = make_event(rng, n_hits, adc_quantize=adc_quantize,
                             undershoot=undershoot, kernel=kernel)
        waves[ev] = w
        for r in rows:
            r["event"] = ev
        truth.extend(rows)
    for k in range(n_one):
        ev = n_events + k
        w, rows = make_event(rng, 1, adc_quantize=adc_quantize,
                             undershoot=undershoot, kernel=kernel)
        waves[ev] = w
        for r in rows:
            r["event"] = ev
        truth.extend(rows)
    return waves, truth


def make_two_hit_scan(rng: np.random.Generator, *,
                      intervals_ns=tuple(range(1, 11)),
                      per_interval: int = 10000,
                      adc_quantize: bool = True,
                      undershoot: bool = True,
                      kernel: str = "r5912",
                      ) -> Dict[int, Tuple[np.ndarray, List[dict]]]:
    """The paper's fig-12 sample: two 1-PE-scale hits at a FIXED separation,
    ``per_interval`` waveforms per Δt ∈ ``intervals_ns``. Returns
    {dt_ns_int: (waves2d, truth_rows)} — one run per Δt so the recognition
    curve is a per-run scoring pass."""
    out: Dict[int, Tuple[np.ndarray, List[dict]]] = {}
    for dt_sep in intervals_ns:
        waves = np.zeros((per_interval, N_SAMPLES), dtype=np.float32)
        truth: List[dict] = []
        for ev in range(per_interval):
            t0 = TRIGGER_NS + float(rng.uniform(0.0, 20.0))
            times = np.array([t0, t0 + float(dt_sep)])
            w, rows = make_event(rng, 2, hit_times_ns=times,
                                 adc_quantize=adc_quantize,
                                 undershoot=undershoot, kernel=kernel)
            waves[ev] = w
            for r in rows:
                r["event"] = ev
            truth.extend(rows)
        out[int(dt_sep)] = (waves, truth)
    return out


# ─── writers: standard flat ROOT + truth + sampled template + card ───────

def write_run(out_dir, run_number: int, waves2d: np.ndarray,
              truth_rows: List[dict]) -> None:
    """Write ``flattened_Run<N>.root`` (flat ``waveforms`` tree, branch
    d0_ch1) + the sibling truth CSV — the standard fixture layout the whole
    pipeline consumes unchanged."""
    import warnings
    import pandas as pd
    import uproot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_path = out_dir / f"flattened_Run{run_number}.root"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with uproot.recreate(root_path) as f:
            # same TTree layout as tests/fixtures/make_synth_root._write_root
            f["waveforms"] = {BRANCH: waves2d}
    from pmtdeconv.io.truth import save_truth
    save_truth(pd.DataFrame(truth_rows),
               out_dir / f"flattened_Run{run_number}.truth.csv")
    (out_dir / ".dt_ns").write_text(f"{DT_NS}\n")


def write_template_csv(out_dir, undershoot: bool = True,
                       kernel: str = "r5912") -> Path:
    """Write the central-parameter kernel as a SAMPLED template CSV
    (schema #4: branch, t_ns, r) — the calibrated r_calib of the paper."""
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kern = central_kernel(undershoot=undershoot, kernel=kernel)
    t = np.arange(len(kern)) * DT_NS
    path = out_dir / "spe_template_tang2024_sampled.csv"
    pd.DataFrame({"branch": BRANCH, "t_ns": t, "r": kern}).to_csv(
        path, index=False)
    return path


def write_veto_profile(out_dir) -> Path:
    """Write the ``tang2024_hits`` file-backed veto profile.

    The ``benchmark_hits`` thresholds under their own name, so a simulation
    run records which profile it used.
    """
    from dataclasses import replace
    from pmtdeconv.presets.veto_profiles import (BENCHMARK_HITS_PROFILE,
                                                 save_veto_profile_yaml)
    prof = replace(BENCHMARK_HITS_PROFILE, name="tang2024_hits",
                   version="tang2024-1")
    return save_veto_profile_yaml(prof, Path(out_dir))


def write_dataset_card(out_dir, template_csv, undershoot: bool = True,
                       kernel: str = "r5912") -> Path:
    """Write a dataset card so ``--dataset <out_dir>`` runs the whole
    pipeline on the simulation with the exact PE anchor + the tailored
    ``tang2024_hits`` veto profile (written beside it)."""
    import yaml
    out_dir = Path(out_dir)
    card = {
        "name": "tang2024",
        "data_dir": ".",
        "file_glob": "flattened_Run*.root",
        "format": "flat",
        "tree": "waveforms",
        "branches": [BRANCH],
        "dt_ns": DT_NS,
        "polarity": 1,
        "cosine_cutoff_mhz": 450.0,      # the paper's headline filter
        "template": str(Path(template_csv).name),
        "spe_charge": {BRANCH: round(central_spe_charge_pvs(undershoot=undershoot, kernel=kernel), 3)},
        "pe_band": [0.0, 20.0],
        "notes": ("Tang et al. 2411.07620 MC reproduction "
                  "(pmtdeconv.sim.tang2024). spe_charge = analytic central-"
                  "kernel net area — an EXACT PE anchor. veto_profile = "
                  "tang2024_hits (benchmark_hits with the post-sat veto off; "
                  "the modeled AC undershoot is signal, not saturation)."),
        "settings": {"veto_profile": "tang2024_hits"},
    }
    write_veto_profile(out_dir)
    path = out_dir / "dataset.yaml"
    path.write_text(yaml.safe_dump(card, sort_keys=False))
    return path
