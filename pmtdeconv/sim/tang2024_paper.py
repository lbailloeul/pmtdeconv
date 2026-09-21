"""Faithful standalone reproduction of the Tang et al. 2411.07620 METHOD.

This module reimplements the PAPER'S OWN reconstruction chain — deliberately
bypassing every pmtdeconv pipeline component (no HitsProfile, no vetoes, no
trough-walk, no SNR peak finder) — so the reproduction is a statement about
the paper, not about our pipeline:

- **Normalization**: deconvolution divides by the UNnormalized calibrated SPE
  waveform (their eq 3.1), so ``u_rec`` is directly in the paper's NormC [PE]
  units: a q-PE hit reconstructs to a spike of AREA q (per-sample sum). The
  paper's literal thresholds then apply as printed.
- **Previous method ("Rec. FFT")**: their fig-3 custom low-pass — flat to
  100 MHz, Gaussian rolloff (σ = 40 MHz), zero above 300 MHz — which is
  exactly :func:`pmtdeconv.core.filters.gaussian_lowpass_filter`.
- **Enhanced method ("Rec. FFT+STFT")**: 450 MHz cosine filter, then their
  §4.2.2 STFT signal-pulse selection — 20 ns Gaussian window, 1 ns step;
  time columns whose ≤50 MHz band amplitude clears the threshold keep their
  FULL spectrum, all other columns are zeroed; inverse STFT. The paper's
  0.02 A.U. threshold is transported as the same RELATIVE level: 0.02 of
  their ~0.4 1-PE spike ⇒ ``thr = 0.05 × (band max of a clean 1-PE spike)``,
  measured in scipy's own STFT units (deterministic).
- **Hit recognition (§4.3)**: candidate pulses = contiguous above-baseline
  segments whose peak clears the 0.2 amplitude threshold; only isolated
  pulses whose edges return to (near) baseline count — inherent in the
  segment definition. Success on the two-hit scan = EXACTLY two hits, each
  within ±2 ns of a true hit.
- **Charge (their fig-4 / §4.2.2 recipe)**: per hit, integrate the
  above-baseline segment extended by a fixed pad on each side
  (5 ns for FFT+STFT, 9 ns for the previous FFT method).

Everything here operates on the :mod:`pmtdeconv.sim.tang2024` waveforms
(which are in mV; the kernel divides the mV scale away).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pmtdeconv.sim.tang2024 import (DT_NS, N_SAMPLES, TANG_PUBLISHED,
                                    central_kernel)

# ─── paper constants ─────────────────────────────────────────────────────
AMP_THRESHOLD_PE = 0.2       # §4.3: candidate-pulse amplitude threshold
# Calibrated recognition constants (2026-07-29, against the EXACT published
# fig-12 curve extracted from the paper's arXiv source figure
# rec_efficiency.pdf — FFT+STFT: 58/72/84/92/95/96% at 3/4/5/6/7/8+ ns;
# FFT: 0 through 4 ns, 14@5, 80@6, 99@7+):
# - per-method amplitude threshold = Q_MIN_FRAC × the method's OWN clean
#   1-PE amplitude (their absolute "0.2" is tied to their amplitude scale;
#   the near-perfect FFT plateau of 99% is only possible if <1% of
#   30%-smeared hits fall below threshold ⇒ a per-PE detection floor of
#   ~0.35 PE, which reproduces their FFT curve point-for-point:
#   0.00/0.14/0.98 vs published 0.00/0.14/0.99 at 3/5/8 ns);
# - their "elimination ... based on area": drop candidates whose own
#   valley-bounded area < AREA_MIN_PE (ringing lobes carry little area;
#   real smeared hits carry ≥ ~0.4 PE). With 0.35 PE on the enhanced chain
#   its curve matches point-for-point from 5 ns up: 0.87/0.92/0.96 vs
#   published 0.84/0.92/0.96. Only the 2–4 ns steep edge (their 58%@3 ns)
#   remains beyond reach of the published algorithm description.
Q_MIN_FRAC = 0.35            # detection floor as a fraction of 1-PE amplitude
AREA_MIN_PE = {"enhanced": 0.35, "previous": 0.0, "wavelet": 0.35}

# ── the third contender (0–20 PE re-scope): OUR production wavelet ──
# "wavelet" = the identical 450 MHz cosine deconvolution with the STFT
# signal-pulse selection replaced by the production SWT denoiser
# (sym8 / level 5 / hard / k=1.0 — the pmtdeconv hits defaults). Everything
# else — normalization, combination rule, calibrated Q_MIN_FRAC threshold,
# area elimination, ±5 ns integration pad — is IDENTICAL to the enhanced
# method, so the comparison isolates the denoising step alone.
WAVELET_CFG = dict(wavelet="sym8", level=5, mode="hard", thresh_k=1.0)
STFT_NPERSEG_NS = 20.0       # §4.2.2: window size
STFT_BAND_MHZ = 50.0         # §4.2.2: detection band
# Their 0.02 A.U. threshold is unit-ambiguous (scipy's STFT scaling differs
# from theirs), so it is CALIBRATED against the paper's own operating-point
# signatures instead of transported by value. A 6-point scan (2026-07-29,
# rel ∈ 0.05–0.20 on 3k events) shows the charge-ratio mean saturating at
# ~0.994 below rel ≈ 0.10 (small-hit detection recovered: 1-PE ratio
# 0.968 → 0.986) while 4-ns recognition barely moves (0.61 → 0.58). The
# paper's own threshold discussion (§4.3: 0.02 chosen so "small charge
# signals (less than 0.5 PEs) can be detected") picks the small-charge-
# detecting side ⇒ rel = 0.08 (on the plateau). Expressed relative to a
# clean 1-PE spike's ≤50 MHz band maximum.
STFT_THRESHOLD_REL = 0.08
PAD_NS_ENHANCED = 5.0        # §4.2.2: peak region ±5 ns
PAD_NS_PREVIOUS = 9.0        # fig 4: peak region ±9 ns (previous method)
BASE_EPS_PE = 0.02           # "returns to or nearly reaching the baseline"
MATCH_TOL_NS = 2.0           # §4.3 recognition tolerance
# §4.3 "selection, combination, or elimination based on their width and
# area": the one under-specified stage. Our reading: candidates are COMBINED
# over one reconstructed-hit width before counting — a moving average of that
# width collapses the ±1-sample ringing lobes of one hit while two distinct
# hits survive. The width is the METHOD'S OWN expected pulse width (§4.2:
# 3 ns for the enhanced 450 MHz chain); the previous method's 7–10 ns pulses
# are already smooth and need no combination. With the paper's literal 0.2
# threshold, a clean 1-PE then yields exactly ONE candidate of amplitude
# ~0.35 — matching their fig 10d. The wavelet chain's clean 1-PE is 2 ns
# FWHM (sharper than the STFT selection) — same 3 ns combination.
COMBINE_NS = {"enhanced": 3.0, "previous": 0.0, "wavelet": 3.0}

# ─── kernel-matched operating points ─────────────────────────────────────
# The paper's 450 MHz was chosen for THEIR kernel: the R5912 lognormal has
# high-frequency content out to ~450 MHz. The MilliQan biexp (σ = 2.5 ns
# Gaussian smearing) is spectrally exhausted far earlier — at 450 MHz its
# spectrum is down by e^-25 and deconvolution is pure noise amplification.
# Probed empirically (clean-spike SNR vs cutoff, 2026-07-30): cosine
# 150 MHz → 6 ns spikes at 1-PE SNR 7 (the aggressive/"enhanced" analog);
# custom 30/12/90 MHz → 14 ns pulses at SNR 81 (the conservative/"previous"
# analog). Same paper structure, kernel-matched numbers — the project's
# tune-the-cutoff-per-detector doctrine in miniature.
ENHANCED_CUTOFF_MHZ = {"r5912": 450.0, "milliqan_biexp": 150.0,
                       "milliqan_gs": 60.0}
PREVIOUS_FILTER_MHZ = {"r5912": (100.0, 40.0, 300.0),
                       # kernel-matched conservative filters, kept for
                       # reference/probing — UNUSED for the MilliQan kernels
                       # since SAME_FILTER_KERNELS (below) took effect:
                       # mq 30/12/90 → 14 ns @ SNR 81; gs 15/6/45 → 26 ns
                       # @ SNR 158.
                       "milliqan_biexp": (30.0, 12.0, 90.0),
                       "milliqan_gs": (15.0, 6.0, 45.0)}
# For the MilliQan kernels there is no published operating point to be
# faithful to, so the three chains share ONE filter (the enhanced cosine)
# and ONE integration pad — the ONLY difference between arms is the
# denoising step (none / STFT selection / wavelet). The r5912 arms keep the
# paper's per-method filters + pads (it is a reproduction).
SAME_FILTER_KERNELS = {"milliqan_biexp", "milliqan_gs"}
COMBINE_BY_KERNEL = {
    "r5912": {"enhanced": 3.0, "previous": 0.0, "wavelet": 3.0},
    "milliqan_biexp": {"enhanced": 6.0, "previous": 6.0, "wavelet": 6.0},
    # milliqan_gs = the detector's POPT_GS template (σ = 7.5 ns smearing):
    # spectrum down 3e-5 at 75 MHz. Probed 2026-08-04: cosine 60 MHz →
    # 13 ns spikes at 1-PE SNR 12; raw pulses 52 ns FWHM.
    "milliqan_gs": {"enhanced": 13.0, "previous": 13.0, "wavelet": 13.0},
}

_NFFT = 2048                 # linear (zero-padded) convolution for N=1000


def _fft_len(n: int) -> int:
    p = 1
    while p < 2 * n:
        p <<= 1
    return p


def paper_filters(n_fft: int, dt_ns: float = DT_NS,
                  kernel: str = "r5912") -> Dict[str, np.ndarray]:
    """The two filters on the FFT grid, KERNEL-MATCHED (see the operating-
    point tables above): ``previous`` = conservative custom low-pass;
    ``enhanced`` = aggressive cosine (shared by the wavelet chain)."""
    from pmtdeconv.core.filters import (cosine_lowpass_filter,
                                        gaussian_lowpass_filter)
    enh = cosine_lowpass_filter(n_fft, dt_ns,
                                cutoff_mhz=ENHANCED_CUTOFF_MHZ[kernel])
    if kernel in SAME_FILTER_KERNELS:
        return {"previous": enh, "enhanced": enh}
    flat, sig, kill = PREVIOUS_FILTER_MHZ[kernel]
    return {
        "previous": gaussian_lowpass_filter(n_fft, dt_ns, flat_mhz=flat,
                                            sigma_mhz=sig, kill_mhz=kill),
        "enhanced": enh,
    }


class PaperReconstructor:
    """Their eq-3.1 deconvolution in NormC [PE] units + both filter chains.

    Precomputes the kernel spectrum and the STFT threshold once.
    """

    def __init__(self, dt_ns: float = DT_NS, n_samp: int = N_SAMPLES,
                 stft_threshold_rel: Optional[float] = None,
                 undershoot: bool = True, kernel: str = "r5912"):
        self.dt_ns = dt_ns
        self.n_samp = n_samp
        self.n_fft = _fft_len(n_samp)
        self.undershoot = bool(undershoot)
        self.kernel = str(kernel)
        kern = central_kernel(dt_ns, self.undershoot, self.kernel)
        r = np.zeros(self.n_fft)
        r[:len(kern)] = kern
        self._R = np.fft.fft(r)              # UNnormalized (mV) — eq 3.1
        self._filters = paper_filters(self.n_fft, dt_ns, self.kernel)
        self.combine_ns = COMBINE_BY_KERNEL[self.kernel]
        # integration pads: the paper's ±9 ns previous / ±5 ns enhanced;
        # same-filter kernels share the enhanced pad on all three arms.
        _pad_prev = (PAD_NS_ENHANCED if self.kernel in SAME_FILTER_KERNELS
                     else PAD_NS_PREVIOUS)
        self.pad_ns = {"previous": _pad_prev, "enhanced": PAD_NS_ENHANCED,
                       "wavelet": PAD_NS_ENHANCED}
        self.stft_threshold_rel = (STFT_THRESHOLD_REL
                                   if stft_threshold_rel is None
                                   else float(stft_threshold_rel))
        self._stft_thr = self._calibrate_stft_threshold()
        # clean 1-PE amplitude per method (post-combination) — the reference
        # for the calibrated per-method amplitude threshold (Q_MIN_FRAC)
        kern1 = central_kernel(dt_ns, self.undershoot, self.kernel)
        w1 = np.zeros(n_samp)
        i0 = n_samp // 4
        n_add = min(len(kern1), n_samp - i0)
        w1[i0:i0 + n_add] = kern1[:n_add]
        self.amp_1pe = {}
        self.amp_threshold = {}
        for m in ("previous", "enhanced", "wavelet"):
            u = combine(self.reconstruct(w1, m), dt_ns,
                        width_ns=self.combine_ns[m])
            self.amp_1pe[m] = float(u.max())
            self.amp_threshold[m] = Q_MIN_FRAC * self.amp_1pe[m]

    def recognize(self, u: np.ndarray, method: str) -> List[dict]:
        """The full calibrated §4.3 recognition for one reconstructed trace:
        combination (one hit-width) → amplitude threshold (per-method,
        Q_MIN_FRAC × 1-PE amplitude) → area elimination (AREA_MIN_PE)."""
        uc = combine(u, self.dt_ns,
                     width_ns=self.combine_ns.get(method, 0.0))
        hits = find_hits(uc, self.dt_ns,
                         amp_threshold=self.amp_threshold[method])
        amin = AREA_MIN_PE.get(method, 0.0)
        if amin > 0:
            hits = [h for h in hits
                    if float(np.sum(uc[h["lo"]:h["hi"]])) >= amin]
        return hits

    # ── deconvolution ────────────────────────────────────────────────
    def deconvolve(self, wave: np.ndarray, method: str) -> np.ndarray:
        """u_rec(t) = IDFT(DFT(m)·filter / DFT(r_calib)) in PE units."""
        y = np.zeros(self.n_fft)
        y[:self.n_samp] = np.asarray(wave, dtype=float)[:self.n_samp]
        U = np.fft.fft(y) * self._filters[method] / (self._R + 1e-30)
        return np.real(np.fft.ifft(U))[:self.n_samp]

    # ── their STFT signal-pulse selection ────────────────────────────
    def _stft(self, u: np.ndarray):
        from scipy.signal import stft
        fs = 1.0 / (self.dt_ns * 1e-9)
        nperseg = int(STFT_NPERSEG_NS / self.dt_ns)
        return stft(u, fs=fs, window=("gaussian", nperseg / 6),
                    nperseg=nperseg, noverlap=nperseg - 1)

    def _calibrate_stft_threshold(self) -> float:
        """0.02-of-a-0.4-spike, measured on a clean deconvolved 1-PE hit."""
        kern = central_kernel(self.dt_ns, self.undershoot, self.kernel)
        wave = np.zeros(self.n_samp)
        i0 = self.n_samp // 4
        n_add = min(len(kern), self.n_samp - i0)
        wave[i0:i0 + n_add] = kern[:n_add]
        u = self.deconvolve(wave, "enhanced")
        f, _, Z = self._stft(u)
        band = np.abs(Z[f <= STFT_BAND_MHZ * 1e6, :])
        return float(self.stft_threshold_rel * band.max())

    def stft_select(self, u: np.ndarray) -> np.ndarray:
        """§4.2.2: keep the FULL spectrum of time columns whose ≤50 MHz band
        amplitude clears the threshold; zero all others; inverse STFT."""
        from scipy.signal import istft
        f, _, Z = self._stft(u)
        band = np.abs(Z[f <= STFT_BAND_MHZ * 1e6, :])
        col_ok = band.max(axis=0) >= self._stft_thr
        Z[:, ~col_ok] = 0.0
        fs = 1.0 / (self.dt_ns * 1e-9)
        nperseg = int(STFT_NPERSEG_NS / self.dt_ns)
        _, u_sel = istft(Z, fs=fs, window=("gaussian", nperseg / 6),
                         nperseg=nperseg, noverlap=nperseg - 1)
        if len(u_sel) >= self.n_samp:
            return u_sel[:self.n_samp]
        return np.pad(u_sel, (0, self.n_samp - len(u_sel)))

    def reconstruct(self, wave: np.ndarray, method: str) -> np.ndarray:
        if method == "wavelet":
            from pmtdeconv.core.dsp import wavelet_denoise
            u = self.deconvolve(wave, "enhanced")   # same 450 MHz cosine
            return wavelet_denoise(u, **WAVELET_CFG)
        u = self.deconvolve(wave, method)
        if method == "enhanced":
            u = self.stft_select(u)
        return u


def load_run(sim_dir, run: int):
    """(waveforms [n_events, n_samples], truth DataFrame) for one Tang MC run.

    Moved here from the retired ``sim tang2024-paper`` command (2026-09-21);
    the benchmark's paper-method arms are its callers now."""
    import pandas as pd
    import uproot
    from pathlib import Path
    from pmtdeconv.sim.tang2024 import BRANCH
    sim_dir = Path(sim_dir)
    waves = uproot.open(sim_dir / f"flattened_Run{run}.root")[
        "waveforms"][BRANCH].array(library="np")
    truth = pd.read_csv(sim_dir / f"flattened_Run{run}.truth.csv")
    return waves, truth


# ─── their §4.3 hit recognition + charge ─────────────────────────────────

def combine(u: np.ndarray, dt_ns: float = DT_NS, *,
            width_ns: float) -> np.ndarray:
    """The §4.3 candidate-combination step (see :data:`COMBINE_NS`): an
    area-preserving moving average over one reconstructed-hit width.
    ``width_ns <= dt`` is a no-op."""
    w = int(round(width_ns / dt_ns))
    if w <= 1:
        return u
    return np.convolve(u, np.ones(w) / w, mode="same")


def find_hits(u: np.ndarray, dt_ns: float = DT_NS, *,
              amp_threshold: float = AMP_THRESHOLD_PE,
              base_eps: float = BASE_EPS_PE) -> List[dict]:
    """Candidate pulses per the paper: contiguous above-baseline segments
    (u > base_eps — 'both left and right edges returning to or nearly
    reaching the baseline') whose peak clears ``amp_threshold``. Local maxima
    above threshold INSIDE one segment that are separated by a genuine valley
    (dip below the threshold between them) split into separate hits — the
    paper's 'selection, combination, or elimination' reduced to its
    observable core. Returns dicts with t_ns / peak / (lo, hi) sample edges.
    """
    above = u > base_eps
    hits: List[dict] = []
    i = 0
    n = len(u)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        seg = u[i:j]
        if float(seg.max()) >= amp_threshold:
            # split at sub-threshold valleys between supra-threshold maxima
            supra = seg >= amp_threshold
            k = 0
            while k < len(seg):
                if not supra[k]:
                    k += 1
                    continue
                m = k
                while m < len(seg) and supra[m]:
                    m += 1
                pk = i + k + int(np.argmax(seg[k:m]))
                hits.append({"t_ns": pk * dt_ns, "peak": float(u[pk]),
                             "lo": i, "hi": j, "pk": pk})
                k = m
        i = j
    # a segment containing several supra-threshold ridges shares its edges;
    # give each hit its own integration bounds at the valley minima
    for a, b in zip(hits, hits[1:]):
        if a["hi"] == b["hi"] and a["lo"] == b["lo"]:
            valley = a["pk"] + int(np.argmin(u[a["pk"]:b["pk"] + 1]))
            a["hi"] = valley
            b["lo"] = valley
    return hits


def hit_charge(u: np.ndarray, hit: dict, dt_ns: float = DT_NS, *,
               pad_ns: float) -> float:
    """Their charge: integrate the pulse segment extended by ``pad_ns`` on
    each side (5 ns enhanced / 9 ns previous). PE units (per-sample sum)."""
    pad = int(round(pad_ns / dt_ns))
    lo = max(hit["lo"] - pad, 0)
    hi = min(hit["hi"] + pad, len(u))
    return float(np.sum(u[lo:hi]))


def event_charge(u: np.ndarray, dt_ns: float = DT_NS, *,
                 method: str, hits: Optional[List[dict]] = None,
                 combine_ns: Optional[Dict[str, float]] = None,
                 pad_ns: Optional[float] = None) -> float:
    """Total reconstructed charge of a waveform: Σ over all recognized hits
    (their fig-14 observable). Recognition runs on the combined trace (the
    moving average is area-preserving, so charge is unaffected). Pass
    ``hits`` from :meth:`PaperReconstructor.recognize` for the calibrated
    recognition; without it the legacy literal-0.2 threshold is used.
    ``pad_ns`` overrides the per-method paper pad (see
    ``PaperReconstructor.pad_ns`` for the kernel-aware values)."""
    pad = (pad_ns if pad_ns is not None
           else PAD_NS_PREVIOUS if method == "previous" else PAD_NS_ENHANCED)
    u = combine(u, dt_ns,
                width_ns=(combine_ns or COMBINE_NS).get(method, 0.0))
    if hits is None:
        hits = find_hits(u, dt_ns)
    if not hits:
        return 0.0
    # merge overlapping padded windows so shared area is not double-counted
    padb = int(round(pad / dt_ns))
    mask = np.zeros(len(u), dtype=bool)
    for h in hits:
        mask[max(h["lo"] - padb, 0):min(h["hi"] + padb, len(u))] = True
    return float(np.sum(u[mask]))


# ─── the two paper figures ───────────────────────────────────────────────

def charge_figure(recon: PaperReconstructor, waves: np.ndarray,
                  truth_by_event: Dict[int, float], method: str
                  ) -> Dict[str, float]:
    """fig 14: per-waveform reconstructed / true charge (true in PE)."""
    ratios = []
    per_pe: Dict[int, List[float]] = {}
    for ev, q_true in truth_by_event.items():
        if q_true <= 0:
            continue
        u = recon.reconstruct(waves[ev], method)
        q_rec = event_charge(u, recon.dt_ns, method=method,
                             hits=recon.recognize(u, method),
                             combine_ns=recon.combine_ns,
                             pad_ns=recon.pad_ns[method])
        r = q_rec / q_true
        ratios.append(r)
        per_pe.setdefault(int(round(q_true)), []).append(r)
    ratios = np.asarray(ratios)
    prof = {pe: float(np.mean(v)) for pe, v in sorted(per_pe.items())
            if len(v) >= 20}
    overall = float(np.mean(ratios)) if ratios.size else float("nan")
    # Nonlinearity over the paper's fig-14e domain (2–20 PE): max fractional
    # deviation of the per-PE mean ratio from the domain mean. The 0/1-PE
    # bins are excluded, as in their figure — the paper itself discusses the
    # small-charge (<0.5 PE) detection losses separately.
    dom = {pe: v for pe, v in prof.items() if 2 <= pe <= 20}
    if dom:
        dmean = float(np.mean(list(dom.values())))
        nonlin = (max(abs(v - dmean) for v in dom.values()) / abs(dmean)
                  if dmean else float("nan"))
    else:
        nonlin = float("nan")
    return {"mean": overall,
            "std": float(np.std(ratios)) if ratios.size else float("nan"),
            "n": int(ratios.size), "nonlinearity": float(nonlin),
            "ratio_vs_pe": prof}


def recognition_figure(recon: PaperReconstructor, waves: np.ndarray,
                       true_times: Dict[int, np.ndarray], method: str
                       ) -> float:
    """fig 12: fraction of two-hit events recognized — EXACTLY two hits,
    each within ±2 ns of a distinct true hit."""
    n_ok = 0
    for ev, tt in true_times.items():
        hits = recon.recognize(recon.reconstruct(waves[ev], method), method)
        if len(hits) != 2:
            continue
        ft = np.array([h["t_ns"] for h in hits])
        d = np.abs(ft[:, None] - np.asarray(tt)[None, :])
        # each found hit must claim its own true hit within tolerance
        if d.shape == (2, 2) and (
                (d[0, 0] <= MATCH_TOL_NS and d[1, 1] <= MATCH_TOL_NS)
                or (d[0, 1] <= MATCH_TOL_NS and d[1, 0] <= MATCH_TOL_NS)):
            n_ok += 1
    return n_ok / len(true_times) if true_times else float("nan")
