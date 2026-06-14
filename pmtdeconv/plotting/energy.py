"""Per-channel energy (charge) histograms with method overlay.

Lifted from ``cd109_energy_hist.py`` (PR 7 / dataset-agnostic). Both
the legacy script and ``pmtdeconv plot energy`` call into
:func:`render_energy_hist`. The only Cd109-specific parts that remain
in the legacy script are the default constants (PROC_ROOT, BOARD, DT,
TEMPLATE_CSV) — when called via ``pmtdeconv plot energy`` everything
flows from the dataset preset + CLI flags.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from pmtdeconv.utils import (                                       # noqa: E402
    deconvolve_hits, cosine_lowpass_filter,
    find_integration_windows, apply_peak_vetoes,
)
# Step 6: the wavelet/STFT curves go through the SAME denoiser wrapper as
# `hits` (pipeline.denoise.denoise), so the plot's "wavelet" curve is provably
# the identical transform — no more divergent hardcoded module constants.
from pmtdeconv.pipeline.denoise import denoise
from pmtdeconv.plotting.style import use_root_style, style_root_axes

# Overlay methods + plot styles (shared with the legacy CLI)
METHODS = ["raw", "fft", "stft", "wavelet"]
MSTYLE = {
    "raw":     ("dimgray",  "raw area (no deconv)"),
    "fft":     ("orange",   "FFT only"),
    "stft":    ("royalblue", "STFT"),
    "wavelet": ("crimson",  "wavelet"),
}

BASE_BINS = 30
PRE_BINS = 50


def _biexp_smeared(t, A, tau_rise, tau_fall, t0, sigma, dt):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t >= t0
    dd = t[m] - t0
    tr = max(tau_rise, 1e-3); tf = max(tau_fall, 1e-3)
    out[m] = np.exp(-dd / tf) - np.exp(-dd / tr)
    return A * gaussian_filter1d(out, max(sigma / dt, 1e-6), mode="constant")


def _build_kernel(row, n: int, dt: float, n_ref: int = 0) -> np.ndarray:
    t = np.arange(n) * dt
    k = _biexp_smeared(t, 1.0, row.tau_rise_ns, row.tau_fall_ns,
                       n_ref * dt, row.sigma_ns, dt)
    s = k.sum()
    return k / s if s > 0 else k


def _load(proc_root: Path, board: int, channels: Iterable[int],
          tree_name: str, entry_stop: Optional[int]) -> dict:
    f = uproot.open(proc_root)
    keys = [k for k in f.keys() if k.split(';')[0] == tree_name]
    if not keys:
        raise RuntimeError(f"No '{tree_name}' tree in {proc_root}")
    t = f[keys[-1]]
    branch_names = [
        f"{p}_{board}_{c}"
        for c in channels
        for p in ("voltages", "tMax", "vMax", "tStart", "tEnd", "area")
    ]
    A = t.arrays(branch_names, library="np", entry_stop=entry_stop)
    out = {}
    for c in channels:
        vv = A[f"voltages_{board}_{c}"]
        tm = A[f"tMax_{board}_{c}"]; vm = A[f"vMax_{board}_{c}"]
        ts = A[f"tStart_{board}_{c}"]; te = A[f"tEnd_{board}_{c}"]
        ar = A[f"area_{board}_{c}"]
        n = len(vv)
        fin = np.all(np.isfinite(vv), axis=1) & np.isfinite(ar) & np.isfinite(vm)
        base = np.full(n, np.nan)
        base[fin] = np.median(vv[fin, :BASE_BINS], axis=1)
        pre = vv[fin, BASE_BINS:BASE_BINS + PRE_BINS] - base[fin, None]
        pm = np.full(n, np.nan); pr = np.full(n, np.nan)
        pm[fin] = pre.mean(axis=1); pr[fin] = pre.std(axis=1)
        mask = (fin & (np.abs(pm) <= 1.0) & (pr <= 1.0) & (ar > 0)
                & (te > ts) & (tm >= 125) & (tm <= 1300) & (vm > 0))
        out[c] = dict(v=vv, base=base, ts=ts, vm=vm, mask=mask)
    return out


def _qualifying_charge(u, raw, ped, noise, pad_bins: int, dt: float,
                       n_samp: int) -> Tuple[float, int]:
    height = max(4.0 * noise, 5.0)
    peaks, _ = find_peaks(u, height=height, distance=max(int(8.0 / dt), 1))
    if len(peaks) == 0:
        return 0.0, 0
    windows = find_integration_windows(u, peaks, pad_bins)
    pulse_widths = [hi - lo for lo, hi in windows]
    t = np.arange(n_samp) * dt
    q = 0.0; nkeep = 0
    for pk_i, pk in enumerate(peaks):
        lo, hi = windows[pk_i]
        success, _reason, charge = apply_peak_vetoes(
            pk, pk_i, peaks, u, t, windows, pulse_widths, n_samp,
            t_peak_min_ns=20.0, t_peak_max_ns=1400.0,
            afterpulse_window_ns=40.0, afterpulse_height_ratio=5.0,
            post_sat_cutoff_bin=n_samp, dt_ns=dt,
            raw_wave=raw, raw_ped=ped, min_raw_amp=10.0,
            min_charge_purity=0.5,
        )
        if success:
            q += charge
            nkeep += 1
    return float(q), nkeep


def _process_channel(d, kern, filt, c_lo, c_hi, pad_bins, dt, n_samp, dn):
    """``dn`` carries the denoiser knobs (wavelet + STFT) so the wavelet/STFT
    curves use the exact same ``pipeline.denoise.denoise`` transform as the
    hits pipeline."""
    idx = np.where(d["mask"])[0]
    whole = {m: np.full(len(idx), np.nan) for m in METHODS}
    qual = {m: np.full(len(idx), np.nan) for m in METHODS}
    pre_n = int(100 / dt)
    for j, i in enumerate(idx):
        ped = d["base"][i]
        raw = d["v"][i].astype(np.float64)
        y = raw - ped
        u_fft = deconvolve_hits(y, kern, filt)
        u_stft = denoise(u_fft, denoiser="stft", dt_ns=dt,
                         stft_nperseg_ns=dn["stft_nperseg_ns"],
                         stft_freq_cutoff_hz=dn["stft_freq_cutoff_mhz"] * 1e6,
                         stft_amp_threshold=dn["stft_amp_threshold"])
        u_wav = denoise(u_fft, denoiser="wavelet",
                        wavelet=dn["wavelet"], wlevel=dn["wlevel"],
                        wmode=dn["wmode"], wthresh_k=dn["wthresh_k"])
        whole["raw"][j] = float(np.sum(y[c_lo:c_hi]))
        for m, u in (("fft", u_fft), ("stft", u_stft), ("wavelet", u_wav)):
            whole[m][j] = float(np.sum(u[c_lo:c_hi]))
            seg = u[c_lo:c_lo + pre_n]
            noise = 1.4826 * np.median(np.abs(seg - np.median(seg)))
            qual[m][j], _ = _qualifying_charge(u, raw, ped, noise,
                                               pad_bins, dt, n_samp)
    return idx, whole, qual


def render_energy_hist(
    *,
    proc_root: Path,
    template_csv: Path,
    out_dir: Path,
    dataset_label: str = "",
    run_label: str = "",
    dt: float = 1.432,
    n_samp: int = 1024,
    board: int = 2984,
    channels: Iterable[int] = (1, 2),
    tree_name: str = "Events",
    cutoff_mhz: float = 50.0,
    nbins: int = 120,
    entry_stop: Optional[int] = None,
    # Step 6: denoiser knobs flow from Settings (default = the hits-pipeline
    # defaults) so the wavelet/STFT curves match `hits --denoiser <x>` exactly.
    wavelet: str = "sym8",
    wlevel: int = 5,
    wmode: str = "hard",
    wthresh_k: float = 1.0,
    stft_nperseg_ns: float = 20.0,
    stft_freq_cutoff_mhz: float = 175.0,
    stft_amp_threshold: float = 10.0,
) -> dict:
    """End-to-end energy-histogram renderer.

    Returns ``{"png": Path, "csv": Path, "summary": dict}``. Both the
    PNG (4 panels: whole vs qualifying × ch1 / ch2) and the per-event
    CSV land in ``out_dir``.
    """
    dn = {"wavelet": wavelet, "wlevel": wlevel, "wmode": wmode,
          "wthresh_k": wthresh_k, "stft_nperseg_ns": stft_nperseg_ns,
          "stft_freq_cutoff_mhz": stft_freq_cutoff_mhz,
          "stft_amp_threshold": stft_amp_threshold}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc_root = Path(proc_root)
    template_csv = Path(template_csv)
    channels = tuple(channels)

    # Build per-channel kernel from the stacked template CSV
    tdf = pd.read_csv(template_csv)
    if "channel" in tdf.columns:
        ch_col = "channel"
    elif "chan" in tdf.columns:
        ch_col = "chan"
    else:
        raise ValueError(
            f"Template CSV {template_csv} has no 'channel' or 'chan' column.")
    kern = {c: _build_kernel(tdf[tdf[ch_col] == c].iloc[0], n_samp, dt)
            for c in channels}
    filt = cosine_lowpass_filter(n_samp, dt, cutoff_mhz=cutoff_mhz)
    c_lo, c_hi = int(20 / dt), n_samp - int(20 / dt)
    pad_bins = int(50 / dt)

    print(f"loading {proc_root.name} (entry_stop={entry_stop}) …")
    data = _load(proc_root, board, channels, tree_name, entry_stop)

    res = {}
    rows = []
    for c in channels:
        n_evt = int(data[c]["mask"].sum())
        print(f"  deconvolving ch{c}: {n_evt} events …")
        idx, whole, qual = _process_channel(
            data[c], kern[c], filt, c_lo, c_hi, pad_bins, dt, n_samp, dn)
        res[c] = (whole, qual)
        for j, i in enumerate(idx):
            rows.append(dict(channel=c, entry=int(i),
                             **{f"whole_{m}": whole[m][j] for m in METHODS},
                             **{f"qual_{m}":  qual[m][j]  for m in METHODS}))

    # 2 rows (whole / qualifying) × len(channels) cols
    use_root_style()
    n_cols = len(channels)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 9),
                              squeeze=False)
    for col, c in enumerate(channels):
        whole, qual = res[c]
        ref = whole["raw"][np.isfinite(whole["raw"]) & (whole["raw"] > 0)]
        hi = float(np.percentile(ref, 99.5)) if ref.size else 1.0
        edges = np.linspace(0, hi, nbins + 1)
        ax = axes[0][col]
        for m in METHODS:
            v = whole[m][np.isfinite(whole[m])]
            col_, lab = MSTYLE[m]
            ax.hist(v, bins=edges, histtype="step", lw=1.6, color=col_, label=lab)
        ax.set_title(f"d0_ch{c} — whole-waveform")
        ax.set_yscale("log"); ax.set_xlabel("charge [a.u.]"); ax.set_ylabel("events / bin")
        ax.legend(); style_root_axes(ax)

        ax = axes[1][col]
        for m in ["fft", "stft", "wavelet"]:
            v = qual[m][np.isfinite(qual[m]) & (qual[m] > 0)]
            col_, lab = MSTYLE[m]
            ax.hist(v, bins=edges, histtype="step", lw=1.6, color=col_, label=lab)
        ax.set_title(f"d0_ch{c} — qualifying pulses")
        ax.set_yscale("log"); ax.set_xlabel("charge [a.u.]"); ax.set_ylabel("events / bin")
        ax.legend(); style_root_axes(ax)

    fig.tight_layout()
    png_path = out_dir / "energy_hist.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "energy_charges.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {png_path}")
    print(f"saved {csv_path}  ({len(df)} rows)")

    summary = {}
    for c in channels:
        sub = df[df.channel == c]
        summary[f"ch{c}_whole_median"] = {
            m: float(sub[f"whole_{m}"].median()) for m in METHODS}
    return {"png": png_path, "csv": csv_path, "summary": summary}
