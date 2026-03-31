import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, find_peaks, stft
import uproot
from scipy.signal import istft

dt_ns = 1.25
DTYPE = np.float32
POPT_GS = [400, 12.3709373015379, 18.0852574335868, 654.265529056178, 2.50780690507711, -3.2105948951947]

def is_empty_event(y, dt_ns, t_min=0.0, t_max=1200.0, baseline_tail_ns=200.0, thr_sigma=6.0):
    """
    Fast pre-check: returns True if event likely contains no pulse.
    Uses robust baseline/noise estimate and checks max deviation above baseline.
    """
    t = np.arange(len(y), dtype=float) * dt_ns
    region = (t >= t_min) & (t <= t_max)
    if not np.any(region):
        return True

    yR = y[region]
    tail_bins = max(int(baseline_tail_ns / dt_ns), 10)
    y_tail = yR[-tail_bins:] if len(yR) >= tail_bins else yR

    base = float(np.median(y_tail))
    noise = float(1.4826 * np.median(np.abs(y_tail - np.median(y_tail))))
    noise = max(noise, 1e-12)

    return (float(np.max(yR) - base) < thr_sigma * noise)

def find_peak_windows_local_ratio(
    t, y, dt_ns,
    t_min=0.0, t_max=1200.0,
    smooth_sigma_ns=2.0,
    half_window_ns=50.0,
    ratio=5.0,
    min_distance_ns=30.0,
    pad_left_ns=25.0,
    pad_right_ns=120.0,
    max_peaks=12,
    baseline_override=None):
    """
    Peak definition:
      peak at p if z[p] > ratio * mean(z in +/- half_window_ns around p, excluding p)
    where z is a smoothed, baseline-subtracted waveform.
    """
    out = []
    region = (t >= t_min) & (t <= t_max)
    if not np.any(region):
        return out

    tR = t[region]
    yR = y[region]
    sigma_bins = max(smooth_sigma_ns / dt_ns, 0.5)
    yS = gaussian_filter1d(yR, sigma_bins)

    if baseline_override is None:
        # old tail-based baseline
        tail_bins = max(int(200.0 / dt_ns), 10)
        y_tail = yS[-tail_bins:] if len(yS) >= tail_bins else yS
        baseline = float(np.median(y_tail))
    else:
        baseline = float(baseline_override)

    z = yS - baseline

    # Candidate peaks: local maxima only
    distance_bins = max(int(min_distance_ns / dt_ns), 1)
    candidates, _ = find_peaks(z, distance=distance_bins)
    if len(candidates) == 0:
        return out

    hw = max(int(half_window_ns / dt_ns), 1)

    keep = []
    for p in candidates:
        lo = max(0, p - hw)
        hi = min(len(z), p + hw + 1)

        # exclude the peak bin itself from neighborhood mean
        neigh = np.concatenate([z[lo:p], z[p+1:hi]])
        if neigh.size < 5:
            continue

        # robust local noise estimate from neighborhood
        med = float(np.median(neigh))
        sigma_local = 1.4826 * float(np.median(np.abs(neigh - med)))  # MAD -> sigma
        sigma_local = max(sigma_local, 1e-12)

        # require peak to be ratio x sigma above
        abs_floor = 5
        if (z[p] > ratio * sigma_local) and (z[p] > abs_floor):
            keep.append(p)

    if len(keep) == 0:
        return out

    # Sort by peak height
    keep = np.array(keep, dtype=int)
    order = np.argsort(z[keep])[::-1]
    keep = keep[order][:max_peaks]

    # Build windows around each kept peak
    for k, p in enumerate(sorted(keep), start=1):
        t_peak = float(tR[p])
        peak_amp = float(z[p])

        fit_start = max(t_min, t_peak - pad_left_ns)
        fit_end   = min(t_max, t_peak + pad_right_ns)

        out.append({
            "peak_id": k,
            "t_peak": t_peak,
            "peak_amp": peak_amp,
            "fit_start": fit_start,
            "fit_end": fit_end,
            "baseline": baseline,
        })

    return out

def baseline_noise_bestwindow(
    t, y, dt_ns,
    window_ns=500.0,
    step_ns=25.0,
    min_sigma=1e-6,
    flat_frac_min=0.98,      # fraction of samples that must be "flat"
    flat_k=3.0,              # flat threshold
    spike_k=8.0,             # veto
):
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    w = max(int(window_ns / dt_ns), 10)
    step = max(int(step_ns / dt_ns), 1)

    if len(y) < w:
        raise RuntimeError(f"Waveform too short for window_ns={window_ns} ns")

    best = None

    for s in range(0, len(y) - w + 1, step):
        seg = y[s:s+w]

        base = float(np.median(seg))
        mad  = float(np.median(np.abs(seg - base)))
        sig  = max(1.4826 * mad, min_sigma)

        if np.max(np.abs(seg - base)) > spike_k * sig:
            continue

        flat = np.abs(seg - base) < (flat_k * sig)
        if flat.mean() < flat_frac_min:
            continue

        cand = (sig, s, base)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        raise RuntimeError(
            f"No baseline window found (window_ns={window_ns}, step_ns={step_ns}, "
            f"flat_k={flat_k}, spike_k={spike_k}, flat_frac_min={flat_frac_min})."
        )

    sig, s, base = best
    e = s + w
    # print(base, sig, s, e)
    return base, sig, (float(t[s]), float(t[e-1])), w

def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)
    mask = t >= t0
    dt = t[mask] - t0
    output[mask] = A * (np.exp(-dt / tau_fall) - np.exp(-dt / tau_rise))
    dt_sample = t[1] - t[0]
    smoothed = gaussian_filter1d(output, sigma / dt_sample, mode="constant", axis=0)
    return smoothed
#
# def true_hits(timeaxis, t_starts, heights, widths):
#     dt = timeaxis[1] - timeaxis[0]
#     x = np.zeros_like(timeaxis, dtype=np.float32)
#     w = widths
#     for ts, h in zip(t_starts, heights):
#         i0 = int((ts - timeaxis[0]) / dt)
#         n  = int(w / dt)
#         x[i0:i0+n] += h
#     return x
#
# def waveform_from_hits(timeaxis, t_hits, h, height, box_width_ns=1.25):
#     x = true_hits(timeaxis, t_hits, height, widths=box_width_ns)
#     y = fftconvolve(x, h, mode="full")[:len(timeaxis)]
#     return y, x

# Filter from paper
def custom_lowpass_filter(N, dt_ns, flat_mhz=15.0, sigma_mhz=40.0, kill_mhz=50.0):
    """
    Custom-defined low-pass filter adapted for MilliQan pulse shape:
    - 1 for f < flat_mhz
    - Gaussian rolloff from flat_mhz to kill_mhz
    - 0 for f > kill_mhz
    """
    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)  # Hz
    freqs_mhz = np.abs(freqs_hz) / 1e6           # MHz

    filt = np.ones(N, dtype=np.float64)
    rolloff = (freqs_mhz >= flat_mhz) & (freqs_mhz <= kill_mhz)
    filt[rolloff] = np.exp(-0.5 * ((freqs_mhz[rolloff] - flat_mhz) / sigma_mhz) ** 2)
    filt[freqs_mhz > kill_mhz] = 0.0
    return filt

def cosine_lowpass_filter(N, dt_ns):
    cutoff = 450
    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)  # Hz
    freqs_mhz = np.abs(freqs_hz) / 1e6  # MHz
    mask = freqs_mhz <= cutoff

    filt = np.zeros(N, dtype=np.float64)
    filt[mask] = np.cos(np.pi * freqs_mhz[mask] / (2 * cutoff))
    return filt

def pure_gaussian_filter(N, dt_ns):
    cutoff_3sigma = 400
    sigma = cutoff_3sigma / 3

    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)
    freqs_mhz = np.abs(freqs_hz) / 1e6

    filt = np.exp(-0.5 * (freqs_mhz / sigma) ** 2)
    filt[freqs_mhz > cutoff_3sigma] = 0

    return filt

# Deconvolution function
def deconvolve_hits(m_t, r_calib, filt, eps=1e-12):
    """
    u_rec(t) = F^{-1}[ F[m(t)] * filter(f) / F[r_calib(t)] ]
    """
    M = fft.fft(m_t)
    R = fft.fft(r_calib)
    U_rec = M * filt / (R + eps)
    return np.real(fft.ifft(U_rec))

def strip_cycle(k: str) -> str:
    return k.split(";")[0]

# t = np.arange(50, 290, dt_ns, dtype=np.float32)
# N = len(t)
#
# t_hits = np.array([65.3, 73.2, 115.2, 213.3], dtype=DTYPE)
# height = np.array([0.91, 1.13, 0.97, 1.25], dtype=DTYPE)

# MilliQan-fitted SPE parameters
A         = POPT_GS[0]     # amplitude scale
tau_rise  = POPT_GS[1]     # ns
tau_fall  = POPT_GS[2]    # ns
sigma     = POPT_GS[4]      # ns

tree_name = "waveforms"
entry = 223
ch = 15
branches = [f"d0_ch{ch:02d}"]
entry_start = 223
entry_stop_exclusive = 224
run_start = 5
run_end = 5



for run in range(run_start, run_end + 1):
    root_path = f"/Users/leobailloeul/Documents/coding/LANL/data/flattened_waveform_MilliQan_Run2150.1_default.root"
    print(f"Opening: {root_path}")

    try:
        with uproot.open(root_path) as f:
            tree = f[tree_name]
            available = {strip_cycle(k) for k in tree.keys()}

            n_entries_total = tree.num_entries
            stop = min(entry_stop_exclusive, n_entries_total)

            for branch in branches:
                if branch not in available:
                    print(f"[skip] missing branch {branch}")
                    continue

                # parse true channel number from branch name, since we skip 2 & 3
                ch = int(branch.split("ch")[-1])

                print(f"\n=== Processing run {run:02d} {branch} ===")
                waves = tree[branch].array(
                    library="np",
                    entry_start=entry_start,
                    entry_stop=stop,
                )
                wave_raw = np.asarray(waves[0], dtype=DTYPE)
                N = len(wave_raw)
                t = np.arange(N, dtype=DTYPE) * dt_ns
                print(f"  Waveform: {N} samples, {N*dt_ns:.1f} ns")

    except (FileNotFoundError, OSError) as e:
        print(f"[skip] cannot open: {root_path} ({e})")

# Baseline subtraction
ped_i, noise_i, (b0, b1), wbins = baseline_noise_bestwindow(
    t, wave_raw, dt_ns,
    window_ns=300.0, step_ns=25.0, min_sigma=1e-3,
    flat_frac_min=0.95, flat_k=4.0, spike_k=5.0,
)
print(f"  Baseline: ped={ped_i:.2f} ADC, noise={noise_i:.3f} ADC, window=[{b0:.1f}, {b1:.1f}] ns")
y_for_deconv = (wave_raw - ped_i).astype(np.float64)

# SPE template (r_calib)
t_template = np.arange(0, N * dt_ns, dt_ns, dtype=DTYPE)[:N]
g = gaussian_smeared_biexponential(t_template, A, tau_rise, tau_fall, t0=16.0, sigma=sigma)

r_calib = np.zeros(N, dtype=np.float64)
r_calib[:len(g)] = g

# Deconvolve
filt = custom_lowpass_filter(N, dt_ns)
u_rec = deconvolve_hits(y_for_deconv, r_calib, filt)

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t, wave_raw, color="#3CC850", alpha=0.7, label="Raw waveform")
# ax1.axhline(ped_i, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({ped_i:.1f} ADC)")
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Amplitude [ADC]")

ax1r = ax1.twinx()
ax1r.plot(t, u_rec , color="blue", linewidth=0.8, label="Rec. hit (u_rec)")
ax1r.set_ylabel("Rec. amplitude [a.u.]", color="blue")
ax1r.tick_params(axis="y", labelcolor="blue")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_title(f"Deconvolution — Run2150 entry {entry} ch{ch:02d}")

plt.tight_layout()
plt.show()

# Peak finding and charge integration on u_rec
peaks, props = find_peaks(u_rec, height=0.02, distance=int(5 / dt_ns))

pad_ns   = 18.0
pad_bins = int(pad_ns / dt_ns)

def find_integration_windows(u_rec, peaks, pad_bins, baseline_frac=0.0):
    """Return (lo, hi) slice indices for each peak.

    Inner boundaries: valley (minimum) between consecutive detected peaks.
    Outer boundaries: walk to zero-crossing + pad_bins padding.
    """
    N = len(u_rec)
    windows = []

    for i, pk in enumerate(peaks):
        # left edge
        if i > 0:
            prev_pk = peaks[i - 1]
            valley = prev_pk + np.argmin(u_rec[prev_pk:pk])
            lo = valley
        else:
            lo = pk
            while lo > 0 and u_rec[lo - 1] > baseline_frac:
                lo -= 1
            lo = max(lo - pad_bins, 0)

        # right edge
        if i < len(peaks) - 1:
            next_pk = peaks[i + 1]
            valley = pk + np.argmin(u_rec[pk:next_pk])
            hi = valley
        else:
            hi = pk
            while hi < N - 1 and u_rec[hi + 1] > baseline_frac:
                hi += 1
            hi = min(hi + pad_bins + 1, N)

        windows.append((lo, hi))
    return windows

windows = find_integration_windows(u_rec, peaks, pad_bins)

print(f"\n{'Peak #':<8} {'t_rec [ns]':<14} {'height [a.u.]':<16} {'charge [a.u.]':<16} {'window [ns]'}")
print("-" * 72)

for i, pk in enumerate(peaks):
    t_peak = t[pk]
    h_peak = u_rec[pk]
    lo, hi = windows[i]
    charge = np.sum(u_rec[lo:hi])

    win_str = f"[{t[lo]:.1f}, {t[min(hi, N-1)]:.1f}]"
    print(f"{i:<8} {t_peak:<14.2f} {h_peak:<16.4f} {charge:<16.4f} {win_str}")
