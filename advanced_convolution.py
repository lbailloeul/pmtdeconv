import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, find_peaks, stft, istft

dt_ns = 1.25
DTYPE = np.float32
POPT_GS = [300, 1.06069764323418, 1.48020379040983, 709.474148772849, 1.1079849521775, -0.336218850361248]

def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)
    mask = t >= t0
    dt = t[mask] - t0
    output[mask] = A * (np.exp(-dt / tau_fall) - np.exp(-dt / tau_rise))
    dt_sample = t[1] - t[0]
    smoothed = gaussian_filter1d(output, sigma / dt_sample, mode="constant", axis=0)
    return smoothed

def true_hits(timeaxis, t_starts, heights, widths):
    dt = timeaxis[1] - timeaxis[0]
    x = np.zeros_like(timeaxis, dtype=np.float32)
    w = widths
    for ts, h in zip(t_starts, heights):
        i0 = int((ts - timeaxis[0]) / dt)
        n  = int(w / dt)
        x[i0:i0+n] += h
    return x

def waveform_from_hits(timeaxis, t_hits, h, height, box_width_ns=1.25):
    x = true_hits(timeaxis, t_hits, height, widths=box_width_ns)
    y = fftconvolve(x, h, mode="full")[:len(timeaxis)]
    return y, x

# Filter from paper
def custom_lowpass_filter(N, dt_ns):
    """
    Custom-defined low-pass filter from the paper (Figure 3):
    - 1 for f < 100 MHz
    - Gaussian rolloff from 100-300 MHz (mean=100, sigma=40)
    - 0 for f > 300 MHz
    """
    freqs_hz = fft.fftfreq(N, d=dt_ns * 1e-9)  # Hz
    freqs_mhz = np.abs(freqs_hz) / 1e6           # MHz

    filt = np.ones(N, dtype=np.float64)
    rolloff = (freqs_mhz >= 100) & (freqs_mhz <= 300)
    filt[rolloff] = np.exp(-0.5 * ((freqs_mhz[rolloff] - 100) / 40) ** 2)
    filt[freqs_mhz > 300] = 0.0
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

t = np.arange(50, 290, dt_ns, dtype=np.float32)
N = len(t)

t_hits = np.array([65.3, 73.2, 115.2, 213.3], dtype=DTYPE)
height = np.array([0.91, 1.13, 0.97, 1.25], dtype=DTYPE)

A = POPT_GS[0]
tau_rise = POPT_GS[1]
tau_fall = POPT_GS[2]
sigma = POPT_GS[4]

t_template = np.arange(0, N * dt_ns, dt_ns, dtype=DTYPE)[:N]
g = gaussian_smeared_biexponential(t_template, A, tau_rise, tau_fall, t0=16.0, sigma=sigma)

# Convolve to get waveform
y, x = waveform_from_hits(t, t_hits, g, height, dt_ns)

# Add noise + pedestal
noise_sigma = 0.8050
n = np.random.normal(0, noise_sigma, size=N).astype(DTYPE)
y_noisy = y + n + POPT_GS[5]  # measured waveform m(t)
# should be:
pedestal = POPT_GS[5]
y_for_deconv = y + n              # zero-mean baseline

# r_calib must be same length as waveform,
r_calib = np.zeros(N, dtype=np.float64)
r_calib[:len(g)] = g

filt = cosine_lowpass_filter(N, dt_ns)
u_rec = deconvolve_hits(y_for_deconv, r_calib, filt)

# STFT
fs = 1.0 / (dt_ns * 1e-9)  # sampling frequency in Hz
nperseg = int(20.0 / dt_ns)       # 16 samples = 20 ns window
noverlap = nperseg - max(int(1.0 / dt_ns), 1)  # at least 1 sample step
f, t_seg, Zxx = stft(u_rec, fs=fs, window=('gaussian', nperseg / 6), nperseg=nperseg,
                      noverlap=noverlap)

Zxx_filtered = Zxx.copy()
Zxx_filtered[f >20e7, :] = 0.0
Zxx_filtered[np.abs(Zxx_filtered) < 0.065] = 0.0

t_istft, u_rec_stft = istft(Zxx_filtered, fs=fs, window=('gaussian', nperseg / 6),
                       nperseg=nperseg, noverlap=noverlap)

# plotting the Amplitude vs Time meshgrid
# amp = np.abs(Zxx)
# amp_filtered = amp.copy()
# # Zero out everything above 50 MHz (noise)
# freq_mask_high = f > 50e6
# amp_filtered[freq_mask_high, :] = 0.0
#
# # Apply 0.02 threshold only in the low-frequency band
# amp_filtered[amp_filtered < 0.02] = 0.0
#
# filtered_waveform = istft(amp_filtered, nperseg=nperseg)
#
# plt.pcolormesh(t_seg * 1e9, f / 1e6, amp_filtered, shading='auto')
# plt.xlabel("Time [ns]")
# plt.ylabel("Frequency [MHz]")
# plt.colorbar(label="Amplitude [A.U.]")
# plt.show()


# # Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t, y_noisy, color="#3CC850", label="Waveform")
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Amplitude [V]")
ax1r = ax1.twinx()
ax1r.vlines(t_hits, 0, height, color="r", linewidth=2, label="True hit")
# ax1r.plot(t, u_rec, color="blue", label="Rec. hit")
plt.plot(t_istft * 1e9+ 50, u_rec_stft, color="blue", label="STFT")
ax1r.set_ylabel("NormC. [PE]", color="r")
ax1r.tick_params(axis="y", labelcolor="r")

# align zeros
y_min, y_max = ax1.get_ylim()
frac_zero = -y_min / (y_max - y_min)
pe_max = 1.5
ax1r.set_ylim(-frac_zero * pe_max / (1 - frac_zero), pe_max)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_title("Deconvolution Reconstruction (Cosine Filter)")

plt.tight_layout()
plt.show()
#
# Find peaks in the reconstructed waveform
peaks, props = find_peaks(u_rec_stft, height=0.1, distance=int(5 / dt_ns))

# Integration windows
# For each peak, walk left/right to baseline crossing or valley between
# consecutive peaks, then add padding (paper: 9 ns each side).
pad_ns   = 9.0
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
            # between this peak and the previous one, split at the valley
            prev_pk = peaks[i - 1]
            valley = prev_pk + np.argmin(u_rec[prev_pk:pk])
            lo = valley
        else:
            # walk left until signal crosses baseline
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

windows = find_integration_windows(u_rec_stft, peaks, pad_bins)

print(f"{'Peak #':<8} {'t_rec [ns]':<14} {'t_true [ns]':<14} {'height [PE]':<14} {'charge [PE]':<14} {'window [ns]'}")
print("-" * 82)

for i, pk in enumerate(peaks):
    t_peak = t[pk]
    h_peak = u_rec_stft[pk]
    lo, hi = windows[i]
    charge = np.sum(u_rec_stft[lo:hi])

    # match to nearest true hit
    j_match = np.argmin(np.abs(t_peak - t_hits))

    win_str = f"[{t[lo]:.1f}, {t[min(hi, N-1)]:.1f}]"
    print(f"{i:<8} {t_peak:<14.2f} {t_hits[j_match]:<14.2f} {h_peak:<14.4f} {charge:<14.4f} {win_str}")

# Timing and charge residuals
print()
for i, pk in enumerate(peaks):
    t_peak = t[pk]
    j_match = np.argmin(np.abs(t_peak - t_hits))
    dt_residual = t_peak - t_hits[j_match]

    lo, hi = windows[i]
    charge = np.sum(u_rec_stft[lo:hi])

    q_true = height[j_match]
    ratio  = charge / q_true if q_true > 0 else float('nan')
    print(f"Hit {j_match}: Δt = {dt_residual:+.2f} ns, charge = {charge:.4f} PE  (true {q_true:.2f}, ratio {ratio:.3f})")
