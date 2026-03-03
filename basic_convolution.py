import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

dt_ns = 1.25
DTYPE = np.float32
POPT_GS = [75.99, 0.100, 1.927, 979.18665858, 1.177, -0.406]

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

# r_calib must be same length as waveform,
r_calib = np.zeros(N, dtype=np.float64)
r_calib[:len(g)] = g

filt = custom_lowpass_filter(N, dt_ns)
u_rec = deconvolve_hits(y_noisy, r_calib, filt)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(t, y_noisy, color="#3CC850", label="Waveform")
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Amplitude [V]")

ax1r = ax1.twinx()
ax1r.vlines(t_hits, 0, height, color="r", linewidth=2, label="True hit")
ax1r.plot(t, u_rec, color="blue", label="Rec. hit")
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
ax1.set_title("Deconvolution Reconstruction")

plt.tight_layout()
plt.show()