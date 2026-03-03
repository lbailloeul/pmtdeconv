import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, deconvolve

# ===== Peak parameters =====
u_peak_0 = 42.0            # mV  (U0_peak)
tau = 8.4                  # ns  (peak width)
sigma = 0.28               # peak shape parameter

# ===== Undershoot amplitudes =====
u_fast_undershoot = 0.11
u_slow_undershoot = 0.03
u_gaus_undershoot = 0.4 * u_fast_undershoot  # given relation

# ===== Time constants =====
tau_fermi = 10.0           # ns
tau_fast = 45.0            # ns
tau_slow = 290.0           # ns

# ===== Delay constants =====
t0 = 50
td1 = 20.0                 # ns
td2 = 5.0                  # ns

# ===== Gaussian undershoot shape =====
sigma_undershoot = 20.0    # ns

# ===== Reflection amplitudes =====
# Index 0 unused so indices match U1, U2, U3...
u_reflection = np.array([
    0.0,   # dummy index 0
    0.24,  # U1_reflection
    0.18,  # U2_reflection
    0.14,  # U3_reflection
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # up to i=12 if needed
])
# SPE waveform of a PMT from Daya Bay data and its fitting result 1707.03699
def u_peak(timeaxis, u_peak_0, t0, sigma, tau):
    timeaxis = np.asarray(timeaxis, dtype=float)
    x = (timeaxis - t0)/tau
    u_peak_t = u_peak_0 * np.exp(-0.5*(np.log(x)/sigma)**2)
    return np.where(timeaxis > t0, u_peak_t, 0.0)

def u_undershoot(timeaxis, u_peak_0, t0, sigma, tau,
               u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot,
               td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot):
    first_term = u_fast_undershoot/ (np.exp((-timeaxis + t0+ td1)/tau_fermi)+1) * np.exp((-timeaxis + t0 - td2)/tau_fast)
    second_term = u_slow_undershoot/ (np.exp((-timeaxis + t0+ td1)/tau_fermi)+1) * np.exp((-timeaxis + t0 - td2)/tau_slow)
    third_term = u_gaus_undershoot * np.exp(-0.5*((timeaxis - t0 - td1)/sigma_undershoot)**2)
    return -u_peak(timeaxis, u_peak_0, t0, sigma, tau)*(first_term + second_term + third_term)

def u_t(timeaxis, u_peak_0, t0, sigma, tau, u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot, td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot, u_reflection):
    sum = 0
    for i in range(1,12):
        sum += u_reflection[i] * u_peak(timeaxis - i*7, u_peak_0, t0, sigma, tau)
    return u_peak(timeaxis, u_peak_0, t0, sigma, tau) + u_undershoot(timeaxis, u_peak_0, t0, sigma, tau,
               u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot,
               td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot) + sum

# deconvolve m(t) functions with and without noise
def deconvolve_hits(m_t, r_calib, filter_f, eps=1e-12):
    M = fft.fft(m_t)
    R = fft.fft(r_calib)
    u_rec = fft.ifft(M * filter_f / (R + eps))
    return np.real(u_rec)

def filter_f(f):
    af = np.abs(f)
    return np.where(af >= 100.0, np.exp(-0.5*((af-100)/40)**2), 1.0)

def deconvolve_fn(u_t,r_t,r_calib, n_t, filter_f):
    signal_fourier = fft.fft(u_t) * filter_f * fft.fft(r_t)/fft.fft(r_calib)
    noise_fourier = filter_f * fft.fft(n_t)/fft.fft(r_calib)
    u_rec = fft.ifft(signal_fourier + noise_fourier)
    return u_rec

timeaxis = np.linspace(0.0, 1.0e3, 1000)
x = u_t(timeaxis, u_peak_0, t0, sigma, tau, u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot, td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot, u_reflection)

# functions to obtain convolved waveforms
def true_hits(timeaxis, t_starts, heights, width_ns):
    dt = timeaxis[1] - timeaxis[0]
    u = np.zeros_like(timeaxis, dtype=float)

    n = max(1, int(round(width_ns / dt)))  # at least 1 bin
    for ts, h in zip(t_starts, heights):
        i0 = int(round((ts - timeaxis[0]) / dt))
        i1 = min(len(u), i0 + n)
        if 0 <= i0 < len(u):
            u[i0:i1] += h
    return u

def waveform_from_hits(timeaxis, t_hits, kernel_r, heights, box_width_ns=1.0):
    u = true_hits(timeaxis, t_hits, heights, width_ns=box_width_ns)
    m = fftconvolve(u, kernel_r, mode="full")[:len(timeaxis)]
    return m, u

r = u_t(timeaxis, u_peak_0, t0, sigma, tau,
        u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot,
        td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot, u_reflection)

m, u = waveform_from_hits(timeaxis, [t0], r, [1.0], box_width_ns=1.25)

dt_s = (timeaxis[1] - timeaxis[0]) * 1e-9
freq = np.fft.fftfreq(len(timeaxis), d=dt_s) * 1e-6   # MHz

new = u_rec(freq)
plt.plot(timeaxis, new)
plt.plot(timeaxis, m)
# draw rectangles on a second y-axis
ax = plt.gca()
ax2 = ax.twinx()
ax2.bar([t0], [1.0], width=1.25, align="edge", alpha=0.4, label="true hit u(t)")

ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.xlim(t0 - 50, t0 + 150)
plt.show()