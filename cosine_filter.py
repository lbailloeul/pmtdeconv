import numpy as np
import matplotlib.pyplot as plt

def cosine_lowpass_filter(f_mhz, cutoff):
    f = np.abs(f_mhz)
    filt = np.zeros_like(f)

    mask = f <= cutoff
    filt[mask] = np.cos(np.pi * f[mask] / (2 * cutoff))

    return filt


freqs = np.linspace(0, 600, 1000)  # MHz
cutoff = 450

# filt = cosine_lowpass_filter(freqs, cutoff)
filt = cosine_lowpass_filter(freqs, 100)
filt1 = cosine_lowpass_filter(freqs, 300)
filt2 = cosine_lowpass_filter(freqs, 500)

# plot
plt.plot(freqs, filt, label="Cosine low-pass filter")
plt.plot(freqs, filt1, label="Cosine low-pass filter")
plt.plot(freqs, filt2, label="Cosine low-pass filter")

# vertical cutoff line
# plt.axvline(x=cutoff, color="black", ls="--", lw=1)
# plt.text(cutoff, 0.9, "$f_c$", color='black', ha='right')

plt.grid(True)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.title("Cosine Low-Pass Filter")
plt.legend()
plt.show()