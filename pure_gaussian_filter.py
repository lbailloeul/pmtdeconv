import numpy as np
import matplotlib.pyplot as plt

def pure_gaussian(f_mhz, sigma):
    f = np.abs(f_mhz)
    filt = np.exp(-0.5 * (f / (sigma/3)) ** 2)
    return filt


freqs = np.linspace(0, 600, 1000)  # MHz
filt = pure_gaussian(freqs, 400)
# filt1 = pure_gaussian(freqs,300)
# filt2 = pure_gaussian(freqs,500)


# plot
plt.plot(freqs, filt, label="$\\sigma$ = 400/3 MHz")
# plt.plot(freqs, filt1, label="$\\sigma$ = 300/3 MHz")
# plt.plot(freqs, filt2, label="$\\sigma$ = 500/3 MHz")

plt.grid(True)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.title("Pure Gaussian Filter")
plt.legend()
plt.show()