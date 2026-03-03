import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import itertools

# Set the global DTYPE
DTYPE = np.float64

def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)
    mask = t >= t0
    dt = t[mask] - t0
    tr, tf = max(tau_rise, 1e-9), max(tau_fall, 1e-9)
    output[mask] = A * (np.exp(-dt / tf) - np.exp(-dt / tr))
    dt_sample = t[1] - t[0]
    sigma_bins = sigma / dt_sample
    return gaussian_filter1d(output, sigma_bins, mode="constant", axis=0)

def model_biexponential(t, A, tau_rise, tau_fall, t0, sigma, pedestal):
    return gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma) + pedestal

def u_peak(timeaxis, u_peak_0, t0, sigma, tau):
    timeaxis = np.asarray(timeaxis, dtype=float)
    x = (timeaxis - t0)/tau
    # Guard against log of non-positive numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        u_peak_t = u_peak_0 * np.exp(-0.5*(np.log(x)/sigma)**2)
    return np.where((timeaxis > t0) & (x > 0), u_peak_t, 0.0)

def u_undershoot(timeaxis, u_peak_0, t0, sigma, tau,
               u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot,
               td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot):
    first_term = u_fast_undershoot/ (np.exp((-timeaxis + t0+ td1)/tau_fermi)+1) * np.exp((-timeaxis + t0 - td2)/tau_fast)
    second_term = u_slow_undershoot/ (np.exp((-timeaxis + t0+ td1)/tau_fermi)+1) * np.exp((-timeaxis + t0 - td2)/tau_slow)
    third_term = u_gaus_undershoot * np.exp(-0.5*((timeaxis - t0 - td1)/sigma_undershoot)**2)
    return -u_peak(timeaxis, u_peak_0, t0, sigma, tau)*(first_term + second_term + third_term)

def u_t(timeaxis, u_peak_0, t0, sigma, tau, u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot, td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot, u_reflection):
    ref_sum = 0
    for i in range(1, 12):
        ref_sum += u_reflection[i] * u_peak(timeaxis - i*7, u_peak_0, t0, sigma, tau)
    return u_peak(timeaxis, u_peak_0, t0, sigma, tau) + u_undershoot(timeaxis, u_peak_0, t0, sigma, tau,
               u_fast_undershoot, u_slow_undershoot, u_gaus_undershoot,
               td1, td2, tau_fermi, tau_fast, tau_slow, sigma_undershoot) + ref_sum

# Wrapper for curve_fit to handle individual reflection parameters
def daya_bay_fit_wrapper(t, u0, t0, sig, tau, uf, us, ug, td1, td2, tf_fermi, t_fast, t_slow, sig_un, ped, r1, r2, r3):
    u_refl = np.zeros(13)
    u_refl[1], u_refl[2], u_refl[3] = r1, r2, r3
    return u_t(t, u0, t0, sig, tau, uf, us, ug, td1, td2, tf_fermi, t_fast, t_slow, sig_un, u_refl) + ped

file = uproot.open("/Users/leobailloeul/Documents/coding/LANL/data/806/flattened_waveform_Run806.100_default.root")
tree = file["waveforms"]
wave = -tree["d0_ch00"].array(library="np")[5]
dt_ns = 1.25
t = np.arange(len(wave)) * dt_ns

# Window - using your fixed values
fit_start, fit_end = 555.5, 579.5
mask = (t >= fit_start) & (t <= fit_end)
t_fit, y_fit = t[mask], wave[mask]

def fit_with_grid(func, t_data, y_data, grid, bounds):
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_p0, lowest_chi2 = None, np.inf
    for p in combinations:
        p_list = list(p.values())
        try:
            popt_temp, _ = curve_fit(func, t_data, y_data, p0=p_list, bounds=bounds, maxfev=1500)
            chi2 = np.sum((y_data - func(t_data, *popt_temp)) ** 2)
            if chi2 < lowest_chi2:
                lowest_chi2, best_p0 = chi2, popt_temp
        except: continue
    final_popt, _ = curve_fit(func, t_data, y_data, p0=best_p0, bounds=bounds, maxfev=10000)
    return final_popt, np.sum((y_data - func(t_data, *final_popt)) ** 2)

# Grid and Bounds for Model A
grid_a = {'A': [50], 'tr': [5], 'tf': [30], 't0': [559], 's': [2.0], 'ped': [0]}
bounds_a = ([0, 0.1, 1, 550, 0.1, -10], [1000, 30, 200, 570, 10, 10])

# Grid and Bounds for Model B (Daya Bay Wrapper)
# Order: u0, t0, sig, tau, uf, us, ug, td1, td2, tf_fermi, t_fast, t_slow, sig_un, ped, r1, r2, r3
grid_b = {'u0':[50], 't0':[558], 'sig':[0.5], 'tau':[8], 'uf':[0.1], 'us':[0.1], 'ug':[0.1],
          'td1':[5], 'td2':[2], 'tf':[2], 't_f':[10], 't_s':[50], 's_un':[2], 'ped':[0],
          'r1':[0.1], 'r2':[0.05], 'r3':[0.01]}

lower_b = [0, 550, 0.1, 1, 0, 0, 0, 0, 0, 0.1, 1, 1, 0.1, -5, 0, 0, 0]
upper_b = [1000, 570, 2, 50, 1, 1, 1, 20, 20, 10, 100, 500, 10, 5, 1, 1, 1]

print("Running competition...")
popt_a, chi2_a = fit_with_grid(model_biexponential, t_fit, y_fit, grid_a, bounds_a)
popt_b, chi2_b = fit_with_grid(daya_bay_fit_wrapper, t_fit, y_fit, grid_b, (lower_b, upper_b))

print(len(popt_a), len(popt_b))
ndof_a = len(y_fit) - len(popt_a)
ndof_b = len(y_fit) - len(popt_b)
reduced_chi2_a = chi2_a/ndof_a
reduced_chi2_b = chi2_b/ndof_b

plt.figure(figsize=(10, 6))
plt.step(t_fit, y_fit, where='mid', color='black', label='Data', alpha=0.5)

t_fine = np.linspace(fit_start, fit_end, 1000)
plt.plot(t_fine, model_biexponential(t_fine, *popt_a), 'r--', label=f'Biexponential model, reduced $\chi^2$: {reduced_chi2_a:.2f}')
plt.plot(t_fine, daya_bay_fit_wrapper(t_fine, *popt_b), 'b-', label=f'Daya Bay model, reduced $\chi^2$: {reduced_chi2_b:.2f}')

plt.title(f"Waveform from run Run806")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.show()

print("-" * 30)
print(f"WINNER: {'BIEXPONENTIAL' if reduced_chi2_a < reduced_chi2_b else 'DAYA BAY'}")
print("-" * 30)