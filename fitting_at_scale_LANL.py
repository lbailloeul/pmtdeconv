#!/usr/bin/env python3
import uproot
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from statsmodels.stats.moment_helpers import cov2corr

DTYPE = np.float64
# MU_PED = -0.8464
CHI2_CUT = 3.0
A_FIXED = 1.0

# def baseline_from_tail(wave, dt_ns, baseline_tail_ns=200.0):
#     """
#     Robust baseline estimate from last baseline_tail_ns of waveform.
#     Returns (median, mean, std) of the tail region.
#     """
#     n_tail = max(int(baseline_tail_ns / dt_ns), 10)
#     tail = wave[-n_tail:] if len(wave) >= n_tail else wave
#     return float(np.median(tail)), float(np.mean(tail)), float(np.std(tail))
#
# def summarize_empty_event_baseline(
#     root_path,
#     tree_name="waveforms",
#     branch="d0_ch00",
#     dt_ns=1.25,
#     n_check=500,              # how many entries to scan
#     baseline_tail_ns=200.0,
# ):
#     with uproot.open(root_path) as f:
#         tree = f[tree_name]
#         waves = tree[branch].array(library="np")
#
#     n = min(n_check, len(waves))
#
#     empty_rows = []
#     for i in range(n):
#         wave = -np.asarray(waves[i], dtype=np.float64)
#
#         # use YOUR peak finder as the definition of "empty"
#         t = np.arange(len(wave), dtype=np.float64) * dt_ns
#         peaks = find_peak_windows_local_ratio(
#             t, wave, dt_ns,
#             t_min=0.0, t_max=1200.0,
#             smooth_sigma_ns=2.0,
#             half_window_ns=50.0,
#             ratio=5.0,
#             min_distance_ns=30.0,
#             pad_left_ns=25.0,
#             pad_right_ns=120.0,
#             max_peaks=12
#         )
#         if len(peaks) != 0:
#             continue
#
#         med, mean, std = baseline_from_tail(wave, dt_ns, baseline_tail_ns=baseline_tail_ns)
#         empty_rows.append({"entry": i, "baseline_median": med, "baseline_mean": mean, "baseline_std": std})
#
#     if len(empty_rows) == 0:
#         print("No empty events found in scanned range.")
#         return
#
#     df0 = pd.DataFrame(empty_rows)
#
#     print(f"Empty events found: {len(df0)} / {n}")
#     print("Baseline (tail) summary over empty events:")
#     print(f"  median of medians = {df0['baseline_median'].median():.6g}")
#     print(f"  mean of means     = {df0['baseline_mean'].mean():.6g}")
#     print(f"  std of means      = {df0['baseline_mean'].std():.6g}")
#
#     # optional: save for inspection
#     df0.to_csv("empty_event_baselines.csv", index=False)
#     print("Saved: empty_event_baselines.csv")

# Model: Gaussian-smeared bi-exponential + pedestal
def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)

    mask = t >= t0
    dt = t[mask] - t0

    tr = max(tau_rise, 1e-12)
    tf = max(tau_fall, 1e-12)
    output[mask] = A * (np.exp(-dt / tf) - np.exp(-dt / tr))

    dt_sample = t[1] - t[0]
    sigma_bins = max(sigma / dt_sample, 1e-6)

    return gaussian_filter1d(output, sigma_bins, mode="constant", axis=0)

def model_biexponential(t, A, tau_rise, tau_fall, t0, sigma, pedestal):
    return gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma) + pedestal

def model_biexponential_fixed_A(t, tau_rise, tau_fall, t0, sigma, pedestal):
    return model_biexponential(t, A_FIXED, tau_rise, tau_fall, t0, sigma, pedestal)

# Fit helper
def fit_biexp_single(t_fit, y_fit, bounds, p0, sigma_signal):
    # sigma_signal is your baseline noise estimate (scalar)
    sigma_signal = float(max(sigma_signal, 1e-12))
    sigma_arr = np.full_like(y_fit, sigma_signal, dtype=np.float64)

    popt, pcov = curve_fit(
        model_biexponential_fixed_A,
        t_fit, y_fit,
        p0=p0,
        bounds=bounds,
        sigma=sigma_arr,
        absolute_sigma=True,
        maxfev=20000,
    )

    resid = y_fit - model_biexponential_fixed_A(t_fit, *popt)
    chi2 = float(np.sum((resid / sigma_arr) ** 2))
    ndof = int(len(y_fit) - len(popt))
    red = chi2 / ndof if ndof > 0 else np.nan
    return popt, pcov, chi2, ndof, red

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

        # require peak to be ratio x sigma above neighborhood
        abs_floor = 5
        if (z[p] > ratio * sigma_local) and (z[p] > abs_floor):
            keep.append(p)

    if len(keep) == 0:
        return out

    # Sort by peak height and keep top N
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

def main():
    tree_name = "waveforms"

    run_start = 0
    run_end = 50  # 00..50

    branches = [f"d0_ch{ch:02d}" for ch in range(16) if ch not in {2, 3}]

    # Entries you want to fit
    entry_start = 0
    entry_stop_exclusive = 1000

    dt_ns = 1.25
    ped_k = 1.0 # number of sigmas your pedestal bounds will allow

    out_dir = Path("biexp_fits_run806_allruns_LANL")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    SAVE_PLOTS = False

    def strip_cycle(k: str) -> str:
        return k.split(";")[0]

    results = []

    for run in range(run_start, run_end + 1):
        root_path = f"/Users/leobailloeul/Documents/coding/LANL/data/806_real/flattened_waveform_Run806.{run:02d}_default.root"
        print(f"\n\n================ RUN {run:02d} ================")
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

                    for idx, i in enumerate(range(entry_start, stop)):
                        wave = -np.asarray(waves[idx], dtype=DTYPE)
                        t = np.arange(len(wave), dtype=DTYPE) * dt_ns
                        try:
                            ped_i, noise_i, (b0, b1), wbins = baseline_noise_bestwindow(
                                t, wave, dt_ns,
                                window_ns=500.0,
                                step_ns=25.0,
                                min_sigma=1e-3,  # 1e-3 V
                                flat_frac_min=0.98,
                                flat_k=3.0,
                                spike_k=4.0,
                            )
                            # ped_lo = ped_i - ped_k * noise_i
                            # ped_hi = ped_i + ped_k * noise_i

                            # bounds_event = ( #A, tau_rise, tau_fall, t0, sigma, pedestal
                            #     [0.01, 0.1, 0, 0.1, ped_lo],
                            #     [30, 200, 1200, 10, ped_hi],
                            # )
                        except RuntimeError as e:
                            row = {
                                "run": run, "chan": ch, "branch": branch, "entry": i,
                                "success": False,
                                "reason": "baseline_failed",
                                "error": str(e),
                            }
                            results.append(row)
                            continue

                        baseline_override = ped_i

                        row = {
                            "run": run,
                            "chan": ch,
                            "branch": branch,
                            "entry": i,
                            "success": False,
                            "reason": "",
                            "peak_id": np.nan,
                            "t_peak": np.nan,
                            "peak_amp": np.nan,
                            "fit_start": np.nan,
                            "fit_end": np.nan,
                            "chi2": np.nan,
                            "ndof": np.nan,
                            "red_chi2": np.nan,
                            "A": np.nan,
                            "tau_rise": np.nan,
                            "tau_fall": np.nan,
                            "t0": np.nan,
                            "sigma": np.nan,
                            "pedestal": np.nan,
                            "A_err": np.nan,
                            "tau_rise_err": np.nan,
                            "tau_fall_err": np.nan,
                            "t0_err": np.nan,
                            "sigma_err": np.nan,
                            "pedestal_err": np.nan,
                            "baseline_t0": b0,
                            "baseline_t1": b1,
                            "baseline_wbins": wbins,
                            "baseline_sig": noise_i,
                            "baseline_ped": ped_i,
                        }

                        peak_windows = find_peak_windows_local_ratio(
                            t, wave, dt_ns,
                            t_min=float(t[0]), t_max=float(t[-1]),
                            smooth_sigma_ns=2.0,
                            half_window_ns=50.0,
                            ratio=4.0,
                            min_distance_ns=10.0,
                            pad_left_ns=15.0,
                            pad_right_ns=15.0,
                            max_peaks=12,
                            baseline_override=baseline_override,
                        )
                        if len(peak_windows) == 0:
                            row["reason"] = "no_peaks_found"
                            results.append(row)
                            continue
                        snr_min = 8.0
                        # print(idx, snr_min * noise_i)
                        peak_windows = [w for w in peak_windows if (w["peak_amp"] > snr_min * noise_i)]

                        if len(peak_windows) == 0:
                            row["reason"] = f"snr_cut_{snr_min}"
                            results.append(row)
                            continue

                        for win in peak_windows:
                            fit_start_i, fit_end_i = win["fit_start"], win["fit_end"]
                            mask = (t >= fit_start_i) & (t <= fit_end_i)
                            t_fit = t[mask]
                            y_fit = wave[mask]

                            if len(t_fit) < 10:
                                continue

                            y_fit = (y_fit - ped_i) / 300.0
                            sigma_normed = noise_i / 300.0
                            ped_norm_slack = 3.0 * sigma_normed

                            prow = row.copy()
                            prow["peak_id"] = win["peak_id"]
                            prow["t_peak"] = win["t_peak"]
                            prow["peak_amp"] = win["peak_amp"]
                            prow["fit_start"] = fit_start_i
                            prow["fit_end"] = fit_end_i

                            bounds_peak = (
                                [0.01, 0.1, fit_start_i, 0.1, -ped_norm_slack],
                                [30.0, 200.0, fit_end_i, 10.0, ped_norm_slack],
                            )

                            p0_event = [
                                1.5,  # tau_rise
                                1.5,  # tau_fall
                                win["t_peak"],  # t0
                                1.0,  # sigma
                                0,  # pedestal start
                            ]

                            try:
                                popt, pcov, chi2, ndof, red = fit_biexp_single(
                                    t_fit, y_fit,  bounds=bounds_peak, p0=p0_event, sigma_signal = sigma_normed)
                                # print(cov2corr(pcov))
                                # print(run,idx,i)
                                perr = np.full_like(popt, np.nan, dtype=float)
                                if pcov is not None and np.all(np.isfinite(pcov)):
                                    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))

                                if (not np.isfinite(red)) or (red >= CHI2_CUT):
                                    prow["reason"] = f"red_chi2_cut_{CHI2_CUT}"
                                    prow["chi2"], prow["ndof"], prow["red_chi2"] = chi2, ndof, red
                                    results.append(prow)
                                    continue

                                prow.update({
                                    "success": True,
                                    "chi2": chi2,
                                    "ndof": ndof,
                                    "red_chi2": red,
                                    "A": A_FIXED,
                                    "tau_rise": popt[0], "tau_fall": popt[1], "t0": popt[2], "sigma": popt[3],
                                    "pedestal": popt[4] * 300.0 + ped_i,
                                    "A_err": 0.0, "tau_rise_err": perr[0], "tau_fall_err": perr[1], "t0_err": perr[2],
                                    "sigma_err": perr[3], "pedestal_err": perr[4] * 300.0,
                                    "reason": "",
                                })

                                if SAVE_PLOTS:
                                    plt.figure(figsize=(10, 6))
                                    plt.step(t_fit, y_fit, where="mid", color="black", label="Data", alpha=0.5)

                                    t_fine = np.linspace(fit_start_i, fit_end_i, 1000)
                                    plt.plot(
                                        t_fine,
                                        model_biexponential(t_fine, A_FIXED, *popt),
                                        "r--",
                                        label=f"Fit peak {win['peak_id']} | red $\\chi^2$={red:.2f}",
                                    )
                                    plt.axvline(win["t_peak"], linestyle="--", alpha=0.25)

                                    ax = plt.gca()
                                    lines = [
                                        f"A = {A_FIXED:.1f} V",
                                        f"#tau_{{rise}} = {popt[0]:.3f} #pm {perr[0]:.3f} ns",
                                        f"#tau_{{fall}} = {popt[1]:.3f} #pm {perr[1]:.3f} ns",
                                        f"t_{{0}} = {popt[2]:.3f} #pm {perr[2]:.3f} ns",
                                        f"#sigma = {popt[3]:.3f} #pm {perr[3]:.3f} ns",
                                        f"ped = {popt[4]:.4f} #pm {perr[4]:.4f}",
                                    ]

                                    # convert ROOT-style #cmd to matplotlib $\cmd$
                                    def root_to_mpl(s):
                                        s = s.replace("#pm", "±")
                                        s = s.replace("#tau", "τ")
                                        s = s.replace("#sigma", "σ")
                                        return s

                                    x_ndc, y_ndc, dy = 0.97, 0.95, 0.045
                                    for k, line in enumerate(lines):
                                        ax.text(
                                            x_ndc, y_ndc - k * dy,
                                            root_to_mpl(line),
                                            transform=ax.transAxes,
                                            fontsize=9, color="red",
                                            verticalalignment="top",
                                            horizontalalignment="right",
                                            fontfamily="monospace",
                                        )

                                    plt.title(f"Run806 entry {i} ({branch}) peak {win['peak_id']}")
                                    plt.xlabel("Time (ns)")
                                    plt.ylabel("Amplitude (V)")
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.savefig(
                                        plots_dir / f"run{run:02d}_fit_{branch}_entry_{i:04d};{win['peak_id']}.png",
                                        dpi=150)
                                    plt.close()

                            except Exception as e:
                                prow["reason"] = "fit_failed"
                                prow["error"] = str(e)

                            results.append(prow)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] cannot open: {root_path} ({e})")
            continue

    df = pd.DataFrame(results)
    # keep only good fits in the saved CSV
    df_good = df[(df["success"]) & (df["red_chi2"] < CHI2_CUT)]
    df_good.to_csv(out_dir / "biexp_fit_results.csv", index=False)

    print(f"Saved CSV (passing only): {out_dir / 'biexp_fit_results.csv'}")
    print(f"Passing fits: {len(df_good)} / {len(df)}")

    # summarize_empty_event_baseline(
    #     "/Users/leobailloeul/Documents/coding/LANL/data/806/flattened_waveform_Run806.100_default.root",
    #     n_check=2000
    # )

    # Quick summary
    n_ok = int(df["success"].sum())
    print("\n" + "-" * 60)
    print(f"Finished. Success: {n_ok}/{len(df)}")
    if n_ok > 0:
        print("Best (lowest red_chi2):")
        best = df[df["success"]].sort_values("red_chi2").iloc[0]
        print(best[["entry", "red_chi2", "A", "tau_rise", "tau_fall", "t0", "sigma", "pedestal"]])
    print(f"Saved CSV: {out_dir / 'biexp_fit_results.csv'}")
    if SAVE_PLOTS:
        print(f"Saved plots to: {plots_dir}")
    print("-" * 60)


if __name__ == "__main__":
    main()