#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

DTYPE = np.float64
CHI2_CUT = 3.0

# ----------------------------
# Model: Gaussian-smeared bi-exponential + pedestal
# ----------------------------
def gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma):
    t = np.asarray(t, dtype=DTYPE)
    output = np.zeros_like(t, dtype=DTYPE)

    mask = t >= t0
    dt = t[mask] - t0

    tr = max(float(tau_rise), 1e-12)
    tf = max(float(tau_fall), 1e-12)
    output[mask] = A * (np.exp(-dt / tf) - np.exp(-dt / tr))

    dt_sample = t[1] - t[0]
    sigma_bins = max(float(sigma) / dt_sample, 1e-6)
    return gaussian_filter1d(output, sigma_bins, mode="constant", axis=0)

def model_biexponential(t, A, tau_rise, tau_fall, t0, sigma, pedestal):
    return gaussian_smeared_biexponential(t, A, tau_rise, tau_fall, t0, sigma) + pedestal

# ----------------------------
# Fit helper
# ----------------------------
def fit_biexp_single(t_fit, y_fit, bounds, p0, sigma_signal):
    sigma_signal = float(max(sigma_signal, 1e-12))
    sigma_arr = np.full_like(y_fit, sigma_signal, dtype=np.float64)

    popt, pcov = curve_fit(
        model_biexponential,
        t_fit, y_fit,
        p0=p0,
        bounds=bounds,
        sigma=sigma_arr,
        absolute_sigma=True,
        maxfev=20000,
    )

    resid = y_fit - model_biexponential(t_fit, *popt)
    chi2 = float(np.sum((resid / sigma_arr) ** 2))
    ndof = int(len(y_fit) - len(popt))
    red = chi2 / ndof if ndof > 0 else np.nan
    return popt, pcov, chi2, ndof, red

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
    baseline_override=None,
):
    out = []
    region = (t >= t_min) & (t <= t_max)
    if not np.any(region):
        return out

    tR = t[region]
    yR = y[region]

    sigma_bins = max(smooth_sigma_ns / dt_ns, 0.5)
    yS = gaussian_filter1d(yR, sigma_bins)

    if baseline_override is None:
        tail_bins = max(int(200.0 / dt_ns), 10)
        y_tail = yS[-tail_bins:] if len(yS) >= tail_bins else yS
        baseline = float(np.median(y_tail))
    else:
        baseline = float(baseline_override)

    z = yS - baseline

    distance_bins = max(int(min_distance_ns / dt_ns), 1)
    candidates, _ = find_peaks(z, distance=distance_bins)
    if len(candidates) == 0:
        return out

    hw = max(int(half_window_ns / dt_ns), 1)
    keep = []
    for p in candidates:
        lo = max(0, p - hw)
        hi = min(len(z), p + hw + 1)
        neigh = np.concatenate([z[lo:p], z[p+1:hi]])
        if neigh.size < 5:
            continue

        med = float(np.median(neigh))
        sigma_local = 1.4826 * float(np.median(np.abs(neigh - med)))
        sigma_local = max(sigma_local, 1e-12)

        abs_floor = 5  # keep your units consistent
        if (z[p] > ratio * sigma_local) and (z[p] > abs_floor):
            keep.append(p)

    if len(keep) == 0:
        return out

    keep = np.array(keep, dtype=int)
    order = np.argsort(z[keep])[::-1]
    keep = keep[order][:max_peaks]

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
    flat_frac_min=0.98,
    flat_k=3.0,
    spike_k=8.0,
):
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    w = max(int(window_ns / dt_ns), 10)
    step = max(int(step_ns / dt_ns), 1)

    if len(y) < w:
        raise RuntimeError(f"Waveform too short for window_ns={window_ns} ns")

    best = None  # (sig, s, base)

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
    return base, sig, (float(t[s]), float(t[e-1])), w

# ----------------------------
# Main (AK + iterate)
# ----------------------------
def main():
    tree_name = "waveforms"

    run_start = 0
    run_end = 50

    branches = [f"d0_ch{ch:02d}" for ch in range(16) if ch not in {2, 3}]
    dt_ns = 1.25
    ped_k = 1.0

    entry_start = 0
    entry_stop_exclusive = 1000

    out_dir = Path("./biexp_fits_run806_allruns_ak")
    out_dir.mkdir(parents=True, exist_ok=True)

    # chunk size: tune this (bigger = faster I/O, more RAM)
    step_size = 2000

    results = []

    for run in range(run_start, run_end + 1):
        root_path = f"/Users/leobailloeul/Documents/coding/LANL/data/806_real/flattened_waveform_Run806.{run:02d}_default.root"
        print(f"\n================ RUN {run:02d} ================")
        print(f"Opening: {root_path}")

        try:
            # iterate gives you (chunk_start, chunk_stop) slices implicitly; we use entry_start/stop
            for arrays in uproot.iterate(
                f"{root_path}:{tree_name}",
                expressions=branches,
                library="ak",
                entry_start=entry_start,
                entry_stop=entry_stop_exclusive,
                step_size=step_size,
                how=dict,   # arrays is a dict: {branch: ak.Array}
            ):
                # find waveform length once per chunk (assumes fixed-length waveforms)
                any_branch = next(iter(arrays.keys()))
                # convert one element to numpy to get nsamples
                nsamp = len(ak.to_numpy(arrays[any_branch][0]))
                t_full = (np.arange(nsamp, dtype=DTYPE) * dt_ns)

                # how many entries in this chunk?
                n_chunk = len(arrays[any_branch])

                # IMPORTANT: we need global entry index. uproot.iterate doesn't directly hand it;
                # easiest is to track it ourselves using a counter.
                # We'll store a running offset per run.
                # (This is correct as long as you don't skip entries inside the chunk.)
                # We'll compute it from a running variable:
                if not hasattr(main, "_entry_cursor"):
                    main._entry_cursor = {}
                if run not in main._entry_cursor:
                    main._entry_cursor[run] = entry_start
                base_entry = main._entry_cursor[run]
                main._entry_cursor[run] += n_chunk

                for branch in branches:
                    if branch not in arrays:
                        continue
                    ch = int(branch.split("ch")[-1])

                    wav_chunk = arrays[branch]  # ak.Array of shape (n_chunk, nsamp)

                    for j in range(n_chunk):
                        entry_i = base_entry + j

                        # pull waveform j -> numpy (fast C++ -> numpy)
                        wave = -ak.to_numpy(wav_chunk[j]).astype(DTYPE, copy=False)
                        t = t_full  # reuse

                        # baseline/noise
                        try:
                            ped_i, noise_i, (b0, b1), wbins = baseline_noise_bestwindow(
                                t, wave, dt_ns,
                                window_ns=500.0,
                                step_ns=25.0,
                                min_sigma=1e-3,
                                flat_frac_min=0.98,
                                flat_k=3.0,
                                spike_k=4.0,
                            )
                            ped_lo = ped_i - ped_k * noise_i
                            ped_hi = ped_i + ped_k * noise_i
                            bounds_event = (
                                [0, 0.1, 1, 0, 0.1, ped_lo],
                                [1000, 30, 200, 1200, 10, ped_hi],
                            )
                        except RuntimeError as e:
                            results.append({
                                "run": run, "chan": ch, "branch": branch, "entry": entry_i,
                                "success": False, "reason": "baseline_failed", "error": str(e),
                            })
                            continue

                        row_base = {
                            "run": run, "chan": ch, "branch": branch, "entry": entry_i,
                            "success": False, "reason": "",
                            "peak_id": np.nan, "t_peak": np.nan, "peak_amp": np.nan,
                            "fit_start": np.nan, "fit_end": np.nan,
                            "chi2": np.nan, "ndof": np.nan, "red_chi2": np.nan,
                            "A": np.nan, "tau_rise": np.nan, "tau_fall": np.nan, "t0": np.nan,
                            "sigma": np.nan, "pedestal": np.nan,
                            "A_err": np.nan, "tau_rise_err": np.nan, "tau_fall_err": np.nan,
                            "t0_err": np.nan, "sigma_err": np.nan, "pedestal_err": np.nan,
                            "baseline_t0": b0, "baseline_t1": b1,
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
                            baseline_override=ped_i,
                        )
                        if len(peak_windows) == 0:
                            r = dict(row_base)
                            r["reason"] = "no_peaks_found"
                            results.append(r)
                            continue

                        snr_min = 8.0
                        peak_windows = [w for w in peak_windows if (w["peak_amp"] > snr_min * noise_i)]
                        if len(peak_windows) == 0:
                            r = dict(row_base)
                            r["reason"] = f"snr_cut_{snr_min}"
                            results.append(r)
                            continue

                        for win in peak_windows:
                            fit_start_i, fit_end_i = win["fit_start"], win["fit_end"]
                            mask = (t >= fit_start_i) & (t <= fit_end_i)
                            t_fit = t[mask]
                            y_fit = wave[mask]
                            if len(t_fit) < 10:
                                continue

                            prow = dict(row_base)
                            prow.update({
                                "peak_id": win["peak_id"],
                                "t_peak": win["t_peak"],
                                "peak_amp": win["peak_amp"],
                                "fit_start": fit_start_i,
                                "fit_end": fit_end_i,
                            })

                            p0_event = [
                                max(1.0, win["peak_amp"]),
                                5.0,
                                30.0,
                                win["t_peak"],
                                2.0,
                                ped_i,
                            ]

                            try:
                                popt, pcov, chi2, ndof, red = fit_biexp_single(
                                    t_fit, y_fit, bounds=bounds_event, p0=p0_event, sigma_signal=noise_i
                                )

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
                                    "chi2": chi2, "ndof": ndof, "red_chi2": red,
                                    "A": popt[0], "tau_rise": popt[1], "tau_fall": popt[2],
                                    "t0": popt[3], "sigma": popt[4], "pedestal": popt[5],
                                    "A_err": perr[0], "tau_rise_err": perr[1], "tau_fall_err": perr[2],
                                    "t0_err": perr[3], "sigma_err": perr[4], "pedestal_err": perr[5],
                                    "reason": "",
                                })
                            except Exception as e:
                                prow["reason"] = "fit_failed"
                                prow["error"] = str(e)

                            results.append(prow)

        except (FileNotFoundError, OSError) as e:
            print(f"[skip] cannot open: {root_path} ({e})")
            continue

    df = pd.DataFrame(results)
    df_good = df[(df["success"]) & (df["red_chi2"] < CHI2_CUT)]
    out_csv = out_dir / "biexp_fit_results.csv"
    df_good.to_csv(out_csv, index=False)

    print(f"\nSaved CSV (passing only): {out_csv}")
    print(f"Passing fits: {len(df_good)} / {len(df)}")

if __name__ == "__main__":
    main()