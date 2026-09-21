#!/usr/bin/env python3
"""
Analyze deconvolution hit results: charge histogram and charge vs time 2D histogram.
Works with CSV output from cosine_advanced_convolve_at_scale_*.py (MilliQan + Na22).

Examples:
  # Default — run all plots for every preset CSV that exists
  python deconvolution/analyze_deconv.py

  # Only the plain charge histogram, Na22 preset
  python deconvolution/analyze_deconv.py --preset na22 --plot charge

  # Arbitrary CSV, only the 2D charge-vs-time plot
  python deconvolution/analyze_deconv.py --csv outputs/my_dir/hits.csv --plot charge_vs_time

  # Two specific plots on Na22
  python deconvolution/analyze_deconv.py --preset na22 --plot charge charge_vs_time
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pmtdeconv.plotting.style import use_minimal_style, minimal_axes, root_viridis
from pmtdeconv.io.deconv_pulse import DECONV_PULSE_CSV, resolve_deconv_pulse_csv
import numpy as np
from pathlib import Path

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # bailloeulScripts/
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def _preset_csv(canonical_dir, legacy_dir=None, prefer_legacy=False):
    if prefer_legacy and legacy_dir is not None:
        found = resolve_deconv_pulse_csv(OUTPUTS_DIR / legacy_dir)
        if found is not None:
            return found
    found = resolve_deconv_pulse_csv(OUTPUTS_DIR / canonical_dir)
    if found is not None:
        return found
    if legacy_dir is not None:
        found = resolve_deconv_pulse_csv(OUTPUTS_DIR / legacy_dir)
        if found is not None:
            return found
    return OUTPUTS_DIR / canonical_dir / DECONV_PULSE_CSV


# Presets. Each entry is a dict with a label + csv + default channels.
PRESETS = {
    "milliqan": dict(
        label="MilliQan Run2150.1 Cosine Advanced",
        csv=_preset_csv(
            "deconv_pulse_milliqan_lab_cosine_advanced",
            "deconv_pulses_run2150.1_MilliQan_cosine_advanced",
            prefer_legacy=True,
        ),
        channels=None,
    ),
    "run1000": dict(
        label="Run1000 Cosine Advanced",
        csv=_preset_csv(
            "deconv_pulse_milliqan_lab_cosine_advanced",
            "deconv_pulses_run1000.1_allruns_MilliQan_cosine_advanced",
            prefer_legacy=True,
        ),
        channels=[38, 39],    # per-channel plots, one figure per channel
    ),
    "na22": dict(
        label="Na22 Lab Cosine Advanced",
        csv=_preset_csv(
            "deconv_pulse_na22_lab_cosine_advanced",
            "deconv_pulses_Na22_lab_cosine_advanced",
        ),
        channels=[1, 2],      # per-channel plots, one figure per channel
    ),
    "na22_wavelet": dict(
        label="Na22 Lab Cosine Wavelet Battle-Lemarie bior3.5",
        csv=_preset_csv(
            "deconv_pulse_na22_lab_cosine_wavelet",
            "deconv_pulses_Na22_lab_cosine_wavelet",
        ),
        channels=[1, 2],
    ),
}

PLOT_CHOICES = ["all", "charge", "charge_tcut", "charge_vs_height",
                "charge_vs_time", "ch1_vs_ch2"]


def _plot_out_dir(csv_path, output_dir=None):
    out_dir = Path(output_dir) if output_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _safe_label(label):
    # Strip anything that matplotlib/OS could interpret as a path separator.
    bad = [" ", "+", "/", "\\", "(", ")", ",", ":"]
    for ch in bad:
        label = label.replace(ch, "_")
    # Collapse accidental runs of underscores.
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def _prepare(df, q_range, bins_q=100):
    """Return (q, t, peak_height, bins, bin_centers, bin_width, q_lo, q_hi).

    bins_q is the number of bins on the charge axis (edges = bins_q + 1)."""
    q = df["charge"].values
    t = df["t_peak"].values
    peak_height = df["height"].values.astype(np.float64)

    q_lo = q_range[0] if q_range[0] is not None else 0.0
    if q_range[1] is not None:
        q_hi = q_range[1]
    else:
        pos = q[q > 0]
        q_hi = np.percentile(pos, 99.5) if len(pos) > 0 else (q.max() if len(q) else 1.0)
    bins = np.linspace(q_lo, q_hi, int(bins_q) + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    return q, t, peak_height, bins, bin_centers, bin_width, q_lo, q_hi


# ───────────────────────── individual plot functions ─────────────────────────

def plot_charge(csv_path, df, label, q_range, bins_q=100, output_dir=None):
    """Charge histogram with bars colored by mean peak height."""
    from scipy.stats import binned_statistic
    q, _, peak_height, bins, bin_centers, bin_width, *_ = _prepare(df, q_range, bins_q)

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, _ = np.histogram(q, bins=bins)
    mean_h, _, _ = binned_statistic(q, peak_height, statistic="mean", bins=bins)

    cmap = root_viridis()
    h_min, h_max = np.nanmin(mean_h), np.nanmax(mean_h)
    norm = LogNorm(vmin=max(h_min, 1e-6), vmax=h_max)
    colors = cmap(norm(np.where(np.isnan(mean_h), h_min, mean_h)))

    ax.bar(bin_centers, counts, width=bin_width, color=colors, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Mean Peak Height [a.u.]")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed Charge [pVs]")
    ax.set_ylabel("Counts")
    minimal_axes(ax)

    fig.tight_layout()
    out = _plot_out_dir(csv_path, output_dir) / f"charge_histogram_{_safe_label(label)}.pdf"
    fig.savefig(out, dpi=200, transparent=True)
    plt.close("all")
    print(f"  Saved: {out}")
    return out


def plot_charge_tcut(csv_path, df, label, q_range, t_lo_cut, t_hi_cut,
                     bins_q=100, output_dir=None):
    """Charge histogram restricted to a time window."""
    from scipy.stats import binned_statistic
    q, t, peak_height, bins, bin_centers, bin_width, *_ = _prepare(df, q_range, bins_q)

    mask = (t >= t_lo_cut) & (t <= t_hi_cut)
    q_cut = q[mask]
    h_cut = peak_height[mask]

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts_cut, _ = np.histogram(q_cut, bins=bins)

    cmap = root_viridis()
    if len(q_cut) > 0:
        mean_h_cut, _, _ = binned_statistic(q_cut, h_cut, statistic="mean", bins=bins)
        hmin = np.nanmin(mean_h_cut) if np.any(~np.isnan(mean_h_cut)) else 1.0
        hmax = np.nanmax(mean_h_cut) if np.any(~np.isnan(mean_h_cut)) else hmin * 1.1
        norm = LogNorm(vmin=max(hmin, 1e-6), vmax=max(hmax, hmin * 1.1))
        colors = cmap(norm(np.where(np.isnan(mean_h_cut), hmin, mean_h_cut)))
    else:
        norm = LogNorm(vmin=1e-6, vmax=1.0)
        colors = ["steelblue"] * len(bin_centers)

    ax.bar(bin_centers, counts_cut, width=bin_width, color=colors, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Mean Peak Height [a.u.]")

    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed Charge [pVs]")
    ax.set_ylabel("Counts")
    minimal_axes(ax)

    fig.tight_layout()
    out = _plot_out_dir(csv_path, output_dir) / (
        f"charge_histogram_tcut{int(t_lo_cut)}-{int(t_hi_cut)}ns_{_safe_label(label)}.pdf"
    )
    fig.savefig(out, dpi=200, transparent=True)
    plt.close("all")
    print(f"  Saved: {out}")
    return out


def plot_charge_vs_height(csv_path, df, label, q_range, bins_q=100, bins_h=80,
                          output_dir=None):
    """2D histogram of charge vs peak height."""
    q, _, peak_height, bins, *_ = _prepare(df, q_range, bins_q)

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    h_lo, h_hi = np.percentile(peak_height, [1, 99])
    h_bins = np.linspace(h_lo, h_hi, int(bins_h) + 1)

    h2 = ax.hist2d(q, peak_height, bins=[bins, h_bins],
                   cmap=root_viridis(), norm=LogNorm(vmin=1), cmin=1)
    fig.colorbar(h2[3], ax=ax).set_label("Counts")

    ax.set_xlabel("Reconstructed Charge [pVs]")
    ax.set_ylabel("Peak Height [a.u.]")
    minimal_axes(ax)

    fig.tight_layout()
    out = _plot_out_dir(csv_path, output_dir) / f"charge_vs_height_{_safe_label(label)}.pdf"
    fig.savefig(out, dpi=200, transparent=True)
    plt.close("all")
    print(f"  Saved: {out}")
    return out


def plot_charge_vs_time(csv_path, df, label, q_range, bins_q=50, bins_t=120,
                        output_dir=None):
    """2D histogram of charge vs hit time."""
    q, t, _, _, _, _, q_lo, q_hi = _prepare(df, q_range, bins_q)

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    t_bins = np.linspace(np.min(t), np.max(t), int(bins_t) + 1)
    q_bins = np.linspace(q_lo, q_hi, int(bins_q) + 1)

    h2 = ax.hist2d(t, q, bins=[t_bins, q_bins],
                   cmap=root_viridis(), norm=LogNorm(vmin=1), cmin=1)
    fig.colorbar(h2[3], ax=ax).set_label("Counts")

    ax.set_xlabel("Hit Time [ns]")
    ax.set_ylabel("Reconstructed Charge [pVs]")
    minimal_axes(ax)

    fig.tight_layout()
    out = _plot_out_dir(csv_path, output_dir) / f"charge_vs_time_{_safe_label(label)}.pdf"
    fig.savefig(out, dpi=200, transparent=True)
    plt.close("all")
    print(f"  Saved: {out}")
    return out

def plot_ch1_vs_ch2(csv_path, df_full, label, q_range, t_cut=None,
                    dt_window=None, ch_pair=(1, 2),
                    calib=None, emax=1500.0,
                    peak_lines=(511.0, 1275.0),
                    bins_2d=80, window_scale=1.0, output_dir=None):
    """2D histogram of E_chA vs E_chB for coincident DRS4 triggers.

    Styled after the reference Na22 HV=1.4 kV plot:
      - 2D hist of coincident pulse energies
      - horizontal + vertical crosshairs at 511 keV (red)
      - horizontal + vertical crosshairs at 1275 keV (green)

    Events are matched on (run, cycle, entry); per channel per trigger we
    SUM the charge of all hits (total charge in the bar = main pulse +
    secondaries), not just the dominant pulse. An optional time-peak window
    cut (t_peak in [t_cut[0], t_cut[1]], using the dominant pulse's time) is
    applied to BOTH channels; |Δt| ≤ dt_window tightens to the prompt
    coincidence.

    Parameters
    ----------
    df_full : DataFrame
        Full CSV (all channels), NOT already filtered to one channel.
    t_cut : tuple(float, float) or None
    dt_window : float or None
    ch_pair : (int, int)
    calib : dict {chA: slope_keV_per_charge, chB: slope_keV_per_charge} or None
        If given, axes are drawn in keV and the 511/1275 reference lines
        are shown. If None, axes stay in arbitrary units and no peak
        lines are drawn.
    emax : float
        Upper axis limit when calib is given [keV] (matches reference
        plot's 0–1500 keV range).
    peak_lines : (float, float)
        Energies [keV] for the red and green crosshair pairs.
    """
    chA, chB = ch_pair
    need = {"run", "cycle", "entry", "chan", "charge", "t_peak",
            "win_lo_ns", "win_hi_ns"}
    missing = need - set(df_full.columns)
    if missing:
        print(f"  [skip ch1_vs_ch2] CSV missing columns: {missing}")
        return None

    # Per (trigger, channel): CLUSTER the pulses by time-proximity and use
    # the dominant cluster's total charge. Each pulse (tallest first) seeds
    # a window proportional to its own trough-walked width
    # (window_scale·width); pulses inside it merge into that cluster and are
    # summed; repeat for the remaining pulses. This groups a deposit's main
    # pulse + immediate pile-up while excluding late ringing/afterpulses,
    # then keeps the largest cluster. The cluster's seed time is the
    # representative pulse time for the optional timing cuts.
    keys = ["run", "cycle", "entry"]
    gk = keys + ["chan"]

    def _dominant_cluster(g):
        t = g["t_peak"].to_numpy(float)
        q = g["charge"].to_numpy(float)
        w = (g["win_hi_ns"] - g["win_lo_ns"]).to_numpy(float)
        o = np.argsort(-q)            # tallest first
        t, q, w = t[o], q[o], w[o]
        used = np.zeros(len(q), bool)
        best_q, best_t = -np.inf, np.nan
        for i in range(len(q)):
            if used[i]:
                continue
            half = 0.5 * window_scale * max(w[i], 1e-6)
            sel = (~used) & (t >= t[i] - half) & (t <= t[i] + half)
            cq = float(q[sel].sum())
            if cq > best_q:
                best_q, best_t = cq, float(t[i])
            used |= sel
        return pd.Series({"charge": best_q, "t_peak": best_t})

    best = (df_full.groupby(gk, group_keys=False)
            .apply(_dominant_cluster).reset_index())
    dA = best[best["chan"] == chA][keys + ["charge", "t_peak"]].rename(
        columns={"charge": "qA", "t_peak": "tA"}
    )
    dB = best[best["chan"] == chB][keys + ["charge", "t_peak"]].rename(
        columns={"charge": "qB", "t_peak": "tB"}
    )
    coinc = dA.merge(dB, on=keys, how="inner")

    n_all_coinc = len(coinc)
    if t_cut is not None:
        t_lo_cut, t_hi_cut = t_cut
        coinc = coinc[coinc["tA"].between(t_lo_cut, t_hi_cut) &
                      coinc["tB"].between(t_lo_cut, t_hi_cut)]
    if dt_window is not None:
        coinc = coinc[(coinc["tA"] - coinc["tB"]).abs() <= dt_window]

    qA = coinc["qA"].values
    qB = coinc["qB"].values
    if len(qA) == 0:
        print(f"  [ch{chA}_vs_ch{chB}] 0 coincident events in window — skip")
        return

    # ── calibration to keV (optional) ──
    in_keV = calib is not None and chA in calib and chB in calib
    if in_keV:
        xA = qA * float(calib[chA])
        xB = qB * float(calib[chB])
        x_lo, x_hi = 0.0, float(emax)
        xlabel = f"CH{chA} pulse energy [keV]"
        ylabel = f"CH{chB} pulse energy [keV]"
    else:
        xA, xB = qA, qB
        x_lo = q_range[0] if q_range[0] is not None else 0.0
        if q_range[1] is not None:
            x_hi = q_range[1]
        else:
            x_hi = float(np.percentile(np.concatenate([xA, xB]), 99.5))
        xlabel = f"Q ch{chA} [pVs]"
        ylabel = f"Q ch{chB} [pVs]"

    edges = np.linspace(x_lo, x_hi, int(bins_2d) + 1)

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    h2 = ax.hist2d(xA, xB, bins=[edges, edges],
                   cmap=root_viridis(), norm=LogNorm(vmin=1), cmin=1)
    fig.colorbar(h2[3], ax=ax).set_label("Coincident events")

    # ── reference peak lines (only drawn when calibrated to keV) ──
    if in_keV:
        e511, e1275 = peak_lines
        # 511 keV — red crosshair
        ax.axvline(e511, color="red",   lw=1.2, alpha=0.85)
        ax.axhline(e511, color="red",   lw=1.2, alpha=0.85)
        # 1275 keV — green crosshair
        ax.axvline(e1275, color="lime", lw=1.2, alpha=0.85)
        ax.axhline(e1275, color="lime", lw=1.2, alpha=0.85)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(x_lo, x_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    minimal_axes(ax)

    fig.tight_layout()
    suffix = ""
    if t_cut is not None:
        suffix += f"_tcut{int(t_cut[0])}-{int(t_cut[1])}ns"
    if dt_window is not None:
        suffix += f"_dt{int(dt_window)}ns"
    if in_keV:
        suffix += "_keV"
    out = _plot_out_dir(csv_path, output_dir) / f"ch{chA}_vs_ch{chB}{suffix}_{_safe_label(label)}.pdf"
    fig.savefig(out, dpi=200, transparent=True)
    plt.close("all")
    print(f"  Saved: {out}  (N_coinc={n_all_coinc}, N_after_cuts={len(qA)})")
    return out


# ───────────────────────── main driver ─────────────────────────

def analyze(csv_path, label="", channels=None, q_range=(None, None),
            ipulse=None, plots=("all",), t_cut=(250.0, 400.0),
            df_override=None,
            bins_q=100, bins_q_2d=50, bins_t=120, bins_h=80,
            output_dir=None):
    """Core analysis. Pass df_override to skip the CSV re-read (used when
    splitting by run at the dispatch layer)."""
    if df_override is None:
        print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = df_override.copy()
        print(f"Reusing pre-loaded rows (N={len(df)})")
    print(f"  Total hits: {len(df)}")

    if channels is not None:
        df = df[df["chan"].isin(channels)]
        label = f"{label} ch{'+'.join(str(c) for c in sorted(channels))}"
        print(f"  Filtered to channels {channels}: {len(df)} hits")

    if ipulse is not None:
        if "ipulse" not in df.columns:
            print(f"  [warn] ipulse column not in CSV — ignoring --ipulse filter")
        else:
            df = df[df["ipulse"] == ipulse]
            label = f"{label} ipulse={ipulse}"
            print(f"  Filtered to ipulse={ipulse}: {len(df)} hits")

    if len(df) == 0:
        print("  No rows after filtering — nothing to plot.")
        return []

    if "det" in df.columns:
        print(f"  Detectors: {sorted(df['det'].unique())}")
    print(f"  Channels: {sorted(df['chan'].unique())}")
    print(f"  Runs: {sorted(df['run'].unique())}")
    print(f"  Charge range: [{df['charge'].min():.4f}, {df['charge'].max():.4f}]")
    print(f"  Mean charge: {df['charge'].mean():.4f}")
    print(f"  Median charge: {df['charge'].median():.4f}")

    want = set(plots)
    do_all = "all" in want
    outputs = []

    if do_all or "charge" in want:
        outputs.append(plot_charge(csv_path, df, label, q_range,
                                   bins_q=bins_q, output_dir=output_dir))
    # Bug-fix (2026-05-30 gallery run): only make the time-cut charge plot
    # when a t_cut window was actually supplied. Presets like na22_wavelet
    # leave t_cut=None; the old code subscripted it unconditionally and
    # crashed with TypeError after the other plots were already written.
    if (do_all or "charge_tcut" in want) and t_cut is not None:
        outputs.append(plot_charge_tcut(csv_path, df, label, q_range,
                                        t_cut[0], t_cut[1], bins_q=bins_q,
                                        output_dir=output_dir))
    if do_all or "charge_vs_height" in want:
        outputs.append(plot_charge_vs_height(csv_path, df, label, q_range,
                                             bins_q=bins_q_2d, bins_h=bins_h,
                                             output_dir=output_dir))
    if do_all or "charge_vs_time" in want:
        outputs.append(plot_charge_vs_time(csv_path, df, label, q_range,
                                           bins_q=bins_q_2d, bins_t=bins_t,
                                           output_dir=output_dir))

    # Per-channel summary
    print(f"\n  Per-channel charge summary:")
    has_det = "det" in df.columns
    if has_det:
        print(f"  {'Det':<6} {'Chan':<6} {'N hits':<8} {'Mean':<12} {'Median':<12} {'Std':<12}")
        print(f"  {'-'*56}")
        for (det, ch), grp in df.groupby(["det", "chan"]):
            s = grp["charge"]
            print(f"  {det:<6} {ch:<6} {len(s):<8} {s.mean():<12.4f} "
                  f"{s.median():<12.4f} {s.std():<12.4f}")
    else:
        print(f"  {'Chan':<6} {'N hits':<8} {'Mean':<12} {'Median':<12} {'Std':<12}")
        print(f"  {'-'*50}")
        for ch in sorted(df["chan"].unique()):
            s = df[df["chan"] == ch]["charge"]
            print(f"  {ch:<6} {len(s):<8} {s.mean():<12.4f} "
                  f"{s.median():<12.4f} {s.std():<12.4f}")
    return [p for p in outputs if p is not None]


# ───────────────────────── CLI ─────────────────────────

def _dispatch(csv_path, base_label, channels, q_range, plots, ipulse, t_cut,
              dt_window, calib, emax, split_runs, df_override=None,
              bins_q=100, bins_q_2d=50, bins_t=120, bins_h=80, bins_2d=80,
              output_dir=None):
    """Run the whole plot suite against a single CSV. When split_runs=True,
    iterate over each run in the CSV and produce an independent set of
    plots (filename + label both tagged with runNNNNN)."""
    per_chan_plots = tuple(p for p in plots if p != "ch1_vs_ch2")
    want_ch = "ch1_vs_ch2" in plots or "all" in plots
    pair_src = channels if (channels and len(channels) >= 2) else [1, 2]
    ch_pair = (int(pair_src[0]), int(pair_src[1]))

    df_full = df_override if df_override is not None else pd.read_csv(csv_path)
    outputs = []

    if split_runs and "run" in df_full.columns:
        runs = sorted(df_full["run"].unique())
    else:
        runs = [None]

    for run in runs:
        if run is None:
            df_run = df_full
            run_label = base_label
            tag = ""
        else:
            df_run = df_full[df_full["run"] == run]
            run_label = f"{base_label} run{run}"
            tag = f" [run {run}, N={len(df_run)}]"
        print(f"\n══════ {base_label}{tag} ══════")

        # Per-channel plots
        if channels is None:
            outputs.extend(analyze(csv_path, label=run_label, channels=None,
                                   q_range=q_range, ipulse=ipulse,
                                   plots=per_chan_plots, t_cut=t_cut,
                                   df_override=df_run,
                                   bins_q=bins_q, bins_q_2d=bins_q_2d,
                                   bins_t=bins_t, bins_h=bins_h,
                                   output_dir=output_dir))
        else:
            for ch in channels:
                outputs.extend(analyze(csv_path, label=run_label, channels=[ch],
                                       q_range=q_range, ipulse=ipulse,
                                       plots=per_chan_plots, t_cut=t_cut,
                                       df_override=df_run,
                                       bins_q=bins_q, bins_q_2d=bins_q_2d,
                                       bins_t=bins_t, bins_h=bins_h,
                                       output_dir=output_dir))

        # Q_ch1 vs Q_ch2
        if want_ch:
            print(f"\n── Q_ch{ch_pair[0]} vs Q_ch{ch_pair[1]}{tag} ──")
            outputs.append(plot_ch1_vs_ch2(
                csv_path, df_run, label=run_label,
                q_range=q_range, t_cut=None, dt_window=dt_window,
                ch_pair=ch_pair, calib=calib, emax=emax,
                bins_2d=bins_2d, output_dir=output_dir))
            outputs.append(plot_ch1_vs_ch2(
                csv_path, df_run, label=run_label,
                q_range=q_range, t_cut=t_cut, dt_window=dt_window,
                ch_pair=ch_pair, calib=calib, emax=emax,
                bins_2d=bins_2d, output_dir=output_dir))
    return [p for p in outputs if p is not None]


def _run_preset(name, q_range, plots, ipulse, chan_override, t_cut,
                dt_window=None, calib=None, emax=1500.0, split_runs=True,
                bins_q=100, bins_q_2d=50, bins_t=120, bins_h=80, bins_2d=80,
                output_dir=None):
    spec = PRESETS[name]
    csv_path = spec["csv"]
    if not csv_path.exists():
        print(f"[skip] {name}: CSV not found at {csv_path}")
        return []
    channels = chan_override if chan_override is not None else spec["channels"]
    return _dispatch(csv_path, spec["label"], channels, q_range, plots, ipulse,
                     t_cut, dt_window, calib, emax, split_runs,
                     bins_q=bins_q, bins_q_2d=bins_q_2d,
                     bins_t=bins_t, bins_h=bins_h, bins_2d=bins_2d,
                     output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze deconvolution hit results")
    parser.add_argument("--preset", choices=list(PRESETS.keys()) + ["all"],
                        default="all",
                        help="Which preset CSV to analyze (default: all presets that exist)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Analyze an arbitrary CSV instead of a preset. "
                             "Label defaults to the parent folder name.")
    parser.add_argument("--plot", choices=PLOT_CHOICES, nargs="+", default=["all"],
                        help="Which plot(s) to produce. Repeatable. Default: all.")
    parser.add_argument("--xmin", type=float, default=None, help="Charge axis lower limit")
    parser.add_argument("--xmax", type=float, default=None, help="Charge axis upper limit")
    parser.add_argument("--chan",  type=int,   default=None, nargs="+",
                        help="Filter to specific channel(s); overrides preset default")
    parser.add_argument("--ipulse", type=int, default=None,
                        help="Filter to hits with this ipulse value (e.g. 0 for source peak)")
    parser.add_argument("--t-cut-lo", type=float, default=None,
                        help="Lower edge of the time-cut charge histogram window [ns] (default: 500)")
    parser.add_argument("--t-cut-hi", type=float, default=None,
                        help="Upper edge of the time-cut charge histogram window [ns] (default: 800)")
    parser.add_argument("--dt-window", type=float, default=None,
                        help="For ch1_vs_ch2: tighten to |t_ch1 − t_ch2| ≤ dt [ns] "
                             "(default: no Δt cut)")
    parser.add_argument("--calib-ch1", type=float, default=None,
                        help="Ch1 calibration slope [keV per charge unit]. "
                             "If given together with --calib-ch2, ch1_vs_ch2 is "
                             "drawn on a keV/keV axis with 511 & 1275 keV lines.")
    parser.add_argument("--calib-ch2", type=float, default=None,
                        help="Ch2 calibration slope [keV per charge unit].")
    parser.add_argument("--emax", type=float, default=1500.0,
                        help="Upper axis limit [keV] for the ch1_vs_ch2 plot "
                             "when calibrated (default: 1500 keV, matching the "
                             "reference Na22 HV=1.4 kV figure)")
    parser.add_argument("--no-split-runs", action="store_true",
                        help="Pool all runs into a single plot set. Default is "
                             "to produce a separate plot suite per run.")
    # ── Binning ──
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of bins on the charge axis for the 1D "
                             "histograms (plot_charge, plot_charge_tcut). "
                             "Default: 100.")
    parser.add_argument("--bins-2d-q", type=int, default=50,
                        help="Charge-axis bins for 2D plots charge_vs_{time,height}. "
                             "Default: 50 (fewer than 1D on purpose — less noisy).")
    parser.add_argument("--bins-t", type=int, default=120,
                        help="Time-axis bins for charge_vs_time. Default: 120.")
    parser.add_argument("--bins-h", type=int, default=80,
                        help="Height-axis bins for charge_vs_height. Default: 80.")
    parser.add_argument("--bins-ch1ch2", "--bins-2d", type=int, default=80,
                        dest="bins_ch1ch2",
                        help="Bins per axis for the ch1_vs_ch2 2D plot. Default: 80. "
                             "(--bins-2d kept as an alias for backwards compatibility.)")
    args = parser.parse_args()

    q_range = (args.xmin, args.xmax)
    plots = tuple(args.plot)
    t_cut = (args.t_cut_lo, args.t_cut_hi)
    split_runs = not args.no_split_runs

    # Build per-channel calibration dict (only applied to ch1_vs_ch2)
    calib = None
    if args.calib_ch1 is not None and args.calib_ch2 is not None:
        calib = {1: args.calib_ch1, 2: args.calib_ch2}
    elif (args.calib_ch1 is not None) ^ (args.calib_ch2 is not None):
        print("[warn] need BOTH --calib-ch1 and --calib-ch2 to enable keV axes; "
              "ignoring partial calibration.")

    bin_kwargs = dict(bins_q=args.bins, bins_q_2d=args.bins_2d_q,
                      bins_t=args.bins_t, bins_h=args.bins_h,
                      bins_2d=args.bins_ch1ch2)

    if args.csv is not None:
        if not args.csv.exists():
            print(f"ERROR: --csv path not found: {args.csv}")
            raise SystemExit(1)
        label = args.csv.parent.name
        channels = args.chan if args.chan else None
        _dispatch(args.csv, label, channels, q_range, plots, args.ipulse,
                  t_cut, args.dt_window, calib, args.emax, split_runs,
                  **bin_kwargs)
    else:
        targets = list(PRESETS.keys()) if args.preset == "all" else [args.preset]
        ran_any = False
        for name in targets:
            if PRESETS[name]["csv"].exists():
                ran_any = True
                _run_preset(name, q_range, plots, args.ipulse, args.chan, t_cut,
                            dt_window=args.dt_window, calib=calib,
                            emax=args.emax, split_runs=split_runs,
                            **bin_kwargs)

        if not ran_any:
            print("No CSV files found for the selected preset(s). "
                  "Run the deconvolution first, or pass --csv <path>.")
