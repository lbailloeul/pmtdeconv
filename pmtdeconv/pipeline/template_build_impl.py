#!/usr/bin/env python3
"""build_spe_templates_perfile.py — fit a stacked SPE template per (dataset, file,
channel), following the Cd109 protocol: low-vMax SPE-like selection -> peak-aligned
peak-normalized stack -> biexp (Gaussian-smeared) fit with unlocked tau_rise. One
template per file per channel; the user wants to verify same-setup files agree.

Output: outputs/spe_templates_perfile/spe_templates_perfile.csv + per-dataset
overlay plot + per-(file,channel) stacked-template .npy.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.core.kernels import spe_overshoot

N_SAMP = 1024


def _resolve_data_dir(d):
    """Resolve a dataset's data directory.

    The legacy ``DATASETS`` entries use paths like ``../../../data/Cd109_lab``
    that were written relative to the *old* top-level script location
    (``bailloeulScripts/``). After the PR-10 lift into
    ``pmtdeconv/pipeline/`` a bare ``os.chdir(__file__.parent)`` made those
    resolve two directories too deep (→ ``timewalksim/data``). Anchor
    relative paths to ``PROJECT_ROOT`` (= ``bailloeulScripts/``) so they
    point back at ``LANL/data`` regardless of cwd. Absolute paths (e.g. a
    ``--data-root`` override injected by the CLI) pass through unchanged.
    """
    p = Path(d)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
BASE_BINS, PRE_BINS = 30, 50
PK_CENTER = 120                    # align all peaks to this sample

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "spe_templates_perfile"

# Per-dataset processing CAPS (build speed only — never science). Everything
# science-bearing (dir, glob, polarity, dt, channels, spe_band) comes from the
# ONE dataset registry (presets/datasets.py + YAML cards) via
# :func:`config_from_preset`; before 2026-07-02 this module carried its own
# parallel 8-entry registry that drifted from the real one and could only be
# reached through a hand-maintained alias map.
BUILD_CAPS = {                       # name -> (max_events, max_files)
    "milliqan":      (10000, None),
    "lanl_806":      (1000, 10),     # many subruns -> cap files for speed
    "lanl_835":      (1000, 10),
    "lanl_movedBar": (1000, 10),
}
DEFAULT_CAPS = (30000, None)


def config_from_preset(ds) -> dict:
    """Adapt a :class:`~pmtdeconv.presets.datasets.DatasetPreset` (or a YAML
    card resolved through the same registry) into the per-dataset build-config
    dict this module's ``main`` consumes.

    Notes vs the retired in-module registry:
    - names are the REGISTRY names (``na22``, ``cd109``, …), so the perfile
      CSV's ``dataset`` column and the ``stacked_*.npy`` tags now match what
      every other subcommand calls the dataset;
    - the old ``MilliQan_2150`` entry (``data/2150``) had no registry
      equivalent — reproduce it with ``--dataset milliqan --data-root
      <...>/data/2150``.
    """
    channels = (list(ds.branch_pattern)
                if isinstance(ds.branch_pattern, list) else "auto")
    max_events, max_files = BUILD_CAPS.get(ds.name, DEFAULT_CAPS)
    cfg = dict(
        name=ds.name, dir=str(ds.data_dir), glob=ds.file_glob,
        format=ds.format, polarity=ds.polarity, dt=ds.dt_ns,
        channels=channels, spe_band=tuple(ds.spe_band_mv),
        max_events=max_events,
    )
    if ds.exclude_glob:
        cfg["exclude_glob"] = ds.exclude_glob
    if max_files:
        cfg["max_files"] = max_files
    return cfg


def configs_from_registry() -> list:
    """One build config per dataset in the shared registry."""
    from pmtdeconv.presets.datasets import DATASETS as _REGISTRY
    return [config_from_preset(ds) for ds in _REGISTRY.values()]


def biexp_smeared(t, A, tau_rise, tau_fall, t0, sigma, ped, dt_s):
    t = np.asarray(t, float); out = np.zeros_like(t)
    m = t >= t0; dd = t[m] - t0
    out[m] = np.exp(-dd / max(tau_fall, 1e-3)) - np.exp(-dd / max(tau_rise, 1e-3))
    sb = max(sigma / dt_s, 1e-6)
    return A * gaussian_filter1d(out, sb, mode="constant") + ped


def biexp_overshoot_smeared(t, A, tau_rise, tau_fall, t0, sigma, ped,
                            U_os0, U_os1, tau_os, t0_os, sigma_os, dt_s):
    """Gaussian-smeared biexp peak MINUS the Daya Bay baseline-restoration
    undershoot (``utils.spe_overshoot``, their Eq. 4). The undershoot's
    time origin is the pulse onset ``t0``, so it is evaluated at ``t - t0``.
    Positive-going convention: the recovery dips below baseline, hence the
    subtraction. ``t_fermi``/``w_fermi`` are held at the Daya Bay 50/10 ns
    constants for now (per the design decision); free them later if a
    detector needs it.
    """
    peak = biexp_smeared(t, A, tau_rise, tau_fall, t0, sigma, 0.0, dt_s)
    over = spe_overshoot(np.asarray(t, float) - t0,
                         U_os0, U_os1, tau_os, t0_os, sigma_os)
    return peak - over + ped


def load_file_waveforms(path, fmt, channels_spec, max_events):
    """Return ({channel_name: (N_ev, N_samp) ndarray}, dt_ns_or_None). Handles both:
       fmt='flat' -> tree 'waveforms' with branches d0_chXX containing arrays.
       fmt='object' -> tree 'Events' with event/digitizers/digitizers.waveform[16][1024]
                       (a 2 digitizers x 16 channels x 1024 samples nested array).
    """
    f = uproot.open(path)
    if fmt == "flat":
        for tn in ("waveforms", "Events", "tree"):
            try: tree = f[tn]; break
            except Exception: continue
        chans = channels_spec if channels_spec != "auto" else find_channels(tree)
        n_read = min(tree.num_entries, max_events)
        out = {}
        for ch in chans:
            try:
                a = tree[ch].array(entry_stop=n_read, library="np")
                ok = [np.asarray(w, float) for w in a if len(w) == N_SAMP]
                if ok: out[ch] = np.stack(ok)
            except Exception: continue
        return out, None
    elif fmt == "object":
        tree = f["Events"]
        n_read = min(tree.num_entries, max_events)
        wf = tree["event/digitizers/digitizers.waveform[16][1024]"].array(
            entry_stop=n_read, library="np")
        try:
            ns = tree["event/digitizers/digitizers.nanosecondsPerSample"].array(
                entry_stop=1, library="np")
            dt = float(np.atleast_1d(ns[0]).ravel()[0])
        except Exception:
            dt = None
        # each wf[i] is shape (2, 16384) -> reshape to (2, 16, 1024)
        N = len(wf)
        W = np.empty((N, 2, 16, N_SAMP), dtype=np.float64)
        for i in range(N):
            W[i] = np.asarray(wf[i], float).reshape(2, 16, N_SAMP)
        # build per-channel arrays  d0_ch{00..31}  (digitizer 0 -> ch00-15, digitizer 1 -> ch16-31)
        all_chans = {f"d0_ch{dig*16+ch:02d}": W[:, dig, ch, :] for dig in range(2) for ch in range(16)}
        if channels_spec == "auto":
            # auto-detect populated channels using POLARITY-AGNOSTIC max(abs(y))
            chans = []
            for name, wfs in all_chans.items():
                base = np.median(wfs[:, :BASE_BINS], axis=1, keepdims=True)
                vabs = np.abs(wfs - base).max(axis=1)
                if (vabs > 10).mean() >= 0.05: chans.append(name)
            return {ch: all_chans[ch] for ch in chans}, dt
        return {ch: all_chans[ch] for ch in channels_spec if ch in all_chans}, dt
    else:
        raise ValueError(f"unknown format: {fmt}")


def find_channels(tree, sample=200, min_frac=0.05, vmax_th=10.0):
    br = [b for b in tree.keys()
          if (b.startswith("d0_") or b.startswith("voltages")) and
          all(x not in b for x in ("tMax", "vMax", "tStart", "tEnd", "area"))]
    n = min(sample, tree.num_entries)
    chans = []
    for b in br:
        a = tree[b].array(entry_stop=n, library="np")
        cnt = 0
        for w in a:
            w = np.asarray(w, float)
            # POLARITY-AGNOSTIC max(|y - baseline|), matching the object-format
            # auto-detect. The old positive-going `.max()` missed negative-going
            # flat channels (e.g. lanl_806, polarity=-1, channels="auto").
            cnt += int((np.abs(w - np.median(w[:30])).max()) > vmax_th)
        if cnt / n >= min_frac: chans.append(b)
    return chans


def _warn_if_no_spe_mode(vmax, spe_band):
    """Non-SPE guardrail (warn-only; vault 16 design item 4 / 17a).

    A single-photoelectron population shows a pedestal → 1-PE MODE in the
    amplitude spectrum. If the vMax histogram has no local mode inside the
    selected band, the band is probably NOT centered on a 1-PE peak (e.g. the
    legacy na22 (50,200) mV "dominant amplitude band" = multi-PE average, or
    Run 66810's ~24 mV SPE mode skipped by a (50,200) default). Warn; the
    expert decides.
    """
    v = np.asarray(vmax, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 200:
        return                       # too few events to judge a spectrum
    hi = min(float(np.percentile(v, 99.5)), 2.0 * spe_band[1])
    if hi <= 0:
        return
    counts, edges = np.histogram(v, bins=100, range=(0.0, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts.astype(float), np.ones(3) / 3.0, mode="same")
    pk_idx, _ = find_peaks(smooth, prominence=0.05 * float(smooth.max()))
    modes = centers[pk_idx]
    in_band_modes = modes[(modes > spe_band[0]) & (modes <= spe_band[1])]
    if len(in_band_modes) == 0:
        print(f"  [warn] non-SPE guardrail: no amplitude MODE inside the "
              f"spe_band ({spe_band[0]:g},{spe_band[1]:g}) mV "
              f"(modes found: {[round(float(m), 1) for m in modes[:5]]}). "
              f"The band may not be centered on a single-PE peak — the "
              f"stacked template could be a multi-PE average. Inspect the "
              f"amplitude spectrum and set --spe-band on the 1-PE mode.")


def select_and_align_Y(Y, dt, spe_band, single_peak=True):
    """SPE-like selection (amplitude band + clean baseline) + CFD50
    rising-edge align + peak-normalize, then average to a stacked template.
    Y is already baseline-subtracted and polarity-corrected (positive-going).

    Alignment is on the 50%-of-max **rising-edge crossing** (sub-sample,
    interpolated), NOT the peak sample. Peak-max alignment pinned every
    pulse's apex to the same bin and value, producing a 1-sample cusp on
    the stacked peak; CFD alignment leaves the apex free so the stack is
    smooth through the top (see vault note 09a)."""
    vmax = Y.max(axis=1); imax = Y.argmax(axis=1)
    pre = Y[:, BASE_BINS:BASE_BINS + PRE_BINS]
    pmean = pre.mean(axis=1); prms = pre.std(axis=1)
    in_band = (vmax > spe_band[0]) & (vmax <= spe_band[1])
    _warn_if_no_spe_mode(vmax, spe_band)
    # baseline quality: |preMean|<=2 mV, preRMS<=2 mV (loose to handle dataset-to-
    # dataset noise differences; toolbox-general). Drops events where the "pre-
    # pulse" window contains a pulse or where baseline drift is large.
    quiet = (np.abs(pmean) <= 2.0) & (prms <= 2.0)
    keep = np.where(in_band & quiet)[0]
    if single_peak:
        # Single-peak veto — DEFAULT (2026-06-10, Runs 66727-29 diagnosis):
        # an SPE template is the single-PE impulse response by definition.
        # The band + quiet-baseline selection alone admits pulse trains
        # (afterpulses / photon pileup) — on 66727 ch2 they were 97% of the
        # band and the stacked "SPE" fit returned the train envelope
        # (tau_fall 40 ns) instead of the impulse response (2.8 ns). Keep
        # only waveforms with no secondary peak >=25% of the primary
        # beyond +-5 ns. Disable with --no-single-peak.
        kept = []
        for i in keep:
            w = Y[i]; pk = int(imax[i])
            pks, _ = find_peaks(w, height=0.25 * w[pk], distance=6,
                                prominence=0.15 * w[pk])
            if len(pks[np.abs(pks - pk) > 10]) == 0:
                kept.append(i)
        keep = np.asarray(kept, dtype=int)
    if len(keep) < 10:
        return None, len(Y), 0, np.zeros(N_SAMP)
    # ── CFD50 rising-edge crossing (sub-sample) per kept pulse ──
    cfds = np.empty(len(keep))
    rises = np.empty(len(keep))
    for k, i in enumerate(keep):
        w = Y[i]; pk = int(imax[i]); vthr = 0.5 * w[pk]
        j = pk
        while j > 0 and w[j] > vthr:
            j -= 1
        denom = w[j + 1] - w[j]
        cfds[k] = j + (vthr - w[j]) / denom if denom > 0 else float(j)
        rises[k] = pk - cfds[k]
    # Reference crossing chosen so the (free) peak still lands ≈ PK_CENTER,
    # keeping the fit window + display centred as before.
    ref_cfd = PK_CENTER - float(np.median(rises))

    x = np.arange(N_SAMP, dtype=np.float64)
    stacked = np.zeros(N_SAMP); n = 0
    for k, i in enumerate(keep):
        vmax_i = Y[i][int(imax[i])]
        if vmax_i <= 0:
            continue
        # interpolate so this pulse's CFD crossing lands at ref_cfd
        aligned = np.interp(x - (ref_cfd - cfds[k]), x, Y[i],
                            left=0.0, right=0.0)
        stacked += aligned / vmax_i
        n += 1
    if n == 0:
        return None, len(Y), 0, np.zeros(N_SAMP)
    return stacked / n, len(Y), n, stacked / n


def fit_stacked(stacked, dt, model="biexp"):
    """Fit the peak-normalized stacked SPE template.

    ``model``:
    - ``"biexp"`` (default): Gaussian-smeared biexponential (6 params).
    - ``"biexp-overshoot"``: the same peak minus the Daya Bay
      baseline-restoration undershoot (Eq. 4), adding 5 free params
      (U_os0, U_os1, tau_os, t0_os, sigma_os). The fit window is widened
      to ~+450 ns past the peak so the ~150 ns recovery tail is in view.
      Overshoot amplitudes are FRACTIONS of the peak (template is
      peak-normalized), seeded at the Daya Bay ~5% level.

    The returned dict always carries the 6 biexp keys; the overshoot
    keys are present (NaN for the plain biexp model) plus a ``model`` tag.
    """
    t = np.arange(N_SAMP) * dt
    lo = max(0, PK_CENTER - int(30 / dt))
    # biexp-overshoot needs the recovery tail in the window; widen to +450 ns.
    tail_ns = 450.0 if model == "biexp-overshoot" else 200.0
    hi = min(N_SAMP, PK_CENTER + int(tail_ns / dt))
    tw, yw = t[lo:hi], stacked[lo:hi]
    tpk = t[PK_CENTER]
    nan_over = dict(U_os0=np.nan, U_os1=np.nan, tau_os_ns=np.nan,
                    t0_os_ns=np.nan, sigma_os_ns=np.nan,
                    undershoot_depth_frac=np.nan, undershoot_charge_frac=np.nan)
    try:
        if model == "biexp-overshoot":
            #            A    tr   tf    t0       sg   ped  U0    U1    tos   t0os sgos
            p0 =        (1.0, 5.0, 15.0, tpk - 5, 2.0, 0.0, 0.05, 0.01, 150., 70., 30.)
            # Pedestal tightly bound near 0: the data is baseline-subtracted,
            # so the template's DC level is ~0. A loose pedestal lets the fit
            # absorb the elevated ringing tail and plateau too high.
            lo_b = [0.1, 0.01, 0.5, tpk - 30, 0.1, -0.03, 0.0,  0.0,  20.,  10., 2.]
            hi_b = [3.0, 100,  300, tpk + 10,  30,  0.03,  0.3,  0.2,  600., 300, 200]
            popt, _ = curve_fit(
                lambda tt, A, tr, tf, t0, sg, pd, U0, U1, tos, t0o, sgo:
                    biexp_overshoot_smeared(tt, A, tr, tf, t0, sg, pd,
                                            U0, U1, tos, t0o, sgo, dt),
                tw, yw, p0=p0, bounds=(lo_b, hi_b), maxfev=20000)
            A, tr, tf, t0, sg, pd, U0, U1, tos, t0o, sgo = popt
            yf = biexp_overshoot_smeared(tw, A, tr, tf, t0, sg, pd,
                                         U0, U1, tos, t0o, sgo, dt)
            chi2 = float(np.sum((yw - yf) ** 2) / max(len(yw) - 11, 1))
            # Degeneracy-robust observables. The raw (U_os0, tau_os)
            # amplitude/time-constant pair trades off (like A/tau_rise), so
            # report the actual undershoot DEPTH (deepest dip as a fraction
            # of peak height) and CHARGE fraction (∫undershoot / ∫peak) —
            # these are fixed by the data regardless of the parametrization.
            peak_curve = biexp_smeared(tw, A, tr, tf, t0, sg, 0.0, dt)
            over_curve = spe_overshoot(tw - t0, U0, U1, tos, t0o, sgo)
            depth_frac = float(over_curve.max()
                               / max(peak_curve.max(), 1e-9))
            charge_frac = float(over_curve.sum()
                                / max(peak_curve[peak_curve > 0].sum(), 1e-9))
            return dict(model="biexp-overshoot",
                        A=A, tau_rise_ns=tr, tau_fall_ns=tf, t0_ns=t0,
                        sigma_ns=sg, pedestal=pd, chi2_red=chi2,
                        U_os0=U0, U_os1=U1, tau_os_ns=tos,
                        t0_os_ns=t0o, sigma_os_ns=sgo,
                        undershoot_depth_frac=depth_frac,
                        undershoot_charge_frac=charge_frac)
        # default: plain biexp. Pedestal tightly bound near 0 (baseline-
        # subtracted data → DC level ~0; a loose pedestal plateaus too high
        # by absorbing the elevated ringing tail).
        p0 = (1.0, 5.0, 15.0, tpk - 5.0, 2.0, 0.0)
        lo_b = [0.1, 0.01, 0.5, tpk - 30, 0.1, -0.03]
        hi_b = [3.0, 100, 300, tpk + 10, 30, 0.03]
        popt, _ = curve_fit(lambda tt, A, tr, tf, t0, sg, pd:
                            biexp_smeared(tt, A, tr, tf, t0, sg, pd, dt),
                            tw, yw, p0=p0, bounds=(lo_b, hi_b), maxfev=10000)
        A, tr, tf, t0, sg, pd = popt
        yf = biexp_smeared(tw, A, tr, tf, t0, sg, pd, dt)
        chi2 = float(np.sum((yw - yf) ** 2) / max(len(yw) - 6, 1))
        return dict(model="biexp",
                    A=A, tau_rise_ns=tr, tau_fall_ns=tf, t0_ns=t0,
                    sigma_ns=sg, pedestal=pd, chi2_red=chi2, **nan_over)
    except Exception as e:
        return dict(model=model,
                    A=np.nan, tau_rise_ns=np.nan, tau_fall_ns=np.nan, t0_ns=np.nan,
                    sigma_ns=np.nan, pedestal=np.nan, chi2_red=np.nan,
                    error=str(e), **nan_over)


def process_file(path, cfg, out_dir, max_events=None, model="biexp"):
    max_events = max_events or cfg.get("max_events", 10000)
    fmt = cfg.get("format", "flat"); polarity = cfg.get("polarity", 1)
    try:
        wf_dict, dt_auto = load_file_waveforms(path, fmt, cfg["channels"], max_events)
    except Exception as e:
        print(f"    load failed: {e}"); return []
    dt = dt_auto if cfg.get("dt") in (None, "auto") else cfg["dt"]
    rows = []
    for ch, W in wf_dict.items():
        # apply baseline subtraction + polarity inversion if needed -> positive-going Y
        base = np.median(W[:, :BASE_BINS], axis=1, keepdims=True)
        Y = polarity * (W - base)
        try:
            stacked, n_total, n_sel, _ = select_and_align_Y(
                Y, dt, cfg["spe_band"],
                single_peak=cfg.get("single_peak", True))
            if stacked is None or n_sel < 10:
                continue                                # silent skip for clutter
            fit = fit_stacked(stacked, dt, model=model)
            tag = (os.path.basename(path).replace(".root", "")
                   .replace("flattened_", "").replace("waveform_", ""))
            np.save(out_dir / f"stacked_{cfg['name']}_{tag}_{ch}.npy", stacked)
            row = dict(dataset=cfg["name"], file=os.path.basename(path), channel=ch,
                       dt_ns=dt, polarity=polarity, n_total=n_total, n_selected=n_sel, **fit)
            rows.append(row)
            extra = ""
            if model == "biexp-overshoot" and np.isfinite(row.get("undershoot_depth_frac", np.nan)):
                extra = (f"  undershoot={row['undershoot_depth_frac']*100:4.1f}%peak  "
                         f"τ_os={row['tau_os_ns']:5.1f}ns")
            print(f"      {ch}: N={n_sel:>4}/{n_total:<5}  "
                  f"τ_rise={row['tau_rise_ns']:5.2f}  τ_fall={row['tau_fall_ns']:5.2f}  "
                  f"σ={row['sigma_ns']:4.2f}  χ²ᵣ={row['chi2_red']:.2f}{extra}")
        except Exception as e:
            print(f"      {ch}: ERR {e}")
    return rows


def plot_comparison(df, out_dir):
    from pmtdeconv.plotting.style import use_clean_style, minimal_axes
    use_clean_style()
    ds_list = df.dataset.unique()
    fig, axes = plt.subplots(len(ds_list), 2, figsize=(15, 3.6 * len(ds_list)), squeeze=False)
    for r, ds in enumerate(ds_list):
        sub = df[df.dataset == ds].reset_index(drop=True)
        # dt from the fit rows, NOT the build config: for dt="auto" datasets
        # (object format) the config holds the string "auto" and the true
        # per-file dt only exists on the row — using cfg["dt"] crashed here
        # at the END of a long build (2026-07-01 review).
        dt = float(sub.dt_ns.iloc[0])
        t = np.arange(N_SAMP) * dt
        ax = axes[r][0]
        cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(sub), 1)))
        for (i, row), col in zip(sub.iterrows(), cmap):
            tag = row.file.replace(".root", "").replace("flattened_", "").replace("waveform_", "")
            spath = out_dir / f"stacked_{ds}_{tag}_{row.channel}.npy"
            if spath.exists():
                s = np.load(spath)
                ax.plot(t, s, lw=1.0, alpha=0.7, color=col,
                        label=f"{tag[:18]} {row.channel} (N={row.n_selected})")
                # Overlay the fitted model (dashed) against the stacked data,
                # for BOTH models so the fit is always visible.
                if np.isfinite(row.get("tau_fall_ns", np.nan)):
                    if (row.get("model") == "biexp-overshoot"
                            and np.isfinite(row.get("U_os0", np.nan))):
                        yfit = biexp_overshoot_smeared(
                            t, row.A, row.tau_rise_ns, row.tau_fall_ns,
                            row.t0_ns, row.sigma_ns, row.pedestal,
                            row.U_os0, row.U_os1, row.tau_os_ns,
                            row.t0_os_ns, row.sigma_os_ns, dt)
                    else:
                        yfit = biexp_smeared(
                            t, row.A, row.tau_rise_ns, row.tau_fall_ns,
                            row.t0_ns, row.sigma_ns, row.pedestal, dt)
                    ax.plot(t, yfit, lw=0.9, ls="--", color=col, alpha=0.9)
                    ax.axhline(0, color="gray", lw=0.4, ls=":")
        # Focus on the pulse: rise, peak, and main decay back toward
        # baseline. The long ringing tail is cropped — it's not part of the
        # modeled SPE shape.
        t_peak = PK_CENTER * dt
        ax.set_xlim(max(0.0, t_peak - 80.0), t_peak + 400.0)
        ax.set_xlabel("Time [ns]"); ax.set_ylabel("peak-normalized")
        ax.legend(fontsize=6, ncol=2, loc="upper right"); minimal_axes(ax)

        ax = axes[r][1]
        x = np.arange(len(sub))
        ax.bar(x - 0.25, sub.tau_rise_ns, width=0.25, label="τ_rise")
        ax.bar(x, sub.tau_fall_ns, width=0.25, label="τ_fall")
        ax.bar(x + 0.25, sub.sigma_ns, width=0.25, label="σ")
        labels = [f"{r.file.replace('.root','').replace('flattened_','').replace('waveform_','')[:12]}\n{r.channel}"
                  for _, r in sub.iterrows()]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("ns"); ax.legend(fontsize=7); minimal_axes(ax)
    fig.tight_layout()
    p = out_dir / "templates_comparison.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\nsaved {p}")


def main(datasets=None, *, model="biexp", out_dir=None):
    """Run the per-(dataset, file, channel) template build.

    ``datasets``: list of build-config dicts (see :func:`config_from_preset`);
    ``None`` builds every dataset in the shared registry. ``out_dir``:
    output directory (created if needed; default ``DEFAULT_OUT``). No module
    globals, no ``os.chdir`` — callers pass everything in.
    """
    datasets = datasets if datasets is not None else configs_from_registry()
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    import fnmatch
    for cfg in datasets:
        data_dir = _resolve_data_dir(cfg["dir"])
        files = sorted(glob.glob(os.path.join(str(data_dir), cfg["glob"])))
        excl = cfg.get("exclude_glob")
        if excl:
            files = [f for f in files if not fnmatch.fnmatch(os.path.basename(f), excl)]
        runs = cfg.get("runs")
        if runs:
            toks = [str(r) for r in runs]
            files = [f for f in files
                     if any(t in os.path.basename(f) for t in toks)]
        if cfg.get("max_files"):
            files = files[:cfg["max_files"]]
        print(f"\n=== {cfg['name']}: {len(files)} files  fmt={cfg.get('format','flat')}  "
              f"polarity={cfg.get('polarity',1)}  dt={cfg.get('dt')}  band={cfg['spe_band']} ===")
        for fp in files:
            print(f"  {os.path.basename(fp)}")
            all_rows.extend(process_file(fp, cfg, out_dir, model=model))
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "spe_templates_perfile.csv", index=False)
    print(f"\nsaved {out_dir}/spe_templates_perfile.csv  ({len(df)} templates total)")
    if len(df):
        plot_comparison(df, out_dir)


if __name__ == "__main__":
    main()
