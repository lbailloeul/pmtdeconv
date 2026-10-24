"""``pmtdeconv tune fom`` — sweep the cosine cutoff on REAL data vs the card FoM.

Unlike ``tune cosine`` (synthetic regimes scored against ground truth), this
runs the actual pipeline on real runs at each cutoff and scores with the
dataset's *declared* figure of merit (e.g. photopeak σ/μ). It REPORTS the
FoM-vs-cutoff curve plus **anti-cheat columns** — reconstruction efficiency and
charge drift — so an apparent FoM win that comes from dropping events or
shifting the charge scale is visible, not hidden. ``--auto-select`` is opt-in;
by default it only reports and the expert reads the curve.

Writes ``sweep_fom.csv``, ``sweep_fom_<branch>.png``, ``recipe.json`` (only with
--auto-select), and a manifest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.commands._tune_shared import area_denoiser
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.pipeline.template import (
    add_generic_fallback_flag, configured_template_csv,
)
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tune fom")
    add_generic_fallback_flag(g)
    g.add_argument("--cutoff-range", type=float, nargs=3, default=None,
                   metavar=("LO", "HI", "STEP"),
                   help="Cosine cutoff grid LO HI STEP in MHz. Default: a "
                        "±50%% band around the card's cutoff at a Δf-aware step.")
    g.add_argument("--fom", default=None,
                   help="FoM name (default: the card's `fom:`).")
    g.add_argument("--area-col", default="area",
                   help="Charge column to score (default: area).")
    g.add_argument("--window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Photopeak window [lo hi] for the FoM fit (real spectra "
                        "need this to isolate the peak from Compton/pileup).")
    g.add_argument("--denoiser", choices=("none", "wavelet"), default=None,
                   help="`none` isolates the cutoff effect; `wavelet` keeps the "
                        "default denoise (matches the area path).")
    g.add_argument("--auto-select", action="store_true",
                   help="Opt in to pick a winner per channel (min FoM subject to "
                        "the anti-cheat gates). Default: report only.")
    g.add_argument("--min-efficiency", type=float, default=0.95,
                   help="Anti-cheat gate: a cutoff is eligible only if its "
                        "efficiency is >= this fraction of the best (default "
                        "0.95) — stops 'winning' by dropping events.")
    g.add_argument("--drift-band", type=float, default=0.02,
                   help="Anti-cheat gate: |charge drift vs the reference cutoff| "
                        "must be <= this fraction (default 0.02) — stops "
                        "'winning' by shifting the charge scale.")


def _waveform_length(root_path: Path, tree: str, branch: str) -> Optional[int]:
    try:
        import uproot
        with uproot.open(root_path) as f:
            arr = f[tree][branch].array(entry_stop=1, library="np")
        return int(len(arr[0]))
    except Exception:                              # noqa: BLE001
        return None


def _default_cutoffs(ref_mhz: float, df_mhz: float) -> List[float]:
    lo = max(df_mhz, 0.5 * ref_mhz)
    hi = 1.5 * ref_mhz
    step = max(df_mhz, round((hi - lo) / 12.0, 1))
    return [round(float(x), 1) for x in np.arange(lo, hi + 1e-6, step)]


def _default_out_dir(settings) -> Path:
    name = settings.dataset or "dataset"
    return PROJECT_ROOT / "outputs" / f"tune_fom_{Path(name).stem}"


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=True)
    denoiser = area_denoiser(settings, provenance)
    # A configured-but-missing template is an error (the rule hits applies).
    template_csv = configured_template_csv(settings, provenance)

    out_dir = Path(settings.out_dir or _default_out_dir(settings))
    out_dir.mkdir(parents=True, exist_ok=True)
    if maybe_dry_run_exit(subcommand="tune fom", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    # FoM: --fom overrides the card's declared fom.
    from pmtdeconv.fom import get_fom, FOM_LOWER_IS_BETTER
    fom_name = args.fom or getattr(settings, "fom", None)
    if fom_name is None and getattr(args, "dataset", None):
        from pmtdeconv.presets.datasets import resolve_dataset
        fom_name = getattr(resolve_dataset(args.dataset), "fom", None)
    if fom_name is None:
        print("no FoM: pass --fom or declare `fom:` in the card")
        return 1
    fom_fn = get_fom(fom_name)
    lower_better = FOM_LOWER_IS_BETTER.get(fom_name, True)

    runs = list(getattr(settings, "runs", None) or getattr(args, "runs", []))
    if not runs:
        print("tune fom needs --runs (real data)")
        return 1
    data_dir = Path(settings.data_root)
    tree = settings.tree or "waveforms"

    from pmtdeconv.io.data_paths import resolve_run_file
    from pmtdeconv.io.templates import load_template
    from pmtdeconv.pipeline.process_area import process_run

    run_files = []
    for r in runs:
        rp = resolve_run_file(data_dir, r, kind="flat")
        if rp is None:
            print(f"[skip] run {r}: no flattened file under {data_dir}")
            continue
        run_files.append((r, rp))
    if not run_files:
        print("no run files resolved")
        return 1

    dt_ns = float(settings.dt_ns) if settings.dt_ns != "auto" else 0.5051
    branches_keep = settings.explicit_branches
    probe_branch = (branches_keep[0] if branches_keep else "d0_ch1")
    nsamp = _waveform_length(run_files[0][1], tree, probe_branch) or 1024
    df_mhz = 1000.0 / (nsamp * dt_ns)              # FFT bin spacing Δf = 1/(N·dt)

    if args.cutoff_range:
        lo, hi, step = args.cutoff_range
        if step < df_mhz:
            print(f"[warn] step {step:g} MHz < FFT bin Δf={df_mhz:.2f} MHz "
                  f"(N={nsamp}, dt={dt_ns} ns): sub-bin steps are not resolvable.")
        cutoffs = [round(float(x), 1) for x in np.arange(lo, hi + 1e-6, step)]
    else:
        cutoffs = _default_cutoffs(float(settings.cosine_cutoff_mhz), df_mhz)

    templates = load_template(template_csv)
    wavelet_arg = "none" if denoiser == "none" else (settings.wavelet or "sym8")

    print(f"tune fom: FoM={fom_name}, {len(cutoffs)} cutoffs "
          f"[{cutoffs[0]}..{cutoffs[-1]}] MHz, {len(run_files)} run(s), "
          f"Δf={df_mhz:.2f} MHz, denoiser={denoiser}")

    rows: List[dict] = []
    for cutoff in cutoffs:
        frames = []
        for r, rp in run_files:
            df = process_run(
                root_path=rp, dt_ns=dt_ns,
                sig_start=10, sig_end=1000, n_intervals=1,
                cosine_cutoff_mhz=cutoff,
                wavelet=wavelet_arg, wlevel=settings.wlevel,
                mode=settings.wmode, thresh_k=settings.wthresh_k,
                no_trigger_amp=15.0, t_peak_min_ns=20.0,
                afterpulse_window_ns=0.0, afterpulse_height_ratio=5.0,
                fft_pad=bool(settings.fft_pad), wrap_ratio=0.5, wrap_abs_k=0.0,
                entry_stop=settings.entry_stop, templates=templates,
                branches_keep=branches_keep, polarity=int(settings.polarity),
                tree_name=settings.tree)
            frames.append(df[df["interval"] >= 0])
        alldf = pd.concat(frames, ignore_index=True)
        for branch, grp in alldf.groupby("branch"):
            charges = grp[args.area_col].values
            res = (fom_fn(charges, window=tuple(args.window))
                   if args.window else fom_fn(charges))
            rows.append(dict(branch=str(branch), cosine_cutoff_mhz=cutoff,
                             fom=res["value"], fom_n=res.get("n", len(charges)),
                             median_charge=float(np.nanmedian(charges)),
                             n_pass=int(len(charges))))
        print(f"  cutoff={cutoff:7.1f} MHz done")

    sweep = pd.DataFrame(rows)
    # Anti-cheat columns: efficiency vs best, drift vs the reference cutoff.
    ref_cut = min(cutoffs, key=lambda c: abs(c - float(settings.cosine_cutoff_mhz)))
    out_frames = []
    for branch, sub in sweep.groupby("branch"):
        sub = sub.copy()
        sub["efficiency"] = sub["n_pass"] / sub["n_pass"].max()
        ref_med = float(sub.loc[sub["cosine_cutoff_mhz"] == ref_cut,
                                "median_charge"].mean())
        sub["charge_drift"] = (sub["median_charge"] - ref_med) / ref_med \
            if ref_med else np.nan
        out_frames.append(sub)
    sweep = pd.concat(out_frames, ignore_index=True)
    sweep_csv = out_dir / "sweep_fom.csv"
    sweep.to_csv(sweep_csv, index=False)
    print(f"\nSweep: {sweep_csv}")

    png_paths = _plot_curves(sweep, fom_name, lower_better, out_dir)

    recipe = None
    if args.auto_select:
        recipe = _auto_select(sweep, fom_name, lower_better,
                              args.min_efficiency, args.drift_band)
        (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2))
        print(f"Recipe (auto-select): {out_dir / 'recipe.json'}")
        for br, p in recipe["winners"].items():
            print(f"  {br}: cutoff={p['cosine_cutoff_mhz']} MHz  "
                  f"FoM={p['fom']:.4g}  eff={p['efficiency']:.2f}  "
                  f"drift={p['charge_drift']:+.2%}")
    else:
        print("(report only; pass --auto-select to pick a winner per channel)")

    write_manifest(
        subcommand="tune fom", settings=settings, provenance=provenance,
        inputs=[{"role": "root_data", "path": str(rp)} for _, rp in run_files],
        outputs=[{"role": "sweep_csv", "path": str(sweep_csv)}]
        + [{"role": "sweep_png", "path": str(p)} for p in png_paths],
        stats={"fom": fom_name, "n_cutoffs": len(cutoffs),
               "df_mhz": df_mhz, "nsamp": nsamp, "auto_select": args.auto_select},
        out_path=(Path(settings.manifest_path) if settings.manifest_path
                  else out_dir / "manifest.json"),
        duration_seconds=time.time() - t0, hash_inputs=settings.hash_inputs)
    return 0


def _auto_select(sweep, fom_name, lower_better, min_eff, drift_band) -> dict:
    winners = {}
    for branch, sub in sweep.groupby("branch"):
        elig = sub[(sub["efficiency"] >= min_eff)
                   & (sub["charge_drift"].abs() <= drift_band)
                   & sub["fom"].notna()]
        if len(elig) == 0:
            elig = sub[sub["fom"].notna()]
        if len(elig) == 0:
            continue
        pick = elig.loc[(elig["fom"].idxmin() if lower_better
                         else elig["fom"].idxmax())]
        winners[str(branch)] = {
            "cosine_cutoff_mhz": float(pick["cosine_cutoff_mhz"]),
            "fom": float(pick["fom"]), "efficiency": float(pick["efficiency"]),
            "charge_drift": float(pick["charge_drift"])}
    return {"fom": fom_name, "lower_is_better": lower_better,
            "min_efficiency": min_eff, "drift_band": drift_band,
            "winners": winners}


def _plot_curves(sweep, fom_name, lower_better, out_dir) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for branch, sub in sweep.groupby("branch"):
        sub = sub.sort_values("cosine_cutoff_mhz")
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(sub["cosine_cutoff_mhz"], sub["fom"], "o-")
        axes[0].set_ylabel(fom_name)
        axes[0].set_title(f"{branch} — FoM vs cutoff "
                          f"({'lower' if lower_better else 'higher'} better)")
        axes[1].plot(sub["cosine_cutoff_mhz"], sub["efficiency"], "s-",
                     color="tab:green")
        axes[1].axhline(1.0, ls=":", color="gray"); axes[1].set_ylabel("efficiency")
        axes[2].plot(sub["cosine_cutoff_mhz"], 100 * sub["charge_drift"], "d-",
                     color="tab:red")
        axes[2].axhline(0.0, ls=":", color="gray")
        axes[2].set_ylabel("charge drift [%]")
        axes[2].set_xlabel("cosine cutoff [MHz]")
        fig.tight_layout()
        p = out_dir / f"sweep_fom_{branch}.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        paths.append(p)
    return paths
