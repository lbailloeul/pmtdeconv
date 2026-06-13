#!/usr/bin/env python3
"""aggregate_spe_templates.py — combine the per-file stacked SPE templates from
build_spe_templates_perfile.py into one template per (dataset, channel) by
weighted-averaging the per-file stacks (weights = N pulses per file). Then refit.

Output: outputs/spe_templates_perfile/spe_templates_perchannel.csv  +
        outputs/spe_templates_perfile/aggregated_templates.png  +
        per (dataset, channel) stacked_AGG_*.npy
"""
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pmtdeconv.pipeline.template_build_impl import fit_stacked, N_SAMP, PK_CENTER, DATASETS

OUT = Path("outputs/spe_templates_perfile")
DT_BY_DS = {c["name"]: c["dt"] if c["dt"] != "auto" else 1.25 for c in DATASETS}


def tag_for(file):
    return (file.replace(".root", "")
                .replace("flattened_", "")
                .replace("waveform_", ""))


def main():
    df = pd.read_csv(OUT / "spe_templates_perfile.csv")
    rows = []
    keys = df.groupby(["dataset", "channel"])
    for (ds, ch), sub in keys:
        dt = sub.dt_ns.iloc[0]
        weights = sub.n_selected.values
        stacks = []
        for _, row in sub.iterrows():
            sp = OUT / f"stacked_{ds}_{tag_for(row.file)}_{row.channel}.npy"
            if sp.exists():
                stacks.append(np.load(sp))
            else:
                weights = None; break
        if not stacks:
            continue
        S = np.stack(stacks)             # (n_files, N_SAMP)
        w = weights[: len(stacks)]
        agg = np.average(S, axis=0, weights=w)
        N_total = int(w.sum())
        # refit
        fit = fit_stacked(agg, dt)
        np.save(OUT / f"stacked_AGG_{ds}_{ch}.npy", agg)
        rows.append(dict(dataset=ds, channel=ch, dt_ns=dt, n_files=len(stacks),
                         n_pulses=N_total, **fit))
        print(f"{ds:>14} {ch:>10}: {len(stacks):>2} files  N={N_total:>5}  "
              f"τ_rise={fit['tau_rise_ns']:5.2f}  τ_fall={fit['tau_fall_ns']:5.2f}  "
              f"σ={fit['sigma_ns']:4.2f}  χ²ᵣ={fit['chi2_red']:.2f}")
    agg_df = pd.DataFrame(rows).sort_values(["dataset", "channel"])
    agg_df.to_csv(OUT / "spe_templates_perchannel.csv", index=False)
    print(f"\nsaved {OUT}/spe_templates_perchannel.csv  ({len(agg_df)} aggregated templates)")

    # overlay plots per dataset
    ds_list = agg_df.dataset.unique()
    fig, axes = plt.subplots(len(ds_list), 1, figsize=(11, 2.3 * len(ds_list)), squeeze=False)
    for r, ds in enumerate(ds_list):
        sub = agg_df[agg_df.dataset == ds]
        dt = DT_BY_DS.get(ds, 1.25); t = np.arange(N_SAMP) * dt
        ax = axes[r][0]
        cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(sub), 1)))
        for (_, row), c in zip(sub.iterrows(), cmap):
            sp = OUT / f"stacked_AGG_{ds}_{row.channel}.npy"
            if sp.exists():
                s = np.load(sp)
                ax.plot(t, s, lw=1.4, color=c,
                        label=f"{row.channel}  N={row.n_pulses:>4}  τr {row.tau_rise_ns:.1f}  "
                              f"τf {row.tau_fall_ns:.1f}  σ {row.sigma_ns:.1f}")
        ax.set_xlim((PK_CENTER - 20 / dt) * dt, (PK_CENTER + 200 / dt) * dt)
        ax.set_ylabel("peak-norm stacked"); ax.grid(alpha=0.3)
        ax.set_title(f"{ds} — per-channel aggregated SPE templates ({len(sub)} channels)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[-1][0].set_xlabel("Time [ns]")
    fig.tight_layout()
    p = OUT / "aggregated_templates.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
