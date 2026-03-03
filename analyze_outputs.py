#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PARAMS = ["A", "tau_rise", "tau_fall", "t0", "sigma", "pedestal"]
META   = ["run", "chan", "entry", "peak_id", "peak_amp", "fit_start", "fit_end", "chi2", "ndof", "red_chi2"]

def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

def wstd(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan
    mu = np.sum(w[m] * x[m]) / np.sum(w[m])
    var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
    return float(np.sqrt(var))

def robust_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, mad=np.nan, p16=np.nan, p84=np.nan)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) if x.size else np.nan
    p16, p84 = np.percentile(x, [16, 84]) if x.size else (np.nan, np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        median=float(med),
        mad=float(mad),
        p16=float(p16),
        p84=float(p84),
    )

def add_pull_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in PARAMS:
        e = f"{p}_err"
        if e in out.columns:
            # "pull" relative to global median (computed later) is done in main, here we only prep columns safely
            out[e] = pd.to_numeric(out[e], errors="coerce")
        out[p] = pd.to_numeric(out[p], errors="coerce")
    return out

def main():
    ap = argparse.ArgumentParser(description="Summarize biexponential fit outputs.")
    ap.add_argument("--csv", required=True, help="Path to biexp_fit_results.csv (passing-only output).")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside CSV).")
    ap.add_argument("--redchi2-max", type=float, default=None,
                    help="Optional extra cut: keep only rows with red_chi2 <= this.")
    ap.add_argument("--snr-min", type=float, default=None,
                    help="Optional extra cut on peak SNR = peak_amp / baseline_sig.")
    ap.add_argument("--make-plots", action="store_true",
                    help="If set, save a few quick histograms (requires matplotlib).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for c in META + PARAMS + [f"{p}_err" for p in PARAMS] + ["baseline_sig", "baseline_ped"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    if "baseline_sig" in df.columns:
        print(f"Mean baseline_sig: {df['baseline_sig'].mean():.4f}")

    if args.redchi2_max is not None and "red_chi2" in df.columns:
        df = df[df["red_chi2"] <= args.redchi2_max].copy()

    if args.snr_min is not None and {"peak_amp", "baseline_sig"}.issubset(df.columns):
        df["snr"] = df["peak_amp"] / df["baseline_sig"]
        df = df[df["snr"] >= args.snr_min].copy()
    elif {"peak_amp", "baseline_sig"}.issubset(df.columns):
        df["snr"] = df["peak_amp"] / df["baseline_sig"]

    key_cols = [c for c in ["run", "chan", "entry", "peak_id"] if c in df.columns]
    if key_cols:
        df = df.sort_values(["red_chi2"] if "red_chi2" in df.columns else key_cols)
        df = df.drop_duplicates(subset=key_cols, keep="first").copy()

    # Overall summary stats
    overall = []
    overall.append(("n_rows", len(df)))

    if {"run", "chan"}.issubset(df.columns):
        overall.append(("n_runs", df["run"].nunique()))
        overall.append(("n_chans", df["chan"].nunique()))

    if "entry" in df.columns:
        overall.append(("n_entries_unique", df["entry"].nunique()))

    if "red_chi2" in df.columns:
        s = robust_stats(df["red_chi2"].values)
        overall += [(f"red_chi2_{k}", v) for k, v in s.items()]

    if "snr" in df.columns:
        s = robust_stats(df["snr"].values)
        overall += [(f"snr_{k}", v) for k, v in s.items()]

    overall_df = pd.DataFrame(overall, columns=["metric", "value"])
    overall_df.to_csv(outdir / "overall_summary.csv", index=False)

    # Parameter summaries
    param_rows = []
    for p in PARAMS:
        if p not in df.columns:
            continue
        x = df[p].values
        s = robust_stats(x)
        row = {"param": p, **{k: v for k, v in s.items()}}
        e = f"{p}_err"
        if e in df.columns:
            err = df[e].values
            w = np.where(np.isfinite(err) & (err > 0), 1.0 / (err * err), np.nan)
            row["wmean_1/err2"] = wmean(x, w)
            row["wstd_1/err2"]  = wstd(x, w)
        param_rows.append(row)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(outdir / "param_summary.csv", index=False)

    # By-run summary (counts + median red_chi2 + median params)
    if "run" in df.columns:
        agg = {"entry": "nunique"} if "entry" in df.columns else {}
        if "red_chi2" in df.columns:
            agg["red_chi2"] = ["count", "median", "mean"]
        if "snr" in df.columns:
            agg["snr"] = ["median", "mean"]
        for p in PARAMS:
            if p in df.columns:
                agg[p] = ["median", "mean"]

        by_run = df.groupby("run").agg(agg)
        by_run.columns = ["_".join([c for c in col if c]) for col in by_run.columns.to_flat_index()]
        by_run = by_run.reset_index().rename(columns={"entry_nunique": "n_entries_unique"})
        by_run.to_csv(outdir / "by_run_summary.csv", index=False)

    # By-channel summary
    if "chan" in df.columns:
        agg = {"entry": "nunique"} if "entry" in df.columns else {}
        if "red_chi2" in df.columns:
            agg["red_chi2"] = ["count", "median", "mean"]
        if "snr" in df.columns:
            agg["snr"] = ["median", "mean"]
        for p in PARAMS:
            if p in df.columns:
                agg[p] = ["median", "mean"]

        by_chan = df.groupby("chan").agg(agg)
        by_chan.columns = ["_".join([c for c in col if c]) for col in by_chan.columns.to_flat_index()]
        by_chan = by_chan.reset_index().rename(columns={"entry_nunique": "n_entries_unique"})
        by_chan.to_csv(outdir / "by_chan_summary.csv", index=False)

    # Optional plots
    if args.make_plots:
        import matplotlib.pyplot as plt

        def hist_save(series, name, bins=80, logy=False):
            x = pd.to_numeric(series, errors="coerce").dropna().values
            if x.size == 0:
                return
            plt.figure()
            plt.hist(x, bins=bins)
            if logy:
                plt.yscale("log")
            plt.xlabel(name)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{name}.png", dpi=150)
            plt.close()

        if "red_chi2" in df.columns:
            hist_save(df["red_chi2"], "red_chi2", bins=100, logy=True)
        if "snr" in df.columns:
            hist_save(df["snr"], "snr", bins=100, logy=True)
        for p in PARAMS:
            if p in df.columns:
                hist_save(df[p], p, bins=120, logy=False)

    print("\nWrote summary files to:", outdir)
    print(" - overall_summary.csv")
    print(" - param_summary.csv")
    if "run" in df.columns:
        print(" - by_run_summary.csv")
    if "chan" in df.columns:
        print(" - by_chan_summary.csv")
    if args.make_plots:
        print(" - hist_*.png")

if __name__ == "__main__":
    main()