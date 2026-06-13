#!/usr/bin/env python3
"""Render the Run66518 raw-fit timewalk reproduction from cached observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import curve_fit

from pmtdeconv.plotting.style import minimal_axes, root_viridis, use_minimal_style


CH1_SLICES_NVS = (
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
)


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_histogram(values, time_range=(-60.0, 60.0), bins=120):
    finite = values[
        np.isfinite(values)
        & (values >= time_range[0])
        & (values <= time_range[1])
    ]
    counts, edges = np.histogram(finite, bins=bins, range=time_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_mean = float(np.mean(finite)) if finite.size else np.nan
    hist_rms = float(np.std(finite)) if finite.size else np.nan

    fit_ok = False
    fit = np.full(3, np.nan)
    errors = np.full(3, np.nan)
    if counts.sum() >= 20:
        weighted_mean = np.average(centers, weights=counts)
        weighted_sigma = np.sqrt(
            np.average((centers - weighted_mean) ** 2, weights=counts)
        )
        try:
            fit, covariance = curve_fit(
                gaussian,
                centers,
                counts.astype(float),
                p0=(counts.max(), weighted_mean, max(weighted_sigma, 1.0)),
                maxfev=2000,
            )
            fit[2] = abs(fit[2])
            errors = np.sqrt(np.diag(covariance))
            fit_ok = True
        except (RuntimeError, ValueError):
            pass

    return {
        "values": finite,
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "hist_mean": hist_mean,
        "hist_rms": hist_rms,
        "fit": fit,
        "errors": errors,
        "fit_ok": fit_ok,
    }


def draw_histogram(
    ax,
    result,
    *,
    run,
    ch1_lo,
    ch1_hi,
    ch2_lo,
    ch2_hi,
    frame_index,
    style,
):
    counts = result["counts"]
    centers = result["centers"]
    amplitude, mean, sigma = result["fit"]
    amp_err, mean_err, sigma_err = result["errors"]

    line_color = "#1f77b4" if style == "minimal" else "blue"
    fit_color = "#d62728" if style == "minimal" else "blue"
    ax.step(
        centers, counts, where="mid", color=line_color, lw=1.6,
        label="data" if style == "minimal" else None,
    )
    if result["fit_ok"]:
        x_fit = np.linspace(result["edges"][0], result["edges"][-1], 500)
        label = (
            fr"$\mu={mean:.2f}$ ns, $\sigma={sigma:.2f}$ ns"
            if style == "minimal" else None
        )
        ax.plot(
            x_fit, gaussian(x_fit, amplitude, mean, sigma),
            color=fit_color, lw=1.8 if style == "minimal" else 1.0,
            label=label,
        )

    ax.set_xlim(result["edges"][0], result["edges"][-1])
    if style == "minimal":
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("CH1 fitted crossing - CH2 fitted crossing [ns]")
        ax.set_ylabel("entries")
        ax.set_title(
            f"Run {run}: CH1 {ch1_lo:g}-{ch1_hi:g} nVs, "
            f"CH2 {ch2_lo:g}-{ch2_hi:g} nVs"
        )
        minimal_axes(ax)
        ax.legend(loc="upper right")
        return

    ax.set_yscale("log")
    ax.set_ylim(0.8, max(500.0, counts.max() * 1.35))
    ax.set_xlabel("Time difference CH1-CH2 using fitted raw crossing (ns)")
    ax.set_ylabel("Number of entries")
    ax.set_title(
        f"fitt1-2 in Run{run} Ch1 area between {ch1_lo:g} and {ch1_hi:g} nVs"
    )
    ax.grid(True, which="both", ls=":", color="0.45", alpha=0.9)

    lines = [
        f"H1F_{frame_index}",
        f"Entries   {len(result['values']):d}",
        f"Mean      {result['hist_mean']:.2f}",
        f"RMS       {result['hist_rms']:.2f}",
    ]
    if result["fit_ok"]:
        lines.extend([
            f"Constant  {amplitude:.0f} +/- {amp_err:.1f}",
            f"Mean      {mean:.2f} +/- {mean_err:.2f}",
            f"Sigma     {sigma:.2f} +/- {sigma_err:.2f}",
        ])
    ax.text(
        0.79, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "black", "pad": 2.0},
    )


def render_frame(
    result,
    *,
    run,
    ch1_lo,
    ch1_hi,
    ch2_lo,
    ch2_hi,
    frame_index,
    scatter,
    with_area_path,
    hist_only_path,
    style,
):
    if style == "minimal":
        use_minimal_style()
    fig, (ax_hist, ax_area) = plt.subplots(
        1, 2,
        figsize=(14.0, 5.0) if style == "minimal" else (14.8, 5.1),
        gridspec_kw={
            "width_ratios": (1.4, 1.0) if style == "minimal" else (1.55, 0.95)
        },
    )
    draw_histogram(
        ax_hist, result, run=run, ch1_lo=ch1_lo, ch1_hi=ch1_hi,
        ch2_lo=ch2_lo, ch2_hi=ch2_hi, frame_index=frame_index, style=style,
    )

    counts_max = max(float(scatter["counts"].max()), 1.0)
    image = ax_area.imshow(
        scatter["counts"].T,
        origin="lower",
        extent=(0, 100, 0, 120),
        aspect="auto",
        cmap=root_viridis(),
        norm=(
            Normalize(vmin=0.5, vmax=counts_max)
            if style == "minimal"
            else LogNorm(vmin=1, vmax=max(1000, counts_max))
        ),
        interpolation="nearest",
    )
    ax_area.add_patch(Rectangle(
        (ch1_lo, ch2_lo), ch1_hi - ch1_lo, ch2_hi - ch2_lo,
        fill=False, edgecolor="red", lw=2.0,
    ))
    ax_area.set_xlim(0, 100)
    ax_area.set_ylim(0, 120)
    ax_area.set_xlabel("CH1 pulse area (nVs)")
    ax_area.set_ylabel("CH2 pulse area (nVs)")
    if style == "minimal":
        minimal_axes(ax_area)
    else:
        ax_area.set_title("Area selection")
        ax_area.grid(True, ls=":", color="0.35", alpha=0.8)
    colorbar = fig.colorbar(image, ax=ax_area, pad=0.025)
    colorbar.set_label("entries" if style == "minimal" else "events / bin")
    fig.tight_layout()
    fig.savefig(with_area_path, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    draw_histogram(
        ax, result, run=run, ch1_lo=ch1_lo, ch1_hi=ch1_hi,
        ch2_lo=ch2_lo, ch2_hi=ch2_hi, frame_index=frame_index, style=style,
    )
    fig.tight_layout()
    fig.savefig(hist_only_path, dpi=110)
    plt.close(fig)


def stitch_gif(frame_paths, output_path, frame_ms):
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()


def render_summary(summary, output_path, compact_output_path):
    centers = 0.5 * (summary["ch1_lo_nvs"] + summary["ch1_hi_nvs"])
    ok = summary["fit_ok"].astype(bool)

    def draw(path, figsize, dpi):
        fig, (ax_mean, ax_sigma) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
        )
        ax_mean.errorbar(
            centers[ok], summary.loc[ok, "fit_mean_ns"],
            yerr=summary.loc[ok, "fit_mean_err_ns"],
            fmt="o-", color="blue", lw=1.5, ms=5.5,
        )
        ax_mean.set_ylabel("Gaussian mean [ns]")
        ax_mean.grid(True, ls=":", color="0.45")

        ax_sigma.errorbar(
            centers[ok], summary.loc[ok, "fit_sigma_ns"],
            yerr=summary.loc[ok, "fit_sigma_err_ns"],
            fmt="s-", color="red", lw=1.5, ms=5.0,
        )
        ax_sigma.set_ylabel("Gaussian sigma [ns]")
        ax_sigma.set_xlabel("CH1 area slice center [nVs]")
        ax_sigma.grid(True, ls=":", color="0.45")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    draw(output_path, (8.0, 6.2), 150)
    draw(compact_output_path, (8.4, 5.4), 150)


def render_timewalk_curve(summary, output_path, run):
    """Render the GIF's fitted mean and ±sigma band versus CH1 area."""
    centers = 0.5 * (summary["ch1_lo_nvs"] + summary["ch1_hi_nvs"])
    ok = summary["fit_ok"].astype(bool)
    x = centers[ok].to_numpy()
    mean = summary.loc[ok, "fit_mean_ns"].to_numpy()
    sigma = summary.loc[ok, "fit_sigma_ns"].to_numpy()
    mean_err = summary.loc[ok, "fit_mean_err_ns"].to_numpy()

    use_minimal_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.fill_between(
        x, mean - sigma, mean + sigma,
        color="#1f77b4", alpha=0.2, label=r"$\pm\sigma$",
    )
    ax.errorbar(
        x, mean, yerr=mean_err,
        fmt="o", color="#1f77b4", capsize=2, label=r"$\mu_{\Delta t}$",
    )
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("CH1 area [nVs]")
    ax.set_ylabel(r"$\Delta t$ (CH1 - CH2) [ns]")
    ax.set_title(f"Run {run} — timewalk")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _ch1_slices(args):
    if args.ch1_scan is None:
        return CH1_SLICES_NVS
    center_lo, center_hi = args.ch1_scan
    if center_hi < center_lo:
        raise SystemExit("--ch1-scan upper bound must be >= lower bound.")
    if args.ch1_step <= 0 or args.ch1_width <= 0:
        raise SystemExit("--ch1-step and --ch1-width must be positive.")
    centers = np.arange(center_lo, center_hi + 1e-9, args.ch1_step)
    half_width = 0.5 * args.ch1_width
    return tuple((center - half_width, center + half_width) for center in centers)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run", default="66518")
    parser.add_argument("--ch2-lock", type=float, nargs=2, default=(10.0, 40.0),
                        metavar=("LO", "HI"))
    parser.add_argument("--frame-ms", type=int, default=300)
    parser.add_argument(
        "--ch1-scan", type=float, nargs=2, default=None,
        metavar=("CENTER_LO", "CENTER_HI"),
        help="Animate CH1 window centers over this range in nVs.",
    )
    parser.add_argument(
        "--ch1-step", type=float, default=2.0,
        help="Spacing between CH1 scan-window centers in nVs.",
    )
    parser.add_argument(
        "--ch1-width", type=float, default=4.0,
        help="Width of each CH1 scan window in nVs.",
    )
    parser.add_argument(
        "--style", choices=("logbook", "minimal"), default="logbook",
        help="Frame presentation. `minimal` matches the current deconv GIF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ch2_lo, ch2_hi = args.ch2_lock
    range_tag = f"{ch2_lo:g}_{ch2_hi:g}"
    style_tag = "" if args.style == "logbook" else f"_{args.style}"
    ch1_slices = _ch1_slices(args)
    scan_tag = ""
    if args.ch1_scan is not None:
        scan_tag = f"_ch1scan_{args.ch1_scan[0]:g}_{args.ch1_scan[1]:g}"

    cache = np.load(args.cache)
    ch1_area = cache["a_scan"] * 1e-3
    ch2_area = cache["a_ref"] * 1e-3
    t_ch1 = cache["t_fit_scan"]
    t_ch2 = cache["t_fit_ref"]
    dt = t_ch1 - t_ch2

    valid_timing = (
        np.isfinite(dt)
        & (t_ch1 >= 0.0) & (t_ch1 <= 1016.0)
        & (t_ch2 >= 0.0) & (t_ch2 <= 1016.0)
    )
    ch2_locked = valid_timing & (ch2_area >= ch2_lo) & (ch2_area <= ch2_hi)

    scatter_counts, _, _ = np.histogram2d(
        ch1_area[np.isfinite(ch1_area) & np.isfinite(ch2_area)],
        ch2_area[np.isfinite(ch1_area) & np.isfinite(ch2_area)],
        bins=(100, 120), range=((0, 100), (0, 120)),
    )
    scatter = {"counts": scatter_counts}

    frames_area = out_dir / (
        f"frames_ch1_slices_ch2_{range_tag}{scan_tag}{style_tag}_with_area"
    )
    frames_hist = out_dir / (
        f"frames_ch1_slices_ch2_{range_tag}{scan_tag}{style_tag}_hist_only"
    )
    frames_area.mkdir(exist_ok=True)
    frames_hist.mkdir(exist_ok=True)

    rows = []
    for index, (ch1_lo, ch1_hi) in enumerate(ch1_slices):
        selected = ch2_locked & (ch1_area >= ch1_lo) & (ch1_area <= ch1_hi)
        result = fit_histogram(dt[selected])
        fit = result["fit"]
        errors = result["errors"]
        rows.append({
            "ch1_lo_nvs": ch1_lo,
            "ch1_hi_nvs": ch1_hi,
            "ch2_lock_lo_nvs": ch2_lo,
            "ch2_lock_hi_nvs": ch2_hi,
            "N_total_selected": int(selected.sum()),
            "N_in_time_window": int(len(result["values"])),
            "hist_mean_ns": result["hist_mean"],
            "hist_rms_ns": result["hist_rms"],
            "fit_mean_ns": fit[1],
            "fit_sigma_ns": fit[2],
            "fit_mean_err_ns": errors[1],
            "fit_sigma_err_ns": errors[2],
            "fit_ok": result["fit_ok"],
        })
        render_frame(
            result,
            run=args.run,
            ch1_lo=ch1_lo,
            ch1_hi=ch1_hi,
            ch2_lo=ch2_lo,
            ch2_hi=ch2_hi,
            frame_index=index,
            scatter=scatter,
            with_area_path=frames_area / f"frame_{index:03d}.png",
            hist_only_path=frames_hist / f"frame_{index:03d}.png",
            style=args.style,
        )

    summary = pd.DataFrame(rows)
    csv_path = out_dir / (
        f"timewalk_ch1_slices_ch2_locked_{range_tag}nvs_rawfit"
        f"{scan_tag}{style_tag}_run{args.run}.csv"
    )
    summary.to_csv(csv_path, index=False)

    summary_path = out_dir / (
        f"timewalk_mean_sigma_ch1_slices_ch2_locked_{range_tag}nvs_"
        f"rawfit{scan_tag}{style_tag}_run{args.run}.png"
    )
    compact_path = out_dir / (
        f"time_difference_vs_ch1_area_ch2_locked_{range_tag}nvs_"
        f"rawfit{scan_tag}{style_tag}_run{args.run}.png"
    )
    render_summary(summary, summary_path, compact_path)
    curve_path = out_dir / (
        f"timewalk_curve_ch1scan_{args.ch1_scan[0]:g}_{args.ch1_scan[1]:g}nvs_"
        f"ch2_locked_{range_tag}nvs_rawfit_run{args.run}.png"
        if args.ch1_scan is not None
        else f"timewalk_curve_ch1_slices_ch2_locked_{range_tag}nvs_"
             f"rawfit_run{args.run}.png"
    )
    render_timewalk_curve(summary, curve_path, args.run)

    gif_hist = out_dir / (
        f"TimeResolutionSlice_rawfit_CH1-CH2_Run{args.run}_"
        f"Animate_repro{scan_tag}{style_tag}_ch2_{range_tag}nvs.gif"
    )
    gif_area = out_dir / (
        f"TimeResolutionSlice_rawfit_CH1-CH2_Run{args.run}_"
        f"Animate_repro{scan_tag}{style_tag}_with_area_ch2_{range_tag}nvs.gif"
    )
    stitch_gif(sorted(frames_hist.glob("frame_*.png")), gif_hist, args.frame_ms)
    stitch_gif(sorted(frames_area.glob("frame_*.png")), gif_area, args.frame_ms)

    outputs = [
        compact_path, summary_path, curve_path, csv_path, gif_hist, gif_area,
    ]
    manifest = {
        "source_cache": str(args.cache.resolve()),
        "run": args.run,
        "observable": (
            "t_fit_ch1 - t_fit_ch2 from 20-80% raw waveform "
            "leading-edge linear crossing"
        ),
        "style": args.style,
        "cuts": {
            "ch2_area_nvs": [ch2_lo, ch2_hi],
            "t_fit_bounds_ns": [0.0, 1016.0],
            "dt_histogram_range_ns": [-60.0, 60.0],
        },
        "slices_ch1_nvs": [list(item) for item in ch1_slices],
        "ch1_scan_centers_nvs": (
            list(args.ch1_scan) if args.ch1_scan is not None else None
        ),
        "ch1_scan_step_nvs": args.ch1_step,
        "ch1_scan_width_nvs": args.ch1_width,
        "outputs": [str(path.resolve()) for path in outputs],
    }
    manifest_path = out_dir / (
        f"rawfit_repro_manifest{scan_tag}{style_tag}_ch2_{range_tag}nvs.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved {summary_path}")
    print(f"Saved {compact_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {gif_hist}")
    print(f"Saved {gif_area}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
