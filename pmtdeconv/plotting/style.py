#!/usr/bin/env python3
"""Publication plotting style for the toolkit (2026-05-31).

Two looks, both white-background and print-ready:

  - ``use_root_style()``  — ROOT/TBrowser aesthetic: framed axes, inward minor
    ticks on all four sides, dotted gray grid, serif-ish math text. Pair with
    :func:`root_rainbow` / viridis for 2D hists, :func:`profile_overlay` for the
    black-circle binned profile, and :func:`stats_box` for the χ²/Mean/Sigma box.
  - ``use_clean_style()`` — minimal matplotlib (SER templates, multi-panel event
    displays): light grid, framed legend, no top/right spine clutter.

Helpers are colormap- and Axes-level so individual CLI plotting functions can
opt in without a global rcParams takeover.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "use_root_style", "use_clean_style", "use_minimal_style",
    "root_rainbow", "root_viridis",
    "style_root_axes", "minimal_axes", "profile_overlay", "stats_box",
]


# ── colormaps ────────────────────────────────────────────────────────────────
def root_rainbow():
    """ROOT ``kRainBow``-like palette (violet→blue→cyan→green→yellow→red),
    with white for empty (under) bins — matches the MC-true charge-bias figure."""
    stops = [
        (0.00, "#3f1f9e"), (0.14, "#1f3fd0"), (0.30, "#1f9fe0"),
        (0.45, "#1fd0a0"), (0.58, "#46d23f"), (0.72, "#c8e000"),
        (0.85, "#ff9a00"), (1.00, "#e11900"),
    ]
    cmap = LinearSegmentedColormap.from_list("root_rainbow", stops, N=256)
    cmap.set_under("white")
    return cmap


def root_viridis():
    """viridis with white under-color (empty bins), as in the Run-65213 area map."""
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_under("white")
    return cmap


# ── rcParams contexts ────────────────────────────────────────────────────────
def use_root_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
        "axes.linewidth": 1.3, "axes.grid": True,
        "grid.color": "0.6", "grid.linestyle": ":", "grid.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 7, "ytick.major.size": 7,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "legend.frameon": True, "legend.edgecolor": "black",
        "legend.fancybox": False,
    })


def use_clean_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.labelsize": 16, "axes.titlesize": 16,
        "axes.linewidth": 1.1, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": True, "legend.edgecolor": "0.7",
    })


def use_minimal_style():
    """Minimalist output: white, open axes (no top/right spines), no grid,
    frameless legend. Keep axis labels; everything else is stripped by the
    plotting code (no stats boxes, no multi-line titles, no marker annotations)."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelsize": 15, "axes.titlesize": 14,
        "axes.linewidth": 1.0, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
    })


def style_root_axes(ax):
    """Apply the ROOT frame/grid/tick look to a single Axes (idempotent)."""
    ax.grid(True, which="major", color="0.6", linestyle=":", linewidth=0.8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    for s in ax.spines.values():
        s.set_linewidth(1.3)
    return ax


def minimal_axes(ax):
    """Strip an Axes to essentials: open frame (no top/right spine), no grid,
    outward ticks. Axis labels are preserved by the caller."""
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", direction="out", top=False, right=False)
    return ax


# ── overlays ─────────────────────────────────────────────────────────────────
def profile_overlay(ax, x, y, bins, stat="median", color="black",
                    ms=8, capsize=0, connect=True):
    """Black filled-circle binned profile with x half-bin error bars — the
    overlay on the ROOT 2D charge/timewalk figures."""
    x = np.asarray(x); y = np.asarray(y)
    edges = np.asarray(bins, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * np.diff(edges)
    cx, cy, ex = [], [], []
    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < 3:
            continue
        cy.append(np.median(y[m]) if stat == "median" else np.mean(y[m]))
        cx.append(centers[i]); ex.append(half[i])
    cx, cy, ex = map(np.asarray, (cx, cy, ex))
    if connect:
        ax.plot(cx, cy, "-", color=color, lw=1.0, zorder=5)
    ax.errorbar(cx, cy, xerr=ex, fmt="o", color=color, ms=ms,
                capsize=capsize, elinewidth=1.0, zorder=6)
    return cx, cy


def stats_box(ax, lines, loc="upper right"):
    """ROOT-style stats box: monospace right-aligned rows in a black frame.

    ``lines`` is a list of (label, value) tuples, e.g.
    ``[("χ² / ndf", "28.6 / 23"), ("Mean", "-8.35 ± 0.07")]``.
    """
    txt = "\n".join(f"{lab:<10}{val:>14}" for lab, val in lines)
    xa, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    ax.text(xa, 0.97, txt, transform=ax.transAxes, ha=ha, va="top",
            family="monospace", fontsize=12,
            bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=1.2))
