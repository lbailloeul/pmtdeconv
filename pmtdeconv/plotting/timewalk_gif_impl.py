#!/usr/bin/env python3
"""Frame rendering and GIF stitching for ``pmtdeconv timewalk gif``.

``render_frame`` draws one animation frame — the Δt histogram with its Gaussian
fit, optionally beside a ch1-vs-ch2 scatter — and ``stitch_gif`` assembles the
frames with PIL. The CLI imports exactly those two.

This module used to be a standalone script that read a DRSprocessing ROOT tree
directly and computed raw rising-edge timing (``t_half`` / ``t_fit``). That
reader, its summary figure and the script entry point were removed on
2026-09-22: the CLI feeds these renderers from the manifest-tracked CSVs that
``raw-area`` and ``hits`` produce, and ``timewalk run`` already emits the
μ/σ-vs-area curve. Recover them from git history if the raw-edge observables
are ever wanted again.

Mirrors the spirit of the MQAna ``timewalk.csh`` from the UCSB E41473 logbook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import curve_fit

MIN_VMAX_MV = 30.0
NSAMPLES_DEFAULT = 1024


def _root_kbird_cmap():
    """ROOT ``kBird`` palette as a matplotlib colormap, with empty (under) = white.

    Colour stops are the official ``kBird`` values from ROOT's TColor.
    Calling code should pair this with ``vmin`` slightly above 0 so zero-count
    bins fall under the threshold and render as white.
    """
    stops = [
        (0.0000, (0.2082, 0.1664, 0.5293)),
        (0.1250, (0.0592, 0.3599, 0.8684)),
        (0.2500, (0.0780, 0.5041, 0.8385)),
        (0.3750, (0.0232, 0.6419, 0.7914)),
        (0.5000, (0.1802, 0.7178, 0.6425)),
        (0.6250, (0.5301, 0.7491, 0.4661)),
        (0.7500, (0.8186, 0.7328, 0.3499)),
        (0.8750, (0.9956, 0.7862, 0.1968)),
        (1.0000, (0.9764, 0.9831, 0.0538)),
    ]
    cmap = LinearSegmentedColormap.from_list("root_kBird", stops, N=256)
    cmap.set_under("white")
    return cmap


_KBIRD = _root_kbird_cmap()


# -----------------------------------------------------------------------------
# Per-event timing observables
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Tree iteration
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Frame rendering & GIF assembly
# -----------------------------------------------------------------------------
def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gauss(centers: np.ndarray, counts: np.ndarray):
    """Robust Gaussian fit. Returns (mu, sigma, A, fit_ok)."""
    if counts.sum() < 20:
        return np.nan, np.nan, np.nan, False
    mu0 = (centers * counts).sum() / counts.sum()
    var0 = ((centers - mu0) ** 2 * counts).sum() / counts.sum()
    sig0 = max(np.sqrt(var0), (centers[1] - centers[0]))
    A0 = counts.max()
    try:
        popt, _ = curve_fit(
            _gauss,
            centers,
            counts,
            p0=[A0, mu0, sig0],
            maxfev=2000,
        )
        A, mu, sig = popt
        if sig < 0:
            sig = abs(sig)
        return mu, sig, A, True
    except Exception:
        return mu0, sig0, A0, False


def render_frame(
    dt: np.ndarray,
    *,
    method_label: str,
    run_id: str,
    a_ref_lo: float,
    a_ref_hi: float,
    a_scan_lo: float,
    a_scan_hi: float,
    xrange: tuple[float, float],
    nbins: int,
    outpath: Path,
    scatter_data: dict | None = None,
    area_unit_scale: float = 1.0,
    area_unit_label: str = "pVs",
    dpi: int = 110,
    ref_label: str = "CH1",
    scan_label: str = "CH2",
    log_y: bool = False,
) -> tuple[float, float, int]:
    """Render one frame; return (mu, sigma, n_in_window).

    If ``scatter_data`` is provided (keys ``H``, ``xedges``, ``yedges`` from
    :func:`numpy.histogram2d` on the CH1/CH2 area plane), a second panel is
    drawn on the right with the COLZ-style 2D heatmap and a red rectangle
    showing the active (``a_ref``, ``a_scan``) selection box. Axis values are
    multiplied by ``area_unit_scale`` for display.
    """
    if isinstance(nbins, np.ndarray):
        # caller passed pre-computed bin edges
        edges = nbins
    else:
        edges = np.linspace(xrange[0], xrange[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_window = (dt > xrange[0]) & (dt < xrange[1]) & np.isfinite(dt)
    counts, _ = np.histogram(dt[in_window], bins=edges)
    mu, sig, A, ok = _fit_gauss(centers, counts.astype(float))

    from pmtdeconv.plotting.style import use_minimal_style, minimal_axes, root_viridis
    use_minimal_style()
    if scatter_data is not None:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax2 = None

    ax.step(centers, counts, where="mid", color="#1f77b4", lw=1.5, label="data")
    if ok:
        xs = np.linspace(xrange[0], xrange[1], 400)
        ax.plot(
            xs,
            _gauss(xs, A, mu, sig),
            color="#d62728",
            lw=1.8,
            label=fr"$\mu={mu:.2f}$ ns, $\sigma={sig:.2f}$ ns",
        )
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    else:
        ax.set_yscale("linear")
    ax.set_xlim(*xrange)
    ax.set_xlabel(f"{scan_label} time - {ref_label} time [ns]")
    ax.set_ylabel("entries")
    minimal_axes(ax)
    ax.legend(loc="upper right")

    if ax2 is not None:
        from matplotlib.colors import LogNorm
        H = scatter_data["H"]
        xe = scatter_data["xedges"] * area_unit_scale
        ye = scatter_data["yedges"] * area_unit_scale
        vmax = max(H.max(), 1)
        im = ax2.imshow(
            H.T,
            origin="lower",
            extent=(xe[0], xe[-1], ye[0], ye[-1]),
            aspect="auto",
            cmap=root_viridis(),
            # log z (2026-06-10): zero-count bins are masked -> white
            norm=LogNorm(vmin=0.5, vmax=vmax),
            interpolation="nearest",
        )
        ax2.add_patch(
            Rectangle(
                (a_ref_lo * area_unit_scale, a_scan_lo * area_unit_scale),
                (a_ref_hi - a_ref_lo) * area_unit_scale,
                (a_scan_hi - a_scan_lo) * area_unit_scale,
                fill=False,
                edgecolor="red",
                lw=2.0,
            )
        )
        ax2.set_xlabel(f"{ref_label} pulse area ({area_unit_label})")
        ax2.set_ylabel(f"{scan_label} pulse area ({area_unit_label})")
        minimal_axes(ax2)
        cbar = fig.colorbar(im, ax=ax2, pad=0.02)
        cbar.set_label("entries")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return (mu if ok else np.nan, sig if ok else np.nan, int(in_window.sum()))


def stitch_gif(frame_paths: Iterable[Path], outpath: Path, duration_ms: int = 300):
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        raise RuntimeError("no frames to stitch")
    frames[0].save(
        outpath,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
