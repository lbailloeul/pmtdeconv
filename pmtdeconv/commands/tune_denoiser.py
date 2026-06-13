"""``pmtdeconv tune denoiser`` — the hits-path STFT-vs-wavelet verdict harness.

The user's long-standing question: does STFT or wavelet denoising give better
*peak-finding* (timing) performance, measured by the reconstructed per-pulse
FWHM at conserved charge? This sweeps the cosine cutoff and compares five
denoiser configs side by side:

  control (none) / conservative-wavelet / aggressive-wavelet / STFT / robust-STFT

For each (regime, config, cutoff) it deconvolves the regime fixture's waveforms,
applies the config's denoiser via the SAME ``pipeline.denoise.denoise`` wrapper
the hits pipeline uses, finds peaks, and measures the reconstructed FWHM with
``core/peaks.fwhm_ns`` — so the FWHM is identical to what ``hits`` would produce
(this measures the Pipeline's reconstructed waveform directly, sidestepping the
wavelet-vs-STFT veto-profile asymmetry so the comparison is apples-to-apples).

Emits ``verdict.json`` (per-config FWHM / charge / peak-count vs cutoff) and an
overlaid ``fwhm_vs_cutoff_<regime>.png``. Additive — touches no goldens.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.commands._tune_shared import ensure_regime_fixture
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.templates import load_template
from pmtdeconv.pipeline.denoise import denoise
from pmtdeconv.pipeline.template import FALLBACK_TEMPLATE
from pmtdeconv.presets import resolve_settings
from pmtdeconv.utils import (
    build_spe_template, cosine_lowpass_filter, deconvolve_hits, fwhm_ns,
)

# The five configs (labels + denoiser knobs). Conservative wavelet = gentle
# soft-threshold; aggressive = hard, higher k; robust-STFT = wider window,
# lower freq cutoff, higher amplitude gate.
CONFIGS = [
    ("control",      "C7", dict(denoiser="none")),
    ("wavelet-cons", "C0", dict(denoiser="wavelet", wavelet="sym8", wlevel=5,
                                wmode="soft", wthresh_k=1.0)),
    ("wavelet-aggr", "C3", dict(denoiser="wavelet", wavelet="sym8", wlevel=5,
                                wmode="hard", wthresh_k=2.0)),
    ("stft",         "C2", dict(denoiser="stft", stft_nperseg_ns=20.0,
                                stft_freq_cutoff_mhz=175.0, stft_amp_threshold=10.0)),
    ("stft-robust",  "C1", dict(denoiser="stft", stft_nperseg_ns=30.0,
                                stft_freq_cutoff_mhz=100.0, stft_amp_threshold=20.0)),
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tune denoiser")
    g.add_argument("--cutoffs", type=float, nargs="+", default=None,
                   help="Cosine cutoffs (MHz) to sweep (default 50 100 200 300 450).")
    g.add_argument("--regimes", type=str, nargs="+", default=None,
                   help="Regime fixtures to test (default level3).")
    g.add_argument("--sample-events", type=int, default=150,
                   help="Events per regime to measure (default 150).")
    g.add_argument("--n-events", type=int, default=None,
                   help="Override regime fixture event count (rebuilds it).")


def _measure(waves, templates, dt_ns, cutoff, cfg):
    cache = {}
    fwhms: List[float] = []
    charges: List[float] = []
    npk = 0
    pre_n = max(int(100 / dt_ns), 1)
    win = max(int(20 / dt_ns), 1)
    dist = max(int(5.0 / dt_ns), 1)
    for wave, branch, N in waves:
        key = (N, branch)
        if key not in cache:
            tmpl = templates.get(branch, templates.get("_default", FALLBACK_TEMPLATE))
            kern = build_spe_template(N, dt_ns, tmpl)
            filt = cosine_lowpass_filter(N, dt_ns, cutoff)
            cache[key] = (kern, filt)
        kern, filt = cache[key]
        ped = float(np.median(wave[:pre_n]))
        u_fft = deconvolve_hits(wave - ped, kern, filt)
        u = denoise(
            u_fft, denoiser=cfg["denoiser"], dt_ns=dt_ns,
            wavelet=cfg.get("wavelet", "sym8"), wlevel=cfg.get("wlevel", 5),
            wmode=cfg.get("wmode", "hard"), wthresh_k=cfg.get("wthresh_k", 1.0),
            stft_nperseg_ns=cfg.get("stft_nperseg_ns", 20.0),
            stft_freq_cutoff_hz=cfg.get("stft_freq_cutoff_mhz", 175.0) * 1e6,
            stft_amp_threshold=cfg.get("stft_amp_threshold", 10.0),
        )
        from scipy.signal import find_peaks
        pre = u[:pre_n]
        noise = max(1.4826 * float(np.median(np.abs(pre - np.median(pre)))), 1e-12)
        peaks, _ = find_peaks(u, height=5.0 * noise, distance=dist)
        for pk in peaks:
            w = fwhm_ns(u, int(pk), dt_ns)
            if np.isfinite(w) and w > 0:
                fwhms.append(float(w))
            npk += 1
        if len(peaks):
            tallest = int(peaks[int(np.argmax(u[peaks]))])
            lo = max(tallest - win, 0)
            hi = min(tallest + win, N)
            charges.append(float(np.sum(u[lo:hi])))
    return (float(np.median(fwhms)) if fwhms else float("nan"),
            float(np.median(charges)) if charges else float("nan"),
            npk)


def _plot(regime, rows, out_dir, ref_cutoff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pmtdeconv.plotting.style import use_minimal_style, minimal_axes
    use_minimal_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, color, _cfg in CONFIGS:
        pts = sorted([(r["cutoff_mhz"], r["median_fwhm_ns"]) for r in rows
                      if r["config"] == label and np.isfinite(r["median_fwhm_ns"])])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=color, ms=4, label=label)
    if ref_cutoff is not None:
        ax.axvline(float(ref_cutoff), color="gray", ls=":", lw=1,
                   label=f"dataset cutoff ({ref_cutoff:g})")
    ax.set_xlabel("cosine cutoff [MHz]")
    ax.set_ylabel("reconstructed FWHM [ns] (median)")
    ax.set_title(f"STFT-vs-wavelet verdict — regime {regime}")
    ax.legend(fontsize=9)
    minimal_axes(ax)
    fig.tight_layout()
    p = out_dir / f"fwhm_vs_cutoff_{regime}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or (PROJECT_ROOT / "outputs"
                                   / f"tune_denoiser_{settings.dataset or 'na22'}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="tune denoiser", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    dt_ns = float(settings.dt_ns) if settings.dt_ns != "auto" else 0.5051
    cutoffs = args.cutoffs or [50.0, 100.0, 200.0, 300.0, 450.0]
    regimes = args.regimes or ["level3"]
    ref_cutoff = (settings.cosine_cutoff_mhz
                  if settings.cosine_cutoff_mhz is not None else None)

    templates = load_template(settings.template_csv, strict=False, verbose=False)
    templates.setdefault("_default", FALLBACK_TEMPLATE)

    from pmtdeconv.pipeline.event_source import iter_flat_run
    polarity = int(settings.polarity)

    verdict = {}
    plots = []
    print(f"tune denoiser: {len(CONFIGS)} configs × {len(cutoffs)} cutoffs "
          f"× {len(regimes)} regimes (dt_ns={dt_ns})")
    for regime in regimes:
        fixture = ensure_regime_fixture(regime, dt_ns=dt_ns, n_events=args.n_events)
        waves = []
        for rec in iter_flat_run(str(fixture), 0, dt_ns=dt_ns, polarity=polarity,
                                 entry_stop=args.sample_events):
            waves.append((rec.wave.astype(np.float64), rec.branch, rec.N))
        print(f"  regime {regime}: {len(waves)} waveforms")
        rows = []
        for label, _color, cfg in CONFIGS:
            for cutoff in cutoffs:
                fwhm, charge, npk = _measure(waves, templates, dt_ns, cutoff, cfg)
                rows.append({"config": label, "cutoff_mhz": float(cutoff),
                             "median_fwhm_ns": fwhm, "median_charge": charge,
                             "n_peaks": npk})
            best = [r for r in rows if r["config"] == label]
        verdict[regime] = rows
        plots.append(str(_plot(regime, rows, out_dir, ref_cutoff)))

    # Verdict summary: at the dataset's reference cutoff, rank configs by FWHM.
    summary = {}
    for regime, rows in verdict.items():
        at_ref = [r for r in rows
                  if ref_cutoff is not None
                  and abs(r["cutoff_mhz"] - float(ref_cutoff)) < 1e-6
                  and np.isfinite(r["median_fwhm_ns"])]
        if not at_ref:
            at_ref = [r for r in rows if np.isfinite(r["median_fwhm_ns"])]
        ranked = sorted(at_ref, key=lambda r: r["median_fwhm_ns"])
        summary[regime] = [
            {"config": r["config"], "cutoff_mhz": r["cutoff_mhz"],
             "median_fwhm_ns": round(r["median_fwhm_ns"], 4)} for r in ranked]
        if ranked:
            print(f"  verdict [{regime}] @ cutoff "
                  f"{ranked[0]['cutoff_mhz']:g}: best = {ranked[0]['config']} "
                  f"(FWHM {ranked[0]['median_fwhm_ns']:.3f} ns)")

    verdict_path = out_dir / "verdict.json"
    verdict_path.write_text(json.dumps(
        {"dataset": settings.dataset, "dt_ns": dt_ns, "ref_cutoff_mhz": ref_cutoff,
         "configs": [c[0] for c in CONFIGS], "sweep": verdict,
         "summary": summary}, indent=2))
    print(f"\nWrote {verdict_path}")

    manifest_path = (Path(settings.manifest_path) if settings.manifest_path
                     else out_dir / "manifest.json")
    write_manifest(
        subcommand="tune denoiser", settings=settings, provenance=provenance,
        inputs=[], outputs=([{"role": "verdict_json", "path": str(verdict_path)}]
                            + [{"role": "fwhm_curve_png", "path": p} for p in plots]),
        stats={"summary": summary, "cutoffs": cutoffs, "regimes": regimes},
        out_path=manifest_path, duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
