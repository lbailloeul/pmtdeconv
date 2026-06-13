"""``pmtdeconv tune stft`` — sweep STFT denoiser config per regime.

Mirrors :mod:`pmtdeconv.commands.tune_wavelet`'s shape; calls
``pipeline.process_hits_stft.process_run`` and varies STFT-specific
knobs (``stft_nperseg_ns``, ``stft_freq_cutoff_mhz``,
``stft_amp_threshold``). Cosine cutoff is held fixed (Q1:
``--cosine-cutoff-mhz`` defaults to the dataset preset's value, which
for na22 is 450 — note this differs from the STFT pipeline's legacy
hardcoded 100 MHz, which is now also overridable).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.deconv_pulse import DECONV_PULSE_CSV
from pmtdeconv.io.truth import find_truth_for
from pmtdeconv.pipeline.scoring import ScoringWeights, score_run
from pmtdeconv.presets import resolve_settings

from pmtdeconv.commands._tune_shared import (
    apply_drift_bias, exact_reference,
    ensure_regime_fixture, regime_fixture,
    save_sweep_curves, write_tuned_presets,
)


DEFAULT_NPERSEG_NS = (10.0, 20.0, 40.0)
DEFAULT_FREQ_CUTOFF_MHZ = (100.0, 175.0, 300.0)
DEFAULT_AMP_THRESHOLD = (3.0, 10.0, 30.0)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tune stft")
    g.add_argument("--regimes", nargs="+", default=None,
                   help="Which regimes to sweep (default: all 5).")
    g.add_argument("--nperseg-ns", type=float, nargs="+", default=None,
                   help=f"STFT window in ns (default: {DEFAULT_NPERSEG_NS}).")
    g.add_argument("--freq-cutoff-mhz", type=float, nargs="+", default=None,
                   help=f"STFT frequency cutoff in MHz "
                        f"(default: {DEFAULT_FREQ_CUTOFF_MHZ}).")
    g.add_argument("--amp-thresholds", type=float, nargs="+", default=None,
                   help=f"STFT amplitude threshold "
                        f"(default: {DEFAULT_AMP_THRESHOLD}).")
    g.add_argument("--cosine-cutoff-mhz", type=float, default=None,
                   help="Q1 isolation knob: cosine held fixed during STFT "
                        "sweep. Default = STFT pipeline's legacy 100 MHz.")
    g.add_argument("--template", type=str, default=None)
    g.add_argument("--write-presets", action="store_true")
    g.add_argument("--n-events", type=int, default=None)


def _default_out_dir(settings) -> Path:
    parts = [f"tune_stft_{settings.dataset or 'na22'}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def _build_legacy_args(settings, cell_out_dir: Path, fixture: Path,
                       dt_ns: float, cosine_cutoff_mhz: float,
                       nperseg_ns: float, fcut_mhz: float, amp_thr: float
                       ) -> SimpleNamespace:
    return SimpleNamespace(
        runs=[0],
        entry_stop=None,
        dt_ns=dt_ns,
        cosine_cutoff_mhz=cosine_cutoff_mhz,
        snr_peak_min=5.0,
        peak_distance_ns=5.0,
        wavelet="sym8", wlevel=5, wmode="hard", wthresh_k=1.0,
        denoise_raw=False,
        no_denoise=False,
        out_suffix="",
        sat_thresh=float(settings.sat_thresh_mv or 960),
        fit_csv=str(settings.template_csv) if settings.template_csv else None,
        no_vetoes=True,
        peak_order="tallest",
        data_root=str(fixture.parent),
        out_dir=str(cell_out_dir),
        stft_nperseg_ns=nperseg_ns,
        stft_freq_cutoff_mhz=fcut_mhz,
        stft_amp_threshold=amp_thr,
    )


def _iter_grid(args) -> List[Tuple[float, float, float]]:
    npersegs = args.nperseg_ns or list(DEFAULT_NPERSEG_NS)
    fcuts = args.freq_cutoff_mhz or list(DEFAULT_FREQ_CUTOFF_MHZ)
    amps = args.amp_thresholds or list(DEFAULT_AMP_THRESHOLD)
    return [(n, f, a) for n in npersegs for f in fcuts for a in amps]


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or _default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="tune stft", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    grid = _iter_grid(args)
    regimes = args.regimes or ["level1", "level2", "level3", "level4", "level5"]
    n_events_override = args.n_events
    dt_ns = (float(settings.dt_ns) if isinstance(settings.dt_ns, float)
             else 0.5051)
    # Bug-fix 10 (2026-05-30 final scan): use the dataset preset's
    # cosine cutoff as the default, not the STFT pipeline's hardcoded
    # 100 MHz. For na22 that means 450 MHz (matching tune wavelet);
    # for cd109 it's 50 MHz; etc. The 100 MHz fallback is only used
    # when no dataset preset and no --cosine-cutoff-mhz are supplied.
    cosine_cutoff_mhz = (float(args.cosine_cutoff_mhz)
                         if args.cosine_cutoff_mhz is not None
                         else (float(settings.cosine_cutoff_mhz)
                               if settings.cosine_cutoff_mhz is not None
                               else 100.0))

    from pmtdeconv.pipeline.process_hits import process_run_stft as process_run

    weights = ScoringWeights()
    rows: List[dict] = []
    print(f"Sweeping {len(grid)} STFT configs × {len(regimes)} regimes "
          f"= {len(grid) * len(regimes)} cells "
          f"(dt_ns={dt_ns}, cosine={cosine_cutoff_mhz} MHz)")

    for regime in regimes:
        fixture = ensure_regime_fixture(regime, dt_ns=dt_ns,
                                        n_events=n_events_override)
        truth_csv = find_truth_for(fixture)
        if truth_csv is None:
            print(f"[skip] no truth CSV beside {fixture}")
            continue
        print(f"\n── regime {regime}  ({fixture.name})")

        for nperseg, fcut, amp_thr in grid:
            t_cell = time.time()
            with tempfile.TemporaryDirectory(dir=out_dir) as tmpd:
                cell_out_dir = Path(tmpd)
                legacy = _build_legacy_args(settings, cell_out_dir, fixture,
                                            dt_ns, cosine_cutoff_mhz,
                                            nperseg, fcut, amp_thr)
                try:
                    process_run(legacy)
                except Exception as e:
                    print(f"  nperseg={nperseg}/fcut={fcut}/amp={amp_thr}  "
                          f"FAILED: {e}")
                    continue
                produced = cell_out_dir / DECONV_PULSE_CSV
                if not produced.exists() or produced.stat().st_size == 0:
                    print(f"  nperseg={nperseg}/fcut={fcut}/amp={amp_thr}  "
                          f"no hits produced")
                    continue
                metrics = score_run(produced, truth_csv, mode="hits",
                                    weights=weights)

            row = dict(regime=regime, stft_nperseg_ns=nperseg,
                       stft_freq_cutoff_mhz=fcut,
                       stft_amp_threshold=amp_thr,
                       **metrics, seconds=time.time() - t_cell)
            rows.append(row)
            print(f"  nperseg={nperseg:5.1f}ns  fcut={fcut:6.1f}MHz  "
                  f"amp={amp_thr:5.1f}  "
                  f"t_σ={metrics['timing_sigma_ns']:.3f}ns  "
                  f"eff={metrics['efficiency']:.0%}  "
                  f"({row['seconds']:.2f}s)")

    if not rows:
        print("No cells scored. Aborting.")
        return 1

    sweep_df = pd.DataFrame(rows)
    # Q2: bias = drift vs dataset preset's STFT defaults (legacy: 20, 175, 10)
    sweep_df = apply_drift_bias(
        sweep_df,
        axis_cols=("stft_nperseg_ns", "stft_freq_cutoff_mhz",
                   "stft_amp_threshold"),
        reference_lookup=exact_reference({
            "stft_nperseg_ns": 20.0,
            "stft_freq_cutoff_mhz": 175.0,
            "stft_amp_threshold": 10.0,
        }),
        weights=weights,
    )

    sweep_csv = out_dir / "sweep_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSweep summary: {sweep_csv}")

    # Winner per regime
    recipe: Dict[str, dict] = {}
    winner_axis: Dict[str, float] = {}
    grid_index: Dict[Tuple, int] = {cfg: i for i, cfg in enumerate(grid)}
    sweep_df["axis_index"] = sweep_df.apply(
        lambda r: grid_index.get((float(r["stft_nperseg_ns"]),
                                  float(r["stft_freq_cutoff_mhz"]),
                                  float(r["stft_amp_threshold"])), -1),
        axis=1)
    for regime, sub in sweep_df.groupby("regime"):
        sub_ok = sub[sub["score"].notna()
                     & sub["score"].apply(lambda v: v != float("inf"))]
        if len(sub_ok) == 0:
            continue
        winner = sub_ok.loc[sub_ok["score"].idxmin()]
        recipe[regime] = {
            "best_params": {
                "stft_nperseg_ns": float(winner["stft_nperseg_ns"]),
                "stft_freq_cutoff_mhz": float(winner["stft_freq_cutoff_mhz"]),
                "stft_amp_threshold": float(winner["stft_amp_threshold"]),
            },
            "metrics": {k: float(winner[k]) for k in
                        ("score", "charge_bias", "charge_scatter",
                         "efficiency", "timing_sigma_ns",
                         "n_truth", "n_matched")},
            "all_cells": int(len(sub)),
        }
        winner_axis[regime] = float(winner["axis_index"])

    recipe_path = out_dir / "recipe.json"
    recipe_path.write_text(json.dumps(
        {"swept_axis": "stft_config",
         "dataset": settings.dataset,
         "dt_ns": dt_ns,
         "cosine_cutoff_mhz": cosine_cutoff_mhz,
         "reference_config": {"stft_nperseg_ns": 20.0,
                              "stft_freq_cutoff_mhz": 175.0,
                              "stft_amp_threshold": 10.0},
         "regimes": recipe}, indent=2))
    print(f"Recipe: {recipe_path}")

    png_paths = save_sweep_curves(
        sweep_df, axis_col="axis_index", out_dir=out_dir,
        axis_label=f"config index (0..{len(grid)-1})",
        winner_axis_value_by_regime=winner_axis,
    )

    n_presets_written = 0
    if args.write_presets:
        stamp = date.today().isoformat()
        new_entries: Dict[str, dict] = {}
        for regime, payload in recipe.items():
            preset_name = (f"{settings.dataset}-stft-tuned-{stamp}-{regime}"
                           if settings.dataset else
                           f"stft-tuned-{stamp}-{regime}")
            description = (
                f"Auto-tuned by `pmtdeconv tune stft` on {stamp}. "
                f"Regime {regime}; "
                f"nperseg={payload['best_params']['stft_nperseg_ns']:.1f} ns, "
                f"fcut={payload['best_params']['stft_freq_cutoff_mhz']:.1f} MHz, "
                f"amp_thr={payload['best_params']['stft_amp_threshold']:.1f}; "
                f"t_σ={payload['metrics']['timing_sigma_ns']:.3f} ns, "
                f"drift={payload['metrics']['charge_bias']:+.3%}."
            )
            new_entries[preset_name] = {
                "name": preset_name,
                "base_dataset": settings.dataset or "na22",
                "overrides": {
                    "denoiser": "stft",
                    # Bug-fix 11 (2026-05-30 final scan): freeze the
                    # cosine cutoff used during the sweep into the
                    # preset, so applying the tuned recipe later
                    # reproduces the exact (cutoff × STFT) cell that
                    # won. Without this, the preset would inherit
                    # whatever the user's dataset preset cutoff
                    # happens to be at apply-time.
                    "cosine_cutoff_mhz": cosine_cutoff_mhz,
                    "stft_nperseg_ns": payload['best_params']['stft_nperseg_ns'],
                    "stft_freq_cutoff_mhz": payload['best_params']['stft_freq_cutoff_mhz'],
                    "stft_amp_threshold": payload['best_params']['stft_amp_threshold'],
                },
                "description": description,
            }
        total_in_file = write_tuned_presets(new_entries)
        n_presets_written = len(new_entries)
        print(f"Wrote {n_presets_written} ConfigPresets "
              f"(total in file: {total_in_file})")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="tune stft",
        settings=settings, provenance=provenance,
        inputs=[{"role": "regime_fixture",
                 "path": str(regime_fixture(r))} for r in regimes],
        outputs=[
            {"role": "tune_recipe", "path": str(recipe_path)},
            {"role": "tune_summary", "path": str(sweep_csv)},
        ] + [{"role": "sweep_curve_png", "path": str(p)}
             for p in png_paths],
        stats={"n_cells": int(len(rows)),
               "n_regimes": len(regimes),
               "n_configs_per_regime": len(grid),
               "cosine_cutoff_mhz": cosine_cutoff_mhz,
               "dt_ns": dt_ns,
               "n_presets_written": n_presets_written},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
