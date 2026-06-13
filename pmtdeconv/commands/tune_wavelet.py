"""``pmtdeconv tune wavelet`` — sweep wavelet config per regime.

For each (regime, family, level, mode, k) cell:
1. Read the regime-tagged synthetic fixture (built on demand at the
   dataset's dt_ns — Q6).
2. Call ``pipeline.process_hits.process_run`` with the candidate
   wavelet config and the chosen cosine cutoff (Q1: ``--cosine-cutoff-mhz``
   defaults to the dataset preset's cutoff). ``--no-vetoes`` +
   ``--peak-order tallest`` is forced so peak_id=1 is the largest peak
   (matches the truth model: one synthetic pulse per event).
3. Score against the sibling truth CSV via
   ``pipeline.scoring.score_run(mode='hits')``.
4. Apply drift-based bias (Q2): per regime, bias = drift vs the
   dataset preset's wavelet config (if it lands in the sweep grid).

Writes recipe.json / sweep_summary.csv / sweep_curves_<regime>.png /
manifest.json. With ``--write-presets`` does an idempotent
read/merge/rewrite of ``pmtdeconv/presets/tuned_configs.py``.
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


DEFAULT_FAMILIES = ("sym6", "sym8", "db4", "bior3.5")
DEFAULT_LEVELS = (3, 4, 5)
DEFAULT_MODES = ("soft", "hard")
DEFAULT_KS = (0.5, 1.0, 1.5, 2.0)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tune wavelet")
    g.add_argument("--regimes", nargs="+", default=None,
                   help="Which regimes to sweep (default: all 5).")
    g.add_argument("--wavelets", nargs="+", default=None,
                   help=f"Wavelet families (default: {DEFAULT_FAMILIES}).")
    g.add_argument("--wlevels", type=int, nargs="+", default=None,
                   help=f"Decomposition levels (default: {DEFAULT_LEVELS}).")
    g.add_argument("--wmodes", nargs="+", default=None,
                   choices=("soft", "hard", "garrote"),
                   help=f"Threshold modes (default: {DEFAULT_MODES}).")
    g.add_argument("--wks", type=float, nargs="+", default=None,
                   help=f"Threshold k multipliers (default: {DEFAULT_KS}).")
    g.add_argument("--cosine-cutoff-mhz", type=float, default=None,
                   help="Q1 isolation knob: cosine cutoff held fixed during "
                        "the wavelet sweep. Defaults to the dataset preset's "
                        "cosine_cutoff_mhz (450 MHz for na22).")
    g.add_argument("--template", type=str, default=None)
    g.add_argument("--write-presets", action="store_true")
    g.add_argument("--n-events", type=int, default=None)


def _default_out_dir(settings) -> Path:
    parts = [f"tune_wavelet_{settings.dataset or 'na22'}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def _build_legacy_args(settings, cell_out_dir: Path,
                       fixture: Path, dt_ns: float, cosine_cutoff_mhz: float,
                       wavelet: str, wlevel: int, wmode: str,
                       wthresh_k: float) -> SimpleNamespace:
    return SimpleNamespace(
        runs=[0],
        entry_stop=None,
        dt_ns=dt_ns,
        cosine_cutoff_mhz=cosine_cutoff_mhz,    # Q1: now plumbed through
        snr_peak_min=5.0,
        peak_distance_ns=5.0,
        wavelet=wavelet, wlevel=wlevel, wmode=wmode, wthresh_k=wthresh_k,
        denoise_raw=False,
        no_denoise=False,
        out_suffix="",
        sat_thresh=float(settings.sat_thresh_mv or 960),
        fit_csv=str(settings.template_csv) if settings.template_csv else None,
        no_vetoes=True,
        peak_order="tallest",
        data_root=str(fixture.parent),
        out_dir=str(cell_out_dir),
    )


def _iter_grid(args) -> List[Tuple[str, int, str, float]]:
    families = args.wavelets or list(DEFAULT_FAMILIES)
    levels = args.wlevels or list(DEFAULT_LEVELS)
    modes = args.wmodes or list(DEFAULT_MODES)
    ks = args.wks or list(DEFAULT_KS)
    return [(f, l, m, k)
            for f in families for l in levels
            for m in modes for k in ks]


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)

    out_dir = settings.out_dir or _default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="tune wavelet", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    grid = _iter_grid(args)
    regimes = args.regimes or ["level1", "level2", "level3", "level4", "level5"]
    n_events_override = args.n_events
    dt_ns = (float(settings.dt_ns) if isinstance(settings.dt_ns, float)
             else 0.5051)
    cosine_cutoff_mhz = (float(args.cosine_cutoff_mhz)
                         if args.cosine_cutoff_mhz is not None
                         else float(settings.cosine_cutoff_mhz))

    from pmtdeconv.pipeline.process_hits import process_run

    weights = ScoringWeights()
    rows: List[dict] = []
    print(f"Sweeping {len(grid)} wavelet configs × {len(regimes)} regimes "
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

        for wavelet, wlevel, wmode, wthresh_k in grid:
            t_cell = time.time()
            with tempfile.TemporaryDirectory(dir=out_dir) as tmpd:
                cell_out_dir = Path(tmpd)
                legacy = _build_legacy_args(
                    settings, cell_out_dir, fixture,
                    dt_ns, cosine_cutoff_mhz,
                    wavelet, wlevel, wmode, wthresh_k)
                try:
                    process_run(legacy)
                except Exception as e:
                    print(f"  {wavelet}/L{wlevel}/{wmode}/k{wthresh_k}  "
                          f"FAILED: {e}")
                    continue
                produced = cell_out_dir / DECONV_PULSE_CSV
                if not produced.exists() or produced.stat().st_size == 0:
                    print(f"  {wavelet}/L{wlevel}/{wmode}/k{wthresh_k}  "
                          f"no hits produced")
                    continue
                metrics = score_run(produced, truth_csv, mode="hits",
                                    weights=weights)

            row = dict(regime=regime, wavelet=wavelet, wlevel=wlevel,
                       wmode=wmode, wthresh_k=wthresh_k,
                       **metrics, seconds=time.time() - t_cell)
            rows.append(row)
            print(f"  {wavelet:>7s}/L{wlevel}/{wmode:>4s}/k{wthresh_k:.1f}  "
                  f"t_σ={metrics['timing_sigma_ns']:.3f}ns  "
                  f"scatter={metrics['charge_scatter']:.3%}  "
                  f"eff={metrics['efficiency']:.0%}  "
                  f"({row['seconds']:.2f}s)")

    if not rows:
        print("No cells scored. Aborting.")
        return 1

    sweep_df = pd.DataFrame(rows)
    # Q2: bias = drift vs dataset preset's wavelet config (if in grid)
    ref_wavelet = settings.wavelet
    ref_wlevel = int(settings.wlevel)
    ref_wmode = settings.wmode
    ref_wthresh_k = float(settings.wthresh_k)
    sweep_df = apply_drift_bias(
        sweep_df,
        axis_cols=("wavelet", "wlevel", "wmode", "wthresh_k"),
        reference_lookup=exact_reference({
            "wavelet": ref_wavelet, "wlevel": ref_wlevel,
            "wmode": ref_wmode, "wthresh_k": ref_wthresh_k,
        }),
        weights=weights,
    )

    sweep_csv = out_dir / "sweep_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSweep summary: {sweep_csv}")

    # Winner per regime
    recipe: Dict[str, dict] = {}
    winner_axis: Dict[str, float] = {}    # 1D axis for the curve plot
    # Sort grid so each cell has a stable axis index
    grid_index: Dict[Tuple, int] = {cfg: i for i, cfg in enumerate(grid)}
    sweep_df["axis_index"] = sweep_df.apply(
        lambda r: grid_index.get((r["wavelet"], int(r["wlevel"]),
                                  r["wmode"], float(r["wthresh_k"])), -1),
        axis=1)
    for regime, sub in sweep_df.groupby("regime"):
        sub_ok = sub[sub["score"].notna()
                     & sub["score"].apply(lambda v: v != float("inf"))]
        if len(sub_ok) == 0:
            continue
        winner = sub_ok.loc[sub_ok["score"].idxmin()]
        recipe[regime] = {
            "best_params": {
                "wavelet": str(winner["wavelet"]),
                "wlevel": int(winner["wlevel"]),
                "wmode": str(winner["wmode"]),
                "wthresh_k": float(winner["wthresh_k"]),
            },
            "metrics": {k: float(winner[k]) for k in
                        ("score", "charge_bias", "charge_scatter",
                         "efficiency", "timing_sigma_ns",
                         "n_truth", "n_matched")},
            "all_cells": int(len(sub)),
        }
        winner_axis[regime] = float(winner["axis_index"])
        print(f"  {regime} winner: {winner['wavelet']}/L{winner['wlevel']}"
              f"/{winner['wmode']}/k{winner['wthresh_k']}  "
              f"score={winner['score']:+.4f}  "
              f"t_σ={winner['timing_sigma_ns']:.3f}ns  "
              f"drift={winner['charge_bias']:+.3%}")

    recipe_path = out_dir / "recipe.json"
    recipe_path.write_text(json.dumps(
        {"swept_axis": "wavelet_config",
         "dataset": settings.dataset,
         "dt_ns": dt_ns,
         "cosine_cutoff_mhz": cosine_cutoff_mhz,
         "reference_config": {
             "wavelet": ref_wavelet, "wlevel": ref_wlevel,
             "wmode": ref_wmode, "wthresh_k": ref_wthresh_k,
         },
         "regimes": recipe}, indent=2))
    print(f"Recipe: {recipe_path}")

    # Q3: sweep curves vs grid index (wavelet config doesn't have a 1D natural axis)
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
            preset_name = (f"{settings.dataset}-wavelet-tuned-{stamp}-{regime}"
                           if settings.dataset else
                           f"wavelet-tuned-{stamp}-{regime}")
            description = (
                f"Auto-tuned by `pmtdeconv tune wavelet` on {stamp}. "
                f"Regime {regime}; "
                f"{payload['best_params']['wavelet']}/L"
                f"{payload['best_params']['wlevel']}/"
                f"{payload['best_params']['wmode']}/"
                f"k={payload['best_params']['wthresh_k']}; "
                f"t_σ={payload['metrics']['timing_sigma_ns']:.3f} ns, "
                f"drift={payload['metrics']['charge_bias']:+.3%}."
            )
            new_entries[preset_name] = {
                "name": preset_name,
                "base_dataset": settings.dataset or "na22",
                "overrides": {
                    "wavelet": payload['best_params']['wavelet'],
                    "wlevel": payload['best_params']['wlevel'],
                    "wmode": payload['best_params']['wmode'],
                    "wthresh_k": payload['best_params']['wthresh_k'],
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
        subcommand="tune wavelet",
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
