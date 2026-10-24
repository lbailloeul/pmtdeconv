"""``pmtdeconv tune cosine`` — sweep cosine cutoff per regime.

For each (regime, cutoff) cell:
1. Read the regime-tagged synthetic fixture under
   ``tests/fixtures/regime/<regime>/flattened_Run0.root`` (built on
   demand at the dataset's dt_ns — Q6).
2. Call ``pipeline.process_area.process_run`` with the candidate cutoff
   and the chosen denoiser (Q1: ``--denoiser none`` isolates the
   cosine effect; default ``wavelet`` matches deconv_area).
3. Score the produced CSV against the sibling truth CSV via
   ``pipeline.scoring.score_run(mode='area')``.
4. Apply drift-based bias (Q2): bias of each cell = its median ratio
   relative to the dataset preset's cutoff cell.

Writes:
- ``recipe.json`` — winner per regime
- ``sweep_summary.csv`` — every (regime, cutoff) row
- ``sweep_curves_<regime>.png`` — efficiency / charge drift / FWHM vs cutoff
  (the ranked metrics, winner marked)
- ``manifest.json``

With ``--write-presets``: idempotently merges ConfigPreset entries
into ``pmtdeconv/presets/tuned_configs.py`` (Q10).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.schemas import AREA_COLUMNS
from pmtdeconv.io.templates import load_template
from pmtdeconv.io.truth import find_truth_for
from pmtdeconv.pipeline.scoring import ScoringWeights, score_run
from pmtdeconv.pipeline.template import (
    add_generic_fallback_flag, configured_template_csv,
)
from pmtdeconv.presets import resolve_settings

from pmtdeconv.commands._tune_shared import (
    apply_drift_bias, area_denoiser, closest_reference,
    ensure_regime_fixture, regime_fixture,
    save_sweep_curves, write_tuned_presets,
)


# Fixed defaults for non-swept knobs.
SIG_START_DEFAULT = 10
SIG_END_DEFAULT = 1000
N_INTERVALS = 1
WAVELET_DEFAULT = "sym8"
WLEVEL_DEFAULT = 5
WMODE_DEFAULT = "hard"
WTHRESH_K_DEFAULT = 2.0
NO_TRIGGER_AMP_DEFAULT = 15.0
T_PEAK_MIN_NS_DEFAULT = 20.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tune cosine")
    g.add_argument("--regimes", nargs="+", default=None,
                   help="Which regimes to sweep (default: all 5). "
                        "Choices: level1 level2 level3 level4 level5.")
    g.add_argument("--cutoffs", type=float, nargs="+", default=None,
                   help="Cosine cutoffs to test in MHz. Default: a fine, "
                        "Nyquist-aware grid (dense 20-300 MHz @10, coarse "
                        "above @50, capped at 0.9*Nyquist for the dataset's "
                        "dt_ns).")
    g.add_argument("--charge-tol", type=float, default=0.02,
                   help="Loose charge-drift gate (default 0.02 = 2%%). A cutoff "
                        "is only eligible if its charge scale stays within this "
                        "fraction of the reference cutoff. Among the eligible "
                        "cutoffs the winner is picked by --min-efficiency, then "
                        "--drift-band, then the thinnest reconstructed FWHM. "
                        "Tighten to 0.01 for Daya-Bay-strict charge linearity.")
    g.add_argument("--min-efficiency", type=float, default=1.0,
                   help="EFFICIENCY PROTECTION (priority 1, default 1.0 = MAXIMIZE "
                        "efficiency): a cutoff is only eligible if its "
                        "reconstruction efficiency is at least this fraction of "
                        "the regime's BEST efficiency. Default 1.0 keeps ONLY "
                        "the cutoffs that achieve the maximum efficiency, then "
                        "picks the thinnest pulse (and best resolution) among "
                        "them. LOWER it (e.g. 0.95) to permit trading a little "
                        "efficiency for a thinner pulse / higher cutoff.")
    g.add_argument("--drift-band", type=float, default=0.005,
                   help="Charge-drift target band (priority 2), as a fraction "
                        "(default 0.005 = 0.5%%). Among the MAX-efficiency "
                        "cutoffs, keep those whose |charge drift| is within this "
                        "band of zero, then minimize FWHM among them. If no "
                        "max-efficiency cutoff reaches the band (efficiency wins "
                        "— it is priority 1), the cutoff with the smallest "
                        "|drift| is used instead.")
    g.add_argument("--denoiser", choices=("none", "wavelet"),
                   default=None,
                   help="Q1 isolation flag: `none` runs deconv with NO "
                        "post-denoise (pure cosine sweep, isolates the "
                        "cutoff effect on charge). `wavelet` keeps the "
                        "default sym8/L5/hard/k=2.0 denoise applied.")
    g.add_argument("--template", type=str, default=None,
                   help="SPE template CSV path. Default: dataset's preset.")
    add_generic_fallback_flag(g)
    g.add_argument("--write-presets", action="store_true",
                   help="Idempotently merge winning ConfigPresets into "
                        "pmtdeconv/presets/tuned_configs.py.")
    g.add_argument("--n-events", type=int, default=None,
                   help="Override --n-events on the regime fixture builds. "
                        "Default: use whatever's already on disk.")


def _default_cutoffs(dt_ns: float) -> List[float]:
    """Fine, Nyquist-aware default cutoff grid.

    The pulse-width-vs-cutoff knee (Tang et al. 2411.07620, Fig. 8a) is
    steep below ~300 MHz and flat above, so we sample densely in the knee
    (10 MHz) and coarsely in the plateau (50 MHz). The upper bound is
    capped at 0.9·Nyquist so we never sweep into the aliasing edge:

      - na22  (dt 0.5051 ns): Nyquist ≈ 990 MHz → grid 20–300 @10, 350–~890 @50
      - cd109 (dt 1.432  ns): Nyquist ≈ 349 MHz → grid 20–300 @10  (no plateau)

    For cd109 note the deconv conditioning cliff (~65 MHz) lives well
    inside the fine region, so the interesting structure is well sampled.
    Override with ``--cutoffs`` for a custom sweep.
    """
    nyquist_mhz = 500.0 / float(dt_ns)
    cap = 0.9 * nyquist_mhz
    knee = min(300.0, cap)
    fine = list(np.arange(20.0, knee + 1e-6, 10.0))
    coarse = list(np.arange(knee + 50.0, cap + 1e-6, 50.0))
    grid = sorted({round(float(x), 1) for x in fine + coarse})
    return grid


def _default_regimes() -> List[str]:
    return ["level1", "level2", "level3", "level4", "level5"]


def _default_out_dir(settings) -> Path:
    parts = [f"tune_cosine_{settings.dataset or 'na22'}"]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)
    denoiser_choice = area_denoiser(settings, provenance)
    # A configured-but-missing template is an error (the rule hits applies).
    template_csv = configured_template_csv(settings, provenance)

    out_dir = settings.out_dir or _default_out_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="tune cosine", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    dt_ns = (float(settings.dt_ns) if isinstance(settings.dt_ns, float)
             else 0.5051)
    cutoffs = args.cutoffs or _default_cutoffs(dt_ns)
    regimes = args.regimes or _default_regimes()
    n_events_override = args.n_events
    charge_tol = float(args.charge_tol)
    min_eff_frac = float(args.min_efficiency)
    drift_band = float(args.drift_band)
    # Translate to process_run signature (wavelet="none" disables denoise)
    wavelet_arg = ("none" if denoiser_choice == "none"
                   else WAVELET_DEFAULT)

    from pmtdeconv.pipeline.process_area import process_run

    templates = load_template(template_csv)

    weights = ScoringWeights()
    rows: List[dict] = []
    print(f"Sweeping {len(cutoffs)} cosine cutoffs × {len(regimes)} regimes "
          f"= {len(cutoffs) * len(regimes)} cells  "
          f"(dt_ns={dt_ns}, denoiser={denoiser_choice})")

    for regime in regimes:
        fixture = ensure_regime_fixture(regime, dt_ns=dt_ns,
                                        n_events=n_events_override)
        truth_csv = find_truth_for(fixture)
        if truth_csv is None:
            print(f"[skip] no truth CSV beside {fixture}")
            continue
        print(f"\n── regime {regime}  ({fixture.name})")

        for cutoff in cutoffs:
            t_cell = time.time()
            df = process_run(
                root_path=fixture,
                dt_ns=dt_ns,
                sig_start=SIG_START_DEFAULT,
                sig_end=SIG_END_DEFAULT,
                n_intervals=N_INTERVALS,
                cosine_cutoff_mhz=cutoff,
                wavelet=wavelet_arg, wlevel=WLEVEL_DEFAULT,
                mode=WMODE_DEFAULT, thresh_k=WTHRESH_K_DEFAULT,
                no_trigger_amp=NO_TRIGGER_AMP_DEFAULT,
                t_peak_min_ns=T_PEAK_MIN_NS_DEFAULT,
                afterpulse_window_ns=0.0,
                afterpulse_height_ratio=5.0,
                fft_pad=bool(settings.fft_pad),
                wrap_ratio=0.5,
                wrap_abs_k=0.0,
                entry_stop=None,
                templates=templates,
                branches_keep=None,
                measure_width=True,
            )
            # Carry width_ns alongside the area schema into the scored cell
            # CSV (temp file → scored → deleted; extending columns is safe).
            cell_cols = list(AREA_COLUMNS)
            if "width_ns" in df.columns:
                cell_cols.append("width_ns")
            df_pass = df[df["interval"] >= 0][cell_cols].copy()
            cell_csv = out_dir / f"_cell_{regime}_cosine{int(cutoff)}.csv"
            df_pass.to_csv(cell_csv, index=False)
            metrics = score_run(cell_csv, truth_csv, mode="area",
                                weights=weights)
            cell_csv.unlink(missing_ok=True)

            row = dict(regime=regime, cosine_cutoff_mhz=cutoff, **metrics,
                       seconds=time.time() - t_cell)
            rows.append(row)
            print(f"  cutoff={cutoff:6.1f} MHz  "
                  f"scatter={metrics['charge_scatter']:.3%}  "
                  f"eff={metrics['efficiency']:.2%}  "
                  f"FWHM={metrics.get('width_fwhm_ns', float('nan')):.2f} ns  "
                  f"ratio_med={metrics['charge_ratio_median']:.3e}  "
                  f"({row['seconds']:.2f}s)")

    if not rows:
        print("No cells were scored. Aborting.")
        return 1

    sweep_df = pd.DataFrame(rows)
    # Q2: bias = drift vs dataset preset's cutoff. closest_reference
    # gracefully degrades when settings.cosine_cutoff_mhz isn't in grid.
    ref_value = float(settings.cosine_cutoff_mhz)
    sweep_df = apply_drift_bias(
        sweep_df,
        axis_cols=("cosine_cutoff_mhz",),
        reference_lookup=closest_reference(ref_value, "cosine_cutoff_mhz"),
        weights=weights,
    )

    sweep_csv = out_dir / "sweep_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSweep summary: {sweep_csv}")

    # Winner per regime — drift gate, then efficiency > drift band > width.
    #
    # ``charge_bias`` here is the charge-scale DRIFT vs the reference cutoff
    # (apply_drift_bias rewrote it above; the dataset preset's cutoff is the
    # reference and has drift≈0). The hard gate is "charge scale stays within
    # charge_tol of the reference" AND "efficiency ≥ floor".
    #
    # Selection among the gated cells: see PRIORITY ORDER below. One
    # representative per cutoff = its best (lowest) scatter row, so multi-row
    # noise cells can't pollute the width comparison.
    recipe: Dict[str, dict] = {}
    winner_axis: Dict[str, float] = {}
    for regime, sub in sweep_df.groupby("regime"):
        sub_ok = sub[sub["score"].notna()
                     & sub["score"].apply(lambda v: v != float("inf"))
                     & (sub["n_matched"].fillna(0) > 0)]
        if len(sub_ok) == 0:
            print(f"[warn] no valid cells for {regime}")
            continue
        # Reference-cutoff efficiency (cell closest to the dataset preset's
        # cutoff) — kept for provenance only.
        ref_idx = (sub_ok["cosine_cutoff_mhz"].astype(float)
                   - ref_value).abs().idxmin()
        ref_eff = float(sub_ok.loc[ref_idx, "efficiency"])
        # Drift gate only (charge-scale sanity / conservation). Drift is the
        # LOWEST-priority metric, so it is just a loose ±charge_tol bound, not
        # the selector.
        gated = sub_ok[(sub_ok["charge_bias"].astype(float).abs() <= charge_tol)
                       & sub_ok["width_fwhm_ns"].notna()].copy()
        if len(gated) > 0:
            for c in ("cosine_cutoff_mhz", "width_fwhm_ns",
                      "charge_scatter", "efficiency"):
                gated[c] = gated[c].astype(float)
            # one representative per cutoff = its best (lowest) charge scatter
            reps = gated.loc[gated.groupby("cosine_cutoff_mhz")
                             ["charge_scatter"].idxmin()]
            emax = float(reps["efficiency"].max())
            eff_floor = min_eff_frac * emax - 1e-9   # eps for float equality
            # PRIORITY ORDER (user 2026-06-01):
            #   (1) maintain efficiency NEAREST 100% (top priority),
            #   (2) within that, bring |charge drift| within --drift-band (0.5%),
            #   (3) then MINIMIZE reconstructed FWHM.
            # (1) MAXIMISE EFFICIENCY: eligible = cutoffs at the regime's best
            #     efficiency (min_eff_frac default 1.0). Stays off the cliff too.
            eligible = reps[reps["efficiency"] >= eff_floor].copy()
            if len(eligible) == 0:
                eligible = reps.copy()
            eligible["abs_drift"] = eligible["charge_bias"].astype(float).abs()
            # (2) DRIFT within target band; (3) MINIMISE WIDTH among those.
            drift_ok = eligible[eligible["abs_drift"] <= drift_band]
            if len(drift_ok) > 0:
                winner_row = drift_ok.sort_values(
                    ["width_fwhm_ns", "abs_drift", "cosine_cutoff_mhz"]).iloc[0]
                selection = "effmax_driftband_minwidth"
            else:
                # drift target unreachable at max efficiency (efficiency wins) →
                # get drift as close to the band as possible; among the cutoffs
                # whose |drift| ties that minimum (within 0.2·drift_band), take
                # the thinnest (so a hair-smaller drift can't cost real width).
                d_min = float(eligible["abs_drift"].min())
                near = eligible[eligible["abs_drift"]
                                <= d_min + 0.2 * drift_band]
                winner_row = near.sort_values(
                    ["width_fwhm_ns", "abs_drift", "cosine_cutoff_mhz"]).iloc[0]
                selection = "effmax_mindrift"
        else:
            # Nothing passed even the loose drift gate with a finite FWHM;
            # fall back to the composite-score minimum (and flag it).
            winner_row = sub_ok.loc[sub_ok["score"].idxmin()]
            selection = "score_fallback"
            eff_floor = float("nan")
            print(f"  [warn] {regime}: no cutoff passed the drift gate "
                  f"(|drift|≤{charge_tol:.1%}) with a finite FWHM; min-score.")
        recipe[regime] = {
            "best_params": {
                "cosine_cutoff_mhz": float(winner_row["cosine_cutoff_mhz"]),
            },
            "selection": selection,
            "charge_tol": charge_tol,
            "min_efficiency_frac": min_eff_frac,
            "reference_efficiency": ref_eff,
            "efficiency_floor": eff_floor,
            "n_gated": int(len(gated)),
            "metrics": {k: float(winner_row[k]) for k in
                        ("score", "charge_bias", "charge_scatter",
                         "width_fwhm_ns", "efficiency", "n_truth", "n_matched")},
            "all_cells": int(len(sub)),
        }
        winner_axis[regime] = float(winner_row["cosine_cutoff_mhz"])
        print(f"  {regime} winner [{selection}]: cutoff="
              f"{winner_row['cosine_cutoff_mhz']:.1f} MHz  "
              f"FWHM={winner_row['width_fwhm_ns']:.2f} ns  "
              f"drift={winner_row['charge_bias']:+.3%}  "
              f"scatter={winner_row['charge_scatter']:.3%}  "
              f"eff={winner_row['efficiency']:.1%}  "
              f"({int(len(gated))}/{len(sub_ok)} cells within gate)")

    recipe_path = out_dir / "recipe.json"
    recipe_path.write_text(json.dumps(
        {"swept_axis": "cosine_cutoff_mhz",
         "dataset": settings.dataset,
         "dt_ns": dt_ns,
         "denoiser": denoiser_choice,
         "reference_cutoff_mhz": ref_value,
         "selection_method": "effmax_driftband_minwidth",
         "priority_order": ("efficiency (1, at max) > |charge_drift| within "
                            "drift_band (2) > min width_fwhm_ns (3)"),
         "primary_constraint": ("efficiency >= min_efficiency_frac * "
                                "max_efficiency  (default 1.0 = at max)"),
         "charge_tol": charge_tol,
         "min_efficiency_frac": min_eff_frac,
         "drift_band": drift_band,
         "secondary_objective": ("among max-efficiency cutoffs, keep |drift| <= "
                                 "drift_band then minimize FWHM; if none reach "
                                 "the band, take the smallest |drift|"),
         "regimes": recipe}, indent=2))
    print(f"Recipe: {recipe_path}")

    # Q3: per-regime sweep curve PNGs
    png_paths = save_sweep_curves(
        sweep_df, axis_col="cosine_cutoff_mhz", out_dir=out_dir,
        axis_label="cosine cutoff [MHz]",
        winner_axis_value_by_regime=winner_axis,
        drift_band=drift_band,
    )
    for p in png_paths:
        print(f"  Sweep curve: {p}")

    n_presets_written = 0
    if args.write_presets:
        stamp = date.today().isoformat()
        new_entries: Dict[str, dict] = {}
        for regime, payload in recipe.items():
            preset_name = (f"{settings.dataset}-cosine-tuned-{stamp}-{regime}"
                           if settings.dataset else
                           f"cosine-tuned-{stamp}-{regime}")
            description = (
                f"Auto-tuned by `pmtdeconv tune cosine` on {stamp}. "
                f"Regime {regime}; cosine cutoff "
                f"{payload['best_params']['cosine_cutoff_mhz']:.1f} MHz; "
                f"score={payload['metrics']['score']:+.4f}, "
                f"drift={payload['metrics']['charge_bias']:+.3%}, "
                f"scatter={payload['metrics']['charge_scatter']:.3%}."
            )
            new_entries[preset_name] = {
                "name": preset_name,
                "base_dataset": settings.dataset or "na22",
                "overrides": {"cosine_cutoff_mhz":
                              payload['best_params']['cosine_cutoff_mhz']},
                "description": description,
            }
        total_in_file = write_tuned_presets(new_entries)
        n_presets_written = len(new_entries)
        print(f"Wrote {n_presets_written} ConfigPresets to "
              f"pmtdeconv/presets/tuned_configs.py "
              f"(total in file: {total_in_file})")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="tune cosine",
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
               "n_cutoffs": len(cutoffs),
               "denoiser": denoiser_choice,
               "dt_ns": dt_ns,
               "selection_method": "epsilon_constraint",
               "charge_tol": charge_tol,
               "min_efficiency_frac": min_eff_frac,
               "n_presets_written": n_presets_written},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
