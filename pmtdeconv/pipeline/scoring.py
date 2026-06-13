"""Score a (produced CSV, truth CSV) pair → metrics dict (PR 10).

Two scoring modes:

- ``mode="area"``: produced CSV follows AREA_COLUMNS. One row per
  ``(event, branch)`` — match directly to truth's ``(event, branch)``
  rows. Metrics: charge bias, charge scatter, efficiency.

- ``mode="hits"``: produced CSV follows HIT_RESULTS_COLUMNS. The
  primary peak (``peak_id == 1``) per ``(event, branch)`` is matched to
  truth. Metrics: charge bias, charge scatter, timing σ (over
  ``t_peak - true_center_ns``), efficiency.

Composite score (lower is better):

    score = w_b*|bias| + w_s*scatter + w_t*timing_sigma_ns - w_e*efficiency

with sensible defaults (overrideable via ``ScoringWeights``). The
``efficiency`` term is **subtracted** so higher efficiency lowers
score — we want fewer rejected events.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from pmtdeconv.io.truth import load_truth


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the composite score. Lower score = better config."""
    w_bias:        float = 1.0   # |relative charge bias|
    w_scatter:     float = 1.0   # relative charge scatter
    w_timing:      float = 0.2   # timing σ in ns (hits mode only)
    w_efficiency:  float = 0.5   # subtracted — higher eff lowers score
    exclude_saturated: bool = True   # drop truth rows with saturated=True


DEFAULTS = ScoringWeights()


def score_run(produced_csv: Path | str,
              truth_csv: Path | str,
              mode: Literal["area", "hits"],
              weights: Optional[ScoringWeights] = None,
              ) -> Dict[str, float]:
    """Score one (produced, truth) pair.

    Returns a dict:
        n_truth                — count of truth rows (unsaturated)
        n_matched              — count of produced rows that matched truth
        efficiency             — n_matched / n_truth (0 if n_truth=0)
        charge_bias            — mean((q_found - q_true) / q_true)
        charge_scatter         — std((q_found - q_true) / q_true)
        charge_ratio_median    — median(q_found / q_true)  [scale check]
        timing_sigma_ns        — std(t_peak_found - true_center_ns)  [hits only]
        timing_bias_ns         — mean(t_peak_found - true_center_ns) [hits only]
        score                  — composite

    Missing data falls back to NaN so a sweep loop can carry on.
    """
    w = weights or DEFAULTS
    truth = load_truth(truth_csv)
    if w.exclude_saturated and "saturated" in truth.columns:
        truth = truth[~truth["saturated"]].copy()
    n_truth = int(len(truth))

    produced = pd.read_csv(produced_csv)
    if len(produced) == 0:
        return _nan_metrics(n_truth, mode)

    # ─── match by (event, branch) ──────────────────────────────────
    # AREA: produced has columns (run, cycle, entry, branch, interval,
    # area, peak_height). Truth has (event, branch, ...).
    # entry ↔ event (the synth fixture has one entry per event).
    if mode == "area":
        prod = (produced[produced["interval"] == 0]
                if "interval" in produced.columns else produced).copy()
        prod = prod.rename(columns={"entry": "event"})
        # Carry width_ns through the merge when present (emitted by
        # process_area's measure_width path, used by `tune cosine`).
        merge_cols = ["event", "branch", "area", "peak_height"]
        has_width = "width_ns" in prod.columns
        if has_width:
            merge_cols.append("width_ns")
        merged = truth.merge(
            prod[merge_cols],
            on=["event", "branch"], how="inner")
        if len(merged) == 0:
            return _nan_metrics(n_truth, mode)
        # area is on a different absolute scale (deconv output ∝ PE count);
        # we work in ratios so scale cancels. NaN/inf produced areas
        # (from degenerate integration windows in noisy regimes) are
        # filtered out before taking the median — otherwise a single
        # bad event leaks NaN through the whole composite score.
        ratio = merged["area"].values / merged["true_charge_pVs"].values
        good = np.isfinite(ratio)
        if not good.any():
            return _nan_metrics(n_truth, mode)
        ratio_clean = ratio[good]
        ratio_med = float(np.median(ratio_clean))
        rel_resid = ratio_clean / ratio_med - 1.0
        bias = float(np.mean(rel_resid))
        scatter = float(np.std(rel_resid))
        n_matched = int(good.sum())
        eff = n_matched / n_truth if n_truth > 0 else 0.0
        # Median reconstructed-pulse FWHM over the matched, finite-charge
        # rows. This is the secondary objective for `tune cosine`'s
        # ε-constraint cutoff selection (minimize width subject to charge
        # fidelity). NaN when no width column / no finite widths.
        width_fwhm = float("nan")
        if has_width:
            w_vals = merged["width_ns"].values[good]
            w_finite = w_vals[np.isfinite(w_vals)]
            if len(w_finite) > 0:
                width_fwhm = float(np.median(w_finite))
        out = dict(
            n_truth=n_truth, n_matched=n_matched, efficiency=eff,
            charge_bias=bias, charge_scatter=scatter,
            charge_ratio_median=ratio_med,
            width_fwhm_ns=width_fwhm,
            timing_sigma_ns=float("nan"),
            timing_bias_ns=float("nan"),
        )
    elif mode == "hits":
        # Pick the primary peak per (event, branch). With
        # --peak-order=tallest this is the largest peak.
        prod = produced.copy()
        if "peak_id" in prod.columns:
            prod = prod[prod["peak_id"] == 1].copy()
        prod = prod.rename(columns={"entry": "event"})
        merged = truth.merge(
            prod[["event", "branch", "t_peak", "charge", "height"]],
            on=["event", "branch"], how="inner")
        if len(merged) == 0:
            return _nan_metrics(n_truth, mode)
        # Filter out NaN/inf charges (degenerate integration windows
        # at low SNR — happens routinely on level1/level2 fixtures).
        ratio = merged["charge"].values / merged["true_charge_pVs"].values
        timing_resid_all = (merged["t_peak"].values
                            - merged["true_center_ns"].values)
        good = np.isfinite(ratio) & np.isfinite(timing_resid_all)
        if not good.any():
            return _nan_metrics(n_truth, mode)
        ratio_clean = ratio[good]
        timing_resid = timing_resid_all[good]
        ratio_med = float(np.median(ratio_clean))
        rel_resid = ratio_clean / ratio_med - 1.0
        bias = float(np.mean(rel_resid))
        scatter = float(np.std(rel_resid))
        timing_bias = float(np.mean(timing_resid))
        timing_sigma = float(np.std(timing_resid))
        n_matched = int(good.sum())
        eff = n_matched / n_truth if n_truth > 0 else 0.0
        out = dict(
            n_truth=n_truth, n_matched=n_matched, efficiency=eff,
            charge_bias=bias, charge_scatter=scatter,
            charge_ratio_median=ratio_med,
            timing_sigma_ns=timing_sigma,
            timing_bias_ns=timing_bias,
        )
    else:
        raise ValueError(f"mode must be 'area' or 'hits', got {mode!r}")

    out["score"] = _composite(out, w)
    return out


def _composite(metrics: Dict[str, float], w: ScoringWeights) -> float:
    """Combine metrics into a single score. Lower is better."""
    timing = metrics.get("timing_sigma_ns", float("nan"))
    timing_term = 0.0 if not np.isfinite(timing) else w.w_timing * timing
    return (w.w_bias * abs(metrics["charge_bias"])
            + w.w_scatter * metrics["charge_scatter"]
            + timing_term
            - w.w_efficiency * metrics["efficiency"])


def _nan_metrics(n_truth: int, mode: str) -> Dict[str, float]:
    """Return an all-NaN metric dict (preserves the schema so sweep
    loops can build a DataFrame without dropping cells)."""
    return dict(
        n_truth=n_truth, n_matched=0, efficiency=0.0,
        charge_bias=float("nan"), charge_scatter=float("nan"),
        charge_ratio_median=float("nan"),
        width_fwhm_ns=float("nan"),
        timing_sigma_ns=float("nan"), timing_bias_ns=float("nan"),
        score=float("inf"),
    )
