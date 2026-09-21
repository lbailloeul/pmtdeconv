"""Score a (produced CSV, truth CSV) pair → metrics dict (PR 10).

Two scoring modes:

- ``mode="area"``: produced CSV follows AREA_COLUMNS. One row per
  ``(event, branch)`` — match directly to truth's ``(event, branch)``
  rows. Metrics: charge bias, charge scatter, efficiency.

- ``mode="hits"``: produced CSV follows HIT_RESULTS_COLUMNS. The
  primary peak (``peak_id == 1``) per ``(event, branch)`` is matched to
  truth — structurally single-pulse (multi-pulse truth needs the
  ``hits-multi`` mode). Metrics: charge bias, charge scatter, timing σ
  (over ``t_peak - true_center_ns``), efficiency.

Scale convention: ``charge_bias`` / ``charge_scatter`` are computed on
ratio residuals normalized by the MEDIAN produced/true ratio — they are
scale-free (an overall gain error does not show up in them by design).
``charge_ratio_median`` is the only absolute-scale check.

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


def match_pulses(true_times, found_times, tol_ns: float = 15.0,
                 exactly_n: Optional[int] = None) -> Dict:
    """Greedy nearest-time matching of truth pulses to found peaks.

    Promoted 2026-07-28 from ``tests/test_accuracy_floor.py::
    test_pileup_separation`` — the library multi-hit matcher. Each truth time
    (in input order) claims the closest UNUSED found peak within ``tol_ns``.

    Returns a dict:
        matched_pairs   — list of (i_true, j_found) index pairs
        n_true, n_found, n_matched
        residuals_ns    — found − true for each matched pair (np.ndarray)
        n_merged        — unmatched truth whose nearest peak was already
                          claimed by another truth pulse (two hits → one peak)
        n_missed        — unmatched truth with no peak within tol at all
        recognized      — ``exactly_n`` mode (the Tang et al. 2411.07620
                          criterion with exactly_n=2, tol_ns=2): True iff
                          exactly ``exactly_n`` peaks were found AND every
                          truth pulse matched within tol.
    """
    tt = np.asarray(true_times, dtype=float)
    ft = np.asarray(found_times, dtype=float)
    used: set = set()
    pairs = []
    resid = []
    n_merged = 0
    n_missed = 0
    for i, t in enumerate(tt):
        if ft.size == 0:
            n_missed += 1
            continue
        d = np.abs(ft - t)
        order = np.argsort(d)
        j = next((int(k) for k in order if int(k) not in used), None)
        if j is not None and d[j] < tol_ns:
            used.add(j)
            pairs.append((i, j))
            resid.append(float(ft[j] - t))
        elif float(d[order[0]]) < tol_ns:
            n_merged += 1          # nearest peak within tol but claimed
        else:
            n_missed += 1
    recognized = (exactly_n is not None
                  and ft.size == exactly_n
                  and len(pairs) == tt.size)
    return {"matched_pairs": pairs, "n_true": int(tt.size),
            "n_found": int(ft.size), "n_matched": len(pairs),
            "residuals_ns": np.asarray(resid, dtype=float),
            "n_merged": int(n_merged), "n_missed": int(n_missed),
            "recognized": bool(recognized)}


def score_run(produced_csv: Path | str,
              truth_csv: Path | str,
              mode: Literal["area", "hits", "hits-multi"],
              weights: Optional[ScoringWeights] = None,
              match_tol_ns: float = 15.0,
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

    # Multi-pulse truth (pulse_id column) auto-upgrades "hits" scoring to the
    # per-pulse mode — the peak_id==1 single-hit matcher caps efficiency at
    # ~1/n_pulses on pileup truth by construction (0–20 PE re-scope).
    if mode == "hits" and "pulse_id" in truth.columns:
        mode = "hits-multi"

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
    elif mode == "hits-multi":
        # Per-pulse scoring against multi-pulse truth (0–20 PE re-scope):
        # greedy time-matching per (event, branch), then per-pulse timing,
        # ABSOLUTE per-pulse charge accuracy (charge is honestly pVs and
        # truth is pVs since the 2026-07-28 linearization), split/merge rates.
        prod = produced.copy()
        if "success" in prod.columns:
            prod = prod[prod["success"] == True]     # noqa: E712
        prod = prod.rename(columns={"entry": "event"})
        has_thalf = "t_half" in prod.columns
        n_matched = 0
        n_found_total = 0
        n_merged = 0
        resid_tpeak: list = []
        resid_thalf: list = []
        ratios: list = []
        for (ev, br), tg in truth.groupby(["event", "branch"]):
            fg = prod[(prod["event"] == ev) & (prod["branch"] == br)]
            n_found_total += len(fg)
            m = match_pulses(tg["true_center_ns"].values,
                             fg["t_peak"].values, tol_ns=match_tol_ns)
            n_matched += m["n_matched"]
            n_merged += m["n_merged"]
            resid_tpeak.extend(m["residuals_ns"].tolist())
            for it, jf in m["matched_pairs"]:
                trow, frow = tg.iloc[it], fg.iloc[jf]
                r = float(frow["charge"]) / float(trow["true_charge_pVs"])
                if np.isfinite(r):
                    ratios.append(r)
                if has_thalf and np.isfinite(frow["t_half"]):
                    resid_thalf.append(float(frow["t_half"])
                                       - float(trow["true_center_ns"]))
        eff = n_matched / n_truth if n_truth else 0.0
        ratios_arr = np.asarray(ratios, dtype=float)
        rt = np.asarray(resid_tpeak, dtype=float)
        rh = np.asarray(resid_thalf, dtype=float)

        def _mad_sigma(x):
            return (float(1.4826 * np.median(np.abs(x - np.median(x))))
                    if x.size else float("nan"))

        if ratios_arr.size:
            ratio_med = float(np.median(ratios_arr))
            rel = ratios_arr / ratio_med - 1.0
            bias, scatter = float(np.mean(rel)), float(np.std(rel))
            ratio_mean = float(np.mean(ratios_arr))
        else:
            ratio_med = bias = scatter = ratio_mean = float("nan")
        out = dict(
            n_truth=n_truth, n_matched=n_matched, efficiency=eff,
            charge_bias=bias, charge_scatter=scatter,
            charge_ratio_median=ratio_med,
            # ABSOLUTE per-pulse charge scale (vs 1.0) — the linearity check
            # the median-normalized bias/scatter cannot see.
            charge_ratio_mean=ratio_mean,
            timing_sigma_ns=_mad_sigma(rt),
            timing_bias_ns=(float(np.mean(rt)) if rt.size else float("nan")),
            thalf_sigma_ns=_mad_sigma(rh),
            # found peaks beyond the matched ones, per truth pulse (fragments)
            split_rate=((n_found_total - n_matched) / n_truth
                        if n_truth else float("nan")),
            # truth pulses whose nearest peak was claimed by a neighbour
            # (two hits reconstructed as one)
            merge_rate=(n_merged / n_truth if n_truth else float("nan")),
        )
    else:
        raise ValueError(
            f"mode must be 'area', 'hits' or 'hits-multi', got {mode!r}")

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
        # hits-multi keys (harmless NaN for area/hits consumers)
        charge_ratio_mean=float("nan"), thalf_sigma_ns=float("nan"),
        split_rate=float("nan"), merge_rate=float("nan"),
        score=float("inf"),
    )
