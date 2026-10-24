"""Shared helpers for the three ``pmtdeconv tune`` leaves.

Three pieces of common logic:

- :func:`ensure_regime_fixture` — build the regime synth fixture on
  demand at a requested dt_ns, rebuild if the cached dt doesn't match.
  Returns the ROOT path.
- :func:`apply_drift_bias` — post-pass over a per-cell sweep CSV that
  rewrites the ``charge_bias`` column as drift vs a per-regime reference
  config (Q2), then recomputes the composite score.
- :func:`save_sweep_curves` — per-regime 3-panel PNG showing scatter /
  drift / efficiency vs the swept axis, with the winner marked (Q3).
- :func:`write_tuned_presets` — idempotent read/merge/rewrite of
  ``pmtdeconv/presets/tuned_configs.py`` so re-running tune doesn't
  duplicate entries (Q10).
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.pipeline.scoring import ScoringWeights


# ─── regime fixture management ─────────────────────────────────────


def area_denoiser(settings, provenance) -> str:
    """The denoiser ``tune cosine`` / ``tune fom`` actually run: the area path
    has only the wavelet, so a preset's STFT tunes with the wavelet and the
    manifest records that, not a CLI choice nobody made."""
    if settings.denoiser not in ("none", "wavelet"):
        settings.denoiser = "wavelet"
        provenance["denoiser"] = "default (area path has no STFT)"
    return settings.denoiser

def regime_fixture(regime: str) -> Path:
    """tests/fixtures/regime/<regime>/flattened_Run0.root"""
    return (PROJECT_ROOT / "tests" / "fixtures" / "regime"
            / regime / "flattened_Run0.root")


def ensure_regime_fixture(regime: str, dt_ns: float,
                          n_events: Optional[int] = None) -> Path:
    """Build the regime fixture if missing or stale. Returns the path.

    Stale = the cached ``.dt_ns`` sentinel disagrees with the requested
    dt. Rebuilding is cheap (<1 s) and uses the seeded RNG so the bytes
    are deterministic. Q4: nothing about the fixture is committed; we
    materialize into ``tests/fixtures/regime/<regime>/`` on demand.
    """
    fixture = regime_fixture(regime)
    sentinel = fixture.parent / ".dt_ns"
    fixture_ok = fixture.exists() and (
        (sentinel.exists() and float(sentinel.read_text().strip()) == dt_ns)
        and n_events is None  # rebuild if user explicitly passed n_events
    )
    if fixture_ok:
        return fixture
    import sys
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    sys.path.insert(0, str(fixtures_dir))
    try:
        msr = importlib.import_module("make_synth_root")
        importlib.reload(msr)
        msr.main(regime=regime,
                 n_events=n_events or msr.N_EVENTS,
                 write_truth=True,
                 dt_ns=dt_ns)
    finally:
        if str(fixtures_dir) in sys.path:
            sys.path.remove(str(fixtures_dir))
    return fixture


# ─── reference-based drift (Q2) ─────────────────────────────────────

def apply_drift_bias(sweep_df: pd.DataFrame,
                     axis_cols: Iterable[str],
                     reference_lookup: Callable[[str, pd.DataFrame],
                                                Optional[pd.Series]],
                     weights: ScoringWeights = ScoringWeights(),
                     ) -> pd.DataFrame:
    """Rewrite ``charge_bias`` as drift vs per-regime reference; recompute
    ``score``.

    Per regime:
      1. ``reference_lookup(regime, regime_subdf)`` returns the row to use
         as reference (or None to skip the regime).
      2. ``drift = charge_ratio_median[row] / ratio_median[ref] - 1``
      3. ``score = w_bias*|drift| + w_scatter*scatter + w_timing*tσ
                   - w_efficiency*eff``

    The original ``charge_bias`` column is preserved in
    ``charge_bias_residual`` for inspection.
    """
    if "charge_bias" in sweep_df.columns:
        sweep_df = sweep_df.rename(columns={"charge_bias": "charge_bias_residual"})
    sweep_df["charge_bias"] = float("nan")

    for regime, sub in sweep_df.groupby("regime"):
        ref_row = reference_lookup(regime, sub)
        ref_ratio: Optional[float] = None
        if ref_row is not None:
            cand = float(ref_row["charge_ratio_median"])
            if np.isfinite(cand) and cand != 0.0:
                ref_ratio = cand

        # Fallback (Sanity 4 fix): the dataset preset's config can be
        # *degenerate* in some regimes — e.g. wavelet hard/k=1.0 zeros
        # the deconv signal at low SNR. Rather than leaving drift = NaN
        # for the whole regime, fall back to the median ratio across
        # the regime's non-degenerate cells, so drift still measures
        # "how this cell compares to the regime's typical scale."
        if ref_ratio is None:
            finite_ratios = sub["charge_ratio_median"].astype(float)
            finite_ratios = finite_ratios[
                np.isfinite(finite_ratios) & (finite_ratios != 0.0)]
            if len(finite_ratios) > 0:
                ref_ratio = float(np.median(finite_ratios))
                print(f"  [drift-ref] {regime}: dataset reference config "
                      f"is degenerate (ratio_med = 0 or NaN); fell back to "
                      f"regime median ratio_med = {ref_ratio:.3e}.")

        if ref_ratio is None or ref_ratio == 0.0:
            continue
        idx = sub.index
        ratios = sweep_df.loc[idx, "charge_ratio_median"].astype(float)
        sweep_df.loc[idx, "charge_bias"] = ratios / ref_ratio - 1.0

    # Recompute composite score with the new bias definition.
    drift = sweep_df["charge_bias"].astype(float).abs().fillna(0)
    scatter = sweep_df["charge_scatter"].astype(float).fillna(0)
    timing = sweep_df.get("timing_sigma_ns",
                          pd.Series(np.nan, index=sweep_df.index)
                          ).astype(float)
    timing_term = timing.where(timing.notna(), 0.0) * weights.w_timing
    eff = sweep_df["efficiency"].astype(float).fillna(0)
    sweep_df["score"] = (weights.w_bias * drift
                         + weights.w_scatter * scatter
                         + timing_term
                         - weights.w_efficiency * eff)
    # Sanity-2 fix: zero-efficiency cells produced no usable data; force
    # +inf so they can't masquerade as winners. Also catches cells whose
    # drift/scatter came back NaN because n_matched=0 (their .fillna(0)
    # would otherwise zero those terms and leave score = -w_eff*0 = 0).
    no_data_mask = (eff == 0.0) | (sweep_df["n_matched"].fillna(0) == 0)
    sweep_df.loc[no_data_mask, "score"] = float("inf")
    return sweep_df


def closest_reference(reference_value: float, axis_col: str
                      ) -> Callable[[str, pd.DataFrame], Optional[pd.Series]]:
    """Reference-row lookup helper: pick the sub-row whose ``axis_col``
    is closest to ``reference_value``. Used by tune cosine (axis =
    cosine_cutoff_mhz, reference = dataset preset's cutoff).

    If the reference value is in the grid exactly, that row is returned.
    Otherwise the nearest is used and a warning is printed.
    """
    def _lookup(regime: str, sub: pd.DataFrame) -> Optional[pd.Series]:
        if len(sub) == 0:
            return None
        diffs = (sub[axis_col].astype(float) - reference_value).abs()
        idx = diffs.idxmin()
        ref = sub.loc[idx]
        if float(diffs.loc[idx]) > 1e-9:
            print(f"  [drift-ref] {regime}: reference {axis_col}="
                  f"{reference_value} not in sweep; using "
                  f"{axis_col}={ref[axis_col]} (closest).")
        return ref
    return _lookup


def exact_reference(reference_values: Mapping[str, object]
                    ) -> Callable[[str, pd.DataFrame], Optional[pd.Series]]:
    """Reference-row lookup: pick the sub-row whose columns equal
    ``reference_values`` exactly (multi-axis case for wavelet/STFT).
    Returns None if no exact match — caller should fall back to a
    median-of-regime reference if needed.
    """
    def _lookup(regime: str, sub: pd.DataFrame) -> Optional[pd.Series]:
        mask = pd.Series(True, index=sub.index)
        for k, v in reference_values.items():
            if k not in sub.columns:
                return None
            mask &= sub[k] == v
        hits = sub[mask]
        if len(hits) == 0:
            print(f"  [drift-ref] {regime}: no exact match for "
                  f"{dict(reference_values)} in sweep; bias = NaN.")
            return None
        return hits.iloc[0]
    return _lookup


# ─── sweep curve plot (Q3) ──────────────────────────────────────────

def save_sweep_curves(sweep_df: pd.DataFrame, axis_col: str,
                      out_dir: Path, axis_label: str,
                      winner_axis_value_by_regime: Mapping[str, float],
                      drift_band: float = 0.005,
                      ) -> List[Path]:
    """Per-regime 3-panel PNG of the RANKED metrics, in priority order:
    efficiency (1) / charge drift (2, with ±drift_band shaded) / FWHM (3).
    The winner cutoff is a red dotted line on each panel.
    """
    out_dir = Path(out_dir)
    written: List[Path] = []
    for regime, sub in sweep_df.groupby("regime"):
        sub_sorted = sub.sort_values(axis_col)
        if len(sub_sorted) == 0:
            continue
        from pmtdeconv.plotting.style import use_root_style, style_root_axes
        use_root_style()
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
        x = sub_sorted[axis_col].astype(float).values
        eff = sub_sorted["efficiency"].astype(float).values
        drift = sub_sorted["charge_bias"].astype(float).values

        axes[0].plot(x, eff * 100, "o-", color="C3", ms=4)
        axes[0].set_ylabel("efficiency [%]")

        axes[1].axhspan(-drift_band * 100, drift_band * 100,
                        color="green", alpha=0.12)
        axes[1].plot(x, drift * 100, "o-", color="C2", ms=4)
        axes[1].axhline(0, color="gray", lw=0.5, ls="--")
        axes[1].set_ylabel("charge drift [%]")

        # 3rd panel = reconstructed FWHM (cosine sweep) or charge scatter
        # (STFT/wavelet sweeps don't measure width).
        if "width_fwhm_ns" in sub_sorted and \
                sub_sorted["width_fwhm_ns"].notna().any():
            fwhm = sub_sorted["width_fwhm_ns"].astype(float).values
            axes[2].plot(x, fwhm, "o-", color="C0", ms=4)
            axes[2].set_ylabel("FWHM [ns]")
            if np.nanmax(fwhm) / max(np.nanmin(fwhm), 1e-9) > 10:
                axes[2].set_yscale("log")
        else:
            scat = sub_sorted["charge_scatter"].astype(float).values * 100.0
            axes[2].plot(x, scat, "o-", color="C0", ms=4)
            axes[2].set_ylabel("charge scatter [%]")
        axes[2].set_xlabel(axis_label)

        winner_x = winner_axis_value_by_regime.get(regime)
        if winner_x is not None and np.isfinite(winner_x):
            for ax in axes:
                ax.axvline(float(winner_x), color="red", lw=1.0, ls=":")
            axes[0].set_title(f"winner = {winner_x:g} MHz", fontsize=11)
        for ax in axes:
            style_root_axes(ax)

        fig.tight_layout()
        png_path = out_dir / f"sweep_curves_{regime}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        written.append(png_path)
    return written


# ─── idempotent preset writer (Q10) ─────────────────────────────────

_DATE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})-")


def _collapse_tuned(merged: Dict[str, dict]) -> Dict[str, dict]:
    """Collapse the accumulated auto-tuned presets (Step 8 hygiene).

    The writer keys entries by date-stamped name, so re-running ``tune`` on
    new dates piled up ~identical snapshots (e.g. six ``na22-cosine-tuned-*-
    level3`` all converging to cosine 100 MHz). This:

    1. drops prior ``*-latest-*`` aliases (regenerated below);
    2. keeps, per ``(base_dataset, frozen overrides)`` signature, only the
       OLDEST dated snapshot — its date = when this result FIRST won. Dated
       names are immutable provenance anchors (manifests record them), so a
       re-run that reproduces the current winner must be a NO-OP, never a
       rename. (Until 2026-07-02 "newest wins" re-stamped the entry with the
       re-run's date: the old name vanished, manifests referencing it stopped
       resolving, and the date no longer said when the numbers were first
       measured.)
    3. emits a stable ``<prefix>-tuned-latest-<suffix>`` alias per name-family
       (date stripped) pointing at the most recent DISTINCT winner — always
       via its retained (first-appearance) name.

    Non-dated, non-alias entries are passed through untouched.
    """
    dated = {n: p for n, p in merged.items() if _DATE_RE.search(n)}
    passthrough = {n: p for n, p in merged.items()
                   if not _DATE_RE.search(n) and "-latest-" not in n}

    def _sig(p: dict) -> tuple:
        return (p["base_dataset"], tuple(sorted(p["overrides"].items())))

    # (2) oldest dated snapshot per identical-config signature
    by_sig: Dict[tuple, str] = {}
    for name in sorted(dated):           # ascending date → FIRST occurrence wins
        by_sig.setdefault(_sig(dated[name]), name)
    kept = {dated[n]["name"]: dated[n] for n in set(by_sig.values())}

    # (3) one `-latest-` alias per family. The newest dated NAME per family
    # picks the winning config; the alias references that config's RETAINED
    # snapshot so it never points at a collapsed (deleted) name.
    families: Dict[str, dict] = {}
    for name in sorted(dated):           # ascending → newest wins per family
        families[_DATE_RE.sub("-latest-", name)] = dated[by_sig[_sig(dated[name])]]
    aliases = {
        fam: {"name": fam, "base_dataset": p["base_dataset"],
              "overrides": dict(p["overrides"]),
              "description": f"Alias → latest tune winner ({p['name']})."}
        for fam, p in families.items()
    }
    return {**passthrough, **kept, **aliases}


def tuned_configs_path() -> Path:
    """Where ``tune --write-presets`` writes the generated presets.

    ``PMTDECONV_TUNED_CONFIGS`` overrides it; otherwise the module inside the
    package, which is where ``presets/__init__`` imports it from.

    The override exists because the default writes INTO the package: in a
    source checkout that dirties a tracked file (the test suite used to leave
    a spurious diff for every collaborator who ran pytest), and in a
    pip-installed environment it targets site-packages — read-only for some
    installs, and discarded on the next upgrade for the rest. Redirecting the
    path keeps the generated presets out of both.
    """
    override = os.environ.get("PMTDECONV_TUNED_CONFIGS")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / "pmtdeconv" / "presets" / "tuned_configs.py"


def write_tuned_presets(new_entries: Dict[str, dict]) -> int:
    """Read tuned_configs.py, merge ``new_entries`` (overwriting same
    names), re-emit the whole file from scratch with deterministic
    per-entry blocks.

    Each value in ``new_entries`` is a dict with keys: ``name``,
    ``base_dataset``, ``overrides`` (dict), ``description``. Returns
    the number of entries in the final TUNED_CONFIGS.
    """
    from pmtdeconv.presets.tuned_configs import TUNED_CONFIGS  # in-memory
    importlib.reload(importlib.import_module("pmtdeconv.presets.tuned_configs"))
    from pmtdeconv.presets.tuned_configs import (
        TUNED_CONFIGS as fresh_configs,
    )
    # Merge: convert existing ConfigPreset objects to dicts for serialization
    merged: Dict[str, dict] = {}
    for name, cfg in fresh_configs.items():
        merged[name] = {
            "name": cfg.name,
            "base_dataset": cfg.base_dataset,
            "overrides": dict(cfg.overrides),
            "description": cfg.description,
        }
    for name, payload in new_entries.items():
        merged[name] = payload

    final = _collapse_tuned(merged)

    # Re-emit
    lines = [
        '"""Auto-generated named config presets — written by '
        '``pmtdeconv tune``.\n\n'
        'DO NOT EDIT BY HAND. Regenerated idempotently each time '
        '``--write-presets``\n'
        'runs: read existing entries, merge new ones, collapse identical-config\n'
        'dated snapshots (keep the oldest per config) and emit a stable\n'
        '``<prefix>-tuned-latest-<suffix>`` alias tracking the most recent\n'
        'winner, then rewrite the whole file.\n"""\n',
        "from __future__ import annotations\n",
        "from typing import Dict\n",
        "from pmtdeconv.presets.configs import ConfigPreset\n",
        "\n",
        "TUNED_CONFIGS: Dict[str, ConfigPreset] = {}\n",
        "\n",
    ]
    for name in sorted(final):
        p = final[name]
        lines.append(
            f"\nTUNED_CONFIGS[{p['name']!r}] = ConfigPreset(\n"
            f"    name={p['name']!r},\n"
            f"    base_dataset={p['base_dataset']!r},\n"
            f"    overrides={p['overrides']!r},\n"
            f"    description={p['description']!r},\n"
            f")\n"
        )
    tuned_path = tuned_configs_path()
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text("".join(lines))
    return len(final)
