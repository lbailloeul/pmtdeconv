"""Schema-agnostic SPE template loader.

Three schemas are accepted (all encountered in the repo):

1. **Stacked per-channel** — one row per channel, columns
   ``det, channel, tau_rise_ns, tau_fall_ns, sigma_ns``.
   Output of ``aggregate_spe_templates.py`` /
   ``outputs/biexp_fits_Cd109_lab_stacked/spe_template_per_channel*.csv``.
2. **Per-event (branch column)** — many rows per channel, columns
   ``branch, tau_rise, tau_fall, sigma, [success]``. Output of
   ``fitting/fitting_at_scale_Na22.py`` once aggregated per branch.
   We take the per-branch median.
3. **Per-event (det, chan columns)** — same as #2 but with separate
   ``det`` and ``chan`` columns instead of a combined ``branch``.

Returns a dict ``{branch: {"A", "tau_rise", "tau_fall", "sigma", "t0"}}``.
``A`` and ``t0`` are taken from a fallback template. ``A`` is mathematically
irrelevant (the kernel is unit-area normalized downstream). ``t0`` is NOT —
a nonzero t0 time-shifts the deconvolved output by −t0 (FFT(r) picks up
e^{-iω·t0}); correctness relies on this loader pinning ``t0 = DEFAULT_T0 = 0``.

Cross-reference: this is the unified version of three near-duplicate
loaders previously living in ``deconv_area_Na22.load_templates``,
``cosine_advanced_convolve_at_scale_Na22.load_channel_templates``, and
``wavelets_cosine_deconvolution_Na22.load_channel_templates``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Default fallback amplitudes — match ``deconv_area_Na22.FALLBACK_TEMPLATE``
# which uses ``MILLIQAN_TEMPLATE["A"] = 400.0``. A is mathematically irrelevant
# after unit-area kernel normalization, but floating-point intermediate
# magnitudes differ between A=1 and A=400 by 1 ulp on some entries — keeping
# A=400.0 lets the legacy and pmtdeconv pipelines produce *byte-identical*
# CSVs. t0 does NOT cancel (a nonzero t0 time-shifts u_rec by −t0):
# DEFAULT_T0 = 0.0 is load-bearing — never propagate a template file's own
# nonzero t0 through here.
DEFAULT_A = 400.0
DEFAULT_T0 = 0.0


class TemplateNotFoundError(FileNotFoundError):
    """Raised when a *named* template CSV path doesn't exist on disk.

    Distinct from `csv_path=None`, which means "no template configured
    for this dataset, fall back to a built-in" — that path returns an
    empty dict, not an exception. Decision 6: silent MilliQan fallback
    for a Na22 deconv produced visibly wrong outputs during the
    2026-05 session; we hard-fail now and point the user at the
    template-build command.
    """


def load_template(
    csv_path: Optional[Path],
    fallback_A: float = DEFAULT_A,
    fallback_t0: float = DEFAULT_T0,
    verbose: bool = True,
    strict: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Load per-channel SPE shape from a fit CSV.

    Behavior:
    - ``csv_path is None`` → return empty dict (no template configured;
      callers fall back to ``utils.LANL_TEMPLATE`` / ``MILLIQAN_TEMPLATE``).
    - ``csv_path`` set but file missing → raise ``TemplateNotFoundError``
      with a pointer to ``pmtdeconv template build && aggregate``.
      Pass ``strict=False`` to revert to the old silent-fallback behavior
      (used in tests and when a caller knows the fallback is acceptable).
    - File present → load + return per-branch dict.
    """
    templates: Dict[str, Dict[str, float]] = {}
    if csv_path is None:
        return templates
    if not Path(csv_path).exists():
        if strict:
            raise TemplateNotFoundError(
                f"SPE template CSV not found: {csv_path}\n"
                f"  Build the Cd109 stacked template first:\n"
                f"      pmtdeconv template build --dataset cd109\n"
                f"      pmtdeconv template aggregate\n"
                f"  Or pass --template <path> with a CSV you already have."
            )
        if verbose:
            print(f"[warn] no fit CSV at {csv_path} — using fallback template")
        return templates

    df = pd.read_csv(csv_path)

    # --- Schema 4 (2026-07-28): SAMPLED kernel ------------------------
    # Long-format columns ``branch, t_ns, r`` — an arbitrary measured/
    # simulated 1-PE waveform (e.g. the Tang-2024 MC template with
    # undershoot + reflections, or a future measured single-peak kernel).
    # ``build_spe_template`` resamples it onto the run grid and unit-area
    # normalizes — the same normalization contract as the biexp path, so
    # deconvolved charge stays pVs.
    if {"t_ns", "r"}.issubset(df.columns):
        if "branch" in df.columns:
            groups = list(df.groupby("branch"))
        else:
            groups = [("d0_ch1", df)]
        for branch, grp in groups:
            grp = grp.sort_values("t_ns")
            templates[str(branch)] = dict(
                kind="sampled",
                t_ns=grp["t_ns"].to_numpy(dtype=float),
                y=grp["r"].to_numpy(dtype=float),
            )
        if verbose:
            print(f"Loaded {len(templates)} SAMPLED kernel template(s) "
                  f"from {csv_path}")
        return templates

    # --- Schema 1: stacked per-channel (Cd109 stacked-fit) -----------
    # Two on-disk variants both land here (both carry ``tau_rise_ns``):
    #   (a) Cd109 stacked fit: separate ``det`` column + NUMERIC ``channel``
    #       (e.g. det="d0", channel=1)  -> branch "d0_ch1".
    #   (b) ``template aggregate`` output (spe_templates_perchannel.csv):
    #       ``channel`` is already the FULL branch name "d0_ch1" and there is
    #       NO ``det`` column. Without (b)'s handling, `hits --template
    #       <aggregate csv>` died with KeyError('det').
    if "tau_rise_ns" in df.columns:
        ch_col = "channel" if "channel" in df.columns else "chan"
        has_det = "det" in df.columns
        for _, row in df.iterrows():
            raw = str(row[ch_col])
            if "ch" in raw:                       # (b) full branch name e.g. "d0_ch1"
                branch = raw
            elif has_det and pd.notna(row["det"]):  # (a) Cd109 det + numeric chan
                branch = f"{row['det']}_ch{int(float(raw))}"
            else:                                 # bare numeric, no det -> assume d0
                branch = f"d0_ch{int(float(raw))}"
            templates[branch] = dict(
                A=fallback_A,
                tau_rise=float(row["tau_rise_ns"]),
                tau_fall=float(row["tau_fall_ns"]),
                sigma=float(row["sigma_ns"]),
                t0=fallback_t0,
            )
        if verbose:
            print(f"Loaded {len(templates)} stacked per-channel templates "
                  f"from {csv_path}")
        return templates

    # --- Schema 2: per-event with branch column ----------------------
    if {"branch", "tau_rise", "tau_fall", "sigma"}.issubset(df.columns):
        if "success" in df.columns:
            df = df[df["success"] == True]
        for branch, grp in df.groupby("branch"):
            templates[str(branch)] = dict(
                A=fallback_A,
                tau_rise=float(grp["tau_rise"].median()),
                tau_fall=float(grp["tau_fall"].median()),
                sigma=float(grp["sigma"].median()),
                t0=fallback_t0,
            )
        if verbose:
            print(f"Loaded {len(templates)} branch templates "
                  f"(per-event median) from {csv_path}")
        return templates

    # --- Schema 3: per-event with det+chan columns -------------------
    if "success" in df.columns:
        df = df[df["success"] == True]
    if {"det", "chan", "tau_rise", "tau_fall", "sigma"}.issubset(df.columns):
        for (det, ch), grp in df.groupby(["det", "chan"]):
            branch = f"{det}_ch{int(ch)}"
            templates[branch] = dict(
                A=fallback_A,
                tau_rise=float(grp["tau_rise"].median()),
                tau_fall=float(grp["tau_fall"].median()),
                sigma=float(grp["sigma"].median()),
                t0=fallback_t0,
            )
        if verbose:
            print(f"Loaded {len(templates)} channel templates "
                  f"(per-event median, det+chan schema) from {csv_path}")
        return templates

    raise ValueError(
        f"Cannot interpret template schema in {csv_path}. "
        f"Expected one of: tau_rise_ns / branch+tau_rise / det+chan+tau_rise.")


def load_templates_by_channel(
    mapping: Dict[str, Path], *, strict: bool = True, verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Resolve a per-channel CSV mapping ``{branch: csv_path}`` into one
    ``{branch: template}`` dict — each channel gets the SPE shape from its OWN
    CSV (the per-tube transferable artifact, S9).

    A per-tube CSV usually holds one channel; we assign its template to the
    target ``branch`` regardless of the CSV's internal channel key. If a CSV
    holds several channels, the one whose key matches ``branch`` is used; if it
    holds several and none match, that is an error (ambiguous — split it).
    """
    out: Dict[str, Dict[str, float]] = {}
    for branch, csv in mapping.items():
        loaded = load_template(Path(csv), strict=strict, verbose=verbose)
        if branch in loaded:
            out[branch] = loaded[branch]
        elif len(loaded) == 1:
            out[branch] = next(iter(loaded.values()))
        elif len(loaded) == 0:
            if strict:
                raise TemplateNotFoundError(
                    f"per-channel template for {branch!r}: no template in {csv}")
        else:
            raise ValueError(
                f"per-channel template {csv} for {branch!r} holds multiple "
                f"channels {sorted(loaded)} and none is named {branch!r}; "
                f"split the CSV or rename its channel.")
    if verbose and out:
        print(f"Loaded {len(out)} per-channel (per-tube) templates: "
              f"{', '.join(sorted(out))}")
    return out
