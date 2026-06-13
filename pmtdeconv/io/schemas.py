"""CSV column schemas for every file pmtdeconv emits.

Three reasons this module exists:

1. **Single source of truth** for column order. Every command imports the
   relevant tuple instead of inline-listing columns, so an accidental
   reorder fails at lint/test time.

2. **Versioning.** Each schema has a stable ``version`` string that
   bumps on breaking changes. The manifest records which version was
   used per output so downstream tools can guard with
   ``csv_schemas[role]["version"] == "1.0"``.

3. **Discoverability.** ``pmtdeconv info template <csv>`` (PR 1) and
   future ``info schema <role>`` commands look up entries here.

Bump policy:
- Add a column at the END: minor schema change, no version bump
  (downstream readers ignore unknown trailing columns).
- Rename / remove / reorder a column: major bump (``2.0``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CsvSchema:
    name: str             # "area", "hit_results", ...
    version: str          # "1.0"
    columns: Tuple[str, ...]
    description: str


# ─── area CSVs (shared by waveform, raw_area mqana, raw_area drs) ─

AREA_COLUMNS: Tuple[str, ...] = (
    "run", "cycle", "entry", "branch", "interval", "area", "peak_height",
)

AREA_SCHEMA = CsvSchema(
    name="area",
    version="1.0",
    columns=AREA_COLUMNS,
    description=(
        "Per-event integrated area. Used by `waveform` (FFT/wavelet "
        "summary over the signal window), `raw_area mqana` (raw narrow-window "
        "around the per-event peak), and `raw_area drs` (precomputed "
        "DRSprocessing tree branch). Area in mV·ns; `interval` indexes "
        "sub-windows when `--n-intervals > 1` (always 0 for mqana/drs)."
    ),
)


# ─── deconv-pulse results (peak-finding output) ───────────────────

HIT_RESULTS_COLUMNS: Tuple[str, ...] = (
    "run", "cycle", "det", "chan", "branch", "entry",
    "success", "reason", "peak_id",
    "t_peak", "height", "charge",
    "win_lo_ns", "win_hi_ns",
    "baseline_ped", "baseline_sig",
    "baseline_t0", "baseline_t1",
    "u_rec_noise",
)

HIT_RESULTS_SCHEMA = CsvSchema(
    name="hit_results",
    version="1.0",
    columns=HIT_RESULTS_COLUMNS,
    description=(
        "Per-peak deconvolution output from `pulse`. "
        "`peak_id` orders peaks chronologically (or by amplitude when "
        "--peak-order=tallest). `reason` is empty for successful peaks "
        "and one of out_of_time / afterpulse_veto / post_sat_shift / "
        "negative_charge / no_raw_peak / no_veto / no_peaks_found for "
        "rejected ones."
    ),
)


# ─── timewalk summary curve (PR 2) ───────────────────────────────

TIMEWALK_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "ch2_lo", "ch2_hi", "ch2_center", "N",
    "mean_dt", "sigma_dt", "mean_err", "sigma_err",
)

TIMEWALK_SUMMARY_SCHEMA = CsvSchema(
    name="timewalk_summary",
    version="1.0",
    columns=TIMEWALK_SUMMARY_COLUMNS,
    description=(
        "Per-(a_scan-window) timewalk curve in MQAna format. ch2_lo/hi "
        "are in mV·ns (matching the area schema); mean_dt/sigma_dt are "
        "Gaussian fit parameters in ns. Reference: ELOG E41473."
    ),
)


# ─── registry indexed by `role` (matches manifest's outputs[].role) ──

REGISTRY: Dict[str, CsvSchema] = {
    # area CSVs — all three sources share the same schema
    "area_csv":          AREA_SCHEMA,
    "area_mqana_csv":    AREA_SCHEMA,
    "area_drs_csv":      AREA_SCHEMA,
    # pulse. The plural role is retained as a compatibility alias.
    "deconv_pulse": HIT_RESULTS_SCHEMA,
    "deconv_pulses": HIT_RESULTS_SCHEMA,
    "all_results":        HIT_RESULTS_SCHEMA,
    # timewalk (PR 2)
    "timewalk_summary":   TIMEWALK_SUMMARY_SCHEMA,
}


def schema_for(role: str) -> CsvSchema:
    """Return the CsvSchema for a manifest output role.

    Raises KeyError if the role is unknown — bump the registry above
    when adding new output kinds.
    """
    if role not in REGISTRY:
        raise KeyError(
            f"No schema registered for role {role!r}. "
            f"Known: {sorted(REGISTRY)}.")
    return REGISTRY[role]


def version_map(roles_present: Tuple[str, ...]) -> Dict[str, str]:
    """Return ``{role: version}`` for the schemas covering the given roles.

    Used by ``write_manifest`` to populate the ``csv_schema_versions``
    manifest field. Unknown roles are silently skipped (they're
    presumably non-CSV outputs like PDFs or PNGs).
    """
    out: Dict[str, str] = {}
    for r in roles_present:
        if r in REGISTRY:
            out[r] = REGISTRY[r].version
    return out
