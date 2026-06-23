"""Shared result-table finalization for deconv-pulse pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from pmtdeconv.io.deconv_pulse import write_deconv_pulse_csv
from pmtdeconv.io.schemas import HIT_RESULTS_COLUMNS


@dataclass(frozen=True)
class DeconvPulseResultTables:
    all_rows: pd.DataFrame
    successful_rows: pd.DataFrame
    negative_charge_rows: pd.DataFrame
    deconv_pulse_csv: Path
    all_results_csv: Path
    negative_charge_csv: Optional[Path]


def write_deconv_pulse_result_tables(
    results: Iterable[Mapping],
    out_dir: Path,
) -> DeconvPulseResultTables:
    """Write the standard deconv-pulse result bundle.

    All pulse pipelines produce the same three logical tables:
    successful pulses, all attempted rows, and optional negative-charge
    rows. Keeping this in one place prevents the STFT, wavelet, and
    object-format loops from drifting.
    """
    out_dir = Path(out_dir)
    df = pd.DataFrame(results)
    # Enforce the canonical column order (schema is the single source of truth).
    # Failure rows that lack peak columns get NaN; this also lands the trailing
    # `saturated` flag consistently regardless of row-dict insertion order.
    if len(df):
        df = df.reindex(columns=list(HIT_RESULTS_COLUMNS))
    df_good = (
        df[df["success"]]
        if len(df) and "success" in df.columns else pd.DataFrame()
    )
    df_neg = (
        df[df["reason"] == "negative_charge"]
        if len(df) and "reason" in df.columns else pd.DataFrame()
    )

    out_csv = write_deconv_pulse_csv(df_good, out_dir)
    negative_csv = None
    if len(df_neg):
        negative_csv = out_dir / "deconv_negative_charge.csv"
        df_neg.to_csv(negative_csv, index=False)
    all_csv = out_dir / "all_results.csv"
    df.to_csv(all_csv, index=False)

    return DeconvPulseResultTables(
        all_rows=df,
        successful_rows=df_good,
        negative_charge_rows=df_neg,
        deconv_pulse_csv=out_csv,
        all_results_csv=all_csv,
        negative_charge_csv=negative_csv,
    )


def print_deconv_pulse_summary(tables: DeconvPulseResultTables,
                               *,
                               show_full_results: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Total rows: {len(tables.all_rows)}")
    if len(tables.all_rows) > 0 and "reason" in tables.all_rows.columns:
        print(tables.all_rows["reason"].value_counts())
    print(f"Successful pulses: {len(tables.successful_rows)}")
    print(f"Saved: {tables.deconv_pulse_csv}")
    if show_full_results:
        print(f"Full results: {tables.all_results_csv}")
    print(f"{'='*60}")
