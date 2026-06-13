"""Load/save the ground-truth CSV emitted by a regime fixture builder.

A truth CSV lives next to a regime fixture
(``tests/fixtures/regime/level<N>/flattened_Run0.truth.csv``) and
captures, for each (event, branch), the synthetic pulse's analytic
amplitude / center / charge / σ — plus a ``saturated`` flag for
clipped events. ``pmtdeconv tune`` reads these to score sweeps; the
``--diagnostic-events`` flag on ``pulse`` also loads them to compute
per-event truth-match if a fixture path is being processed.

Schema (frozen for PR 10; bump :attr:`TRUTH_SCHEMA_VERSION` to invalidate):

    event, branch, true_amp_mv, true_center_ns, true_charge_pVs,
    true_sigma_ns, saturated
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


TRUTH_SCHEMA_VERSION = "1.0"

REQUIRED_COLUMNS: List[str] = [
    "event",
    "branch",
    "true_amp_mv",
    "true_center_ns",
    "true_charge_pVs",
    "true_sigma_ns",
    "saturated",
]


class TruthSchemaError(ValueError):
    """Raised when a truth CSV is missing required columns."""


def load_truth(path: Path | str, strict: bool = True) -> pd.DataFrame:
    """Load a truth CSV. With ``strict``, raise on missing columns.

    Returns a DataFrame with the columns listed in
    :data:`REQUIRED_COLUMNS`. ``saturated`` is cast to bool.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"truth CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing and strict:
        raise TruthSchemaError(
            f"{path} is missing required columns {missing}. "
            f"Got {list(df.columns)}.")
    if "saturated" in df.columns:
        df["saturated"] = df["saturated"].astype(bool)
    return df


def save_truth(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a truth DataFrame, validating columns first."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TruthSchemaError(
            f"DataFrame missing required columns {missing}. "
            f"Got {list(df.columns)}.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[REQUIRED_COLUMNS].to_csv(path, index=False)
    return path


def find_truth_for(fixture_root: Path | str) -> Path | None:
    """Best-effort: given a fixture ROOT path, return its truth CSV
    sibling if one exists.

    e.g. ``tests/fixtures/regime/level3/flattened_Run0.root``
    →    ``tests/fixtures/regime/level3/flattened_Run0.truth.csv``
    """
    p = Path(fixture_root)
    sib = p.with_suffix(".truth.csv")
    if sib.exists():
        return sib
    # Try same-stem .truth.csv (handles cases where suffix replacement collides)
    alt = p.parent / (p.stem + ".truth.csv")
    if alt.exists():
        return alt
    return None
