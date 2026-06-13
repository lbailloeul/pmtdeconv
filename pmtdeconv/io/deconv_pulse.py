"""Canonical deconv-pulse output naming.

The user-facing name is ``deconv_pulse``.  The older plural
``deconv_pulses`` is kept as a compatibility filename and manifest role for
one transition period so existing scripts can keep reading old paths.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

DECONV_PULSE_ROLE = "deconv_pulse"
DECONV_PULSE_COMPAT_ROLE = "deconv_pulses"

DECONV_PULSE_CSV = "deconv_pulse.csv"
DECONV_PULSE_COMPAT_CSV = "deconv_pulses.csv"


def deconv_pulse_paths(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    return out_dir / DECONV_PULSE_CSV, out_dir / DECONV_PULSE_COMPAT_CSV


def write_deconv_pulse_csv(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write canonical CSV and a plural compatibility copy.

    Returns the canonical path. The compatibility file contains identical
    bytes so old consumers can transition without a schema split.
    """
    canonical, compat = deconv_pulse_paths(out_dir)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(canonical, index=False)
    shutil.copyfile(canonical, compat)
    return canonical


def resolve_deconv_pulse_csv(out_dir: Path) -> Optional[Path]:
    """Return canonical CSV if present, otherwise the plural legacy path."""
    canonical, compat = deconv_pulse_paths(out_dir)
    if canonical.exists():
        return canonical
    if compat.exists():
        return compat
    return None
