"""Per-hit output naming for the ``hits`` command.

Writes ``hits.csv``; reads ``hits.csv`` then the pre-rename
``deconv_pulse.csv`` / ``deconv_pulses.csv`` so older output directories
keep resolving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Manifest ROLE key — deliberately unchanged: it is also the key in the CSV
# schema registry (io/schemas.py) and the results index. Nobody types it.
DECONV_PULSE_ROLE = "deconv_pulse"

# The per-hit output file, named after the command that writes it (2026-09-21:
# `hits` wrote `deconv_pulse.csv`, so the command and its output disagreed).
DECONV_PULSE_CSV = "hits.csv"

# Earlier names, still READ so output directories from before each rename keep
# resolving. Newest first. Never written.
LEGACY_PULSE_CSVS = ("deconv_pulse.csv", "deconv_pulses.csv")


def write_deconv_pulse_csv(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write the per-hit CSV (``hits.csv``). Returns its path.

    Until 2026-09-14 this also wrote a byte-identical plural copy, duplicating
    the whole table on every run. Only the canonical name is written now;
    :func:`resolve_deconv_pulse_csv` still reads the older names.
    """
    canonical = Path(out_dir) / DECONV_PULSE_CSV
    canonical.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(canonical, index=False)
    return canonical


def resolve_deconv_pulse_csv(out_dir: Path) -> Optional[Path]:
    """The per-hit CSV in ``out_dir``: ``hits.csv``, else a legacy name."""
    out_dir = Path(out_dir)
    for name in (DECONV_PULSE_CSV, *LEGACY_PULSE_CSVS):
        cand = out_dir / name
        if cand.exists():
            return cand
    return None
