"""Reader for LANL ``*_offline.root`` pulse-list files.

These files (data/movedBar/MilliQan_Run868.*_triple_offline.root and
data/835/MilliQan_Run835.*_triple_offline.root) carry the MQAna
post-processing output: one tree named ``t`` with one **event** per
entry, and a jagged set of *pulse* fields per event (board, chan,
area, height, time, ipulse, npulses, …).

For the standard area + pulse-timing CSV outputs the user wants, we aggregate
per (event, board, channel):
- ``area``       = Σ over pulses (decision 1)
- ``peak_height``= max over pulses (decision 8)
- the **primary pulse** row (``ipulse == 0``) is also captured so the
  pulse-timing CSV carries per-channel timing for downstream ``timewalk``
  analysis (decision 11).

Branch labels follow the object-format reader convention
(decision 2): ``d0_ch{board*16 + chan:02d}`` so cross-subcommand
merges work without renaming.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import awkward as ak
import numpy as np
import uproot


# Fields read from the ``t`` tree. Time field is selectable by caller.
_BASE_FIELDS = ("event", "chan", "board", "area", "height", "ipulse")

# Canonical primary timing field (decision 12 default).
DEFAULT_TIME_FIELD = "timeFit_module_calibrated"
ALLOWED_TIME_FIELDS = (
    "time", "timeFit",
    "time_module_calibrated", "timeFit_module_calibrated",
    "time_module_finalCalibration", "timeFit_module_finalCalibration",
)

# How many pulses past ipulse==0 we still write into the pulse-timing CSV.
# Set to 1 for "primary pulse only"; downstream timewalk only uses
# peak_id == 1 anyway.
N_PRIMARY_PULSES = 1


@dataclass
class OfflineEventRow:
    """One row per (event, board, channel) — aggregated."""
    run:          int
    cycle:        int
    entry:        int     # global counter across sub-files
    branch:       str
    interval:     int
    area_sum:     float   # Σ pulse.area
    peak_height:  float   # max pulse.height
    # Primary-pulse fields (ipulse == 0). NaN if no ipulse==0 row.
    primary_area:   float
    primary_height: float
    primary_t_peak: float
    has_primary:    bool


def _branch_for(board: int, chan: int, n_channels_per_dig: int = 16) -> str:
    """``d0_ch{board*16 + chan:02d}`` — matches the object-format reader."""
    return f"d0_ch{board * n_channels_per_dig + chan:02d}"


def detect_offline_channels(
    root_path: Path,
    *,
    tree_name: str = "t",
    sample_n: int = 50,
) -> List[str]:
    """Return the channel labels (d0_chXX) that have at least one pulse
    in the first ``sample_n`` events. Useful for the auto-branch path.
    """
    seen = set()
    with uproot.open(root_path) as f:
        t_key = next((k for k in f.keys()
                      if k.split(";")[0] == tree_name), None)
        if t_key is None:
            raise RuntimeError(f"No '{tree_name}' tree in {root_path}")
        arr = f[t_key].arrays(["board", "chan"], entry_stop=sample_n,
                              library="ak")
        for i in range(len(arr["board"])):
            boards = np.asarray(ak.to_numpy(arr["board"][i]))
            chans = np.asarray(ak.to_numpy(arr["chan"][i]))
            for b, c in zip(boards, chans):
                seen.add(_branch_for(int(b), int(c)))
    return sorted(seen)


def iter_offline_run(
    root_path: Path,
    *,
    tree_name: str = "t",
    entry_stop: Optional[int] = None,
    time_field: str = DEFAULT_TIME_FIELD,
    branches: Optional[Iterable[str]] = None,
    start_entry: int = 0,
) -> Iterator[OfflineEventRow]:
    """Yield aggregated (event, branch) rows from one ``*_offline.root``
    file.

    Parameters
    ----------
    root_path : path
        The offline ROOT file.
    tree_name : str
        Tree name (default ``"t"``).
    entry_stop : int or None
        Cap on events read.
    time_field : str
        Which time field to write as primary t_peak. Must be one of
        :data:`ALLOWED_TIME_FIELDS`.
    branches : iterable of str or None
        Restrict output to these channel labels. ``None`` = all
        populated channels.
    start_entry : int
        Offset added to the per-file event index. Used by the command
        layer to maintain a **global** entry counter across sub-files
        (decision 5).
    """
    if time_field not in ALLOWED_TIME_FIELDS:
        raise ValueError(
            f"time_field={time_field!r} not in {ALLOWED_TIME_FIELDS}")

    wanted = set(branches) if branches is not None else None

    with uproot.open(root_path) as f:
        t_key = next((k for k in f.keys()
                      if k.split(";")[0] == tree_name), None)
        if t_key is None:
            raise RuntimeError(f"No '{tree_name}' tree in {root_path}")
        tree = f[t_key]
        n_total = tree.num_entries
        stop = min(entry_stop, n_total) if entry_stop is not None else n_total

        fields = list(_BASE_FIELDS) + [time_field]
        arr = tree.arrays(fields, entry_start=0, entry_stop=stop, library="ak")

        for i in range(stop):
            boards = np.asarray(ak.to_numpy(arr["board"][i]), dtype=np.int32)
            chans  = np.asarray(ak.to_numpy(arr["chan"][i]),  dtype=np.int32)
            areas  = np.asarray(ak.to_numpy(arr["area"][i]),  dtype=np.float64)
            heights = np.asarray(ak.to_numpy(arr["height"][i]), dtype=np.float64)
            ipulses = np.asarray(ak.to_numpy(arr["ipulse"][i]), dtype=np.int32)
            times = np.asarray(ak.to_numpy(arr[time_field][i]), dtype=np.float64)

            # Aggregate per (board, chan): sum(area), max(height), and
            # capture the ipulse==0 row's (area, height, time).
            # Use a dict keyed by (board, chan).
            seen = {}
            for j in range(len(boards)):
                key = (int(boards[j]), int(chans[j]))
                rec = seen.get(key)
                if rec is None:
                    rec = {
                        "area_sum": 0.0, "max_h": -np.inf,
                        "p_area": np.nan, "p_height": np.nan,
                        "p_time": np.nan, "has_primary": False,
                    }
                    seen[key] = rec
                rec["area_sum"] += float(areas[j])
                if heights[j] > rec["max_h"]:
                    rec["max_h"] = float(heights[j])
                if int(ipulses[j]) == 0 and not rec["has_primary"]:
                    rec["p_area"] = float(areas[j])
                    rec["p_height"] = float(heights[j])
                    rec["p_time"] = float(times[j])
                    rec["has_primary"] = True

            # Emit one row per (board, chan), with optional filter
            for (board, chan), rec in seen.items():
                branch = _branch_for(board, chan)
                if wanted is not None and branch not in wanted:
                    continue
                yield OfflineEventRow(
                    run=0,    # filled in by caller
                    cycle=1,
                    entry=start_entry + i,
                    branch=branch,
                    interval=0,
                    area_sum=rec["area_sum"],
                    peak_height=(rec["max_h"]
                                 if np.isfinite(rec["max_h"]) else np.nan),
                    primary_area=rec["p_area"],
                    primary_height=rec["p_height"],
                    primary_t_peak=rec["p_time"],
                    has_primary=rec["has_primary"],
                )


def count_offline_events(root_path: Path, tree_name: str = "t") -> int:
    """Return the number of events in the ``t`` tree (one entry = one event)."""
    with uproot.open(root_path) as f:
        t_key = next((k for k in f.keys()
                      if k.split(";")[0] == tree_name), None)
        if t_key is None:
            return 0
        return int(f[t_key].num_entries)
