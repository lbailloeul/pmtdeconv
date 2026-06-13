#!/usr/bin/env python3
"""Build a deterministic synthetic LANL ``*_offline.root`` fixture.

Mirrors the layout of `data/movedBar/MilliQan_Run868.*_triple_offline.root`:
- tree ``t``
- one entry per event
- per-event jagged fields: ``event, chan, board, area, height, ipulse,
  npulses, time, timeFit, time_module_calibrated, …``

We write ~50 events × 4 populated channels (across 2 boards) × ~5
pulses each. Fixed RNG seed for byte-reproducible goldens.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


N_EVENTS = 50
POPULATED_BOARDS_CHANS = [(0, 2), (0, 4), (1, 0), (1, 4)]
MIN_PULSES, MAX_PULSES = 3, 8
SEED = 42

# Time fields written verbatim (offline_format reader honors any of these
# via --time-field).
TIME_FIELDS = (
    "time", "timeFit",
    "time_module_calibrated", "timeFit_module_calibrated",
    "time_module_finalCalibration", "timeFit_module_finalCalibration",
)


def main():
    out_path = Path(__file__).resolve().parent / "MilliQan_Run99.synth_triple_offline.root"
    rng = np.random.default_rng(SEED)

    # Per-event jagged arrays
    per_event = {
        "event":   [],
        "chan":    [],
        "board":   [],
        "area":    [],
        "height":  [],
        "ipulse":  [],
        "npulses": [],
        "runNumber": [],
    }
    for tf in TIME_FIELDS:
        per_event[tf] = []

    for evt in range(N_EVENTS):
        chans, boards, areas, heights, ipulses, npulses_field = [], [], [], [], [], []
        times = {tf: [] for tf in TIME_FIELDS}
        for (board, chan) in POPULATED_BOARDS_CHANS:
            n_p = int(rng.integers(MIN_PULSES, MAX_PULSES + 1))
            # First pulse = the primary; subsequent are afterpulses
            primary_area = float(rng.uniform(500.0, 2000.0))
            primary_height = float(rng.uniform(150.0, 800.0))
            primary_time = float(rng.uniform(100.0, 600.0))
            for ip in range(n_p):
                if ip == 0:
                    a, h = primary_area, primary_height
                    base_time = primary_time
                else:
                    a = float(rng.normal(0.0, 200.0))
                    h = float(rng.uniform(5.0, 80.0))
                    base_time = primary_time + float(rng.uniform(20.0, 800.0))
                chans.append(chan)
                boards.append(board)
                areas.append(a)
                heights.append(h)
                ipulses.append(ip)
                npulses_field.append(n_p)
                # Each "calibration level" applies a small fixed offset
                # so we can tell them apart in tests.
                times["time"].append(base_time)
                times["timeFit"].append(base_time + 0.05)
                times["time_module_calibrated"].append(base_time + 0.10)
                times["timeFit_module_calibrated"].append(base_time + 0.15)
                times["time_module_finalCalibration"].append(base_time + 0.20)
                times["timeFit_module_finalCalibration"].append(base_time + 0.25)
        per_event["event"].append(evt)
        per_event["chan"].append(np.array(chans, dtype=np.int32))
        per_event["board"].append(np.array(boards, dtype=np.int32))
        per_event["area"].append(np.array(areas, dtype=np.float64))
        per_event["height"].append(np.array(heights, dtype=np.float64))
        per_event["ipulse"].append(np.array(ipulses, dtype=np.int32))
        per_event["npulses"].append(np.array(npulses_field, dtype=np.int32))
        per_event["runNumber"].append(99)
        for tf in TIME_FIELDS:
            per_event[tf].append(np.array(times[tf], dtype=np.float64))

    # Wrap jagged columns as awkward; scalars as numpy
    aw_dict = {
        "event": np.array(per_event["event"], dtype=np.int32),
        "runNumber": np.array(per_event["runNumber"], dtype=np.int32),
    }
    for k in ("chan", "board", "area", "height", "ipulse", "npulses",
              *TIME_FIELDS):
        aw_dict[k] = ak.Array(per_event[k])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with uproot.recreate(out_path) as f:
            f["t"] = aw_dict
    print(f"wrote {out_path}  ({N_EVENTS} events, "
          f"{len(POPULATED_BOARDS_CHANS)} channels populated, seed={SEED})")


if __name__ == "__main__":
    main()
