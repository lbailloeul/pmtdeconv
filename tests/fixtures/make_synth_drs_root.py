#!/usr/bin/env python3
"""Build the synthetic DRSprocessing-format fixture for ``raw-area drs``.

``pmtdeconv raw-area drs`` reads PRECOMPUTED per-event ``area_<board>_<ch>``
and ``vMax_<board>_<ch>`` branches from a flat ``Events`` tree (no waveforms;
the integration was done upstream by DRSprocessing). This is the only
golden-covered path that consumes that format — it had no fixture before
step 8 of the architecture restructure.

Writes ``tests/fixtures/processed/Run0.root`` from a fixed RNG seed so the
golden is byte-deterministic. Board ``2984``, channels ``1``/``2`` — matching
the ``raw-area drs`` defaults.

Rebuild:  python tests/fixtures/make_synth_drs_root.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

SEED = 20260613
N_EVENTS = 200
BOARD = "2984"
OUT = Path(__file__).resolve().parent / "processed" / "Run0.root"


def _synth_channel(rng, area_scale, height_scale):
    """A correlated (area, vMax) pair per event: a lognormal charge spectrum
    with height roughly proportional to area plus noise."""
    area = rng.lognormal(mean=np.log(area_scale), sigma=0.5, size=N_EVENTS)
    height = height_scale * (area / area_scale) * rng.uniform(0.8, 1.2, N_EVENTS)
    return area.astype(np.float64), height.astype(np.float64)


def main() -> Path:
    rng = np.random.default_rng(SEED)
    a1, v1 = _synth_channel(rng, area_scale=1200.0, height_scale=180.0)
    a2, v2 = _synth_channel(rng, area_scale=900.0, height_scale=150.0)
    branches = {
        f"area_{BOARD}_1": a1, f"vMax_{BOARD}_1": v1,
        f"area_{BOARD}_2": a2, f"vMax_{BOARD}_2": v2,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(OUT) as f:
        # Explicit TTree (not RNTuple) so `raw-area drs` reads it the same
        # way across uproot versions (>=5.7 defaults writes to RNTuple).
        f.mktree("Events", {name: np.float64 for name in branches})
        f["Events"].extend(branches)
    print(f"wrote {OUT}  ({N_EVENTS} events, board {BOARD}, ch 1/2)")
    return OUT


if __name__ == "__main__":
    main()
