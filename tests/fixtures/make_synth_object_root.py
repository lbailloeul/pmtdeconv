#!/usr/bin/env python3
"""Build a deterministic synthetic object-format ROOT fixture.

Mirrors the layout of LANL_835 / LANL_movedBar:
- tree "Events"
- branch ``event/digitizers/digitizers.waveform[16][1024]`` shape
  ``(n_events, n_digitizers=2, n_channels=16, n_samples=1024)`` flattened
  per-event to ``(n_events, n_digitizers, n_channels * n_samples)``
- branch ``event/digitizers/digitizers.nanosecondsPerSample`` carrying dt

For the fixture we use 2 digitizers but only populate 4 channels with
real signal (d0_ch00, d0_ch04, d0_ch16, d0_ch20) and leave the rest as
noise — matches LANL's "many channels, few populated" reality and
exercises the auto-detect path.

RNG seed = 42 for byte-reproducibility.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import uproot


N_EVENTS = 200
N_DIGITIZERS = 2
N_CHANS_PER_DIG = 16
N_SAMP = 1024
DT_NS = 1.25
SEED = 42

# Channels with real signal: ch00, ch04 on dig 0; ch00, ch04 on dig 1.
# Becomes d0_ch00, d0_ch04, d0_ch16, d0_ch20 after flattening.
POPULATED = [(0, 0), (0, 4), (1, 0), (1, 4)]


def _make_dataset(rng: np.random.Generator):
    """Build the per-event waveform array.

    Returns the array in the shape uproot's writer accepts for a branch
    declared as ``[N_DIG][N_CHANS][N_SAMP]`` — fully nested 4D
    ``(n_events, n_dig, n_chans, n_samp)``. uproot's TTree writer
    serializes this as one entry per event with the dimensional
    metadata baked into the branch name.
    """
    W = rng.normal(0.0, 1.5, (N_EVENTS, N_DIGITIZERS,
                              N_CHANS_PER_DIG, N_SAMP)).astype(np.float32)
    samp = np.arange(N_SAMP)
    sigma_samp = 25.0
    for (dig, ch) in POPULATED:
        amps = rng.uniform(80.0, 250.0, N_EVENTS).astype(np.float32)
        centers = rng.uniform(350.0, 650.0, N_EVENTS)
        for i in range(N_EVENTS):
            pulse = amps[i] * np.exp(-0.5 * ((samp - centers[i]) / sigma_samp) ** 2)
            W[i, dig, ch] -= pulse.astype(np.float32)
    dt_arr = np.full((N_EVENTS, N_DIGITIZERS), DT_NS, dtype=np.float64)
    return W, dt_arr


def main():
    # Name matches the lanl_835 dataset's file_glob
    # (``MilliQan_Run835.*_triple.root``) so the dataset preset can find
    # it. The literal ``99`` sub-run is reserved for this fixture and
    # doesn't collide with any real LANL_835 sub-run on disk.
    out_path = Path(__file__).resolve().parent / "MilliQan_Run835.99_triple.root"
    rng = np.random.default_rng(SEED)
    W, dt_arr = _make_dataset(rng)

    # Object-format ROOT layout. uproot's writer needs the dimensional
    # axes encoded directly in the branch name `[N][M]` and the array
    # shape to match.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with uproot.recreate(out_path) as f:
            f.mktree("Events", {
                "event/digitizers/digitizers.waveform": (
                    np.dtype((np.float32, (N_DIGITIZERS, N_CHANS_PER_DIG, N_SAMP)))),
                "event/digitizers/digitizers.nanosecondsPerSample": (
                    np.dtype((np.float64, (N_DIGITIZERS,)))),
            })
            f["Events"].extend({
                "event/digitizers/digitizers.waveform": W,
                "event/digitizers/digitizers.nanosecondsPerSample": dt_arr,
            })
    print(f"wrote {out_path}  ({N_EVENTS} events × {N_DIGITIZERS} dig × "
          f"{N_CHANS_PER_DIG} ch × {N_SAMP} samp, dt={DT_NS} ns, seed={SEED})")
    print(f"  populated channels (d0_chXX): {[f'd0_ch{d*16+c:02d}' for d, c in POPULATED]}")


if __name__ == "__main__":
    main()
