"""Reader for object-format LANL ROOT files (LANL_835, LANL_movedBar).

These files store waveforms as a nested array
``event/digitizers/digitizers.waveform[16][1024]`` — for each event,
two digitizers × 16 channels × 1024 samples (flattened by uproot to
shape ``(n_digitizers, 16384)`` per event). We reshape to per-channel
arrays named ``d0_ch00`` through ``d0_ch31`` (digitizer 0 → ch00-15,
digitizer 1 → ch16-31), matching ``build_spe_templates_perfile`` and
the pmtdeconv branch convention.

The dt_ns ("sample spacing") is per-event on these files
(``digitizers.nanosecondsPerSample``); we read it from the first event.

Iteration is chunked via ``uproot.iterate`` so memory stays bounded
even on large LANL_movedBar files. Each yielded chunk is a dict
``{"d0_chXX": ndarray(n_chunk, N_SAMP), ...}`` analogous to the
chunks the flat-format readers (``raw_area_mqana`` etc.) consume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot


N_SAMP_DEFAULT = 1024
N_DIGITIZERS_DEFAULT = 2
N_CHANNELS_PER_DIG_DEFAULT = 16

# Branch name templates. LANL_835 / LANL_movedBar use the bracketed
# form ``waveform[16][1024]`` because the shape is encoded in the
# branch name; uproot's own writer (mktree) puts the shape in the
# dtype instead and emits the un-bracketed form. Try both.
WAVEFORM_BRANCH_VARIANTS = (
    "event/digitizers/digitizers.waveform[16][1024]",
    "event/digitizers/digitizers.waveform",
)
DT_BRANCH = "event/digitizers/digitizers.nanosecondsPerSample"


def _resolve_waveform_branch(tree) -> str:
    """Return the first available waveform branch name."""
    keys = set(tree.keys())
    for cand in WAVEFORM_BRANCH_VARIANTS:
        if cand in keys:
            return cand
    raise KeyError(
        f"No waveform branch found. Tried: {WAVEFORM_BRANCH_VARIANTS}. "
        f"Tree has: {sorted(keys)[:5]}...")


def read_dt_ns(root_path: Path, tree_name: str = "Events") -> float:
    """Read sample spacing (ns) from the first event."""
    with uproot.open(root_path) as f:
        tree = f[tree_name]
        ns = tree[DT_BRANCH].array(entry_stop=1, library="np")
    return float(np.atleast_1d(ns[0]).ravel()[0])


def list_channels(n_digitizers: int = N_DIGITIZERS_DEFAULT,
                  n_channels_per_dig: int = N_CHANNELS_PER_DIG_DEFAULT
                  ) -> List[str]:
    """Return canonical channel names for an object-format file."""
    return [f"d0_ch{dig*n_channels_per_dig + ch:02d}"
            for dig in range(n_digitizers)
            for ch in range(n_channels_per_dig)]


def _reshape_chunk(wf_chunk, n_samp: int, n_dig: int, n_ch_per_dig: int
                   ) -> np.ndarray:
    """Reshape a uproot chunk of waveform records → (n_events, n_dig, n_ch_per_dig, n_samp).

    On-disk layouts vary:
    - LANL_835 / movedBar (branch name ``digitizers.waveform[16][1024]``):
      per event is shape ``(n_dig, n_ch_per_dig * n_samp)`` (flat payload).
    - uproot's own ``mktree`` writer (no bracketed shape): per event is
      already shape ``(n_dig, n_ch_per_dig, n_samp)``.

    We just reshape via the explicit element count, which works either
    way.
    """
    n_events = len(wf_chunk)
    out = np.empty((n_events, n_dig, n_ch_per_dig, n_samp), dtype=np.float64)
    for i in range(n_events):
        out[i] = np.asarray(wf_chunk[i], dtype=np.float64).reshape(
            n_dig, n_ch_per_dig, n_samp)
    return out


def detect_populated_channels(
    root_path: Path,
    *,
    tree_name: str = "Events",
    sample_n: int = 200,
    min_frac: float = 0.05,
    vabs_threshold: float = 10.0,
    n_samp: int = N_SAMP_DEFAULT,
    n_digitizers: int = N_DIGITIZERS_DEFAULT,
    n_channels_per_dig: int = N_CHANNELS_PER_DIG_DEFAULT,
    base_bins: int = 30,
) -> List[str]:
    """Find which of the 32 channels actually carry signal.

    Reads up to ``sample_n`` events, computes baseline-relative
    ``max(abs(y))`` per channel, and keeps channels where that max
    exceeds ``vabs_threshold`` mV for at least ``min_frac`` of the
    sampled events. Polarity-agnostic (LANL_835 has flipped channels;
    the absolute amplitude matters, not the sign). Lifted from
    ``build_spe_templates_perfile.load_file_waveforms``.
    """
    all_chans = list_channels(n_digitizers, n_channels_per_dig)
    with uproot.open(root_path) as f:
        tree = f[tree_name]
        branch = _resolve_waveform_branch(tree)
        n_read = min(sample_n, tree.num_entries)
        wf = tree[branch].array(entry_stop=n_read, library="ak")
        W = _reshape_chunk(wf, n_samp, n_digitizers, n_channels_per_dig)
    # W shape: (n_read, n_dig, n_ch_per_dig, n_samp)
    W_flat = W.reshape(len(W), n_digitizers * n_channels_per_dig, n_samp)
    base = np.median(W_flat[:, :, :base_bins], axis=2, keepdims=True)
    vabs = np.abs(W_flat - base).max(axis=2)        # (n_read, 32)
    frac_above = (vabs > vabs_threshold).mean(axis=0)
    return [name for name, frac in zip(all_chans, frac_above) if frac >= min_frac]


def iter_object_run(
    root_path: Path,
    *,
    tree_name: str = "Events",
    branches: Optional[List[str]] = None,
    entry_stop: Optional[int] = None,
    step_size: int = 100,
    n_samp: int = N_SAMP_DEFAULT,
    n_digitizers: int = N_DIGITIZERS_DEFAULT,
    n_channels_per_dig: int = N_CHANNELS_PER_DIG_DEFAULT,
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Chunked iterator over per-channel waveforms.

    Yields ``(base_entry, chunk_dict)`` pairs where ``chunk_dict`` maps
    each channel name (``d0_ch00`` etc.) to ``ndarray(n_chunk, n_samp)``.

    Parameters
    ----------
    branches : list of str or None
        Restrict to these channel names. ``None`` returns all 32.
        Channels not in the canonical 0-31 range are silently ignored.
    entry_stop : int or None
        Cap on entries read (None = whole file).
    step_size : int
        Events per chunk. 100 events × 2 × 16 × 1024 samples × 8 B ≈ 26 MB,
        comfortable.
    """
    all_chans = list_channels(n_digitizers, n_channels_per_dig)
    if branches is not None:
        wanted = set(branches)
        keep_idx = [i for i, name in enumerate(all_chans) if name in wanted]
        keep_names = [all_chans[i] for i in keep_idx]
    else:
        keep_idx = list(range(len(all_chans)))
        keep_names = all_chans

    with uproot.open(root_path) as f:
        tree = f[tree_name]
        branch = _resolve_waveform_branch(tree)
        n_total = tree.num_entries
        stop = min(entry_stop, n_total) if entry_stop is not None else n_total

        base = 0
        for arrays in tree.iterate(
            expressions=[branch], library="ak",
            entry_start=0, entry_stop=stop, step_size=step_size,
        ):
            wf_chunk = arrays[branch]
            n_chunk = len(wf_chunk)
            W = _reshape_chunk(wf_chunk, n_samp, n_digitizers, n_channels_per_dig)
            # W shape: (n_chunk, n_dig, n_ch_per_dig, n_samp)
            # Flatten (dig, ch_per_dig) → 32 channels, pick the wanted ones
            W_flat = W.reshape(n_chunk, n_digitizers * n_channels_per_dig, n_samp)
            chunk = {keep_names[k]: W_flat[:, keep_idx[k], :]
                     for k in range(len(keep_idx))}
            yield base, chunk
            base += n_chunk
