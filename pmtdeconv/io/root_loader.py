"""ROOT-file loader helpers — handle nested DRS4 auto-save cycles.

The DRS4 acquisition writes cycles incrementally: each new cycle is a
*superset* of the previous (e.g. ``waveforms;1``, ``waveforms;2``, ...
where the highest-numbered cycle contains all events from earlier ones).
We always want the last cycle; the multiplier scales any ``entry_stop``
that was specified against a single cycle.

Lifted verbatim from
``deconvolution/raw_area_Na22._select_tree_key`` /
``deconvolution/deconv_area_Na22._select_tree``.
"""
from __future__ import annotations

from typing import Tuple

import awkward as ak
import numpy as np


def select_tree_key(ufile, tree_name: str = "waveforms"
                    ) -> Tuple[str, int]:
    """Return ``(tree_key, per_cycle_mult)`` for the last cycle.

    Parameters
    ----------
    ufile : uproot.ReadOnlyFile
    tree_name : str
        Tree to look for (e.g. "waveforms" for flat format, "Events" for
        object format).

    Raises
    ------
    RuntimeError
        If no tree of that name exists in the file.
    """
    all_keys = sorted(
        [k for k in ufile.keys(cycle=True) if k.split(";")[0] == tree_name],
        key=lambda k: int(k.split(";")[-1]),
    )
    if not all_keys:
        raise RuntimeError(f"No '{tree_name}' tree found in file")
    if len(all_keys) == 1:
        return all_keys[0], 1

    # Compare the very first waveform across cycles to detect nesting.
    # The waveforms tree might not have anything to "compare" if branches
    # disagree, so be defensive.
    ref_branch = sorted(ufile[all_keys[-1]].keys())[0]
    try:
        w0 = ak.to_numpy(ufile[all_keys[0]][ref_branch].array(
            entry_start=0, entry_stop=1)[0])
        nested = all(
            np.allclose(w0, ak.to_numpy(
                ufile[k][ref_branch].array(entry_start=0, entry_stop=1)[0]))
            for k in all_keys[1:]
        )
    except Exception:
        nested = False

    mult = len(all_keys) if nested else 1
    return all_keys[-1], mult


def to_numpy2d(ak_array) -> np.ndarray:
    """Convert an awkward array of fixed-length waveforms to a 2-D numpy array.

    Matches ``raw_area_Na22._to_numpy2d``. Falls back to a per-row stack
    for truly jagged arrays (shouldn't happen with DRS4 but defensive).
    """
    try:
        return ak.to_numpy(ak.to_regular(ak_array)).astype(np.float64)
    except Exception:
        return np.stack([ak.to_numpy(row) for row in ak_array]).astype(np.float64)
