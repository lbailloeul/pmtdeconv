"""Unified per-event waveform source for the deconvolution pipeline.

Abstracts the two on-disk waveform formats behind one iterator that
yields a common :class:`EventRecord`:

- **flat** — ``flattened_Run<N>.root`` with a ``waveforms`` tree whose
  branches are per-channel fixed-length arrays (``d0_ch1`` …). Handles
  the DRS4 nested auto-save cycles.
- **object** — LANL ``MilliQan_Run*_triple.root`` with the nested
  ``event/digitizers/digitizers.waveform[16][1024]`` payload, read via
  :func:`pmtdeconv.io.object_format.iter_object_run`.

**Polarity is applied here, in exactly one place** (``polarity * wave``).
Before this module the flat path negated only ``if polarity < 0`` while
the object path always multiplied — the asymmetry that produced the
2026-06-10 flat-path polarity bug. Routing both formats through
``iter_events`` removes any path that can "forget" the flip.

The per-event processing body (baseline → deconv → denoise → peaks →
vetoes → integrate) lives in :mod:`pmtdeconv.pipeline.process_hits` and
consumes these records; it is identical for both formats.

The flat cycle-resolution logic is ported verbatim from the pre-collapse
``process_hits.process_run`` (NOT delegated to
``io.root_loader.select_tree_key``, which collapses a non-nested
multi-cycle file to its last cycle only — the inline loop iterates each
such cycle independently). Iteration order is load-bearing for the
byte-identical golden fixtures, so the nesting of
``cycle → chunk → branch → event`` is preserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import awkward as ak
import numpy as np
import uproot

from pmtdeconv.utils import DTYPE


@dataclass
class EventRecord:
    """One channel's waveform for one event, polarity already applied."""
    run: int
    cycle: int
    det: str
    chan: int
    branch: str
    entry: int
    wave: np.ndarray      # polarity applied, dtype DTYPE
    t: np.ndarray         # arange(N) * dt_ns
    N: int
    dt_ns: float


def _apply_polarity(wave: np.ndarray, polarity: int) -> np.ndarray:
    """The single polarity-flip point. ``polarity * wave`` is byte-identical
    to both legacy behaviors (flat: no-op at +1, negate at -1; object:
    always multiply)."""
    return (polarity * wave).astype(DTYPE, copy=False)


def iter_flat_run(
    root_path: str,
    run: int,
    *,
    dt_ns: float,
    polarity: int,
    branches: Optional[List[str]] = None,
    entry_start: int = 0,
    entry_stop: Optional[int] = None,
    step_size: int = 500,
    tree_name: str = "waveforms",
) -> Iterator[EventRecord]:
    """Yield ``EventRecord`` for a flat ``flattened_Run<N>.root`` file.

    Ports the nested-cycle handling + chunked iteration from the
    pre-collapse ``process_hits.process_run`` verbatim, so row order is
    preserved. ``branches=None`` auto-detects from the last cycle.
    """
    print(f"\n================ Run {run} ================")
    print(f"Opening: {root_path}")

    # ── Nested-cycle handling (verbatim from legacy process_run) ──
    try:
        with uproot.open(root_path) as ufile:
            all_keys = sorted(
                [k for k in ufile.keys(cycle=True) if k.split(";")[0] == tree_name],
                key=lambda k: int(k.split(";")[-1]),
            )
            if not all_keys:
                print(f"[skip] no '{tree_name}' cycles in {root_path}")
                return
            if branches is None:
                branches = sorted(ufile[all_keys[-1]].keys())
                print(f"  Auto-detected branches: {branches}")
            nested = True
            if len(all_keys) > 1:
                ref_branch = branches[0]
                w_first = ak.to_numpy(
                    ufile[all_keys[0]][ref_branch].array(
                        entry_start=0, entry_stop=1)[0])
                for k in all_keys[1:]:
                    w = ak.to_numpy(
                        ufile[k][ref_branch].array(
                            entry_start=0, entry_stop=1)[0])
                    if not np.allclose(w_first, w):
                        nested = False
                        break
            if nested and len(all_keys) > 1:
                tree_keys = [all_keys[-1]]
                per_cycle_mult = len(all_keys)
                print(f"  Cycles {all_keys} detected; NESTED → using {tree_keys[0]} with {per_cycle_mult}x entries")
            else:
                tree_keys = all_keys
                per_cycle_mult = 1
                print(f"  Cycles {all_keys} detected; iterating each independently")
    except (FileNotFoundError, OSError) as e:
        print(f"[skip] cannot open: {root_path} ({e})")
        return

    entry_stop_this_file = (entry_stop * per_cycle_mult
                            if entry_stop is not None else None)

    for tree_key in tree_keys:
        cycle_num = int(tree_key.split(";")[-1])
        print(f"\n  -- Processing {tree_key} (target {entry_stop_this_file} entries) --")
        entry_cursor = entry_start
        _printed_dets = set()

        ufile = None
        try:
            ufile = uproot.open(root_path)
            tree_obj = ufile[tree_key]
            for arrays in tree_obj.iterate(
                expressions=branches, library="ak",
                entry_start=entry_start, entry_stop=entry_stop_this_file,
                step_size=step_size, how=dict,
            ):
                any_branch = next(iter(arrays.keys()))
                nsamp = len(ak.to_numpy(arrays[any_branch][0]))
                t_full = np.arange(nsamp, dtype=DTYPE) * dt_ns
                n_chunk = len(arrays[any_branch])
                base_entry = entry_cursor
                entry_cursor += n_chunk

                if entry_stop_this_file is not None and base_entry >= entry_stop_this_file:
                    print(f"    [safety] hit entry_stop={entry_stop_this_file}; breaking")
                    break
                if entry_stop_this_file is not None and entry_cursor > entry_stop_this_file:
                    n_chunk = entry_stop_this_file - base_entry

                for branch in branches:
                    if branch not in arrays:
                        continue
                    det = branch.split("_ch")[0]
                    ch = int(branch.split("ch")[-1])

                    if det not in _printed_dets:
                        print(f"    {tree_key}  {det}", flush=True)
                        _printed_dets.add(det)

                    for j in range(n_chunk):
                        entry_i = base_entry + j
                        wave = ak.to_numpy(arrays[branch][j]).astype(DTYPE, copy=False)
                        wave = _apply_polarity(wave, polarity)
                        yield EventRecord(
                            run=run, cycle=cycle_num, det=det, chan=ch,
                            branch=branch, entry=entry_i,
                            wave=wave, t=t_full, N=len(wave), dt_ns=dt_ns,
                        )
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] error reading {tree_key}: {e}")
            continue
        finally:
            if ufile is not None:
                try:
                    ufile.close()
                except Exception:
                    pass


def iter_object_run_events(
    root_path: Path,
    run: int,
    *,
    dt_ns: float,
    polarity: int,
    branches: List[str],
    entry_stop: Optional[int] = None,
    step_size: int = 100,
    tree_name: str = "Events",
) -> Iterator[EventRecord]:
    """Yield ``EventRecord`` for an object-format LANL ROOT file.

    Ports the per-event loop from the pre-collapse
    ``process_hits.process_run_object`` verbatim. ``cycle`` is always 1
    (object files have no DRS4 auto-save cycles).
    """
    from pmtdeconv.io.object_format import iter_object_run

    for chunk_base, chunk in iter_object_run(
        root_path, branches=branches,
        entry_stop=entry_stop, step_size=step_size,
        tree_name=tree_name,
    ):
        n_chunk = len(next(iter(chunk.values())))
        N = chunk[branches[0]].shape[1]
        t_full = np.arange(N, dtype=DTYPE) * dt_ns

        for branch in branches:
            det = branch.split("_ch")[0]
            ch = int(branch.split("ch")[-1])

            for j in range(n_chunk):
                entry_i = chunk_base + j
                wave = _apply_polarity(chunk[branch][j], polarity)
                yield EventRecord(
                    run=run, cycle=1, det=det, chan=ch,
                    branch=branch, entry=entry_i,
                    wave=wave, t=t_full, N=len(wave), dt_ns=dt_ns,
                )
