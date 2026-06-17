"""Layout-flexible resolution of per-run ROOT files, with actionable errors.

Collaborators' data often doesn't sit exactly where a dataset preset's default
layout expects. The old behavior was to build one fixed path, ``[skip]`` it when
missing, and return success with empty output — which then cascaded into a
confusing downstream error (e.g. ``timewalk run: "Could not resolve --area-csv"``)
far from the real cause.

These helpers instead **search the common layouts** for a run's file and, when
nothing is found, raise a ``SystemExit`` that names every path searched and how
to point the tool at the data (``--data-root`` / ``$PMTDECONV_DATA_ROOT`` / the
explicit ``--root``/``--area-root`` file flag).

Two file kinds:
  - ``"flat"`` — the flattened waveform file ``flattened_Run<N>.root`` (hits, area,
    raw-area mqana).
  - ``"drs"``  — the precomputed-area DRSprocessing file ``Run<N>.root``
    (raw-area drs).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

_KIND_LABEL = {
    "flat": "flattened waveform",
    "drs": "precomputed-area (DRSprocessing)",
}
# Explicit single-file override flag offered to the user per kind (None = the
# flat path has no single-file flag; the remedy is --data-root).
_KIND_FILE_FLAG = {"flat": None, "drs": "--root / --area-root"}


def _flat_candidates(data_root: Path, run) -> List[Path]:
    """Where a ``flattened_Run<N>.root`` might live under ``data_root``:
    the root itself, then one level of subdirectories."""
    name = f"flattened_Run{run}.root"
    cands = [data_root / name]
    if data_root.is_dir():
        cands += sorted(data_root.glob(f"*/{name}"))
    return _dedup(cands)


def _drs_candidates(data_root: Path, run) -> List[Path]:
    """Where a DRSprocessing ``Run<N>.root`` might live: ``processed/``, the
    root itself, one level of subdirs, and a bounded ``*Run<N>*.root`` glob.

    Flattened waveform files (``flattened_Run<N>.root``) are excluded from the
    fuzzy glob: they are the ``"flat"`` kind and carry no precomputed-area
    branches, so returning one here makes ``raw-area drs`` open the wrong file
    and choke on a missing ``Events`` tree."""
    name = f"Run{run}.root"
    cands = [data_root / "processed" / name, data_root / name]
    if data_root.is_dir():
        cands += sorted(data_root.glob(f"*/{name}"))
        cands += sorted(p for p in data_root.glob(f"*Run{run}*.root")
                        if not p.name.startswith("flattened"))
    return _dedup(cands)


def _dedup(paths: List[Path]) -> List[Path]:
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _candidates(data_root: Path, run, kind: str) -> List[Path]:
    if kind == "flat":
        return _flat_candidates(data_root, run)
    if kind == "drs":
        return _drs_candidates(data_root, run)
    raise ValueError(f"unknown file kind {kind!r} (expected 'flat' or 'drs')")


def _data_root_origin() -> str:
    """Best-effort description of how the data root was chosen, for the error."""
    if os.environ.get("PMTDECONV_DATA_ROOT"):
        return "$PMTDECONV_DATA_ROOT"
    return "dataset default / --data-root"


def resolve_run_file(data_root, run, *, kind: str,
                     explicit: Optional[str] = None) -> Optional[Path]:
    """Return the first existing candidate ROOT for ``run``, or ``None``.

    ``explicit`` (an exact ``--root``/``--area-root`` path) wins when given and
    is returned only if it exists.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    data_root = Path(data_root)
    for cand in _candidates(data_root, run, kind):
        if cand.exists():
            return cand
    return None


def not_found_message(data_root, run, *, kind: str,
                      explicit: Optional[str] = None) -> str:
    """Actionable multi-line message naming what was searched + the remedies."""
    label = _KIND_LABEL[kind]
    if explicit:
        return (
            f"\n{label} file for run {run}: the path you passed does not exist:\n"
            f"    {Path(explicit).expanduser()}\n"
            f"  Pass an existing file, or drop the flag and use --data-root.\n"
        )
    data_root = Path(data_root)
    cands = _candidates(data_root, run, kind)
    searched = ("\n".join(f"    {c}" for c in cands)
                if cands else "    (data root does not exist)")
    file_flag = _KIND_FILE_FLAG[kind]
    lines = [
        f"\n{label} file for run {run} not found.",
        f"  data root: {data_root}  (from {_data_root_origin()})",
        f"  searched:\n{searched}",
        "  Fix one of:",
        "    • point --data-root at the directory holding your run files:",
        "        --data-root /path/to/your/data",
        "        (or: export PMTDECONV_DATA_ROOT=/path/to/your/data)",
    ]
    if file_flag:
        exact = f"Run{run}.root" if kind == "drs" else f"flattened_Run{run}.root"
        lines.append(f"    • point at the exact file:  {file_flag} /path/to/{exact}")
    return "\n".join(lines) + "\n"


def require_run_file(data_root, run, *, kind: str,
                     explicit: Optional[str] = None) -> Path:
    """Resolve one run's file or ``raise SystemExit`` with the actionable error."""
    p = resolve_run_file(data_root, run, kind=kind, explicit=explicit)
    if p is not None:
        return p
    raise SystemExit(not_found_message(data_root, run, kind=kind, explicit=explicit))


def resolve_runs(data_root, runs, *, kind: str,
                 explicit: Optional[str] = None) -> List[Tuple[int, Path]]:
    """Resolve every run's file. Returns ``[(run, path), ...]`` for the found
    ones. If **all** runs are missing, raise the actionable error (using the
    first run for the searched-path list); if only **some** are missing, warn
    per-run and continue (preserves legitimate multi-run partial skips)."""
    runs = list(runs)
    found: List[Tuple[int, Path]] = []
    missing: List[int] = []
    for run in runs:
        p = resolve_run_file(data_root, run, kind=kind, explicit=explicit)
        if p is not None:
            found.append((run, p))
        else:
            missing.append(run)
    if missing and not found:
        raise SystemExit(
            not_found_message(data_root, missing[0], kind=kind, explicit=explicit))
    for run in missing:
        print(f"[skip] run {run}: no {_KIND_LABEL[kind]} file found under {data_root}")
    return found
