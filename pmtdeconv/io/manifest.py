"""manifest.json writer — provenance record for every CLI invocation.

Schema is defined in the plan §5. Bump ``SCHEMA_VERSION`` on breaking changes.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import getpass
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pmtdeconv import __version__, get_git_info
from pmtdeconv.io.schemas import version_map as csv_schemas_version_map

SCHEMA_VERSION = "1.0"
HASH_CHUNK = 1 << 20  # 1 MiB


def file_sha256(path: Path) -> str:
    """Compute SHA256 of a file. Returns "" if path doesn't exist."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stat_input(path: Path, do_hash: bool) -> Dict[str, Any]:
    """Stat + optionally hash a single input file."""
    entry: Dict[str, Any] = {"path": str(path)}
    if path.exists() and path.is_file():
        st = path.stat()
        entry["size_bytes"] = int(st.st_size)
        entry["mtime_utc"] = _dt.datetime.utcfromtimestamp(
            st.st_mtime).isoformat() + "Z"
        if do_hash:
            entry["sha256"] = file_sha256(path)
    else:
        entry["size_bytes"] = None
        entry["mtime_utc"] = None
    return entry


def _probe_environment() -> Dict[str, str]:
    """Record Python and key library versions for reproducibility."""
    env = {"python": sys.version.split()[0]}
    for lib in ("numpy", "scipy", "pandas", "uproot", "awkward",
                "matplotlib", "pywt"):
        try:
            mod = importlib.import_module(lib)
            env[lib] = getattr(mod, "__version__", "?")
        except ImportError:
            env[lib] = "absent"
    return env


def write_manifest(
    *,
    subcommand: str,
    settings: Any,                    # Settings dataclass
    provenance: Dict[str, str],
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    out_path: Path,
    duration_seconds: Optional[float] = None,
    hash_inputs: bool = False,
) -> Path:
    """Write the manifest JSON.

    Parameters
    ----------
    subcommand : str
        Full subcommand path, e.g. "raw_area mqana" or "pulse".
    settings : Settings
        The resolved Settings dataclass (from ``resolve_settings``).
    provenance : Dict[str, str]
        Mapping from field name to source level ("dataset"/"config"/"cli"/"default").
    inputs : list of dicts
        Each dict has at least {"role": str, "path": str or Path}.
        Sizes / SHA256 are auto-filled. `do_hash` honored per-entry by
        passing ``hash_inputs=True``.
    outputs : list of dicts
        Each dict has at least {"role": str, "path": str or Path}.
    stats : dict
        Optional command-specific stats (e.g. {"n_triggers_read": 150087}).
    out_path : Path
        Manifest JSON destination.
    duration_seconds : float
        Wall time of the command.

    Returns the path written.
    """
    git = get_git_info()
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Convert Settings dataclass → dict, casting Path → str
    settings_dict: Dict[str, Any] = {}
    for f in dataclasses.fields(settings):
        v = getattr(settings, f.name)
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, tuple):
            v = list(v)
        settings_dict[f.name] = v

    # Stat / hash inputs and outputs (outputs hashed only if requested)
    enriched_inputs = []
    for ent in inputs:
        path = Path(ent["path"])
        enriched_inputs.append({
            "role": ent.get("role", "input"),
            **_stat_input(path, do_hash=hash_inputs),
        })
    enriched_outputs = []
    for ent in outputs:
        path = Path(ent["path"])
        enriched_outputs.append({
            "role": ent.get("role", "output"),
            **_stat_input(path, do_hash=False),  # outputs never hashed
        })

    # CSV schema versions, indexed by manifest output role
    csv_schema_versions = csv_schemas_version_map(
        tuple(o.get("role", "") for o in outputs)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pmtdeconv_version": __version__,
        "subcommand": subcommand,
        "timestamp_utc": timestamp,
        "duration_seconds": duration_seconds,
        "host": platform.node(),
        "user": getpass.getuser(),
        "git": git,
        "environment": _probe_environment(),
        "preset": {
            "dataset": settings.dataset,
            "config": settings.config_preset_name,
        },
        "settings": settings_dict,
        "settings_provenance": provenance,
        "csv_schema_versions": csv_schema_versions,
        "inputs": enriched_inputs,
        "outputs": enriched_outputs,
        "stats": stats or {},
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    return out_path


def maybe_dry_run_exit(*, subcommand: str, settings: Any,
                       provenance: Dict[str, str],
                       out_dir: Path,
                       duration_seconds: float = 0.0) -> bool:
    """If ``settings.dry_run`` is True: write a manifest with empty
    inputs/outputs to ``<out_dir>/manifest.json`` and return True.
    Otherwise return False so the caller proceeds with real work.

    Decision 7. Lets the user verify preset/CLI resolution + manifest
    schema before committing to a multi-minute deconvolution.
    """
    if not getattr(settings, "dry_run", False):
        return False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand=subcommand,
        settings=settings,
        provenance=provenance,
        inputs=[],
        outputs=[],
        stats={"dry_run": True},
        out_path=manifest_path,
        duration_seconds=duration_seconds,
        hash_inputs=False,
    )
    print(f"DRY RUN — wrote {manifest_path} and skipped all ROOT I/O.")
    return True
