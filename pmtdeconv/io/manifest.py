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


def _json_safe(value: Any) -> Any:
    """Coerce a Settings value into something ``json.dump`` accepts.

    Most fields are scalars, Paths or tuples. A card's free-form ``meta`` block
    can hold anything YAML parses — dates, nested maps, lists — and an
    unserializable value used to abort ``json.dump`` MID-WRITE, leaving a
    truncated manifest.json on disk: the run succeeded, its provenance record
    was corrupt. Anything unrecognized degrades to ``str`` instead.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stat_input(path: Path, do_hash: bool) -> Dict[str, Any]:
    """Stat + optionally hash a single input file."""
    entry: Dict[str, Any] = {"path": str(path)}
    if path.exists() and path.is_file():
        st = path.stat()
        entry["size_bytes"] = int(st.st_size)
        entry["mtime_utc"] = _dt.datetime.fromtimestamp(
            st.st_mtime, _dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
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
    timestamp = _dt.datetime.now(_dt.timezone.utc).replace(
        tzinfo=None).isoformat(timespec="seconds") + "Z"

    # Convert Settings dataclass → dict, casting Path → str
    settings_dict: Dict[str, Any] = {}
    for f in dataclasses.fields(settings):
        settings_dict[f.name] = _json_safe(getattr(settings, f.name))

    # The dataset CARD is an input like any data file: it is the configuration
    # the run was built from. Injected centrally so every command records it
    # (and --hash-inputs fingerprints it), rather than each command remembering.
    card = getattr(settings, "dataset_card_path", None)
    if card and Path(card).exists():
        already = {str(Path(i.get("path", ""))) for i in inputs}
        if str(Path(card)) not in already:
            inputs = [{"role": "dataset_card", "path": str(card)}, *inputs]

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

    # Drop the console transcript beside the manifest, and record it as an
    # output so `results show` lists it. The manifest is the structured record;
    # run.log is what the run actually SAID (warnings, resolved paths, veto
    # rates) — together they make an output directory self-describing.
    from pmtdeconv.io.run_log import RUN_LOG_NAME, RUN_LOG_ROLE, write_run_log
    if write_run_log(out_path.parent) is not None:
        manifest["outputs"].append({
            "role": RUN_LOG_ROLE,
            "path": str(out_path.parent / RUN_LOG_NAME),
        })

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
