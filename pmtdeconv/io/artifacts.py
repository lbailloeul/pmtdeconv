"""Artifact index over CLI manifests in ``outputs/``.

The index is intentionally derived from existing ``manifest.json`` files.
It gives downstream tools a stable metadata surface so output directory
names can become shorter without losing dataset/config/run provenance.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pmtdeconv import PROJECT_ROOT, __version__

INDEX_SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INDEX_NAME = "artifact_index.json"


def _utc_now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _load_manifest(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(obj, dict):
        return None, "manifest root is not an object"
    return obj, ""


def _top_child(path: Path, root: Path) -> Optional[Path]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return root / rel.parts[0]


def _entry_roles(entries: Iterable[Dict[str, Any]]) -> List[str]:
    roles = []
    for ent in entries:
        role = ent.get("role")
        if role is not None:
            roles.append(str(role))
    return roles


def _manifest_artifact(manifest_path: Path,
                       manifest: Dict[str, Any]) -> Dict[str, Any]:
    settings = manifest.get("settings") or {}
    preset = manifest.get("preset") or {}
    stats = manifest.get("stats") or {}
    outputs = manifest.get("outputs") or []
    inputs = manifest.get("inputs") or []
    return {
        "kind": "manifest",
        "manifest_path": str(manifest_path),
        "out_dir": str(manifest_path.parent),
        "subcommand": manifest.get("subcommand"),
        "timestamp_utc": manifest.get("timestamp_utc"),
        "duration_seconds": manifest.get("duration_seconds"),
        "dataset": preset.get("dataset") or settings.get("dataset"),
        "config": preset.get("config") or settings.get("config_preset_name"),
        "runs": stats.get("runs") if "runs" in stats else settings.get("runs"),
        "entry_stop": settings.get("entry_stop"),
        "out_suffix": settings.get("out_suffix"),
        "input_roles": _entry_roles(inputs),
        "output_roles": _entry_roles(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "stats": stats,
        "manifest_schema_version": manifest.get("schema_version"),
    }


def _dir_file_stats(path: Path) -> Dict[str, int]:
    file_count = 0
    size_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            try:
                size_bytes += child.stat().st_size
            except OSError:
                pass
    return {"file_count": file_count, "size_bytes": size_bytes}


def build_artifact_index(outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
                         *,
                         include_orphans: bool = True) -> Dict[str, Any]:
    """Build a JSON-serializable index for an outputs directory."""
    outputs_dir = Path(outputs_dir)
    manifest_paths = (
        sorted(outputs_dir.rglob("manifest.json"))
        if outputs_dir.exists() else []
    )

    artifacts: List[Dict[str, Any]] = []
    invalid_manifests = []
    manifest_top_dirs = set()
    role_counts: Dict[str, int] = {}

    for manifest_path in manifest_paths:
        top = _top_child(manifest_path, outputs_dir)
        if top is not None:
            manifest_top_dirs.add(top.resolve())
        manifest, error = _load_manifest(manifest_path)
        if manifest is None:
            invalid = {
                "kind": "manifest_error",
                "manifest_path": str(manifest_path),
                "out_dir": str(manifest_path.parent),
                "error": error,
            }
            artifacts.append(invalid)
            invalid_manifests.append(invalid)
            continue
        artifact = _manifest_artifact(manifest_path, manifest)
        artifacts.append(artifact)
        for role in artifact["output_roles"]:
            role_counts[role] = role_counts.get(role, 0) + 1

    orphan_count = 0
    if include_orphans and outputs_dir.exists():
        for child in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
            if child.resolve() in manifest_top_dirs:
                continue
            stats = _dir_file_stats(child)
            artifacts.append({
                "kind": "orphan_dir",
                "out_dir": str(child),
                "manifest_path": None,
                **stats,
            })
            orphan_count += 1

    artifacts.sort(key=lambda ent: (
        str(ent.get("timestamp_utc") or ""),
        str(ent.get("out_dir") or ""),
    ), reverse=True)

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "pmtdeconv_version": __version__,
        "generated_at_utc": _utc_now(),
        "outputs_dir": str(outputs_dir),
        "summary": {
            "n_artifacts": len(artifacts),
            "n_manifests": len(manifest_paths),
            "n_invalid_manifests": len(invalid_manifests),
            "n_orphan_dirs": orphan_count,
            "output_role_counts": role_counts,
        },
        "artifacts": artifacts,
    }


def write_artifact_index(index: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=False)
    return path
