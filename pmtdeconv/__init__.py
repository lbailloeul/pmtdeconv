"""pmtdeconv — unified CLI for the PMT-waveform deconvolution toolkit.

As of the PR 10 cleanup the package is **self-contained**: all the math
(``pmtdeconv.core``), per-run pipelines (``pmtdeconv.pipeline``), plot
helpers (``pmtdeconv.plotting``), preset registry (``pmtdeconv.presets``),
and CLI command modules (``pmtdeconv.commands``) live inside this
package. The repo root only carries ``pmtdeconv/`` + ``tests/`` + docs;
pre-CLI scripts (``utils.py`` itself, the per-stage Na22 / Cd109 /
LANL analysis scripts, ``fitting/``, ``deconvolution/``, ``archive/``)
have all moved to ``legacy/`` and are gitignored.

See ``pmtdeconv.cli.main`` for the entry point and
``pmtdeconv/__main__.py`` for the ``python -m pmtdeconv`` shim.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

__version__ = "0.4.0"

# Repo root (still used by several commands to resolve default output
# directories, fixtures, etc.).
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Cached so multiple subcommands in one process don't re-shell every call
_GIT_INFO_CACHE: Optional[dict] = None


def get_git_info(repo_root: Optional[Path] = None) -> dict:
    """Return {'sha', 'branch', 'dirty', 'dirty_files'} for the toolkit.

    ``dirty`` is scoped to the code that actually RAN — ``pmtdeconv/`` and
    ``pyproject.toml`` — not the whole repository. The git root is the parent
    ``timewalksim/`` checkout, so an unscoped ``git status`` flagged a run as
    dirty because of an unrelated ``.DS_Store`` or notebook: a warning that
    cries wolf gets ignored. ``dirty_files`` (capped) says exactly which code
    differs from ``sha``, which is what a reproduction attempt needs.

    Returns empty strings / False if not in a git checkout or git not
    available. Cached after the first call.
    """
    global _GIT_INFO_CACHE
    if _GIT_INFO_CACHE is not None:
        return _GIT_INFO_CACHE
    root = repo_root or PROJECT_ROOT
    info = {"sha": "", "branch": "", "dirty": False, "dirty_count": 0,
            "dirty_files": []}
    try:
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--",
             "pmtdeconv", "pyproject.toml"], cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode()
        changed = [ln[3:] for ln in status.splitlines() if ln.strip()]
        info["dirty"] = bool(changed)
        info["dirty_count"] = len(changed)          # true count
        info["dirty_files"] = changed[:50]          # capped listing
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    _GIT_INFO_CACHE = info
    return info
