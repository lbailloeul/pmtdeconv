"""pmtdeconv — unified CLI for the PMT-waveform deconvolution toolkit.

As of the PR 10 cleanup the package is **self-contained**: all the math
(``pmtdeconv.utils``), per-run pipelines (``pmtdeconv.pipeline``), plot
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

__version__ = "0.2.0"

# Repo root (still used by several commands to resolve default output
# directories, fixtures, etc.).
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Cached so multiple subcommands in one process don't re-shell every call
_GIT_INFO_CACHE: Optional[dict] = None


def get_git_info(repo_root: Optional[Path] = None) -> dict:
    """Return {'sha', 'branch', 'dirty'} for the toolkit's git repo.

    Returns empty strings / False if not in a git checkout or git not
    available. Cached after the first call.
    """
    global _GIT_INFO_CACHE
    if _GIT_INFO_CACHE is not None:
        return _GIT_INFO_CACHE
    root = repo_root or PROJECT_ROOT
    info = {"sha": "", "branch": "", "dirty": False}
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
            ["git", "status", "--porcelain"], cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = bool(status)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    _GIT_INFO_CACHE = info
    return info
