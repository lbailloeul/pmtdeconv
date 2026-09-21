"""Run banner — the human-visible provenance summary printed at run start.

A multi-minute analysis should show WHAT it is about to run: the dataset (and
whether it came from a card or the in-code seed), the key knobs with WHERE each
was resolved from, the declared FoM, and the git SHA. The manifest records the
full provenance; this is the at-a-glance version (S10 transparency).
"""
from __future__ import annotations

from typing import Any, Dict

from pmtdeconv import __version__, get_git_info

# (Settings attr, banner label) for the knobs worth surfacing up front.
_KEY_FIELDS = [
    ("dataset", "dataset"),
    ("cosine_cutoff_mhz", "cutoff [MHz]"),
    ("denoiser", "denoiser"),
    ("no_vetoes", "no_vetoes"),
    ("template_csv", "template"),
    ("fom", "fom"),
]


def format_run_banner(subcommand: str, settings: Any,
                      provenance: Dict[str, str]) -> str:
    """Return the multi-line banner string (no trailing newline)."""
    ds_src = provenance.get("dataset", "?")
    lines = [f"┌─ pmtdeconv {subcommand}"]
    for field, label in _KEY_FIELDS:
        if not hasattr(settings, field):
            continue
        val = getattr(settings, field)
        if val is None:
            continue
        src = provenance.get(field, ds_src if field == "dataset" else "default")
        lines.append(f"│  {label:<13} {val}   [{src}]")
    git = get_git_info() or {}
    sha = (git.get("sha") or "")[:8]
    branch = git.get("branch") or ""
    dirty = " (dirty)" if git.get("dirty") else ""
    tail = f"pmtdeconv {__version__}"
    if sha:
        tail += f"  git {sha}{dirty}"
        if branch:
            tail += f" @{branch}"
    if git.get("dirty"):
        # Loud on purpose: a run from modified code cannot be reproduced from
        # the SHA the manifest records. The manifest lists the files.
        n = git.get("dirty_count") or len(git.get("dirty_files") or [])
        lines.append(f"│  ⚠ NOT REPRODUCIBLE FROM GIT: {n} uncommitted change(s) "
                     f"to pmtdeconv since {sha or 'HEAD'}.")
        lines.append("│    Commit (and tag) before any run you will report; "
                     "manifest git.dirty_files lists them.")
    lines.append(f"└─ {tail}")
    return "\n".join(lines)


def print_run_banner(subcommand: str, settings: Any,
                     provenance: Dict[str, str]) -> None:
    """Print the run banner to stdout (transparency at run start)."""
    print(format_run_banner(subcommand, settings, provenance))
