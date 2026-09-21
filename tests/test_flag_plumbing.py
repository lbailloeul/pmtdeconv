#!/usr/bin/env python3
"""Meta-tests: every CLI flag must actually REACH the processing engine.

A `hits` flag travels FIVE hops:

    argparse (commands/hits.py)
      → CLI_TO_SETTINGS (presets/__init__.py)
        → Settings field
          → Detector field (pipeline/detector.py, via Detector.from_settings)
            → read in the engine (pipeline.py getattr(args, ...) — where
              ``args`` IS the Detector)

Any missed hop fails SILENTLY: the flag parses fine, `getattr(args, X,
default)` quietly returns the default, and the user's setting is ignored.
This has shipped four times (Bug-fix 1, 2, 11 comments; the 2026-07-01
`--timing` incident). These tests make hop-skipping a test failure instead
of a field debugging session.

If a test here fails after you add a flag, do ONE of:
  * plumb it: add the CLI_TO_SETTINGS entry + Settings field + Detector
    field (and copy it in Detector.from_settings); or
  * declare it command-local: add it to the explicit whitelist below —
    meaning the command's own run() consumes it directly off the argparse
    namespace and the engine never needs it.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pmtdeconv.presets import CLI_TO_SETTINGS, Settings  # noqa: E402
from pmtdeconv.pipeline.detector import Detector  # noqa: E402


# Flags consumed directly by the command module's run() (never by the
# engine). Adding here is a CONSCIOUS decision — see module docstring.
HITS_COMMAND_LOCAL = {
    "diagnostic_events",   # commands/hits.py builds the DiagnosticSink
}

# Consumed by resolve_settings itself (not per-field mapped).
RESOLVER_LOCAL = {"dataset", "preset"}


def _settings_fields() -> set:
    return {f.name for f in dataclasses.fields(Settings)}


def _detector_fields() -> set:
    return {f.name for f in dataclasses.fields(Detector)}


def test_cli_to_settings_targets_exist():
    """Every CLI_TO_SETTINGS value must be a real Settings field (a typo
    here silently drops the flag at hop 2→3)."""
    missing = [v for v in CLI_TO_SETTINGS.values()
               if v not in _settings_fields()]
    assert not missing, (
        f"CLI_TO_SETTINGS maps to nonexistent Settings fields: {missing}")


def test_every_hits_flag_is_plumbed_or_declared_local():
    """Every flag on the `hits` leaf must be mapped into Settings or be an
    explicitly declared command-local flag. (Catches: argparse flag added,
    CLI_TO_SETTINGS entry forgotten — hop 1 without hop 2.)"""
    import argparse
    from pmtdeconv.cli import add_common_args
    from pmtdeconv.commands import hits
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    hits.add_arguments(parser)
    dests = {a.dest for a in parser._actions
             if a.option_strings and a.dest != "help"}
    unplumbed = sorted(
        d for d in dests
        if d not in CLI_TO_SETTINGS
        and d not in HITS_COMMAND_LOCAL
        and d not in RESOLVER_LOCAL)
    assert not unplumbed, (
        f"hits flags parsed but never reaching Settings (silently ignored): "
        f"{unplumbed}. Plumb them (CLI_TO_SETTINGS + Settings + Detector) "
        f"or add to HITS_COMMAND_LOCAL — see this file's docstring.")


def test_engine_args_reads_are_detector_fields():
    """Every ``args.<name>`` / ``getattr(args, "<name>", ...)`` the hits
    engine reads must be a Detector field — because the ``args`` object the
    engine receives IS the Detector. (Catches: engine reads a knob the
    Detector never carries — the exact `--timing` failure, where the
    getattr default was silently used for every event.)"""
    sources = [
        REPO_ROOT / "pmtdeconv" / "pipeline" / "pipeline.py",
        REPO_ROOT / "pmtdeconv" / "pipeline" / "process_hits.py",
    ]
    attr_re = re.compile(
        r"""(?:\bargs\.([a-zA-Z_][a-zA-Z0-9_]*)          # args.name
            |getattr\(\s*args\s*,\s*["']([^"']+)["'])    # getattr(args, "name")
        """, re.VERBOSE)
    det_fields = _detector_fields()
    offenders = []
    for src in sources:
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            for m in attr_re.finditer(line):
                name = m.group(1) or m.group(2)
                if name not in det_fields:
                    offenders.append(f"{src.name}:{lineno}: args.{name}")
    assert not offenders, (
        "Engine reads args attributes that Detector does not carry — these "
        "silently resolve to the getattr default (or AttributeError):\n  "
        + "\n  ".join(offenders)
        + "\nAdd the field to Detector (+ Settings + CLI_TO_SETTINGS).")


def test_detector_from_settings_covers_all_fields():
    """Detector.from_settings must populate every Detector field (a field
    added to the dataclass but not the constructor arrives as the dataclass
    default regardless of Settings — hop 3→4 dropped)."""
    import inspect
    src = inspect.getsource(Detector.from_settings)
    required = {f.name for f in dataclasses.fields(Detector)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}
    unassigned = sorted(n for n in required
                        if not re.search(rf"\b{n}\s*=", src))
    assert not unassigned, (
        f"Detector.from_settings never assigns: {unassigned} — the Settings "
        f"value can't reach the engine.")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
