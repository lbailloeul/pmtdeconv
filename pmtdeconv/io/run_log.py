"""Capture a run's console output into ``<out-dir>/run.log``.

The manifest is the *structured* record of a run: resolved settings, provenance,
inputs, outputs, stats. But a lot of what a collaborator needs when something
looks wrong is only ever printed — the per-branch template lines, the
``[warn] no fit CSV`` notices, the veto-rate banner, the resolved data paths.
Until now that stream vanished with the terminal, so "it gave weird results"
arrived with no way to see what the run actually said.

This module tees stdout/stderr into an in-memory buffer while still printing
live, and :func:`pmtdeconv.io.manifest.write_manifest` flushes the buffer next
to the manifest. Send someone an output directory and they get both halves:
``manifest.json`` for what was configured, ``run.log`` for what it said.

Deliberately NOT a logging framework. Warnings that a reader must be able to
act on belong in the manifest as structured fields (see the template
provenance block); this is the human-readable transcript beside it, not a
second source of truth.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional, TextIO

RUN_LOG_NAME = "run.log"
RUN_LOG_ROLE = "run_log"

# Cap the buffer so a pathological --log-level debug run cannot eat memory.
# 8 MiB of console text is far beyond any real run; past it we keep the head
# (where the configuration and warnings are) and say so.
_MAX_CHARS = 8 * 1024 * 1024

_buffer: Optional[io.StringIO] = None
_truncated = False
_saved: Optional[tuple] = None


class _Tee:
    """Write to the real stream AND the capture buffer."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        global _truncated
        if _buffer is not None:
            if _buffer.tell() < _MAX_CHARS:
                _buffer.write(text)
            elif not _truncated:
                _truncated = True
                _buffer.write("\n[run.log truncated: output exceeded "
                              f"{_MAX_CHARS // (1024 * 1024)} MiB]\n")
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        # Preserved because template resolution asks, to decide whether it may
        # prompt interactively.
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_capture() -> None:
    """Begin teeing stdout/stderr. Safe to call more than once."""
    global _buffer, _saved, _truncated
    if _saved is not None:
        return
    _buffer = io.StringIO()
    _truncated = False
    _saved = (sys.stdout, sys.stderr)
    sys.stdout = _Tee(sys.stdout)       # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr)       # type: ignore[assignment]


def stop_capture() -> None:
    """Restore the real streams. Always call this, or an in-process caller
    (pytest invoking ``cli.main()`` directly) keeps a dangling tee."""
    global _buffer, _saved
    if _saved is None:
        return
    sys.stdout, sys.stderr = _saved
    _saved = None
    _buffer = None


def captured_text() -> str:
    return _buffer.getvalue() if _buffer is not None else ""


def write_run_log(out_dir: Path) -> Optional[Path]:
    """Write the captured console output to ``<out_dir>/run.log``.

    Returns the path, or None when capture was never started (a library caller
    using ``write_manifest`` directly) or produced nothing.
    """
    text = captured_text()
    if not text:
        return None
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / RUN_LOG_NAME
        path.write_text(text)
    except OSError:
        # A log is a convenience; never fail a completed run over it.
        return None
    return path
