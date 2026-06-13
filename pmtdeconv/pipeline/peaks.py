"""Peak finding + integration windows.

Wraps ``scipy.signal.find_peaks`` and ``utils.find_integration_windows``.
When ``no_vetoes=True``, peaks are reordered so peak_id=1 is the TALLEST
peak (not the chronologically earliest). This matches the
2026-05-session finding that without vetoes, the chronologically first
``peak_id`` is often a noise spike or pre-pulse ringing — taking the
tallest peak gives back the photopeak as peak 1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks

from pmtdeconv.utils import (                                      # noqa: E402
    find_integration_windows,
    _walk_to_trough_left,
    _walk_to_trough_right,
)


def find_and_order(
    u_rec: np.ndarray,
    *,
    peak_thresh: float,
    distance_bins: int,
    no_vetoes: bool = False,
) -> np.ndarray:
    """Find peaks; optionally reorder by descending u_rec amplitude.

    Returns an ``ndarray`` of peak sample indices in the order ``peak_id``
    should follow. With ``no_vetoes=False`` this preserves the
    chronological order returned by ``scipy.signal.find_peaks``.
    With ``no_vetoes=True`` it sorts by descending ``u_rec[pk]``.
    """
    peaks, _ = find_peaks(u_rec, height=peak_thresh, distance=distance_bins)
    if no_vetoes and len(peaks) > 1:
        order = np.argsort(-u_rec[peaks])
        peaks = peaks[order]
    return peaks


def integration_windows(
    u_rec: np.ndarray,
    peaks: np.ndarray,
    pad_bins: int,
) -> List[Tuple[int, int]]:
    """Trough-walked integration windows for each peak."""
    return find_integration_windows(u_rec, peaks, pad_bins)


def pulse_widths(
    u_rec: np.ndarray,
    peaks: np.ndarray,
    pad_bins: int,
) -> List[int]:
    """Trough-to-trough widths in sample bins (matches the legacy logic
    in wavelets_cosine_deconvolution_Na22.py:407-409)."""
    widths = []
    for pk in peaks:
        lo_tr = _walk_to_trough_left(u_rec, pk, pad_bins)
        hi_tr = _walk_to_trough_right(u_rec, pk, pad_bins)
        widths.append(hi_tr - lo_tr)
    return widths
