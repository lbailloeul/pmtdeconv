#!/usr/bin/env python3
"""Per-peak charge integration windows, FWHM, and veto logic.

Moved verbatim from ``pmtdeconv/utils.py`` (2026-05-31 domain split). The
``_walk_to_trough_*`` helpers are intentionally public-importable (the legacy
scripts and pipeline both use them) and re-exported through ``pmtdeconv.utils``.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Integration windows
# ═══════════════════════════════════════════════════════════════════

def _walk_to_trough_right(u_rec, start, max_bins):
    """Walk right until two consecutive upticks OR u_rec crosses zero.
    The zero-cross stop prevents the window from extending into deconvolution
    undershoot regions, which would subtract real positive charge.
    """
    N = len(u_rec)
    i = start
    while i < min(N - 1, start + max_bins):
        if u_rec[i] <= 0:
            break
        if u_rec[i + 1] > u_rec[i] and (i + 2 >= N or u_rec[i + 2] > u_rec[i + 1]):
            break
        i += 1
    return i


def _walk_to_trough_left(u_rec, start, max_bins):
    """Walk left until two consecutive upticks OR u_rec crosses zero."""
    i = start
    while i > max(0, start - max_bins):
        if u_rec[i] <= 0:
            break
        if u_rec[i - 1] > u_rec[i] and (i - 2 < 0 or u_rec[i - 2] > u_rec[i - 1]):
            break
        i -= 1
    return i


def find_integration_windows(u_rec, peaks, pad_bins, neighbor_min_ratio=0.1):
    """
    For each peak:
      1. Walk outward to the left/right troughs (natural pulse boundaries).
      2. Extend by one pulse_width on each side (default = lo_trough - pulse_width
         to hi_trough + pulse_width).
      3. Cap at the shared trough with each adjacent peak only if that neighbor
         is at least neighbor_min_ratio × the current peak's height. Noise peaks
         below this threshold are ignored so they don't clip the main window.
    """
    N = len(u_rec)
    windows = []
    for i, pk in enumerate(peaks):
        hi_trough = _walk_to_trough_right(u_rec, pk, pad_bins)
        lo_trough = _walk_to_trough_left(u_rec, pk, pad_bins)
        pulse_width = hi_trough - lo_trough
        lo = max(lo_trough - pulse_width, 0)
        hi = min(hi_trough + pulse_width + 1, N)
        h_pk = u_rec[pk]
        if i > 0:
            prev_pk = peaks[i - 1]
            if u_rec[prev_pk] >= neighbor_min_ratio * h_pk:
                shared = _walk_to_trough_right(u_rec, prev_pk, min(pad_bins, pk - prev_pk))
                lo = max(lo, shared)
        if i < len(peaks) - 1:
            next_pk = peaks[i + 1]
            if u_rec[next_pk] >= neighbor_min_ratio * h_pk:
                shared = _walk_to_trough_left(u_rec, next_pk, min(pad_bins, next_pk - pk))
                hi = min(hi, shared + 1)
        windows.append((lo, hi))
    return windows


def fwhm_ns(u, pk, dt_ns, max_half_ns=120.0):
    """Full-width-at-half-maximum (ns) of the reconstructed peak at index ``pk``.

    Ported verbatim from the legacy ``cd109_pulse_width.fwhm_ns`` so the
    ``pmtdeconv tune cosine`` width-vs-cutoff study matches the prior
    methodology (and the Fig.-8-style pulse-width curves in Tang et al.
    2411.07620). Linear-interpolates each half-maximum crossing.

    Returns ``NaN`` when the peak is non-positive or the half-max level is
    not reached within ``max_half_ns`` on either side (spike merged into a
    neighbour / ringing) — those events simply don't contribute to the
    per-cutoff median width.
    """
    n = len(u)
    peak = u[pk]
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = 0.5 * peak
    max_bins = int(max_half_ns / dt_ns)

    iR = None
    i = pk
    hi = min(n - 1, pk + max_bins)
    while i < hi:
        if u[i] >= half > u[i + 1]:
            frac = (u[i] - half) / (u[i] - u[i + 1])
            iR = i + frac
            break
        i += 1

    iL = None
    i = pk
    lo = max(0, pk - max_bins)
    while i > lo:
        if u[i] >= half > u[i - 1]:
            frac = (u[i] - half) / (u[i] - u[i - 1])
            iL = i - frac
            break
        i -= 1

    if iL is None or iR is None:
        return np.nan
    return (iR - iL) * dt_ns


# ═══════════════════════════════════════════════════════════════════
# Per-peak veto logic
# ═══════════════════════════════════════════════════════════════════

def apply_peak_vetoes(
    pk, pk_i, peaks, u_rec, t, windows, pulse_widths, N,
    *,
    t_peak_min_ns=20.0,
    t_peak_max_ns=None,
    afterpulse_window_ns=40.0,
    afterpulse_height_ratio=5.0,
    post_sat_cutoff_bin=None,
    dt_ns=1.25,
    raw_wave=None,
    raw_ped=0.0,
    min_raw_amp=15.0,
    min_charge_purity=0.5,
):
    """
    Apply per-peak vetoes. Returns (success, reason, charge).

    Veto chain:
      1. out_of_time       — peak before t_peak_min_ns or after t_peak_max_ns
                             (a plain time-window cut; renamed 2026-05-30 from
                             the misnomer "fft_wraparound" — with linear
                             (zero-padded) deconvolution there is no circular
                             wraparound, this just enforces the physical window)
      2. no_raw_peak       — max raw amplitude in window < min_raw_amp ADC above baseline
      3. afterpulse_veto   — small peak within fixed ns window of a much taller peak
      4. post_sat_shift    — peak at or after the first baseline-shifted bin
      5. negative_charge   — integrated charge < 0
      6. poor_pulse_quality — positive area fraction < min_charge_purity
                              (window dominated by ringing/negative lobes)
    """
    lo, hi = windows[pk_i]
    t_peak = float(t[pk])
    h_peak = float(u_rec[pk])
    window_samples = u_rec[lo:hi]
    charge = float(np.sum(window_samples))

    # Out-of-time veto: reject peaks outside the physical signal window
    # [t_peak_min_ns, t_peak_max_ns]. (Formerly "fft_wraparound" — see the
    # veto-chain docstring; the name predates linear deconvolution.)
    if t_peak < t_peak_min_ns:
        return False, "out_of_time", charge
    if t_peak_max_ns is not None and t_peak > t_peak_max_ns:
        return False, "out_of_time", charge

    # Per-peak raw amplitude veto — reject if no real signal in the raw waveform
    if raw_wave is not None:
        raw_amp_in_window = float(np.max(raw_wave[lo:min(hi, N)]) - raw_ped)
        if raw_amp_in_window < min_raw_amp:
            return False, "no_raw_peak", charge

    # Bidirectional afterpulse veto (fixed window from integration edge)
    veto_bins = int(afterpulse_window_ns / dt_ns)
    for other_i, other_pk in enumerate(peaks):
        if other_pk == pk:
            continue
        if u_rec[other_pk] >= afterpulse_height_ratio * h_peak:
            other_lo, other_hi = windows[other_i]
            # Peak inside the dominant pulse's integration window
            if other_lo <= pk <= other_hi:
                return False, "afterpulse_veto", charge
            # Peak within afterpulse_window_ns of the window edge
            dist_to_edge = min(abs(pk - other_lo), abs(pk - other_hi))
            if dist_to_edge <= veto_bins:
                return False, "afterpulse_veto", charge

    # Post-saturation shift veto (blanket: everything after first shifted bin)
    if post_sat_cutoff_bin is not None and pk >= post_sat_cutoff_bin:
        return False, "post_sat_shift", charge

    # Negative charge veto
    if charge < 0:
        return False, "negative_charge", charge

    # Charge purity veto — reject if integration window is dominated by ringing
    # purity = positive_area / (positive_area + |negative_area|)
    pos_area = float(np.sum(window_samples[window_samples > 0]))
    neg_area = float(np.sum(np.abs(window_samples[window_samples < 0])))
    total_abs = pos_area + neg_area
    purity = pos_area / total_abs if total_abs > 0 else 1.0
    if purity < min_charge_purity:
        return False, "poor_pulse_quality", charge

    return True, "", charge
