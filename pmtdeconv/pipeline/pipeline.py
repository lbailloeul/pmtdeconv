"""Unified per-event pipeline core + processing-mode strategies.

The whole-waveform **area** summary and the per-hit **hits** path share the same
front half — saturation gate, no-trigger gate, adaptive baseline, FFT cosine
deconvolution (optionally zero-padded), optional wavelet denoising — and differ
only in the *tail*: how the reconstructed waveform ``u_rec`` becomes output rows.

Step 4 of the architecture restructure models that tail as a strategy object
(:class:`AreaMode`) and shares the prefix in :func:`run_area`. The flat and
object area readers (in :mod:`pmtdeconv.pipeline.process_area`) previously
inlined the *same* ~180-line loop twice; they now both feed :func:`run_area`
and differ only in two veto flags carried on :class:`AreaMode`:

- ``use_wrap_abs_k`` — the flat path gates the left-edge veto with an absolute
  noise floor (``wrap_abs_k``); the object path does not.
- ``use_afterpulse`` — the flat path runs the afterpulse veto; the object path
  does not.

(The hits tail — ``find_peaks`` → ``apply_peak_vetoes`` → trough-walked charge
integration — still lives in ``process_hits._process_event``; folding it behind
a ``HitsMode`` here is a tracked follow-up. The two paths already share the math
via ``pmtdeconv.core``/``utils``; this step removes the area flat/object
duplication and makes ``mode`` a first-class object.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from pmtdeconv.core.dsp import detect_saturation_plateau
from pmtdeconv.pipeline.template import FALLBACK_TEMPLATE
from pmtdeconv.presets.veto_profiles import VETO_PROFILES
from pmtdeconv.utils import (
    build_spe_template,
    cosine_lowpass_filter,
    deconvolve_hits,
    baseline_noise_bestwindow,
    fwhm_ns,
    wavelet_denoise,
    stft_denoise,
    sliding_local_mean,
    apply_peak_vetoes,
    find_integration_windows,
    _walk_to_trough_left,
    _walk_to_trough_right,
)

# One event yielded by an area reader: (run, cycle, entry, branch, wave, N).
# ``wave`` is float64 with polarity already applied by the reader (the flat
# reader is polarity-naive today, matching legacy; the object reader flips by
# ``polarity``). Kept as a plain tuple so readers stay cheap generators.
AreaEvent = Tuple[int, int, int, str, np.ndarray, int]

N_PRE_DEFAULT = 50    # fallback baseline samples if the best-window estimator fails


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass(frozen=True)
class AreaMode:
    """Whole-waveform area-summary tail strategy.

    Bundles the signal window, interval count, optional FFT zero-padding, the
    (looser-than-hits) baseline gate, and the inline left-edge + afterpulse
    vetoes. :meth:`finalize` owns the tail: vetoes → interval integration.
    """
    sig_start: int
    sig_end: int
    n_intervals: int
    fft_pad: bool
    no_trigger_amp: float
    t_peak_min_ns: float
    afterpulse_window_ns: float
    afterpulse_height_ratio: float
    wrap_ratio: float
    wrap_abs_k: float
    measure_width: bool
    # flat vs object tail differences (see module docstring)
    use_wrap_abs_k: bool
    use_afterpulse: bool
    # baseline gate (looser than the hits profile; verbatim legacy constants)
    flat_frac_min: float = 0.50
    flat_k: float = 10.0
    spike_k: float = 20.0
    search_end_ns: float = 500.0
    max_baseline_adc: float = 900.0
    peak_height_adc: float = 10.0
    n_pre: int = N_PRE_DEFAULT

    def finalize(self, u_rec: np.ndarray, N: int, dt_ns: float,
                 row_base: dict) -> List[dict]:
        """Vetoes + per-interval integration. Returns the rows this event
        contributes (a single ``interval=-1`` veto row, or ``n_intervals``
        data rows). Byte-identical to the legacy inline tail."""
        rows: List[dict] = []
        s0 = max(self.sig_start, 0)
        s1 = min(self.sig_end, N)

        veto_reason = ""
        # ── Left-edge veto (early-time energy; ex-"fft_wraparound") ──
        n_edge = int(self.t_peak_min_ns / dt_ns)
        if n_edge > 0 and self.wrap_ratio > 0:
            edge_max = float(u_rec[:n_edge].max())
            sig_max = float(u_rec[s0:s1].max()) if s1 > s0 else 0.0
            abs_ok = True
            if self.use_wrap_abs_k and self.wrap_abs_k > 0:
                med = float(np.median(u_rec))
                noise_u = 1.4826 * float(np.median(np.abs(u_rec - med))) + 1e-12
                abs_ok = edge_max > self.wrap_abs_k * noise_u
            if sig_max > 0 and edge_max > self.wrap_ratio * sig_max and abs_ok:
                veto_reason = "left_edge"

        # ── Afterpulse veto (flat path only) ──
        if (not veto_reason and self.use_afterpulse
                and self.afterpulse_window_ns > 0):
            ap_bins = int(self.afterpulse_window_ns / dt_ns)
            noise_lvl = max(float(np.std(u_rec[:max(n_edge, 10)])), 1e-12)
            pk_idx, _ = find_peaks(u_rec[s0:s1], height=2.0 * noise_lvl,
                                   distance=max(int(5.0 / dt_ns), 1))
            if len(pk_idx) > 1:
                heights = u_rec[s0:s1][pk_idx]
                main_i = int(np.argmax(heights))
                main_pk = pk_idx[main_i]
                main_h = heights[main_i]
                for i, (pk, h) in enumerate(zip(pk_idx, heights)):
                    if i == main_i:
                        continue
                    if (main_h > self.afterpulse_height_ratio * h
                            and abs(int(pk) - int(main_pk)) <= ap_bins):
                        veto_reason = "afterpulse_veto"
                        break

        if veto_reason:
            rows.append({**row_base, "interval": -1, "area": np.nan,
                         "peak_height": np.nan, "reason": veto_reason})
            return rows

        # ── integrate over each interval ──
        width = max((s1 - s0) // self.n_intervals, 1)
        for iv in range(self.n_intervals):
            iv_lo = s0 + iv * width
            iv_hi = s0 + (iv + 1) * width if iv < self.n_intervals - 1 else s1
            seg = u_rec[iv_lo:iv_hi]
            area = float(np.sum(seg))
            peak_height = float(np.max(seg))
            row = {**row_base, "interval": iv, "area": area,
                   "peak_height": peak_height, "reason": ""}
            if self.measure_width:
                pk_idx = iv_lo + int(np.argmax(seg))
                row["width_ns"] = fwhm_ns(u_rec, pk_idx, dt_ns)
            rows.append(row)
        return rows


def run_area(events: Iterable[AreaEvent], *, dt_ns: float,
             templates: Dict[str, dict], cosine_cutoff_mhz: float,
             wavelet, wlevel, denoise_mode, thresh_k,
             area: AreaMode) -> pd.DataFrame:
    """Shared area prefix + tail over a stream of :data:`AreaEvent`.

    For each event: saturation gate → no-trigger gate → adaptive baseline →
    FFT cosine deconvolution (optional zero-pad) → optional wavelet denoise →
    ``area.finalize``. Byte-identical to the legacy ``process_area`` inner loop;
    only the I/O reader differs between the flat and object callers.
    """
    rows: List[dict] = []
    g_cache: Dict[Tuple[int, str], tuple] = {}   # (N, branch) -> (r_cal, filt, n_fft)
    t_cache: Dict[int, np.ndarray] = {}          # N -> t_full

    do_denoise = bool(wavelet) and str(wavelet).lower() not in {"none", "off", "no"}

    for run, cycle, entry_i, branch, wave, N in events:
        # Saturation is a data-quality FLAG (plateau-detected; no rail
        # threshold), carried on every row. Saturated events are KEPT and
        # integrated normally — charge is flagged, not discarded (cut
        # downstream via the flag).
        noise0 = float(np.nanstd(wave[:100])) or 1.0
        is_sat, _run = detect_saturation_plateau(wave, noise0)
        row_base = {"run": run, "cycle": cycle, "entry": entry_i,
                    "branch": branch, "saturated": bool(is_sat)}

        if wave.max() < area.no_trigger_amp:
            rows.append({**row_base, "area": np.nan, "peak_height": np.nan,
                         "reason": "no_trigger"})
            continue

        t_full = t_cache.get(N)
        if t_full is None:
            t_full = np.arange(N, dtype=np.float64) * dt_ns
            t_cache[N] = t_full

        try:
            ped, _, _, _, _ = baseline_noise_bestwindow(
                t_full, wave, dt_ns,
                window_ns=200.0, step_ns=25.0, min_sigma=1e-1,
                flat_frac_min=area.flat_frac_min, flat_k=area.flat_k,
                spike_k=area.spike_k, search_end_ns=area.search_end_ns,
                max_baseline_adc=area.max_baseline_adc,
                peak_height_adc=area.peak_height_adc,
            )
        except RuntimeError:
            ped = float(np.mean(wave[:area.n_pre]))

        y_sub = wave - ped

        key = (N, branch)
        if key not in g_cache:
            tmpl = templates.get(branch, FALLBACK_TEMPLATE)
            n_fft = _next_pow2(2 * N) if area.fft_pad else N
            r_cal = build_spe_template(n_fft, dt_ns, tmpl)
            filt = cosine_lowpass_filter(n_fft, dt_ns, cosine_cutoff_mhz)
            g_cache[key] = (r_cal, filt, n_fft)
        r_cal, filt, n_fft = g_cache[key]

        if n_fft != N:
            y_pad = np.zeros(n_fft, dtype=np.float64)
            y_pad[:N] = y_sub
            u_rec = deconvolve_hits(y_pad, r_cal, filt)[:N]
        else:
            u_rec = deconvolve_hits(y_sub, r_cal, filt)

        if do_denoise:
            u_rec = wavelet_denoise(u_rec, wavelet=wavelet, level=wlevel,
                                    mode=denoise_mode, thresh_k=thresh_k)

        rows.extend(area.finalize(u_rec, N, dt_ns, row_base))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Hits mode — per-pulse peak-finding → vetoes → trough-walked integration.
# ═══════════════════════════════════════════════════════════════════════
# Relocated here (architecture-restructure follow-up) so this module owns BOTH
# processing modes; ``process_hits.py`` is now thin format readers feeding
# :func:`run_hits`, mirroring how ``process_area.py`` feeds :func:`run_area`.
# The hits PREFIX is genuinely richer than the area prefix (no-vetoes baseline
# fallback, no_raw_pulse gate, post-saturation cutoff, the STFT denoiser branch,
# the diagnostic tee), so ``run_hits`` keeps its own loop rather than forcing a
# shared prefix with ``run_area`` — they share this module + the ``core``/utils
# math, not a tangled common front-half. The named ``VetoProfile`` catalog is the
# single source of truth for the per-denoiser thresholds (byte-identity of the
# hits goldens guards the values).


@dataclass(frozen=True)
class HitsProfile:
    """Per-denoiser parameter set for the unified hit-finding body.

    The wavelet/none path and the (formerly separate) STFT path differ only in
    these constants + a few behavior flags; everything else in
    :func:`_process_event` is shared. ``WAVELET_PROFILE`` reproduces the strict
    Na22 wavelet-hits constants; ``STFT_PROFILE`` the relaxed STFT ones. Both are
    built from the :data:`~pmtdeconv.presets.veto_profiles.VETO_PROFILES` catalog.
    """
    denoiser_kind: str            # "wavelet" (honors no_denoise) | "stft"
    # baseline quality gate
    flat_frac_min: float
    flat_k: float
    spike_k: float
    max_baseline_adc: float
    peak_height_adc: float
    # vetoes / windows
    no_trigger_amp: float
    raw_snr_mult: float           # raw-peak cut: raw_peak < raw_snr_mult * noise
    t_peak_min_ns: float
    t_peak_max_ns: float
    afterpulse_window_ns: float
    afterpulse_height_ratio: float
    post_sat_window_ns: float
    post_sat_sigma_mult: float
    post_sat_offset_ns: float
    min_charge_purity: float
    # behavior switches
    allow_no_vetoes: bool         # if False, --no-vetoes is ignored (STFT always vetoes)
    allow_peak_order: bool        # if False, peak_id is always chronological
    allow_denoise_raw: bool       # if False, --denoise-raw is ignored (STFT never)
    baseline_search_end_ns: float = 600.0
    pad_ns: float = 18.0


WAVELET_PROFILE = HitsProfile(
    denoiser_kind="wavelet",
    allow_no_vetoes=True, allow_peak_order=True, allow_denoise_raw=True,
    **VETO_PROFILES.lookup(name="wavelet_hits").hits_fields(),
)

STFT_PROFILE = HitsProfile(
    denoiser_kind="stft",
    allow_no_vetoes=False, allow_peak_order=False, allow_denoise_raw=False,
    **VETO_PROFILES.lookup(name="stft_hits").hits_fields(),
)


def _denoise_u(u, args):
    """Apply the configured wavelet denoiser (or pass through)."""
    if args.no_denoise:
        return u
    return wavelet_denoise(u, wavelet=args.wavelet, level=args.wlevel,
                           mode=args.wmode, thresh_k=args.wthresh_k)


def _process_event(rec, args, profile, ch_templates, cosine_cutoff_mhz,
                   sink, g_cache):
    """Run the peak-finding deconvolution body on one ``EventRecord``.

    Returns the list of result-row dicts the event contributes. Polarity is
    already applied on ``rec.wave`` by the event source — no flip here.
    ``profile`` (:class:`HitsProfile`) carries the per-denoiser constants +
    behavior flags; byte-identity is preserved for both ``WAVELET_PROFILE`` and
    ``STFT_PROFILE`` (including the no-vetoes baseline fallback ``b0,b1=0,100``).
    """
    wave = rec.wave
    t = rec.t
    N = rec.N
    dt_ns = rec.dt_ns
    entry_i = rec.entry
    branch = rec.branch

    no_vetoes = bool(args.no_vetoes) if profile.allow_no_vetoes else False
    snr_peak_min = args.snr_peak_min
    peak_distance_ns = args.peak_distance_ns
    pad_ns = profile.pad_ns

    rows = []
    row_id = {"run": rec.run, "cycle": rec.cycle, "det": rec.det,
              "chan": rec.chan, "branch": branch, "entry": entry_i}

    # Saturation is a DATA-QUALITY FLAG, not a veto. Detect the clipped-top
    # plateau (universal — no per-detector rail threshold) and carry
    # ``saturated`` on every row. Saturated events are KEPT and fully processed:
    # their edge timing is recovered (validated robust to clipping); only their
    # charge is unreliable, and is cut downstream via the flag (e.g.
    # ``plot energy --exclude-saturated``). This replaces the old
    # ``wave.max() >= sat_threshold`` drop, which both needed each detector's
    # rail AND silently discarded usable timing.
    noise0 = float(np.nanstd(wave[:100])) or 1.0
    is_sat, _run = detect_saturation_plateau(wave, noise0)
    row_id["saturated"] = bool(is_sat)

    if np.max(wave) < profile.no_trigger_amp and not no_vetoes:
        rows.append({**row_id, "success": False, "reason": "no_trigger"})
        return rows

    try:
        ped_i, noise_i, (b0, b1), wbins, baseline_passed = \
            baseline_noise_bestwindow(
                t, wave, dt_ns,
                window_ns=200.0, step_ns=25.0, min_sigma=1e-1,
                flat_frac_min=profile.flat_frac_min,
                flat_k=profile.flat_k, spike_k=profile.spike_k,
                search_end_ns=profile.baseline_search_end_ns,
                max_baseline_adc=profile.max_baseline_adc,
                peak_height_adc=profile.peak_height_adc,
            )
    except RuntimeError:
        if not no_vetoes:
            rows.append({**row_id, "success": False, "reason": "baseline_failed"})
            return rows
        fb = wave[:100]
        ped_i = float(np.median(fb))
        noise_i = float(np.std(fb)) or 1.0
        b0, b1 = 0, 100
        baseline_passed = False

    if not baseline_passed and not no_vetoes:
        rows.append({**row_id, "success": False, "reason": "baseline_quality"})
        return rows

    raw_peak = float(np.max(wave) - ped_i)
    if noise_i > 0 and raw_peak < profile.raw_snr_mult * noise_i and not no_vetoes:
        rows.append({**row_id, "success": False, "reason": "no_raw_pulse"})
        return rows

    post_sat_bins = max(int(profile.post_sat_window_ns / dt_ns), 1)
    local_mean = sliding_local_mean(wave, post_sat_bins)
    shifted_mask = (ped_i - local_mean) > profile.post_sat_sigma_mult * noise_i
    shifted_idx = np.where(shifted_mask)[0]
    post_sat_offset_bins = int(profile.post_sat_offset_ns / dt_ns)
    post_sat_cutoff = (
        max(int(shifted_idx[0]) - post_sat_offset_bins, 0)
        if len(shifted_idx) > 0 else N
    )

    cache_key = (N, branch)
    if cache_key not in g_cache:
        tmpl = ch_templates.get(branch, FALLBACK_TEMPLATE)
        r_calib = build_spe_template(N, dt_ns, tmpl)
        filt = cosine_lowpass_filter(N, dt_ns, cosine_cutoff_mhz)
        g_cache[cache_key] = (r_calib, filt)
    r_calib, filt = g_cache[cache_key]

    y_sub = (wave - ped_i).astype(np.float64)
    if profile.allow_denoise_raw and args.denoise_raw and not args.no_denoise:
        y_sub = _denoise_u(y_sub, args)

    u_rec = deconvolve_hits(y_sub, r_calib, filt)

    # ── diagnostic tee (Q5): short-circuits when sink inactive ──
    u_pre_denoise = None
    if sink.should_dump(entry_i):
        u_pre_denoise = u_rec.copy()
        sink.save_array(entry_i, branch, "raw_waveform", wave)
        sink.save_array(entry_i, branch, "y_sub", y_sub)
        sink.save_array(entry_i, branch, "cosine_filter", np.abs(filt))
        sink.save_array(entry_i, branch, "spe_kernel", r_calib)
        sink.save_array(entry_i, branch, "deconv_pre_denoise", u_pre_denoise)

    if profile.denoiser_kind == "stft":
        u_rec = stft_denoise(
            u_rec, dt_ns,
            nperseg_ns=getattr(args, "stft_nperseg_ns", 20.0),
            freq_cutoff_hz=getattr(args, "stft_freq_cutoff_mhz", 175.0) * 1e6,
            amp_threshold=getattr(args, "stft_amp_threshold", 10.0),
        )
    else:
        u_rec = _denoise_u(u_rec, args)
    if sink.should_dump(entry_i):
        sink.save_array(entry_i, branch, "deconv_post_denoise", u_rec)

    bl_lo = max(int(b0 / dt_ns), 0)
    bl_hi = min(int(b1 / dt_ns) + 1, N)
    u_rec_bl = u_rec[bl_lo:bl_hi]
    u_rec_noise = max(
        1.4826 * float(np.median(np.abs(u_rec_bl - np.median(u_rec_bl)))),
        1e-12,
    )
    peak_thresh = snr_peak_min * u_rec_noise
    dist_bins = max(int(peak_distance_ns / dt_ns), 1)
    peaks, _ = find_peaks(u_rec, height=peak_thresh, distance=dist_bins)

    if sink.should_dump(entry_i):
        sink.save_array(entry_i, branch, "found_peaks", peaks)
        sink.save_peaks(
            entry_i, branch, peaks,
            u_rec[peaks] if len(peaks) else np.array([]),
            [], peak_thresh, u_rec_noise)
        sink.save_truth_match(entry_i, branch)
        sink.save_pdf(
            entry_i, branch,
            raw=wave, t_ns=t,
            cosine_filter=np.abs(filt),
            deconv_pre_denoise=(u_pre_denoise
                                if u_pre_denoise is not None else u_rec),
            deconv_post_denoise=u_rec,
            peaks=peaks,
            ped=ped_i,
        )

    row_base = {
        **row_id, "success": False, "reason": "",
        "peak_id": np.nan, "t_peak": np.nan, "height": np.nan,
        "charge": np.nan, "win_lo_ns": np.nan, "win_hi_ns": np.nan,
        "baseline_ped": ped_i, "baseline_sig": noise_i,
        "baseline_t0": b0, "baseline_t1": b1,
        "u_rec_noise": u_rec_noise,
    }
    if len(peaks) == 0:
        rows.append({**row_base, "reason": "no_peaks_found"})
        return rows

    # Peak ordering (decision 8). STFT has no reorder (always chronological).
    if profile.allow_peak_order:
        peak_order_arg = getattr(args, "peak_order", "auto")
        if peak_order_arg in (None, "auto"):
            effective = "tallest" if no_vetoes else "chrono"
        else:
            effective = peak_order_arg
        if effective == "tallest":
            order = np.argsort(-u_rec[peaks])
            peaks = peaks[order]

    pad_bins = int(pad_ns / dt_ns)
    windows = find_integration_windows(u_rec, peaks, pad_bins)
    pulse_widths = []
    for pk in peaks:
        lo_tr = _walk_to_trough_left(u_rec, pk, pad_bins)
        hi_tr = _walk_to_trough_right(u_rec, pk, pad_bins)
        pulse_widths.append(hi_tr - lo_tr)

    for pk_i, pk in enumerate(peaks):
        lo, hi = windows[pk_i]
        if no_vetoes:
            charge = float(np.sum(u_rec[lo:hi]))
            success, reason = True, "no_veto"
        else:
            success, reason, charge = apply_peak_vetoes(
                pk, pk_i, peaks, u_rec, t, windows, pulse_widths, N,
                t_peak_min_ns=profile.t_peak_min_ns,
                t_peak_max_ns=profile.t_peak_max_ns,
                afterpulse_window_ns=profile.afterpulse_window_ns,
                afterpulse_height_ratio=profile.afterpulse_height_ratio,
                post_sat_cutoff_bin=post_sat_cutoff,
                dt_ns=dt_ns,
                raw_wave=wave,
                raw_ped=ped_i,
                min_raw_amp=profile.no_trigger_amp,
                min_charge_purity=profile.min_charge_purity,
            )
        t_peak_ns = float(t[pk])
        prow = dict(row_base)
        prow.update({
            "peak_id": pk_i + 1, "t_peak": t_peak_ns,
            "height": float(u_rec[pk]), "charge": charge,
            "win_lo_ns": float(t[lo]),
            "win_hi_ns": float(t[min(hi, N - 1)]),
            "success": success, "reason": reason,
        })
        rows.append(prow)
    return rows


def run_hits(events: Iterable, *, args, profile: HitsProfile,
             templates: Dict[str, dict], cosine_cutoff_mhz: float,
             sink) -> List[dict]:
    """Run the hits engine over a stream of ``EventRecord`` (mirrors
    :func:`run_area`). The per-channel kernel/filter cache is shared across the
    whole stream. Returns the flat list of result-row dicts; the caller
    (``process_hits`` reader) writes the CSV tables + veto-rate report."""
    g_cache: Dict[Tuple[int, str], tuple] = {}
    rows: List[dict] = []
    for rec in events:
        rows.extend(_process_event(rec, args, profile, templates,
                                   cosine_cutoff_mhz, sink, g_cache))
    return rows
