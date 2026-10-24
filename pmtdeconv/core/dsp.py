#!/usr/bin/env python3
"""Core signal processing: FFT deconvolution, STFT/wavelet denoising, and the
adaptive baseline estimator.

Moved verbatim from ``pmtdeconv/utils.py`` (2026-05-31 domain split).
"""
import numpy as np
from numpy import fft
from scipy.signal import find_peaks, stft, istft


# ═══════════════════════════════════════════════════════════════════
# FFT deconvolution
# ═══════════════════════════════════════════════════════════════════

def deconvolve_hits(m_t, r_calib, filt, eps=1e-12):
    """FFT deconvolution: u_rec = IFFT(FFT(m) * filter / FFT(r)).

    ``eps`` is a divide-by-zero guard on the (complex) kernel spectrum, NOT a
    regularizer: added to complex R it does not bound ``|filter/R|`` (R can
    pass arbitrarily close to zero at any phase). The low-pass ``filt`` is the
    de-facto regularization. A proper Tikhonov/Wiener form
    (``conj(R)·M·filt / (|R|² + ε)``) is a documented follow-up — changing it
    would change every output byte.
    """
    M = fft.fft(m_t)
    R = fft.fft(r_calib)
    U_rec = M * filt / (R + eps)
    return np.real(fft.ifft(U_rec))


# ═══════════════════════════════════════════════════════════════════
# STFT denoising
# ═══════════════════════════════════════════════════════════════════

def stft_denoise(u_rec, dt_ns, nperseg_ns=20.0, freq_cutoff_hz=175e6,
                 amp_threshold=10.0, threshold_mode="median",
                 fixed_threshold=None):
    """
    STFT denoising: zero out high frequencies and small amplitudes.
    Returns the denoised signal (same length as u_rec).

    Defaults match the production ``Settings`` values (20 ns / 175 MHz / 10.0);
    unified 2026-07-28 — they previously diverged (200 MHz / 3.0), so wrapper
    callers silently got non-pipeline behavior.

    Known limitations (documented, deliberately unchanged — this is one of the
    instruments under benchmark):
    - the hop is pinned to ~1 ns (1 sample at sub-ns dt): a heavily
      over-complete frame, O(N·nperseg) cost;
    - ``noise_level`` is a single global median over the whole time-frequency
      plane, so the effective threshold depends on record length and signal
      occupancy;
    - hard-zeroing sub-threshold bins deletes genuine small (few-PE) pulses
      outright and yields compact "island" pulses whose FWHM is not comparable
      to un-thresholded output.
    """
    fs = 1.0 / (dt_ns * 1e-9)
    nperseg = int(nperseg_ns / dt_ns)
    noverlap = nperseg - max(int(1.0 / dt_ns), 1)
    f, t_seg, Zxx = stft(u_rec, fs=fs, window=('gaussian', nperseg / 6),
                         nperseg=nperseg, noverlap=noverlap)
    if threshold_mode not in ("median", "fixed"):
        # Validate rather than fall through: the median branch is the `else`,
        # so an unrecognized mode (e.g. the pre-2026-09-18 spelling
        # "absolute" left in a card or preset) would otherwise silently run
        # the WRONG algorithm.
        hint = (" (renamed to 'fixed' on 2026-09-18)"
                if threshold_mode == "absolute" else "")
        raise ValueError(f"stft threshold_mode {threshold_mode!r} is not "
                         f"'median' or 'fixed'{hint}")
    if threshold_mode == "fixed":
        # Tang et al. 2411.07620 §4.2.2 — TIME-COLUMN selection: the
        # low-frequency band (≤ freq_cutoff_hz, their <50 MHz) is only the
        # DETECTION band. Time columns whose band amplitude clears the FIXED
        # threshold (their 0.02 A.U.) keep their FULL spectrum (preserving
        # ~3 ns spike structure the band itself could never represent);
        # all other columns are zeroed. Record-length independent; the
        # threshold is in scipy STFT units and must be calibrated for our
        # scaling (sim.tang2024_validate.paper_stft_fixed_threshold).
        thr = float(fixed_threshold if fixed_threshold is not None
                    else amp_threshold)
        det = np.abs(Zxx[f <= freq_cutoff_hz, :])
        col_ok = det.max(axis=0) >= thr if det.size else \
            np.zeros(Zxx.shape[1], dtype=bool)
        # Dilate the kept-column mask by half a window: the Gaussian STFT
        # window smears a spike's band energy across ~±nperseg/2 columns, so
        # bare thresholding keeps only the spike-center columns and truncates
        # the wings (charge loss). The paper's fig-10c candidate mask covers
        # the full pulse region; dilation reproduces that.
        if col_ok.any():
            w = max(nperseg // 2, 1)
            kern = np.ones(2 * w + 1)
            col_ok = np.convolve(col_ok.astype(float), kern,
                                 mode="same") > 0
        Zxx_filtered = Zxx.copy()
        Zxx_filtered[:, ~col_ok] = 0.0
    else:
        Zxx_filtered = Zxx.copy()
        Zxx_filtered[f > freq_cutoff_hz, :] = 0.0
        abs_Zxx = np.abs(Zxx_filtered)
        noise_level = np.median(abs_Zxx[abs_Zxx > 0])
        Zxx_filtered[abs_Zxx < amp_threshold * noise_level] = 0.0
    _, u_denoised = istft(Zxx_filtered, fs=fs, window=('gaussian', nperseg / 6),
                          nperseg=nperseg, noverlap=noverlap)
    N = len(u_rec)
    if len(u_denoised) >= N:
        u_denoised = u_denoised[:N]
    else:
        u_denoised = np.pad(u_denoised, (0, N - len(u_denoised)))
    return u_denoised


# ═══════════════════════════════════════════════════════════════════
# Baseline estimation
# ═══════════════════════════════════════════════════════════════════

def baseline_noise_bestwindow(
    t, y, dt_ns,
    window_ns=200.0, step_ns=25.0, min_sigma=1e-6,
    flat_frac_min=0.98, flat_k=3.0, spike_k=8.0,
    max_baseline_adc=500.0,
    search_end_ns=None,
    peak_height_adc=50.0,
    peak_margin_ns=50.0,
    window_reduce_ns=34.0,
    min_window_ns=64.0,
    sigma_thresh_scale=1.0,
):
    """
    Cascading baseline estimator with logarithmic noise threshold.

    Search strategy:
      1. Try all window sizes (200 → 166 → 132 → 98 → 64 ns) in first 200 ns.
      2. If nothing passes, repeat in first search_end_ns (default 600 ns).
      3. Among candidates passing the log-σ threshold, take the earliest.
      4. If nothing passes threshold, return best noisy fallback with passed=False.

    Logarithmic σ threshold: 7 ADC at 200 ns → 1.5 ADC at 64 ns.

    Units: despite the historical ``_adc`` names, ``max_baseline_adc`` /
    ``peak_height_adc`` and the σ thresholds are in mV. The log-σ anchors
    (7 mV @ 200 ns → 1.5 mV @ 64 ns) are absolute, detector-independent
    constants — a high-noise detector (e.g. the σ=12 mV ``noisy`` regime) can
    never pass this gate; exposing the anchors is a tracked follow-up. Note the
    forward scan returns the EARLIEST passing window, not the quietest.

    Returns:
        (baseline, sigma, (t_start, t_end), window_bins, passed)
        where passed=True if the log-σ threshold was met.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    step = max(int(step_ns / dt_ns), 1)

    # Logarithmic sigma threshold: 7 ADC at 200 ns → 1.5 ADC at 64 ns
    _A_LOG = 5.5 / np.log(200.0 / 64.0)
    _B_LOG = 7.0 - _A_LOG * np.log(200.0)

    # Detect peaks to exclude from baseline search
    rough_baseline = float(np.median(y))
    peak_indices, _ = find_peaks(
        y, height=rough_baseline + peak_height_adc,
        distance=max(int(5.0 / dt_ns), 1),
    )
    in_peak = np.zeros(len(y), dtype=bool)
    margin_bins = int(peak_margin_ns / dt_ns)
    for pk in peak_indices:
        lo = max(0, pk - margin_bins)
        hi = min(len(y), pk + margin_bins + 1)
        in_peak[lo:hi] = True

    # Search regions: first 200 ns, then first search_end_ns
    search_limits = [int(200.0 / dt_ns)]
    if search_end_ns is not None:
        search_limits.append(int(search_end_ns / dt_ns))

    overall_best = None

    for search_end_bin in search_limits:
        search_end_bin = min(search_end_bin, len(y))  # clamp to waveform length
        cur_win_ns = window_ns
        while cur_win_ns >= min_window_ns:
            w = max(int(cur_win_ns / dt_ns), 10)
            if search_end_bin < w:
                cur_win_ns -= window_reduce_ns
                continue
            # sigma_thresh_scale scales the absolute-mV anchors (default 1.0 =
            # legacy; >1 admits noisier detectors — profile knob
            # `baseline_sigma_scale`).
            sigma_thresh = (_A_LOG * np.log(cur_win_ns) + _B_LOG) \
                * sigma_thresh_scale
            earliest_good = None
            best_noisy = None
            for s in range(0, search_end_bin - w + 1, step):
                if np.any(in_peak[s:s + w]):
                    continue
                seg = y[s:s + w]
                base = float(np.median(seg))
                mad = float(np.median(np.abs(seg - base)))
                sig = max(1.4826 * mad, min_sigma)
                if abs(base) > max_baseline_adc:
                    continue
                if np.max(np.abs(seg - base)) > spike_k * sig:
                    continue
                flat = np.abs(seg - base) < (flat_k * sig)
                if flat.mean() < flat_frac_min:
                    continue
                cand = (sig, s, base, w)
                if sig <= sigma_thresh:
                    if earliest_good is None:
                        earliest_good = cand
                        break  # scanning forward — first hit is earliest
                else:
                    if best_noisy is None or sig < best_noisy[0]:
                        best_noisy = cand
            if earliest_good is not None:
                sig, s, base, w = earliest_good
                return base, sig, (float(t[s]), float(t[s + w - 1])), w, True
            if best_noisy is not None:
                if overall_best is None or best_noisy[0] < overall_best[0]:
                    overall_best = best_noisy
            cur_win_ns -= window_reduce_ns

    if overall_best is not None:
        sig, s, base, w = overall_best
        return base, sig, (float(t[s]), float(t[s + w - 1])), w, False

    raise RuntimeError("No baseline window found")


def sample_noise_sigma(x) -> float:
    """Sample-to-sample noise: robust σ of the first differences / √2.

    Blind to slow baseline structure (tails or recovery of earlier pulses,
    pile-up bumps), which inflates a plain σ of the same window."""
    d = np.diff(np.asarray(x, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    if mad == 0:
        # Integer-ADC records with sub-count noise: most differences are exactly
        # 0, so the MAD is 0 and the raw-SNR gate would switch itself off.
        return float(np.std(d) / np.sqrt(2.0))
    return float(1.4826 * mad / np.sqrt(2.0))


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values in a bool mask."""
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int((ends - starts).max())


def detect_saturation_plateau(wave, noise_sigma, *,
                              k_samples: int = 10,
                              eps_sigma_frac: float = 0.5,
                              rel_tol: float = 1e-4):
    """Detect ADC saturation by its flat-plateau signature — universal, no rail.

    A clipped pulse pins to the digitizer rail, producing a run of (near-)
    identical samples at the waveform extreme; a genuine unclipped peak is
    rounded (only 1-3 samples at the max). This replaces the fixed
    ``wave.max() >= sat_thresh`` gate, which needed each detector's rail and
    conflated "tall" with "clipped".

    Returns ``(is_saturated, run_len)`` where ``run_len`` is the longest run of
    consecutive samples within ``eps`` of ``vmax`` (the max of the
    polarity-corrected waveform), and ``eps = eps_sigma_frac * noise_sigma``.
    Saturated iff ``run_len >= k_samples``.

    The tolerance is deliberately SUB-NOISE (``eps_sigma_frac`` < 1): a hard
    digital clip yields identical samples (flatter than the noise floor), so a
    fraction of σ separates it from a rounded, noisy peak. ``eps ≈ σ`` over-flags
    broad unclipped tops; ``eps ≈ 0.5σ`` does not.

    ``k_samples`` defaults to 10 (~5 ns at na22 dt). Real saturation is a LONG
    plateau: synthetic level5 clips run 25-41 samples, real na22 Run 65213 clips
    run 150+. The worst rounded-peak false positive — the legacy bare-Gaussian
    golden fixture (σ=25 samples, far broader than any real PMT pulse) — reaches
    only a 7-sample near-max run, so k=10 leaves a comfortable margin on BOTH
    sides (rejects ≤7, catches ≥25). Validated on real na22 Run 65213 (rail =
    hard 950 mV clip): ε=0.5σ reproduces the rail threshold to ~1% with ~97%
    per-event overlap and ZERO rail knowledge, and works for +ve / -ve / bipolar
    pulses once polarity-corrected (vmax = rail). NaN samples (DRS dead cells)
    compare False and so conservatively break runs.
    """
    w = np.asarray(wave, dtype=float)
    finite = w[np.isfinite(w)]
    if finite.size == 0:
        return False, 0
    vmax = float(finite.max())
    # A digital clip repeats the rail value exactly (950.0 mV on our DRS4), so
    # the tolerance is also capped at rel_tol·|vmax|: 0.5σ alone let the
    # rounded top of a broad few-mV pulse count as a plateau whenever σ was
    # inflated by slow baseline wander (27% of Cd109 CH2, 2026-10-23).
    eps = max(min(eps_sigma_frac * float(noise_sigma), rel_tol * abs(vmax)), 1e-9)
    near = np.abs(w - vmax) < eps        # NaN -> False -> breaks the run
    run_len = _longest_true_run(near)
    return run_len >= int(k_samples), run_len


# ═══════════════════════════════════════════════════════════════
# Wavelet denoiser (used by Na22 deconvolution scripts)
# ═══════════════════════════════════════════════════════════════
def is_saturated(wave) -> bool:
    """The one clipping rule hits, area, raw-area, plot energy and vetoes
    calibrate share: a plateau at the extreme, noise from the first 100
    samples (~50 ns at na22 dt, ~143 ns at cd109)."""
    noise0 = float(np.nanstd(wave[:100])) or 1.0
    return bool(detect_saturation_plateau(wave, noise0)[0])


def wavelet_denoise(
    u: np.ndarray,
    wavelet: str = "sym6",
    level: int = 5,
    mode: str = "soft",
    thresh_k: float = 1.0,
) -> np.ndarray:
    """
    Stationary (undecimated) wavelet denoiser — shift-invariant.

    Thresholding strategy: per-decomposition-level, robust noise estimate
    σ = 1.4826 · MAD(detail), threshold = thresh_k · σ · √(2·log N)
    (Donoho's universal threshold, scaled).

    Parameters
    ----------
    u : 1-D array
        Input signal.
    wavelet : str
        Any valid pywt discrete wavelet. `sym8` is the Liu 2024 PMT-tested
        baseline; `haar` for the most localized output.
    level : int
        SWT decomposition depth. Requires len(u) divisible by 2**level
        (we zero-pad + crop to enforce that).
    mode : 'soft' | 'hard' | 'garrote'
        Threshold function. `soft` smooths, `hard` preserves peak amplitude.
    thresh_k : float
        Multiplier on the universal threshold. 1.0 = vanilla Donoho;
        2.0–3.0 for STFT-like sparse output.

    Returns
    -------
    Denoised 1-D array with the same length as `u`.

    Raises
    ------
    ImportError if PyWavelets is not installed.

    Notes
    -----
    Known limitations (documented, deliberately unchanged — this is one of the
    instruments under benchmark):
    - the universal threshold √(2·ln N) assumes independent coefficients; SWT
      details are strongly correlated, so the effective threshold is more
      aggressive than the nominal Donoho level;
    - ``mode="edge"`` padding to a multiple of 2**level introduces a boundary
      step the detail bands see as a discontinuity (cropped afterwards, so
      silent);
    - a level with σ ≤ 0 (MAD = 0) is passed through untouched with no
      diagnostic.
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "PyWavelets is required for wavelet_denoise. "
            "Install it with: pip install PyWavelets"
        ) from e

    N = len(u)
    if N == 0:
        return u

    if wavelet in pywt.wavelist(kind="continuous"):
        raise ValueError(
            f"Wavelet '{wavelet}' is continuous-only (CWT). "
            "Use a discrete wavelet: sym*, db*, coif*, bior*, haar."
        )

    # SWT requires length divisible by 2**level
    pad_total = (-N) % (2 ** level)
    u_work = np.pad(u, (0, pad_total), mode="edge") if pad_total else np.asarray(u, dtype=np.float64)

    # norm=True requires orthogonality; biorthogonal wavelets need norm=False.
    try:
        is_ortho = bool(pywt.Wavelet(wavelet).orthogonal)
    except Exception:
        is_ortho = True

    coeffs = pywt.swt(u_work, wavelet, level=level,
                      trim_approx=False, norm=is_ortho)
    new_coeffs = []
    for (cA, cD) in coeffs:
        mad = float(np.median(np.abs(cD - np.median(cD))))
        sigma = 1.4826 * mad
        if sigma <= 0:
            new_coeffs.append((cA, cD))
            continue
        thr = thresh_k * sigma * np.sqrt(2.0 * np.log(len(cD)))
        cD_thr = pywt.threshold(cD, thr, mode=mode)
        new_coeffs.append((cA, cD_thr))

    u_rec = pywt.iswt(new_coeffs, wavelet, norm=is_ortho)
    return np.asarray(u_rec[:N], dtype=np.float64)
