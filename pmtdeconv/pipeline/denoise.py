"""Denoiser dispatch — wavelet | stft | none."""
from __future__ import annotations


import numpy as np

from pmtdeconv.core.dsp import wavelet_denoise, stft_denoise


def denoise(
    u_rec: np.ndarray,
    *,
    denoiser: str = "wavelet",
    dt_ns: float = 0.5051,
    # wavelet knobs
    wavelet: str = "sym8",
    wlevel: int = 5,
    wmode: str = "hard",
    wthresh_k: float = 1.0,
    # stft knobs — defaults match Settings (unified 2026-07-28; previously
    # 200 MHz / 3.0 here, so wrapper callers silently diverged from the
    # pipeline's 175 MHz / 10.0). Amplitude threshold 5 since 2026-10.
    stft_nperseg_ns: float = 20.0,
    stft_freq_cutoff_hz: float = 175e6,
    stft_amp_threshold: float = 5.0,
    stft_threshold_mode: str = "median",
    stft_fixed_threshold: float | None = None,
) -> np.ndarray:
    """Apply the selected denoiser to a deconvolved waveform.

    ``denoiser="none"`` returns ``u_rec`` unchanged (ablation baseline).
    """
    if denoiser == "wavelet":
        return wavelet_denoise(u_rec, wavelet=wavelet, level=wlevel,
                               mode=wmode, thresh_k=wthresh_k)
    if denoiser == "stft":
        return stft_denoise(u_rec, dt_ns=dt_ns,
                            nperseg_ns=stft_nperseg_ns,
                            freq_cutoff_hz=stft_freq_cutoff_hz,
                            amp_threshold=stft_amp_threshold,
                            threshold_mode=stft_threshold_mode,
                            fixed_threshold=stft_fixed_threshold)
    if denoiser == "none":
        return u_rec
    raise ValueError(f"Unknown denoiser: {denoiser!r} "
                     f"(expected wavelet|stft|none)")
