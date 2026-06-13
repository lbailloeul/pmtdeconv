"""Detector — the resolved per-run processing configuration.

:meth:`Detector.from_settings` is the ONE place that reads a resolved
:class:`~pmtdeconv.presets.Settings` for the hit-finding pipeline, replacing
``commands/pulse.py::_build_legacy_args`` (the SimpleNamespace round-trip that
caused the 2026-06-10 polarity-bug class). The result is a typed carrier: it
exposes exactly the attribute names the
``pipeline.process_hits.process_run*`` functions read (so it is a drop-in for
the old namespace) and additionally carries a resolved :class:`Template` (with
provenance) for the manifest.

Values are identical to the legacy ``_build_legacy_args`` output — this is the
"pure plumbing" step; the goldens stay byte-identical. The per-(mode, denoiser)
veto/baseline constants still live in ``process_hits.HitsProfile``
(``WAVELET_PROFILE`` / ``STFT_PROFILE``); step 5 lifts those into catalogued
``VetoProfile``s resolved here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pmtdeconv.pipeline.template import Template
from pmtdeconv.presets.veto_profiles import VETO_PROFILES


@dataclass
class Detector:
    """Typed processing config for one ``hits`` invocation.

    Field names mirror what ``process_run`` / ``process_run_object`` /
    ``process_run_stft`` read off their ``args`` parameter, so a ``Detector``
    is a drop-in carrier. ``out_dir`` / ``dt_ns`` may be refined by the command
    after construction (explicit ``--out-dir``, object-format ``dt_ns=="auto"``
    per-file resolution) exactly as the legacy code did.
    """
    # --- run selection / IO ---
    runs: List[int]
    entry_stop: Optional[int]
    dt_ns: float
    polarity: int
    sat_thresh: float
    format: str
    file_glob: Optional[str]
    tree: Optional[str]
    branches: Optional[List[str]]
    data_root: Optional[str]
    data_subdir: Optional[str]
    out_dir: Optional[str]
    out_suffix: str
    # --- deconvolution / filter ---
    cosine_cutoff_mhz: Optional[float]
    fit_csv: Optional[str]
    # --- denoiser ---
    denoiser: str
    no_denoise: bool
    wavelet: str
    wlevel: int
    wmode: str
    wthresh_k: float
    denoise_raw: bool
    stft_nperseg_ns: float
    stft_freq_cutoff_mhz: float
    stft_amp_threshold: float
    # --- peaks / vetoes ---
    snr_peak_min: float
    peak_distance_ns: float
    no_vetoes: bool
    peak_order: str
    veto_warn_frac: float
    # name of the resolved VetoProfile (catalog, by denoiser) — manifest provenance
    veto_profile: str = "wavelet_hits"
    # --- resolved template (provenance for the manifest) ---
    template: Optional[Template] = None

    @classmethod
    def from_settings(cls, settings, provenance=None, *, mode: str = "hits"
                      ) -> "Detector":
        out_suffix = ""
        if settings.config_preset_name:
            out_suffix += "__" + settings.config_preset_name
        if settings.out_suffix:
            out_suffix += settings.out_suffix

        template = Template.from_settings(settings, provenance, command=mode)

        denoiser = settings.denoiser
        branches = (settings.branch_pattern
                    if isinstance(settings.branch_pattern, list) else None)

        return cls(
            runs=settings.runs or [65213],
            entry_stop=settings.entry_stop,
            # auto -> 0.5051 placeholder (matches legacy _build_legacy_args);
            # the command overrides per-file for object-format dt_ns=="auto".
            dt_ns=(float(settings.dt_ns) if settings.dt_ns != "auto" else 0.5051),
            polarity=int(settings.polarity),
            sat_thresh=float(settings.sat_thresh_mv),
            format=settings.format,
            file_glob=settings.file_glob,
            tree=settings.tree,
            branches=branches,
            data_root=(str(settings.data_root) if settings.data_root else None),
            data_subdir=None,
            out_dir=(str(settings.out_dir) if settings.out_dir else None),
            out_suffix=out_suffix,
            cosine_cutoff_mhz=(float(settings.cosine_cutoff_mhz)
                               if settings.cosine_cutoff_mhz is not None else None),
            fit_csv=template.fit_csv,
            denoiser=denoiser,
            no_denoise=(denoiser == "none"),
            wavelet=settings.wavelet,
            wlevel=settings.wlevel,
            wmode=settings.wmode,
            wthresh_k=settings.wthresh_k,
            denoise_raw=settings.denoise_raw,
            stft_nperseg_ns=float(settings.stft_nperseg_ns),
            stft_freq_cutoff_mhz=float(settings.stft_freq_cutoff_mhz),
            stft_amp_threshold=float(settings.stft_amp_threshold),
            snr_peak_min=settings.snr_peak_min,
            peak_distance_ns=settings.peak_distance_ns,
            no_vetoes=settings.no_vetoes,
            peak_order=settings.peak_order,
            veto_warn_frac=float(getattr(settings, "veto_warn_frac", 0.5)),
            veto_profile=VETO_PROFILES.resolve_name(denoiser),
            template=template,
        )
