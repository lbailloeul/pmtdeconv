"""Detector — the resolved per-run processing configuration.

:meth:`Detector.from_settings` is the ONE place that reads a resolved
:class:`~pmtdeconv.presets.Settings` for the hit-finding pipeline, replacing
``commands/hits.py::_build_legacy_args`` (the SimpleNamespace round-trip that
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

from pmtdeconv.pipeline.gain import Gain
from pmtdeconv.pipeline.template import Template
from pmtdeconv.presets.veto_profiles import VETO_PROFILES


@dataclass
class Detector:
    """Typed processing config for one ``hits`` invocation.

    Field names mirror what ``process_run`` / ``process_run_object`` /
    ``process_run_stft`` read off their ``args`` parameter, so a ``Detector``
    is a drop-in carrier. ``out_dir`` may be refined by the command after
    construction (explicit ``--out-dir``).
    """
    # --- run selection / IO ---
    runs: List[int]
    entry_stop: Optional[int]
    dt_ns: Optional[float]          # None: read from each run file (object, auto)
    polarity: int
    format: str
    file_glob: Optional[str]
    tree: Optional[str]
    branches: Optional[List[str]]
    data_root: Optional[str]
    out_dir: Optional[str]
    out_suffix: str
    # --- deconvolution / filter ---
    cosine_cutoff_mhz: Optional[float]
    fit_csv: Optional[str]
    fft_pad: bool
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
    stft_threshold_mode: str
    stft_fixed_threshold: Optional[float]
    # --- peaks / vetoes ---
    snr_peak_min: float
    peak_distance_ns: float
    no_vetoes: bool
    peak_order: str
    veto_warn_frac: float
    # name of the resolved VetoProfile — manifest provenance
    veto_profile: str = "wavelet_hits"
    # 0–20 PE re-scope (2026-07-28): the RESOLVED profile object + how it
    # resolved. `veto_profile_explicit` = user named it via --veto-profile /
    # card — process_hits then builds the HitsProfile from it with ALL
    # behavior switches enabled; default paths keep the legacy per-denoiser
    # constants (byte-identical).
    veto_profile_obj: Optional[object] = None
    veto_profile_source: str = "builtin"
    veto_profile_version: str = "0"
    veto_profile_explicit: bool = False
    # --- 0–20 PE regime (2026-07-28) ---
    pe_band: tuple = (0.0, 20.0)
    in_regime_only: bool = False
    # resolved per-branch 1-PE charge anchor (pipeline.gain.Gain; may be
    # unresolved — nPE columns blank)
    gain: Optional[object] = None
    # `--entry-list` path (readers filter the event stream); None = all
    entry_list: Optional[str] = None
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

        # Veto-profile resolution: explicit name → file-backed catalog
        # (shadows builtins) → error listing known names. Default (no name) →
        # legacy per-denoiser builtin, NEVER file-shadowed (byte safety: a
        # stray wavelet_hits.yaml in a data root must not silently change
        # default outputs).
        explicit_name = getattr(settings, "veto_profile", None)
        if explicit_name:
            from pmtdeconv.presets.veto_profiles import resolve_veto_profile
            try:
                vp_obj, vp_source = resolve_veto_profile(
                    explicit_name, data_root=settings.data_root)
            except KeyError:
                from pmtdeconv.presets.veto_profiles import VETO_PROFILES as _VP
                raise SystemExit(
                    f"--veto-profile {explicit_name!r}: unknown profile. "
                    f"Builtins: {', '.join(_VP.names())}; file-backed "
                    f"profiles live under "
                    f"<data_root>/calibrations/veto_profiles/.") from None
            vp_name = vp_obj.name
            vp_version = vp_obj.version
            vp_explicit = True
        else:
            vp_name = VETO_PROFILES.resolve_name(denoiser)
            vp_obj = VETO_PROFILES.lookup(name=vp_name)
            vp_source = "builtin"
            vp_version = vp_obj.version if vp_obj is not None else "0"
            vp_explicit = False

        return cls(
            runs=settings.runs or [],
            entry_stop=settings.entry_stop,
            dt_ns=(None if settings.dt_ns == "auto" else float(settings.dt_ns)),
            polarity=int(settings.polarity),
            format=settings.format,
            file_glob=settings.file_glob,
            tree=settings.tree,
            branches=settings.explicit_branches,
            data_root=(str(settings.data_root) if settings.data_root else None),
            out_dir=(str(settings.out_dir) if settings.out_dir else None),
            out_suffix=out_suffix,
            cosine_cutoff_mhz=(float(settings.cosine_cutoff_mhz)
                               if settings.cosine_cutoff_mhz is not None else None),
            fit_csv=template.fit_csv,
            fft_pad=bool(settings.fft_pad),
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
            stft_threshold_mode=str(getattr(settings, "stft_threshold_mode",
                                            "median")),
            stft_fixed_threshold=(float(settings.stft_fixed_threshold)
                                  if getattr(settings, "stft_fixed_threshold",
                                           None) is not None else None),
            snr_peak_min=settings.snr_peak_min,
            peak_distance_ns=settings.peak_distance_ns,
            no_vetoes=settings.no_vetoes,
            peak_order=settings.peak_order,
            veto_warn_frac=float(getattr(settings, "veto_warn_frac", 0.5)),
            veto_profile=vp_name,
            veto_profile_obj=vp_obj,
            veto_profile_source=vp_source,
            veto_profile_version=vp_version,
            veto_profile_explicit=vp_explicit,
            pe_band=tuple(settings.pe_band),
            in_regime_only=bool(settings.in_regime_only),
            gain=Gain.from_settings(settings, provenance, command=mode),
            entry_list=(str(settings.entry_list)
                        if getattr(settings, "entry_list", None) else None),
            template=template,
        )
