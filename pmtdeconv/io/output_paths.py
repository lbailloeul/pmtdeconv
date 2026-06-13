"""Canonical output directory and artifact resolution helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.deconv_pulse import resolve_deconv_pulse_csv


def _dataset(settings, default: str = "unknown") -> str:
    return settings.dataset or default


def suffixed_output_dir(base: str, settings) -> Path:
    parts = [base]
    if settings.config_preset_name:
        parts.append("__" + settings.config_preset_name)
    if settings.out_suffix:
        parts.append(settings.out_suffix)
    return PROJECT_ROOT / "outputs" / "".join(parts)


def pulse_output_dir(settings, denoiser: str) -> Path:
    flavor = "advanced" if denoiser == "stft" else "wavelet"
    return suffixed_output_dir(
        f"deconv_pulse_{_dataset(settings, 'na22')}_lab_cosine_{flavor}",
        settings,
    )


def legacy_pulse_output_dir(settings, denoiser: str) -> Path:
    flavor = "advanced" if denoiser == "stft" else "wavelet"
    return suffixed_output_dir(
        f"deconv_pulses_{_dataset(settings, 'na22')}_lab_cosine_{flavor}",
        settings,
    )


def waveform_output_dir(settings) -> Path:
    return suffixed_output_dir(f"deconv_area_{_dataset(settings)}", settings)


def raw_area_output_dir(settings, flavor: str, *, legacy: bool = False) -> Path:
    prefix = "area" if legacy else "raw_area"
    return suffixed_output_dir(f"{prefix}_{flavor}_{_dataset(settings)}", settings)


def timewalk_output_dir(settings) -> Path:
    run = (settings.runs or ["unknown"])[0]
    return suffixed_output_dir(f"timewalk_{_dataset(settings)}_run{run}", settings)


def presentation_plots_dir(subdir: str = "") -> Path:
    """Canonical directory for publication-ready figures (step 8 convention).

    Final presentation/paper figures are collected under
    ``outputs/presentation_plots/[<subdir>]`` — kept distinct from the
    per-command working output dirs (which carry preset/suffix in their name).
    Point any plotting command's ``--out-dir`` here to publish a figure, e.g.
    ``--out-dir $(... )/outputs/presentation_plots/timewalk``.
    """
    base = PROJECT_ROOT / "outputs" / "presentation_plots"
    return base / subdir if subdir else base


def resolve_area_csv(settings) -> Optional[Path]:
    """Resolve the area CSV for timewalk/plot consumers.

    New ``raw_area_*`` directories are checked before the pre-rename
    ``area_*`` directories. The waveform summary directory remains a final
    fallback because it is a deconvolved area source, not a raw-area source.
    """
    if not settings.runs:
        return None
    run = settings.runs[0]

    if settings.format == "object":
        for legacy in (False, True):
            candidate = (
                raw_area_output_dir(settings, "offline", legacy=legacy)
                / f"area_offline_run{run}.csv"
            )
            if candidate.exists():
                return candidate

    for flavor, filename in (
        ("drs", f"area_drs_run{run}.csv"),
        ("mqana", f"area_mqana_run{run}.csv"),
    ):
        for legacy in (False, True):
            candidate = raw_area_output_dir(settings, flavor, legacy=legacy) / filename
            if candidate.exists():
                return candidate

    candidate = waveform_output_dir(settings) / f"areas_run{run}.csv"
    if candidate.exists():
        return candidate
    return None


def resolve_pulse_csv(settings) -> Optional[Path]:
    if not settings.runs:
        return None
    run = settings.runs[0]

    if settings.format == "object":
        for legacy in (False, True):
            candidate = (
                raw_area_output_dir(settings, "offline", legacy=legacy)
                / f"offline_hit_results_run{run}.csv"
            )
            if candidate.exists():
                return candidate

    found = resolve_deconv_pulse_csv(pulse_output_dir(settings, "wavelet"))
    if found is not None:
        return found
    return resolve_deconv_pulse_csv(legacy_pulse_output_dir(settings, "wavelet"))
