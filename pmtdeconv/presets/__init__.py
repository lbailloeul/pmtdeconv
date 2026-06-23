"""Settings resolver — layer dataset preset, config preset, and CLI flags.

The CLI builds a parsed ``argparse.Namespace`` and calls
:func:`resolve_settings` to get back a fully-populated :class:`Settings`
dataclass plus a ``provenance`` dict recording which level set each
field. The provenance dict goes straight into the manifest JSON.

Precedence (lowest → highest):
    dataset default  <  named config preset  <  explicit CLI flag

A field not touched by any of the above keeps its ``Settings`` default
(provenance = "default").
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pmtdeconv.presets.configs import CONFIGS, ConfigPreset, get_config
from pmtdeconv.presets.datasets import (
    DATASETS,
    DatasetPreset,
    get_dataset,
    resolve_dataset,
)
# Merge auto-generated tuned presets (PR 10) into CONFIGS so the
# resolver, info presets, and `--preset <name>` all see them
# indistinguishably from the hand-curated entries.
from pmtdeconv.presets.tuned_configs import TUNED_CONFIGS as _TUNED_CONFIGS
CONFIGS.update(_TUNED_CONFIGS)

# Mapping argparse-arg name (after argparse's hyphen → underscore) →
# Settings field name. Argparse uses underscores already so most are
# identity, but some legacy spellings differ (e.g. CLI --branches
# argparse-key branches ↔ Settings branch_pattern).
CLI_TO_SETTINGS = {
    "data_root":           "data_root",
    "tree":                "tree",
    "branches":            "branch_pattern",
    "dt_ns":               "dt_ns",
    "out_dir":             "out_dir",
    "out_suffix":          "out_suffix",
    # Bug-fix 2 (2026-05-30 scan): --manifest was being parsed but
    # never reached settings.manifest_path. Every command resolved
    # manifest_path from settings, so the flag silently no-op'd.
    "manifest":            "manifest_path",
    "runs":                "runs",
    "entry_stop":          "entry_stop",
    "hash_inputs":         "hash_inputs",
    "dry_run":             "dry_run",
    "log_level":           "log_level",
    # Subcommand-specific flags share names with Settings fields
    "wavelet":             "wavelet",
    "wlevel":              "wlevel",
    "wmode":               "wmode",
    "wthresh_k":           "wthresh_k",
    # Bug-fix 11: route the --stft-* CLI flags into the Settings
    # dataclass (previously they only lived on the args Namespace,
    # so dataset/preset overrides had no path to reach the processor).
    "stft_nperseg_ns":     "stft_nperseg_ns",
    "stft_freq_cutoff_mhz": "stft_freq_cutoff_mhz",
    "stft_amp_threshold":  "stft_amp_threshold",
    "denoise_raw":         "denoise_raw",
    "denoiser":            "denoiser",
    "cosine_cutoff_mhz":   "cosine_cutoff_mhz",
    "snr_peak_min":        "snr_peak_min",
    "peak_distance_ns":    "peak_distance_ns",
    "no_vetoes":           "no_vetoes",
    "peak_order":          "peak_order",
    "template":            "template_csv",
    "polarity":            "polarity",
    "allow_generic_fallback": "allow_generic_fallback",
    "veto_warn_frac":      "veto_warn_frac",
}


@dataclass
class Settings:
    """Fully-resolved configuration for one CLI invocation.

    Defaults below are the *neutral* defaults — they kick in only when
    neither the dataset preset, the config preset, nor a CLI flag
    supplied a value.
    """
    # Dataset-derived (filled in from the DatasetPreset)
    dataset:               Optional[str] = None
    data_root:             Optional[Path] = None
    file_glob:             str = "flattened_Run*.root"
    exclude_glob:          Optional[str] = None
    format:                Literal["flat", "object"] = "flat"
    tree:                  str = "waveforms"
    branch_pattern:        Union[List[str], Literal["auto"]] = "auto"
    dt_ns:                 Union[float, Literal["auto"]] = 1.25
    polarity:              int = +1
    cosine_cutoff_mhz:     float = 100.0
    default_template_csv:  Optional[Path] = None
    default_template_csv_na22fmt: Optional[Path] = None
    offline_data_dir:      Optional[Path] = None
    offline_file_glob:     Optional[str] = None
    spe_band_mv:           Tuple[float, float] = (5.0, 30.0)
    # Step 3: per-detector PMT metadata (auto-copied from the DatasetPreset).
    # `pmt_type` keys the template catalog for metadata-only runs; `hv_volts`
    # is manifest/charge-scale context only (never alters the kernel).
    pmt_type:              Optional[str] = None
    hv_volts:              Optional[Dict[str, float]] = None

    # Common CLI flags
    runs:                  Optional[List[int]] = None
    entry_stop:            Optional[int] = None
    out_dir:               Optional[Path] = None
    out_suffix:            str = ""
    manifest_path:         Optional[Path] = None
    hash_inputs:           bool = False
    dry_run:               bool = False
    log_level:             str = "info"

    # Pipeline / subcommand flags
    template_csv:          Optional[Path] = None
    denoiser:              str = "wavelet"           # wavelet | stft | none
    wavelet:               str = "sym8"
    wlevel:                int = 5
    wmode:                 str = "hard"
    wthresh_k:             float = 1.0
    # Bug-fix 11 (2026-05-30 final scan): STFT knobs as first-class
    # Settings fields. Without these, `tune stft --write-presets`
    # would persist a ConfigPreset whose overrides target unknown
    # Settings attributes, raising KeyError when `--preset` is
    # applied. Defaults match the STFT pipeline's legacy hardcoded
    # values so behavior is unchanged when no override is supplied.
    stft_nperseg_ns:       float = 20.0
    stft_freq_cutoff_mhz:  float = 175.0
    stft_amp_threshold:    float = 10.0
    denoise_raw:           bool = False
    snr_peak_min:          float = 5.0
    peak_distance_ns:      float = 5.0
    no_vetoes:             bool = False
    # `auto`: chronological if vetoes are on, tallest-first when --no-vetoes
    # (decision 8). Override with `chrono` or `tallest` to decouple.
    peak_order:            str = "auto"
    # Step 3: opt-in to the generic MilliQan fallback template when no
    # per-channel CSV is configured and the PMT type is uncatalogued
    # (restores the previously-silent fallback, now loudly logged).
    allow_generic_fallback: bool = False
    # Step 5: warn (never stop) when the vetoed fraction of a hits run
    # exceeds this — the saturation/noise-veto guardrail. Breakdown also
    # lands in the manifest `stats`.
    veto_warn_frac:        float = 0.5

    # Timewalk / scan flags (used by timewalk subcommands later)
    bin_width_ns:          float = 0.5051
    a_scan_step_pvs:       float = 5000.0
    scatter_nbins:         int = 40

    # Named config preset (just for the manifest preset block)
    config_preset_name:    Optional[str] = None


def _set(settings: Settings, provenance: Dict[str, str],
         field_name: str, value: Any, source: str) -> None:
    """Set a Settings field + record provenance. Skips if value is None."""
    if value is None:
        return
    # Cast Path-typed fields
    if field_name in ("data_root", "out_dir", "manifest_path",
                      "template_csv", "default_template_csv"):
        value = Path(value) if not isinstance(value, Path) else value
    setattr(settings, field_name, value)
    provenance[field_name] = source


def _apply_dataset(settings: Settings, provenance: Dict[str, str],
                   ds: DatasetPreset) -> None:
    """Copy every DatasetPreset field onto the Settings."""
    settings.dataset = ds.name
    provenance["dataset"] = "dataset"
    for f in fields(ds):
        name = f.name
        # DatasetPreset uses `data_dir` -> Settings uses `data_root`
        target = "data_root" if name == "data_dir" else name
        if not hasattr(settings, target):
            continue
        _set(settings, provenance, target, getattr(ds, name), "dataset")
    # The dataset's recommended template lives in default_template_csv
    # AND becomes the resolved template_csv unless overridden later.
    if ds.default_template_csv is not None:
        _set(settings, provenance, "template_csv",
             ds.default_template_csv, "dataset")


def _apply_config(settings: Settings, provenance: Dict[str, str],
                  cfg: ConfigPreset) -> None:
    """Apply a named config preset's overrides on top of the dataset."""
    settings.config_preset_name = cfg.name
    provenance["config_preset_name"] = "config"
    for field_name, value in cfg.overrides.items():
        if not hasattr(settings, field_name):
            raise KeyError(
                f"ConfigPreset {cfg.name!r} sets unknown Settings field "
                f"{field_name!r}. Check pmtdeconv/presets/configs.py.")
        _set(settings, provenance, field_name, value, "config")


def _apply_cli(settings: Settings, provenance: Dict[str, str],
               args: argparse.Namespace) -> None:
    """Apply explicit CLI flags last. Only flags whose value is not None
    (i.e. the user actually passed them) override.
    """
    args_dict = vars(args)
    for cli_key, settings_key in CLI_TO_SETTINGS.items():
        if cli_key not in args_dict:
            continue
        value = args_dict[cli_key]
        if value is None:
            continue
        # Special case: --dt-ns can be the string "auto"
        if settings_key == "dt_ns" and isinstance(value, str):
            if value.lower() == "auto":
                value = "auto"
            else:
                try:
                    value = float(value)
                except ValueError as exc:
                    raise SystemExit(f"--dt-ns must be a float or 'auto', "
                                     f"got {value!r}") from exc
        _set(settings, provenance, settings_key, value, "cli")


def resolve_settings(args: argparse.Namespace,
                     require_data_root: bool = True
                     ) -> Tuple[Settings, Dict[str, str]]:
    """Layer dataset → config → CLI flags into a final Settings object.

    Returns ``(settings, provenance)``.

    By default raises ``SystemExit`` if neither ``--dataset`` nor an
    explicit ``--data-root`` was supplied. Pass
    ``require_data_root=False`` from subcommands that don't need a
    single data root (e.g. ``template build``, which iterates the legacy
    DATASETS registry internally).
    """
    settings = Settings()
    provenance: Dict[str, str] = {}

    for field_name, value in getattr(args, "_settings_defaults", {}).items():
        if not hasattr(settings, field_name):
            raise KeyError(f"Unknown command default Settings field {field_name!r}.")
        _set(settings, provenance, field_name, value, "default")

    dataset_name = getattr(args, "dataset", None)
    if dataset_name is not None:
        _apply_dataset(settings, provenance, resolve_dataset(dataset_name))

    preset_name = getattr(args, "preset", None)
    if preset_name is not None:
        cfg = get_config(preset_name)
        if dataset_name is None:
            # If user didn't pick a dataset, fall back to the preset's base
            _apply_dataset(settings, provenance, get_dataset(cfg.base_dataset))
        _apply_config(settings, provenance, cfg)

    _apply_cli(settings, provenance, args)

    if (require_data_root
            and settings.data_root is None
            and not getattr(args, "dry_run", False)):
        raise SystemExit("Must specify either --dataset or --data-root "
                         "(use --dry-run to skip this check).")

    return settings, provenance
