"""Named configuration presets — recipes layered on top of a dataset preset.

A ``ConfigPreset`` declares a base dataset and a dict of overrides keyed
by Settings field. The resolver in ``pmtdeconv/presets/__init__.py``
applies dataset → config → CLI-flag, with each level beating the one
below.

To add a new preset: append to ``CONFIGS`` below. Document it under
``description`` so ``pmtdeconv info presets`` is self-explanatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ConfigPreset:
    name: str
    base_dataset: str
    overrides: Dict[str, Any]
    description: str
    references: List[str] = field(default_factory=list)


# Settled overrides for the 2026-05 Na22 timewalk recipe. Defined once
# and shared between the date-stamped snapshot and the stable alias
# (decision 5).
_NA22_TIMEWALK_2026_05_OVERRIDES = {
    "cosine_cutoff_mhz": 450.0,
    "wavelet":           "sym8",
    "wlevel":            5,
    "wmode":             "hard",
    "wthresh_k":         1.0,
    "no_vetoes":         True,
    # Explicit per decision 8 — don't rely on the auto-coupling from
    # --no-vetoes. The recipe is complete in one place.
    "peak_order":        "tallest",
    "bin_width_ns":      0.5051,
    "a_scan_step_pvs":   5000.0,
    "scatter_nbins":     40,
}


CONFIGS: Dict[str, ConfigPreset] = {
    # Date-stamped immutable snapshot — never changes once published, so
    # manifests can record exactly which values produced a given result.
    "na22-timewalk-winning-2026-05": ConfigPreset(
        name="na22-timewalk-winning-2026-05",
        base_dataset="na22",
        overrides=dict(_NA22_TIMEWALK_2026_05_OVERRIDES),
        references=["E41473"],
        description=(
            "Na22 Run 65213 timewalk: Cd109-unlocked template, sat 960, "
            "wavelet sym8/L5/hard k=1, no peak vetoes (tallest-first), "
            "raw-area selection. ~7.74 ns walk vs ~13.4 ns raw t_half; "
            "σ=1.33 ns at [170,175] nVs (vs ELOG ~1.56)."
        ),
    ),
    # Stable alias — currently points at the 2026-05 snapshot. When a
    # newer best-practice config is settled, repoint this entry's
    # overrides without renaming. Scripts and CLAUDE.md should use this
    # name; manifests still record the resolved preset name so the
    # alias stays auditable (decision 5).
    "na22-timewalk": ConfigPreset(
        name="na22-timewalk",
        base_dataset="na22",
        overrides=dict(_NA22_TIMEWALK_2026_05_OVERRIDES),
        references=["E41473"],
        description=(
            "Stable alias for the latest Na22 timewalk recipe. "
            "Currently equivalent to na22-timewalk-winning-2026-05. "
            "Use this name in scripts; manifests record which snapshot "
            "was resolved."
        ),
    ),
}


def list_configs() -> List[str]:
    return list(CONFIGS.keys())


def get_config(name: str) -> ConfigPreset:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config preset {name!r}. "
                       f"Known: {', '.join(CONFIGS) or '(none)'}")
    return CONFIGS[name]
