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
    # Explicit since 2026-10, when the default denoiser became STFT: this
    # snapshot was validated with the wavelet and must keep using it.
    "denoiser":          "wavelet",
    "wavelet":           "sym8",
    "wlevel":            5,
    "wmode":             "hard",
    "wthresh_k":         1.0,
    "no_vetoes":         True,
    # Explicit per decision 8 — don't rely on the auto-coupling from
    # --no-vetoes. The recipe is complete in one place.
    "peak_order":        "tallest",
    # Until 2026-09-23 this also listed bin_width_ns / a_scan_step_pvs /
    # scatter_nbins. They were never applied (timewalk reads its own
    # --bin-width / --a-scan-step / --scatter-nbins, whose defaults are the
    # same values), so dropping them changes no result.
}

# The 2026-10 recipe: the same selection on the STFT denoiser (the default
# since the 2026-10 denoiser study). Validated against the same E41473
# benchmark as the 2026-05 snapshot — numbers in its description.
_NA22_TIMEWALK_2026_10_OVERRIDES = {
    "cosine_cutoff_mhz": 450.0,
    "denoiser":          "stft",
    "no_vetoes":         True,
    "peak_order":        "tallest",
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
    "na22-timewalk-2026-10": ConfigPreset(
        name="na22-timewalk-2026-10",
        base_dataset="na22",
        overrides=dict(_NA22_TIMEWALK_2026_10_OVERRIDES),
        references=["E41473"],
        description=(
            "Na22 Run 65213 timewalk on the STFT denoiser: same selection as "
            "na22-timewalk-winning-2026-05 (450 MHz cosine, no peak vetoes, "
            "tallest-first). Validated 2026-10 (DRS areas, t_half): sigma "
            "1.55 +- 0.07 ns at [170,175] nVs (ELOG ~1.56), walk 9.45 ns vs "
            "13.4 ns raw t_half. The wavelet snapshot measured 1.39 ns and "
            "6.95 ns on the same inputs: same median resolution (1.68 vs "
            "1.67 ns), ~2.5 ns more residual walk with STFT."
        ),
    ),
    # Stable alias — points at the newest snapshot (2026-10, STFT). When a
    # newer best-practice config is settled, repoint this entry's
    # overrides without renaming. Scripts and CLAUDE.md should use this
    # name; manifests still record the resolved preset name so the
    # alias stays auditable (decision 5).
    "na22-timewalk": ConfigPreset(
        name="na22-timewalk",
        base_dataset="na22",
        overrides=dict(_NA22_TIMEWALK_2026_10_OVERRIDES),
        references=["E41473"],
        description=(
            "Stable alias for the latest Na22 timewalk recipe. "
            "Currently equivalent to na22-timewalk-2026-10 (STFT); the "
            "wavelet recipe stays available as na22-timewalk-winning-2026-05. "
            "Use this name in scripts; manifests record which snapshot "
            "was resolved."
        ),
    ),
}


def get_config(name: str) -> ConfigPreset:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config preset {name!r}. "
                       f"Known: {', '.join(CONFIGS) or '(none)'}")
    return CONFIGS[name]
