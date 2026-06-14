"""Auto-generated named config presets — written by ``pmtdeconv tune``.

DO NOT EDIT BY HAND. Regenerated idempotently each time ``--write-presets``
runs: read existing entries, merge new ones, collapse identical-config
dated snapshots (keep the newest per config) and emit a stable
``<prefix>-tuned-latest-<suffix>`` alias tracking the most recent
winner, then rewrite the whole file.
"""
from __future__ import annotations
from typing import Dict
from pmtdeconv.presets.configs import ConfigPreset

TUNED_CONFIGS: Dict[str, ConfigPreset] = {}

def is_tuned(name: str) -> bool:
    return name in TUNED_CONFIGS

TUNED_CONFIGS['na22-cosine-tuned-2026-05-31-level3'] = ConfigPreset(
    name='na22-cosine-tuned-2026-05-31-level3',
    base_dataset='na22',
    overrides={'cosine_cutoff_mhz': 450.0},
    description='Auto-tuned by `pmtdeconv tune cosine` on 2026-05-31. Regime level3; cosine cutoff 450.0 MHz; score=-0.3524, drift=+0.000%, scatter=6.011%.',
)

TUNED_CONFIGS['na22-cosine-tuned-2026-06-14-level3'] = ConfigPreset(
    name='na22-cosine-tuned-2026-06-14-level3',
    base_dataset='na22',
    overrides={'cosine_cutoff_mhz': 100.0},
    description='Auto-tuned by `pmtdeconv tune cosine` on 2026-06-14. Regime level3; cosine cutoff 100.0 MHz; score=-0.4413, drift=-1.221%, scatter=4.649%.',
)

TUNED_CONFIGS['na22-cosine-tuned-latest-level3'] = ConfigPreset(
    name='na22-cosine-tuned-latest-level3',
    base_dataset='na22',
    overrides={'cosine_cutoff_mhz': 100.0},
    description='Alias → latest tune winner (na22-cosine-tuned-2026-06-14-level3).',
)
