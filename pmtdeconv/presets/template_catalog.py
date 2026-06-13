"""Template catalog — ``pmt_type`` -> an existing real SPE template.

When a run has no per-channel template CSV (no ``--template``, no dataset
default) but its PMT type is known, the resolver
(:meth:`pmtdeconv.pipeline.template.Template.from_settings`) looks the PMT type
up here and uses that real template's SHAPE instead of the silent generic
MilliQan fallback. Provenance is recorded as ``"catalog"`` so the manifest
flags the run as using an approximate (transferred) template.

The catalog is seeded with templates we have already built. PMT-type labels are
the keys, backfilled onto the dataset presets (``DatasetPreset.pmt_type``).

NOTE: the labels below are PROVISIONAL descriptive names for the lab tubes
(real hardware model numbers TBD by the experimenter) — they are internal keys
and can be renamed without affecting byte output, because every golden supplies
an explicit/default template and never reaches the catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pmtdeconv.pipeline.calibration import Catalog
from pmtdeconv.presets.datasets import CD109_STACKED_UNLOCKED_CSV


@dataclass(frozen=True)
class TemplateCatalogEntry:
    pmt_type: str
    csv_path: Path
    note: str = ""


# Seeded with real, on-disk templates. ``milliqan_lab`` = the MilliQan-style
# lab tubes used for Na22/Cd109 lab running; their SPE shape is the Cd109
# stacked-unlocked biexp fit (the canonical Na22 deconvolution template).
_ENTRIES: List[TemplateCatalogEntry] = [
    TemplateCatalogEntry(
        pmt_type="milliqan_lab",
        csv_path=CD109_STACKED_UNLOCKED_CSV,
        note="Cd109-derived stacked-unlocked biexp SPE shape (Na22/Cd109 lab).",
    ),
]


class TemplateCatalog(Catalog):
    """Read-only ``pmt_type`` -> template registry (step 3).

    Step 5's unified ``Calibration``/``Catalog`` work may add a writable
    backing store; for now this is seeded in-code from the templates we ship.
    """
    family = "template"

    def __init__(self, entries: List[TemplateCatalogEntry]):
        self._by_pmt: Dict[str, TemplateCatalogEntry] = {
            e.pmt_type: e for e in entries}

    def lookup(self, *, pmt_type: Optional[str] = None,
               detector: Optional[str] = None, **_: Any
               ) -> Optional[TemplateCatalogEntry]:
        if not pmt_type:
            return None
        return self._by_pmt.get(pmt_type)

    def pmt_types(self) -> List[str]:
        return sorted(self._by_pmt)


TEMPLATE_CATALOG = TemplateCatalog(_ENTRIES)
