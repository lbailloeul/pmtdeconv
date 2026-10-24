"""Shared calibration interface — ``ProvenanceRecord`` + ``Catalog``.

The three calibration features in pmtdeconv — **template** (``template build`` +
the catalog resolver), **cutoff/denoiser** (``tune cosine|wavelet|stft``),
and **vetoes** (``vetoes calibrate``) — are the same SHAPE: scan a data sample,
derive candidate parameters, describe them for human review, and save the winner
to a per-family catalog with a uniform provenance record. They keep their
distinct command names and objectives (documented per-family below), but share
this internal interface so the workflow and provenance are consistent.

Consumers: the template resolver (:mod:`pmtdeconv.pipeline.template`), the
veto-profile and gain catalogs, and every ``*calibrate`` command, which record
a :class:`ProvenanceRecord` in the manifest.

**The interface unifies the WORKFLOW, not the OBJECTIVE.** Each calibration
keeps its own criterion + data source:

- **Template** — goodness-of-fit of the SPE kernel shape (gaussian-smeared
  biexponential to clean single-PE candidates; ``tau_rise`` locked 0.5 ns;
  stacked per-file -> per-channel median). Data: SPE/Cd109 calibration pulses.
- **Cutoff/denoiser** — epsilon-constraint (vault note 13): minimize
  reconstructed FWHM subject to charge drift <= 2% AND an efficiency floor.
  NOTE the implementation gates efficiency at ``min_eff_frac × max(sweep
  efficiency)`` (default 1.0), not at 0.9× the REFERENCE config as vault 13
  proposed — the reference efficiency is computed for provenance only
  (see ``tune_cosine``). Data: ground-truth regime fixtures.
- **Vetoes — HYBRID** — data-driven quantiles of the run's own raw
  distributions by default; ground-truth scoring when a truth CSV exists.
  Data: raw waveforms (always) + truth CSV (when available).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date as _date
from typing import Any, Dict, Generic, List, Optional, TypeVar


@dataclass
class ProvenanceRecord:
    """How a calibrated artifact was resolved/derived. Goes into the manifest.

    ``source`` is the resolution provenance:

    - ``"explicit"``     — user supplied it directly (e.g. ``--template`` CSV)
    - ``"dataset"``      — the dataset preset's configured default
    - ``"catalog"``      — resolved from a family catalog by detector metadata
    - ``"data-derived"`` — produced by a ``*calibrate`` / ``tune`` scan
    - ``"generic"``      — generic built-in fallback (loudly logged)
    - ``"unresolved"``   — not resolved (e.g. a ``--dry-run`` with no template)
    """
    source: str
    family: str = ""                       # "template" | "veto" | "denoiser" | "gain"
    command: Optional[str] = None
    dataset: Optional[str] = None
    runs: Optional[List[int]] = None
    git_sha: Optional[str] = None
    date: Optional[str] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.date is None:
            self.date = _date.today().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


V = TypeVar("V")


class Catalog(ABC, Generic[V]):
    """A per-family registry: detector / pmt_type / regime -> saved artifact.

    Consumed uniformly by the ``*.from_settings`` resolvers via the chain
    ``explicit -> catalog-by-metadata -> [data-derived] -> fallback``.
    """
    family: str = ""

    @abstractmethod
    def lookup(self, **key: Any) -> Optional[V]:
        """Return the catalog entry matching the key, or ``None``."""

    def save(self, name: str, artifact: V,
             provenance: ProvenanceRecord) -> None:
        """Persist a derived artifact under ``name``. Default: unsupported;
        family catalogs that back a file override this (step 5)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a read-only catalog (no save).")
