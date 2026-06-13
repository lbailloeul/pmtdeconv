"""SPE template resolution — ``Template{csv_path, fallback, provenance}`` + resolver.

:meth:`Template.from_settings` is the single resolution chain for the
per-channel SPE kernel, replacing the na22fmt-preference logic + silent generic
fallback that were buried in ``commands/pulse.py::_build_legacy_args`` and the
duplicated ``FALLBACK_TEMPLATE`` constants across the pipeline god-modules.

Resolution order:

1. explicit ``--template`` CSV       -> use it             (provenance ``explicit``)
2. dataset default template CSV      -> use it             (provenance ``dataset``)
3. no CSV, ``pmt_type`` known        -> catalog template   (provenance ``catalog``)
4. no CSV, unknown                   -> interactive TTY prompt /
                                        non-interactive error (unless
                                        ``--allow-generic-fallback`` -> ``generic``)

Byte safety: steps 1-2 are unchanged from the legacy carrier (including the
na22fmt schema swap), so every golden — which supplies an explicit/default
template — stays byte-identical. Only the previously-silent fallback branch
(step 4) changes behavior, and no golden exercises it (the object pulse golden
passes ``--allow-generic-fallback`` to keep today's generic path).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from pmtdeconv.pipeline.calibration import ProvenanceRecord
from pmtdeconv.presets.template_catalog import TEMPLATE_CATALOG
from pmtdeconv.utils import MILLIQAN_TEMPLATE

# The generic built-in fallback the pipeline god-modules use when a branch has
# no per-channel template (Na22 lab uses MilliQan PMTs). ``A`` and ``t0`` cancel
# in the FFT division; ``t0`` is pinned to 0.0 to match the legacy
# ``process_hits.FALLBACK_TEMPLATE`` exactly.
FALLBACK_TEMPLATE: Dict[str, float] = {**MILLIQAN_TEMPLATE, "t0": 0.0}


class TemplateResolutionError(RuntimeError):
    """Raised (non-interactively) when no template can be resolved and
    ``--allow-generic-fallback`` was not passed."""


@dataclass
class Template:
    """A resolved SPE template reference + how it was resolved.

    ``csv_path is None`` means "use the generic :data:`FALLBACK_TEMPLATE`" — the
    pipeline god-modules load ``csv_path`` (via ``io.templates.load_template``)
    and fall back to ``fallback`` per-branch, exactly as before. This object
    only changes WHERE the path/provenance is decided (here, once), not the
    template-loading math.
    """
    csv_path: Optional[Path]
    provenance: ProvenanceRecord
    pmt_type: Optional[str] = None
    fallback: Dict[str, float] = field(
        default_factory=lambda: dict(FALLBACK_TEMPLATE))

    @property
    def fit_csv(self) -> Optional[str]:
        """The ``fit_csv`` string the legacy ``process_run*`` expect
        (``None`` -> the god-module's generic fallback)."""
        return str(self.csv_path) if self.csv_path else None

    def manifest_block(self) -> Dict[str, Any]:
        block = self.provenance.to_dict()
        block["csv"] = self.fit_csv
        if self.pmt_type:
            block["pmt_type"] = self.pmt_type
        return block

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings,
                      provenance: Optional[Dict[str, str]] = None,
                      *, command: str = "hits") -> "Template":
        prov = provenance or {}
        template_csv = getattr(settings, "template_csv", None)
        default_csv = getattr(settings, "default_template_csv", None)
        na22fmt = getattr(settings, "default_template_csv_na22fmt", None)
        dataset = getattr(settings, "dataset", None)
        runs = list(getattr(settings, "runs", None) or []) or None
        pmt_type = getattr(settings, "pmt_type", None)
        allow_generic = bool(getattr(settings, "allow_generic_fallback", False))
        dry_run = bool(getattr(settings, "dry_run", False))

        def _rec(source: str, detail: Optional[str] = None) -> ProvenanceRecord:
            return ProvenanceRecord(source=source, family="template",
                                    command=command, dataset=dataset,
                                    runs=runs, detail=detail)

        # --- Steps 1-2: explicit or dataset-default CSV (byte-unchanged) ----
        if template_csv is not None:
            resolved = Path(template_csv)
            # na22fmt schema swap: only when the resolved CSV IS the dataset
            # default (i.e. user did NOT override with --template). Preserves
            # the legacy ``_build_legacy_args`` byte path exactly.
            if (na22fmt is not None and default_csv is not None
                    and Path(template_csv) == Path(default_csv)):
                resolved = Path(na22fmt)
            source = "explicit" if prov.get("template_csv") == "cli" else "dataset"
            return cls(csv_path=resolved, pmt_type=pmt_type,
                       provenance=_rec(source, detail=f"csv={resolved}"))

        # --- The previously-silent generic-fallback branch -----------------
        # 3) explicit opt-in restores today's generic behavior, loudly logged.
        if allow_generic:
            print("[template] --allow-generic-fallback: using generic MilliQan "
                  "fallback template (no per-channel SPE fit).")
            return cls(csv_path=None, pmt_type=pmt_type,
                       provenance=_rec("generic", detail="allow_generic_fallback"))

        # 4) catalog by PMT type.
        if pmt_type:
            entry = TEMPLATE_CATALOG.lookup(pmt_type=pmt_type, detector=dataset)
            if entry is not None:
                print(f"[template] resolved pmt_type={pmt_type!r} via catalog -> "
                      f"{entry.csv_path}")
                return cls(csv_path=Path(entry.csv_path), pmt_type=pmt_type,
                           provenance=_rec(
                               "catalog",
                               detail=f"pmt_type={pmt_type} -> {entry.csv_path}"))

        # 5) --dry-run never hard-fails (config inspection only).
        if dry_run:
            return cls(csv_path=None, pmt_type=pmt_type,
                       provenance=_rec("unresolved", detail="dry-run"))

        # 6) interactive TTY: let the user pick a PMT type / generic.
        resolved_tmpl = cls._interactive(settings, command=command)
        if resolved_tmpl is not None:
            return resolved_tmpl

        # 7) non-interactive: hard error with guidance.
        raise TemplateResolutionError(_guidance(dataset, pmt_type))

    # ------------------------------------------------------------------
    @classmethod
    def _interactive(cls, settings, *, command: str) -> Optional["Template"]:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return None
        dataset = getattr(settings, "dataset", None)
        print(f"\n[template] No SPE template found for dataset={dataset!r} "
              "and PMT type is unknown.")
        print("  Options:")
        print(f"    - enter a known PMT type "
              f"({', '.join(TEMPLATE_CATALOG.pmt_types())})")
        print("    - 'g' to use the generic MilliQan fallback")
        print("    - empty to abort (then run `pmtdeconv template build`)")
        try:
            choice = input("  PMT type / 'g' / empty: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice:
            return None
        if choice.lower() == "g":
            return cls(csv_path=None, pmt_type=getattr(settings, "pmt_type", None),
                       provenance=ProvenanceRecord(
                           source="generic", family="template", command=command,
                           dataset=dataset, detail="interactive-generic"))
        entry = TEMPLATE_CATALOG.lookup(pmt_type=choice)
        if entry is None:
            print(f"  unknown PMT type {choice!r}; aborting.")
            return None
        return cls(csv_path=Path(entry.csv_path), pmt_type=choice,
                   provenance=ProvenanceRecord(
                       source="catalog", family="template", command=command,
                       dataset=dataset,
                       detail=f"interactive pmt_type={choice} -> {entry.csv_path}"))


def _guidance(dataset: Optional[str], pmt_type: Optional[str]) -> str:
    return (
        f"No SPE template could be resolved for dataset={dataset!r} "
        f"(pmt_type={pmt_type!r}).\n"
        "  Resolution tried: explicit --template, dataset default, "
        "PMT-type catalog.\n"
        "  Fixes:\n"
        "    - pass --template <csv> with a per-channel SPE fit, or\n"
        "    - build one:  pmtdeconv template build --dataset <name> && "
        "pmtdeconv template aggregate, or\n"
        "    - set the dataset's pmt_type to a catalogued type "
        f"({', '.join(TEMPLATE_CATALOG.pmt_types())}), or\n"
        "    - pass --allow-generic-fallback to use the generic MilliQan "
        "template (loudly logged).")
