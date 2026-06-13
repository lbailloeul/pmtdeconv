"""Per-run veto accounting + the high-veto-rate warning (step 5).

The guardrail that would have surfaced the 2026 saturation-veto problem
(vault 11f: deconv-hits-with-vetoes kept only 7.6% of a 91%-saturated run,
vs ~99% for the raw-clump method — the saturation veto was silently
discarding usable timing).

After a hits run, :func:`compute` tallies the result rows by survival +
reason and :func:`format_warning` emits a prominent overall + per-reason
message when the vetoed fraction exceeds ``--veto-warn-frac`` (default 0.5).
It never stops the run; the full breakdown also goes into the manifest
``stats`` so it is auditable after the fact. The message points the user at
``pmtdeconv vetoes calibrate``.

This is computed from the already-written result rows, so it is purely
additive — no CSV byte changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class VetoAccounting:
    total: int
    surviving: int
    vetoed: int
    vetoed_frac: float
    by_reason: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute(all_rows: pd.DataFrame) -> VetoAccounting:
    """Tally survival + rejection reasons over all attempted result rows.

    ``total`` counts every attempt row (event-level rejections — saturated /
    no_trigger / baseline_* — plus per-peak rows). ``surviving`` is the
    ``success == True`` count; ``by_reason`` is the reason histogram over the
    non-surviving rows. Saturation counts as a rejection even in ``no_vetoes``
    mode (it is the event-level gate that fires first) — which is exactly the
    signal the 2026 case needed.
    """
    total = int(len(all_rows))
    if total == 0 or "success" not in all_rows.columns:
        return VetoAccounting(total, total, 0, 0.0, {})
    success_mask = all_rows["success"].astype(bool)
    surviving = int(success_mask.sum())
    vetoed = total - surviving
    frac = (vetoed / total) if total else 0.0
    by_reason: Dict[str, int] = {}
    if "reason" in all_rows.columns:
        failed = all_rows.loc[~success_mask, "reason"]
        by_reason = {str(k): int(v) for k, v in failed.value_counts().items()}
    return VetoAccounting(total, surviving, vetoed, round(frac, 6), by_reason)


def format_warning(acc: VetoAccounting, threshold: float) -> Optional[str]:
    """Return a prominent overall + per-reason warning string when the vetoed
    fraction exceeds ``threshold`` (else ``None``)."""
    if acc.total == 0 or acc.vetoed_frac <= threshold:
        return None
    top = sorted(acc.by_reason.items(), key=lambda kv: -kv[1])[:6]
    parts = ", ".join(f"{r} {100.0 * c / acc.total:.0f}%" for r, c in top)
    return (
        f"[veto-warning] vetoed {100.0 * acc.vetoed_frac:.0f}% of {acc.total} "
        f"attempts ({acc.surviving} survive) — {parts}. "
        f"If this run is saturation- or noise-dominated, the vetoes may be "
        f"discarding usable hits; consider `pmtdeconv vetoes calibrate` "
        f"(or --no-vetoes for a timing-only pass)."
    )


def report(all_rows: pd.DataFrame, threshold: float,
           *, label: str = "") -> VetoAccounting:
    """Compute accounting and print the warning if over ``threshold``.
    Returns the accounting so the caller can fold it into the manifest."""
    acc = compute(all_rows)
    msg = format_warning(acc, threshold)
    if msg:
        banner = "!" * 72
        prefix = f"{label} " if label else ""
        print(f"\n{banner}\n{prefix}{msg}\n{banner}")
    return acc
