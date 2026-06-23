"""``pulse --diagnostic-events`` intermediate dump (PR 10 / Q5).

When a list of event indices is passed via ``--diagnostic-events``,
``process_hits.process_run`` (and ``process_hits_stft.process_run``)
tee np.save calls for each matching event into

    outputs/<deconv_pulse_out_dir>/diagnostic_event<N>/

containing:

- raw_waveform.npy            — the input waveform (pedestal-subtracted)
- fft_pre_filter.npy          — complex FFT before cosine filter
- cosine_filter.npy           — the cosine low-pass shape
- fft_post_filter.npy         — FFT after applying cosine filter
- deconv_pre_denoise.npy      — IFFT result (before STFT/wavelet step)
- denoise_intermediate.npy    — wavelet coeff list OR STFT spectrogram
- deconv_post_denoise.npy     — final deconv output u_rec
- peaks.json                  — found peaks + per-peak veto verdicts
- truth_match.json            — present iff fixture has a .truth.csv
- diagnostic_event<N>.pdf     — 6-panel multi-panel summary

Non-diagnostic invocations skip every np.save by checking
``DiagnosticSink.active`` cheaply (≈1% overhead worst case). Byte
identity of all golden CSVs is preserved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DiagnosticSink:
    """Per-event intermediate-saving sink. ``active=False`` instances
    no-op everywhere so they're safe to leave in the hot path.

    Each sink is bound to one (run, branch, entry) tuple at the point
    the wavelet/STFT pipeline reaches a matching event. Multiple events
    per run produce multiple sinks; each writes to its own
    ``diagnostic_event<N>/`` directory.
    """
    out_root: Path
    target_events: Set[int] = field(default_factory=set)
    truth_csv: Optional[Path] = None
    denoiser: str = "wavelet"        # for label only
    cosine_cutoff_mhz: float = 450.0
    dt_ns: float = 0.5051

    @property
    def active(self) -> bool:
        return bool(self.target_events)

    def event_dir(self, entry: int, branch: str) -> Path:
        d = self.out_root / f"diagnostic_event{entry}__{branch}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def should_dump(self, entry: int) -> bool:
        return self.active and (entry in self.target_events)

    # ─── individual savers (each guarded by should_dump on caller side) ─

    def save_array(self, entry: int, branch: str, name: str,
                   arr: np.ndarray) -> None:
        np.save(self.event_dir(entry, branch) / f"{name}.npy", arr)

    def save_peaks(self, entry: int, branch: str,
                   peaks: np.ndarray, heights: np.ndarray,
                   reasons: List[Dict[str, Any]],
                   peak_thresh: float, u_rec_noise: float) -> None:
        payload = {
            "n_peaks": int(len(peaks)),
            "peak_indices": peaks.tolist() if len(peaks) else [],
            "peak_heights": heights.tolist() if len(heights) else [],
            "peak_thresh": float(peak_thresh),
            "u_rec_noise": float(u_rec_noise),
            "per_peak_rows": reasons,
        }
        (self.event_dir(entry, branch) / "peaks.json").write_text(
            json.dumps(payload, indent=2))

    def save_truth_match(self, entry: int, branch: str) -> None:
        if self.truth_csv is None or not Path(self.truth_csv).exists():
            return
        import pandas as pd
        try:
            df = pd.read_csv(self.truth_csv)
        except Exception:
            return
        row = df[(df["event"] == entry) & (df["branch"] == branch)]
        if len(row) == 0:
            return
        r = row.iloc[0]
        payload = {
            "event": int(entry), "branch": branch,
            "true_amp_mv": float(r["true_amp_mv"]),
            "true_center_ns": float(r["true_center_ns"]),
            "true_charge_pVs": float(r["true_charge_pVs"]),
            "true_sigma_ns": float(r["true_sigma_ns"]),
            "saturated": bool(r["saturated"]),
        }
        (self.event_dir(entry, branch) / "truth_match.json").write_text(
            json.dumps(payload, indent=2))

    def save_pdf(self, entry: int, branch: str,
                 raw: np.ndarray, t_ns: np.ndarray,
                 cosine_filter: np.ndarray,
                 deconv_pre_denoise: np.ndarray,
                 deconv_post_denoise: np.ndarray,
                 peaks: np.ndarray,
                 ped: float) -> None:
        """6-panel summary PDF for this event."""
        from pmtdeconv.plotting.style import use_minimal_style, minimal_axes
        use_minimal_style()
        fig, axes = plt.subplots(3, 2, figsize=(11, 9))
        ax = axes.ravel()

        ax[0].plot(t_ns, raw, "k-", lw=0.5)
        ax[0].axhline(ped, color="C2", lw=0.6, ls="--",
                      label=f"baseline {ped:.1f}")
        ax[0].set_title(f"raw waveform — event {entry} {branch}")
        ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel("mV")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

        ax[1].plot(np.arange(len(cosine_filter)), cosine_filter,
                   "C1-", lw=0.7)
        ax[1].set_title(f"cosine filter (cutoff={self.cosine_cutoff_mhz} MHz)")
        ax[1].set_xlabel("FFT bin"); ax[1].set_ylabel("|H|")
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(t_ns[:len(deconv_pre_denoise)], deconv_pre_denoise,
                   "C0-", lw=0.6)
        ax[2].set_title("deconv pre-denoise"); ax[2].set_xlabel("t [ns]")
        ax[2].grid(True, alpha=0.3)

        ax[3].plot(t_ns[:len(deconv_post_denoise)], deconv_post_denoise,
                   "C0-", lw=0.6)
        if len(peaks):
            ax[3].scatter(t_ns[peaks], deconv_post_denoise[peaks],
                          color="red", marker="v", s=40, zorder=5,
                          label=f"{len(peaks)} peak{'s' if len(peaks)!=1 else ''}")
            ax[3].legend(fontsize=8)
        ax[3].set_title(f"deconv post-denoise ({self.denoiser})")
        ax[3].set_xlabel("t [ns]"); ax[3].grid(True, alpha=0.3)

        # Overlay raw + post-denoise (scaled) on same axis
        ax[4].plot(t_ns, raw - ped, "k-", lw=0.4, alpha=0.5, label="raw - ped")
        scale = (np.max(np.abs(raw - ped))
                 / max(np.max(np.abs(deconv_post_denoise)), 1e-12))
        ax[4].plot(t_ns[:len(deconv_post_denoise)],
                   deconv_post_denoise * scale,
                   "C0-", lw=0.6, label="deconv (scaled)")
        ax[4].set_title("overlay: raw vs deconv (scaled)")
        ax[4].set_xlabel("t [ns]"); ax[4].legend(fontsize=8)
        ax[4].grid(True, alpha=0.3)

        # Truth overlay if present
        if self.truth_csv is not None and Path(self.truth_csv).exists():
            import pandas as pd
            df = pd.read_csv(self.truth_csv)
            row = df[(df["event"] == entry) & (df["branch"] == branch)]
            if len(row) > 0:
                r = row.iloc[0]
                ax[5].plot(t_ns, raw, "k-", lw=0.5, label="raw")
                ax[5].axvline(r["true_center_ns"], color="green",
                              lw=1.0, ls="--",
                              label=f"true center {r['true_center_ns']:.1f} ns")
                ax[5].axhline(ped + r["true_amp_mv"], color="green",
                              lw=1.0, ls=":",
                              label=f"true amp {r['true_amp_mv']:.1f} mV")
                ax[5].set_title("truth overlay")
                ax[5].set_xlabel("t [ns]"); ax[5].set_ylabel("mV")
                ax[5].legend(fontsize=8); ax[5].grid(True, alpha=0.3)
            else:
                ax[5].text(0.5, 0.5, "(no truth row for this event)",
                           ha="center", va="center")
                ax[5].set_axis_off()
        else:
            ax[5].text(0.5, 0.5, "(no truth CSV next to fixture)",
                       ha="center", va="center")
            ax[5].set_axis_off()

        # Focus the time-domain panels on the pulse (centred on the raw
        # peak), rather than showing the whole record. ax[1] is the cosine
        # filter (FFT-bin axis), so it's left alone.
        if len(raw) and len(t_ns):
            pk = int(np.argmax(raw - ped))
            tc = float(t_ns[pk])
            half_w = 150.0
            tlo = max(float(t_ns[0]), tc - half_w)
            thi = min(float(t_ns[-1]), tc + half_w)
            for a in (ax[0], ax[2], ax[3], ax[4], ax[5]):
                a.set_xlim(tlo, thi)

        for a in (ax[0], ax[1], ax[2], ax[3], ax[4]):
            minimal_axes(a)
        fig.tight_layout()
        pdf_path = (self.event_dir(entry, branch)
                    / f"diagnostic_event{entry}__{branch}.pdf")
        fig.savefig(pdf_path, dpi=130)
        plt.close(fig)


_NOOP_SINK: Optional[DiagnosticSink] = None


def noop_sink() -> DiagnosticSink:
    """Module-level no-op sink for non-diagnostic runs. ``active`` is
    False so the hot-path checks short-circuit."""
    global _NOOP_SINK
    if _NOOP_SINK is None:
        _NOOP_SINK = DiagnosticSink(out_root=Path("/tmp/.pmtdeconv_noop"))
    return _NOOP_SINK
