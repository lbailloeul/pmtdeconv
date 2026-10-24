"""``pmtdeconv hits`` — peak-finding FFT deconvolution.

Delegates to ``pmtdeconv.pipeline.process_hits.process_run`` (flat) /
``process_run_object`` (object) for the wavelet/none denoisers, and to
``process_hits.process_run_stft`` for the STFT denoiser. All three share
the unified ``_process_event`` body; the denoiser only selects the
``HitsProfile`` (wavelet vs STFT constants).

Step 3 (architecture restructure): the resolved :class:`~pmtdeconv.presets
.Settings` is turned into a :class:`~pmtdeconv.pipeline.detector.Detector`
*once* (``Detector.from_settings``), replacing the old ``_build_legacy_args``
SimpleNamespace round-trip. The ``Detector`` carries the resolved
:class:`~pmtdeconv.pipeline.template.Template` (with provenance) for the
manifest, and the template resolver decides explicit/dataset/catalog/generic
in one place.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pmtdeconv.io.data_paths import find_run_files
from pmtdeconv.io.deconv_pulse import DECONV_PULSE_CSV, DECONV_PULSE_ROLE
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import hits_output_dir
from pmtdeconv.pipeline.detector import Detector
from pmtdeconv.pipeline.template import add_generic_fallback_flag
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("hits")
    g.add_argument("--template", type=str, default=None,
                   help="SPE template CSV path. Default: dataset's preset.")
    add_generic_fallback_flag(g)
    g.add_argument("--denoiser", choices=("stft", "wavelet", "none"),
                   default=None,
                   help="Post-deconvolution denoiser. Default: stft. wavelet "
                        "is experimental since the 2026-10 denoiser study "
                        "(STFT measured charge best on every test set).")
    g.add_argument("--snr-peak-min", type=float, default=None,
                   help="Peak-finder SNR threshold (default 5.0).")
    g.add_argument("--peak-distance-ns", type=float, default=None,
                   help="Min separation between peaks in ns (default 5.0).")
    g.add_argument("--veto-warn-frac", type=float, default=None,
                   help="Warn (never stop) when the vetoed fraction of a run "
                        "exceeds this (default 0.5). The per-reason breakdown "
                        "also lands in the manifest stats; points at "
                        "`vetoes calibrate`.")
    g.add_argument("--wavelet", type=str, default=None)
    g.add_argument("--wlevel", type=int, default=None)
    g.add_argument("--wmode", choices=("soft", "hard", "garrote"),
                   default=None)
    g.add_argument("--wthresh-k", type=float, default=None)
    g.add_argument("--denoise-raw", action="store_true", default=None)
    vetoes = g.add_mutually_exclusive_group()
    vetoes.add_argument("--no-vetoes", dest="no_vetoes",
                        action="store_true", default=None)
    vetoes.add_argument("--vetoes", dest="no_vetoes", action="store_false",
                        help="Enable vetoes, overriding a preset that disables them.")
    g.add_argument("--peak-order", choices=("chrono", "tallest", "first", "auto"),
                   default=None,
                   help="How to assign peak_id within a waveform. "
                        "`chrono` (legacy): left-to-right in time. "
                        "`tallest`: descending u_rec amplitude (peak_id=1 "
                        "= largest peak, useful when vetoes are off). "
                        "`first`: the first peak >=15%% of the tallest gets "
                        "peak_id=1 (skips deconv ringing side-lobes; "
                        "sub-threshold peaks keep their rows, ordered last) — "
                        "pair with the t_half CSV column (CFD-50 rising-edge "
                        "crossing, MQAna `halftime`) for low-jitter timewalk. "
                        "`auto` (default): chrono if vetoes on, tallest "
                        "if --no-vetoes (decision 8).")
    g.add_argument("--stft-nperseg-ns", type=float, default=None,
                   help="STFT window size in ns (STFT denoiser only, default 20).")
    g.add_argument("--stft-freq-cutoff-mhz", type=float, default=None,
                   help="STFT high-frequency cutoff in MHz (default 175).")
    g.add_argument("--stft-amp-threshold", type=float, default=None,
                   help="STFT amplitude threshold (default 10).")
    g.add_argument("--stft-threshold-mode", choices=("median", "fixed"),
                   default=None,
                   help="STFT masking. 'median' (default): element-wise mask "
                        "at global-median × --stft-amp-threshold, so the cut "
                        "adapts to each waveform. 'fixed' (Tang 2411.07620): "
                        "keep whole time columns whose sub-cutoff amplitude "
                        "clears --stft-fixed-threshold — the same cut for "
                        "every waveform.")
    g.add_argument("--stft-fixed-threshold", type=float, default=None,
                   help="T-F amplitude threshold for --stft-threshold-mode "
                        "fixed (scipy STFT units).")
    # Bug-fix 1 (2026-05-30 scan): expose cosine_cutoff_mhz so tuned
    # presets actually reach the processor. Without this, tuned recipes
    # silently revert to the pipeline's hardcoded defaults (450 MHz
    # wavelet, 100 MHz STFT).
    g.add_argument("--cosine-cutoff-mhz", type=float, default=None,
                   help="Cosine low-pass cutoff in MHz. Default: dataset "
                        "preset. Tuned presets via --preset propagate here.")
    g.add_argument("--spe-charge", dest="spe_charge", type=str, nargs="+",
                   default=None, metavar="BR=PVS",
                   help="Per-branch 1-PE charge anchor in pVs (e.g. "
                        "d0_ch1=820 d0_ch2=790). One-off override of the "
                        "card's spe_charge; fit it with `gain calibrate`.")
    g.add_argument("--pe-band", dest="pe_band", type=float, nargs=2,
                   default=None, metavar=("LO", "HI"),
                   help="Deconvolution applicability window in nPE "
                        "(default 0 20 — the linear-model domain). Drives the "
                        "in_regime flag + --in-regime-only.")
    g.add_argument("--veto-profile", dest="veto_profile", type=str,
                   default=None, metavar="NAME",
                   help="Run under a NAMED veto profile (builtins: "
                        "wavelet_hits, stft_hits, benchmark_hits; a YAML "
                        "under <data_root>/calibrations/veto_profiles/ "
                        "shadows builtins for explicit names). Applies "
                        "IDENTICAL thresholds regardless of --denoiser and "
                        "honors --no-vetoes/--peak-order/--denoise-raw on "
                        "every denoiser — the fair-benchmark switch.")
    g.add_argument("--in-regime-only", dest="in_regime_only",
                   action="store_true", default=None,
                   help="Process ONLY waveforms with raw nPE inside --pe-band "
                        "(rejected events get reason=out_of_pe_range in "
                        "all_results.csv). Default: flag-only, include all. "
                        "Requires a spe_charge anchor (see `gain calibrate`).")
    g.add_argument("--entry-list", dest="entry_list", type=str, default=None,
                   metavar="CSV",
                   help="Process only the entries listed (CSV with an "
                        "`entry` column — e.g. `select multihit`'s "
                        "event_list.csv — or one index per line).")
    fftpad = g.add_mutually_exclusive_group()
    fftpad.add_argument("--fft-pad", dest="fft_pad", action="store_true",
                        default=None,
                        help="Linear (zero-padded to 2N) deconvolution "
                             "(default ON since 2026-07-28).")
    fftpad.add_argument("--no-fft-pad", dest="fft_pad", action="store_false",
                        help="Circular (unpadded) deconvolution — reproduces "
                             "pre-2026-07-28 output bytes exactly.")
    # PR 10 / Q5: diagnostic intermediate dump per event.
    g.add_argument("--diagnostic-events", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Save intermediate arrays + a multi-panel PDF for "
                        "each listed event index into "
                        "outputs/.../diagnostic_event<N>__<branch>/. "
                        "Multiple events accepted (nargs='+'). Non-empty "
                        "list activates the diagnostic sink; an empty list "
                        "(default) short-circuits everywhere.")


def _template_stats(detector: Detector) -> dict:
    """Template provenance for the manifest ``stats`` block (flags
    approximate/catalog/generic templates)."""
    if detector.template is not None:
        return {"template": detector.template.manifest_block()}
    return {}


def _veto_stats(acc) -> dict:
    """Veto accounting for the manifest ``stats`` block (step 5)."""
    if acc is not None:
        return {"veto_accounting": acc.to_dict()}
    return {}


def _gain_stats(detector: Detector, settings) -> dict:
    """PE-scale provenance + gate config for the manifest ``stats`` block."""
    g = detector.gain
    out = {"pe_gate": {"pe_band": list(detector.pe_band),
                       "in_regime_only": bool(detector.in_regime_only)}}
    if g is not None:
        out["gain"] = {"source": g.provenance.source,
                       "spe_charge_pvs": g.per_branch,
                       **({"detail": g.provenance.detail}
                          if g.provenance.detail else {})}
    return out


def _require_gain_for_gate(detector: Detector) -> None:
    """`--in-regime-only` without a PE anchor is a hard, loud error — the tool
    never invents a charge scale (governing philosophy)."""
    if detector.in_regime_only and (detector.gain is None
                                    or not detector.gain.resolved):
        raise SystemExit(
            "--in-regime-only needs a per-branch 1-PE charge anchor and none "
            "is configured (no card `spe_charge`, no --spe-charge, no gain-"
            "catalog entry for this pmt_type).\n"
            "  Fit one first:  pmtdeconv gain calibrate --dataset <ds> "
            "--runs <N>\n"
            "  then paste the proposed `spe_charge:` block into the dataset "
            "card (or pass --spe-charge BR=VAL).")


def _build_diagnostic_sink(settings, out_dir, diag_events):
    """Build the per-event diagnostic sink if --diagnostic-events was passed
    (else None). Shared by the wavelet/object and STFT paths."""
    if not diag_events:
        return None
    from pmtdeconv.io.truth import find_truth_for
    from pmtdeconv.pipeline.diagnostic import DiagnosticSink
    truth_csv = None
    if settings.data_root is not None:
        for cand in Path(settings.data_root).glob("flattened_Run*.root"):
            truth_csv = find_truth_for(cand)
            if truth_csv:
                break
    sink = DiagnosticSink(
        out_root=Path(out_dir),
        target_events=set(diag_events),
        truth_csv=truth_csv,
        denoiser=settings.denoiser,
        cosine_cutoff_mhz=float(settings.cosine_cutoff_mhz),
        dt_ns=(float(settings.dt_ns) if isinstance(settings.dt_ns, float)
               else 0.5051),
    )
    print(f"  diagnostic-events ({settings.denoiser}): {sorted(diag_events)} "
          f"→ {out_dir} (truth={'yes' if truth_csv else 'no'})")
    return sink


def _data_inputs(settings, runs) -> list:
    """Manifest entries for the run files the readers open, found the same
    way they find them (flat: ``resolve_run_file``; object: ``file_glob``)."""
    from pmtdeconv.io.data_paths import resolve_run_file
    data_dir = Path(settings.data_root)
    entries = []
    for run in runs:
        if settings.format == "object":
            glob = settings.file_glob or "MilliQan_Run*_triple.root"
            found = find_run_files(data_dir, glob, run)
            path = found[0] if found else None
        else:
            path = resolve_run_file(data_dir, run, kind="flat")
        if path is not None:
            entries.append({"role": "root_data", "path": str(path)})
    return entries


def _run_stft(settings, provenance, t0, detector, diag_events) -> int:
    """STFT-denoiser path. Calls ``process_hits.process_run_stft``
    (the unified hit-finder with ``STFT_PROFILE``) with the resolved
    :class:`Detector`.
    """
    out_dir = settings.out_dir or hits_output_dir(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector.out_dir = str(out_dir)

    if maybe_dry_run_exit(subcommand="hits", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    from pmtdeconv.pipeline.process_hits import process_run_stft

    diagnostic_sink = _build_diagnostic_sink(settings, out_dir, diag_events)

    print("Delegating to STFT process_run")
    acc = process_run_stft(detector, diagnostic_sink=diagnostic_sink)

    outputs_meta = []
    for role, name in [
        (DECONV_PULSE_ROLE, DECONV_PULSE_CSV),
        ("all_results",        "all_results.csv"),
    ]:
        p = out_dir / name
        if p.exists():
            outputs_meta.append({"role": role, "path": str(p)})

    inputs_meta = []
    if settings.template_csv and Path(settings.template_csv).exists():
        inputs_meta.append({"role": "template_csv",
                            "path": str(settings.template_csv)})
    inputs_meta += _data_inputs(settings, detector.runs)

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="hits",
        settings=settings, provenance=provenance,
        inputs=inputs_meta, outputs=outputs_meta,
        stats={"denoiser": "stft", "runs": settings.runs or [],
               "stft_nperseg_ns": detector.stft_nperseg_ns,
               "stft_freq_cutoff_mhz": detector.stft_freq_cutoff_mhz,
               "stft_amp_threshold": detector.stft_amp_threshold,
               "veto_profile": detector.veto_profile,
               "veto_profile_source": detector.veto_profile_source,
               "veto_profile_version": detector.veto_profile_version,
               **_template_stats(detector), **_veto_stats(acc),
               **_gain_stats(detector, settings)},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)
    from pmtdeconv.io.banner import print_run_banner
    print_run_banner("hits", settings, provenance)
    if not settings.runs and not settings.dry_run:
        raise SystemExit("--runs is required (got an empty list).")
    diag_events = list(getattr(args, "diagnostic_events", None) or [])

    # Single resolution point: Settings -> Detector (kills _build_legacy_args).
    # Carries the resolved Template (explicit/dataset/catalog/generic) +
    # provenance. Polarity now flows through the shared event source.
    detector = Detector.from_settings(settings, provenance, mode="hits")
    _require_gain_for_gate(detector)

    if settings.denoiser == "wavelet":
        print("[experimental] wavelet denoiser — STFT is the default since the "
              "2026-10 denoiser study (STFT reconstructed charge best on every "
              "simulated test set). Wavelet left less timewalk on the Run 65213 "
              "benchmark and remains usable.")

    # Flat STFT has its own reader; object-format files (LANL) go through
    # process_run_object for every denoiser.
    if settings.denoiser == "stft" and settings.format != "object":
        return _run_stft(settings, provenance, t0, detector, diag_events)

    # PR 2 / decision 9: process_run lives in pmtdeconv.pipeline now.
    # PR 5: object-format dispatch for LANL_835 / movedBar.
    from pmtdeconv.pipeline.process_hits import (    # noqa
        process_run, process_run_object,
    )

    is_object_format = (settings.format == "object")

    out_dir = (Path(settings.out_dir) if settings.out_dir else
               hits_output_dir(settings))
    out_dir.mkdir(parents=True, exist_ok=True)
    # The readers write here, and so does the manifest: one directory.
    detector.out_dir = str(out_dir)

    if maybe_dry_run_exit(subcommand="hits", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    diagnostic_sink = _build_diagnostic_sink(settings, out_dir, diag_events)

    if is_object_format:
        print("Delegating to process_run_object (object format)")
        acc = process_run_object(detector, diagnostic_sink=diagnostic_sink)
    else:
        print("Delegating to process_run (flat format)")
        acc = process_run(detector, diagnostic_sink=diagnostic_sink)

    inputs_meta = []
    if settings.template_csv and Path(settings.template_csv).exists():
        inputs_meta.append({"role": "template_csv",
                            "path": str(settings.template_csv)})
    inputs_meta += _data_inputs(settings, detector.runs)

    outputs_meta = [
        {"role": DECONV_PULSE_ROLE,
         "path": str(out_dir / DECONV_PULSE_CSV)},
        {"role": "all_results",
         "path": str(out_dir / "all_results.csv")},
    ]
    neg_path = out_dir / "deconv_negative_charge.csv"
    if neg_path.exists():
        outputs_meta.append({"role": "deconv_negative_charge",
                             "path": str(neg_path)})

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="hits",
        settings=settings,
        provenance=provenance,
        inputs=inputs_meta,
        outputs=outputs_meta,
        stats={"runs": detector.runs, "veto_profile": detector.veto_profile,
               "veto_profile_source": detector.veto_profile_source,
               "veto_profile_version": detector.veto_profile_version,
               **_template_stats(detector), **_veto_stats(acc),
               **_gain_stats(detector, settings)},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
