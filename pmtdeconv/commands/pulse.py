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

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.deconv_pulse import (
    DECONV_PULSE_COMPAT_CSV,
    DECONV_PULSE_CSV,
    DECONV_PULSE_COMPAT_ROLE,
    DECONV_PULSE_ROLE,
)
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.io.output_paths import pulse_output_dir
from pmtdeconv.pipeline.detector import Detector
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("pulse")
    g.add_argument("--template", type=str, default=None,
                   help="SPE template CSV path. Default: dataset's preset.")
    g.add_argument("--allow-generic-fallback", dest="allow_generic_fallback",
                   action="store_true", default=None,
                   help="When no per-channel template CSV is configured and the "
                        "PMT type is uncatalogued, use the generic MilliQan "
                        "fallback template instead of erroring (loudly logged). "
                        "Restores the pre-step-3 silent-fallback behavior.")
    g.add_argument("--denoiser", choices=("wavelet", "stft", "none"),
                   default=None,
                   help="Post-deconv denoiser (default: wavelet).")
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
    g.add_argument("--peak-order", choices=("chrono", "tallest", "auto"),
                   default=None,
                   help="How to assign peak_id within a waveform. "
                        "`chrono` (legacy): left-to-right in time. "
                        "`tallest`: descending u_rec amplitude (peak_id=1 "
                        "= largest peak, useful when vetoes are off). "
                        "`auto` (default): chrono if vetoes on, tallest "
                        "if --no-vetoes (decision 8).")
    g.add_argument("--stft-nperseg-ns", type=float, default=None,
                   help="STFT window size in ns (STFT denoiser only, default 20).")
    g.add_argument("--stft-freq-cutoff-mhz", type=float, default=None,
                   help="STFT high-frequency cutoff in MHz (default 175).")
    g.add_argument("--stft-amp-threshold", type=float, default=None,
                   help="STFT amplitude threshold (default 10).")
    # Bug-fix 1 (2026-05-30 scan): expose cosine_cutoff_mhz so tuned
    # presets actually reach the processor. Without this, tuned recipes
    # silently revert to the pipeline's hardcoded defaults (450 MHz
    # wavelet, 100 MHz STFT).
    g.add_argument("--cosine-cutoff-mhz", type=float, default=None,
                   help="Cosine low-pass cutoff in MHz. Default: dataset "
                        "preset. Tuned presets via --preset propagate here.")
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


def _run_stft(settings, provenance, t0, detector, diag_events) -> int:
    """STFT-denoiser path. Calls ``process_hits.process_run_stft``
    (the unified hit-finder with ``STFT_PROFILE``) with the resolved
    :class:`Detector`.
    """
    out_dir = settings.out_dir or pulse_output_dir(settings, "stft")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector.out_dir = str(out_dir)

    if maybe_dry_run_exit(subcommand="pulse", settings=settings,
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
        (DECONV_PULSE_COMPAT_ROLE, DECONV_PULSE_COMPAT_CSV),
        ("all_results",        "all_results.csv"),
    ]:
        p = out_dir / name
        if p.exists():
            outputs_meta.append({"role": role, "path": str(p)})

    inputs_meta = []
    if settings.template_csv and Path(settings.template_csv).exists():
        inputs_meta.append({"role": "template_csv",
                            "path": str(settings.template_csv)})
    for run_num in (settings.runs or []):
        if settings.data_root:
            rp = Path(settings.data_root) / f"flattened_Run{run_num}.root"
        else:
            rp = (PROJECT_ROOT.parent.parent.parent
                  / "data" / "Na22_lab" / f"flattened_Run{run_num}.root")
        if rp.exists():
            inputs_meta.append({"role": "root_data", "path": str(rp)})

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="pulse",
        settings=settings, provenance=provenance,
        inputs=inputs_meta, outputs=outputs_meta,
        stats={"denoiser": "stft", "runs": settings.runs or [],
               "stft_nperseg_ns": detector.stft_nperseg_ns,
               "stft_freq_cutoff_mhz": detector.stft_freq_cutoff_mhz,
               "stft_amp_threshold": detector.stft_amp_threshold,
               "veto_profile": detector.veto_profile,
               **_template_stats(detector), **_veto_stats(acc)},
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
    diag_events = list(getattr(args, "diagnostic_events", None) or [])

    # Single resolution point: Settings -> Detector (kills _build_legacy_args).
    # Carries the resolved Template (explicit/dataset/catalog/generic) +
    # provenance. Polarity now flows through the shared event source.
    detector = Detector.from_settings(settings, provenance, mode="hits")

    if settings.denoiser == "stft":
        return _run_stft(settings, provenance, t0, detector, diag_events)

    # PR 2 / decision 9: process_run lives in pmtdeconv.pipeline now.
    # PR 5: object-format dispatch for LANL_835 / movedBar.
    from pmtdeconv.pipeline.process_hits import (    # noqa
        process_run, process_run_object,
    )

    is_object_format = (settings.format == "object")

    # Object-format dt_ns=="auto": resolve from the first matching ROOT file
    # (per-run; LANL sub-runs differ). Placeholder 1.25 matches legacy.
    if settings.dt_ns == "auto":
        detector.dt_ns = 1.25

    out_dir = (Path(settings.out_dir) if settings.out_dir else
               pulse_output_dir(settings, "wavelet"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="pulse", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    diagnostic_sink = _build_diagnostic_sink(settings, out_dir, diag_events)

    if is_object_format:
        if settings.dt_ns == "auto":
            data_dir = Path(settings.data_root) if settings.data_root else (
                PROJECT_ROOT.parent.parent.parent / "data" / "835_real")
            for run_num in detector.runs:
                cands = sorted(data_dir.glob(settings.file_glob))
                cands = [p for p in cands if str(run_num) in p.name]
                if cands:
                    from pmtdeconv.io.object_format import read_dt_ns
                    detector.dt_ns = read_dt_ns(
                        cands[0], tree_name=settings.tree or "Events")
                    print(f"  dt_ns (auto-detected): {detector.dt_ns} ns")
                    break
        detector.out_dir = str(out_dir)
        print("Delegating to process_run_object (object format)")
        acc = process_run_object(detector, diagnostic_sink=diagnostic_sink)
    else:
        # Flat path: only pin out_dir when explicit (legacy parity — otherwise
        # process_run computes its own default directory).
        if settings.out_dir is not None:
            detector.out_dir = str(settings.out_dir)
        print(f"Delegating to legacy process_run "
              f"(out_suffix={detector.out_suffix!r})")
        acc = process_run(detector, diagnostic_sink=diagnostic_sink)

    inputs_meta = []
    if settings.template_csv and Path(settings.template_csv).exists():
        inputs_meta.append({"role": "template_csv",
                            "path": str(settings.template_csv)})
    for run_num in detector.runs:
        rp = (PROJECT_ROOT.parent.parent.parent
              / "data" / "Na22_lab" / f"flattened_Run{run_num}.root")
        if rp.exists():
            inputs_meta.append({"role": "root_data", "path": str(rp)})

    outputs_meta = [
        {"role": DECONV_PULSE_ROLE,
         "path": str(out_dir / DECONV_PULSE_CSV)},
        {"role": DECONV_PULSE_COMPAT_ROLE,
         "path": str(out_dir / DECONV_PULSE_COMPAT_CSV)},
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
        subcommand="pulse",
        settings=settings,
        provenance=provenance,
        inputs=inputs_meta,
        outputs=outputs_meta,
        stats={"runs": detector.runs, "veto_profile": detector.veto_profile,
               **_template_stats(detector), **_veto_stats(acc)},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
