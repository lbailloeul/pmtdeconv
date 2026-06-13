"""Thin per-run readers for the peak-finding (hits) deconvolution path.

The hits ENGINE — :class:`~pmtdeconv.pipeline.pipeline.HitsProfile`, the
per-denoiser profiles, the per-event body (``_process_event``) and its loop
(:func:`~pmtdeconv.pipeline.pipeline.run_hits`) — lives in
:mod:`pmtdeconv.pipeline.pipeline` so that module owns BOTH processing modes
(hits + area). The functions here are thin format readers: resolve the ROOT
file(s), stream ``EventRecord``s from :mod:`pmtdeconv.pipeline.event_source`,
hand them to ``run_hits``, write the CSV tables, and emit the veto-rate report.
This mirrors how :mod:`pmtdeconv.pipeline.process_area` feeds ``run_area``.

Polarity is applied once, in the event source (the 2026-06-10 flat-path bug
class). Both the legacy scripts and ``pmtdeconv.commands.pulse`` call these; the
byte-identical regression tests (``tests/test_golden_fixture.py``) confirm the
relocation didn't drift.
"""
from __future__ import annotations

from pathlib import Path

from pmtdeconv.pipeline.deconv_pulse_results import (
    print_deconv_pulse_summary,
    write_deconv_pulse_result_tables,
)
from pmtdeconv.pipeline.veto_accounting import report as _veto_report
# Engine relocated to pipeline.pipeline; re-exported here so callers/tests that
# did ``from ...process_hits import WAVELET_PROFILE`` (etc.) keep working.
from pmtdeconv.pipeline.pipeline import (   # noqa: F401
    HitsProfile,
    WAVELET_PROFILE,
    STFT_PROFILE,
    _process_event,
    run_hits,
)
from pmtdeconv.pipeline.template import FALLBACK_TEMPLATE   # noqa: F401

# Defaults matching the legacy module so behavior is byte-identical when no
# override is supplied via args.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR_DEFAULT = PROJECT_ROOT / "outputs"
DATA_DIR_DEFAULT = PROJECT_ROOT.parent.parent.parent / "data"
FIT_CSV_DEFAULT = OUTPUTS_DIR_DEFAULT / "biexp_fits_Na22_lab" / "biexp_fit_results.csv"


def _legacy_load_channel_templates(csv_path):
    """Per-event biexp CSV → per-branch median, via the unified loader.

    Thin wrapper around ``pmtdeconv.io.templates.load_template`` that preserves
    the original verbose per-channel printing of the pre-PR-3 wavelet pipeline.
    """
    from pmtdeconv.io.templates import load_template
    templates = load_template(csv_path, strict=False, verbose=True)
    for branch, tmpl in templates.items():
        print(f"  {branch}: τ_rise={tmpl['tau_rise']:.3f} ns, "
              f"τ_fall={tmpl['tau_fall']:.3f} ns, σ={tmpl['sigma']:.3f} ns")
    return templates


def process_run_object(args, diagnostic_sink=None):
    """Object-format hits reader for LANL_835/movedBar.

    Resolves the ROOT file + channel set per run, streams events through
    :func:`pmtdeconv.pipeline.event_source.iter_object_run_events`, and feeds
    :func:`pmtdeconv.pipeline.pipeline.run_hits` (wavelet profile). Only the I/O
    differs from the flat :func:`process_run`.
    """
    from pmtdeconv.io.object_format import detect_populated_channels
    from pmtdeconv.pipeline.diagnostic import noop_sink
    from pmtdeconv.pipeline.event_source import iter_object_run_events
    sink = diagnostic_sink if diagnostic_sink is not None else noop_sink()

    dt_ns = args.dt_ns
    polarity = int(getattr(args, "polarity", -1))
    entry_stop = args.entry_stop
    step_size = 100
    cosine_cutoff_mhz = float(getattr(args, "cosine_cutoff_mhz", None) or 450.0)
    sat_threshold = args.sat_thresh

    explicit_out_dir = getattr(args, "out_dir", None)
    if explicit_out_dir is not None:
        out_dir = Path(explicit_out_dir)
    else:
        out_dir = (OUTPUTS_DIR_DEFAULT
                   / f"deconv_pulse_object_lab_cosine_wavelet{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_templates = _legacy_load_channel_templates(
        Path(args.fit_csv) if args.fit_csv else FIT_CSV_DEFAULT)

    print(f"Runs: {args.runs}  (object format)")
    print(f"Sample interval: dt_ns = {dt_ns}  ({1.0/dt_ns:.3f} GSa/s)")
    print(f"Filter: cosine (cutoff={cosine_cutoff_mhz} MHz)")
    print(f"Polarity: {polarity:+d}")
    print(f"Saturation threshold: {sat_threshold} mV")
    print(f"Output: {out_dir}")

    explicit_data_root = getattr(args, "data_root", None)
    if explicit_data_root is not None:
        data_dir = Path(explicit_data_root)
    else:
        data_dir = (Path(DATA_DIR_DEFAULT)
                    / (getattr(args, "data_subdir", None) or "835_real"))
    file_glob = getattr(args, "file_glob", None) or "MilliQan_Run*_triple.root"
    tree_name = getattr(args, "tree", None) or "Events"

    def _events():
        for run in args.runs:
            candidates = sorted(data_dir.glob(file_glob))
            candidates = [p for p in candidates if str(run) in p.name]
            if not candidates:
                print(f"[skip] no object-format ROOT matching run {run} under {data_dir}")
                continue
            root_path = candidates[0]
            print(f"\n================ Run {run} ================")
            print(f"Opening: {root_path}")

            explicit_branches = getattr(args, "branches", None)
            if isinstance(explicit_branches, list) and explicit_branches:
                branches = explicit_branches
            else:
                branches = detect_populated_channels(root_path, tree_name=tree_name)
                print(f"  Auto-detected {len(branches)} populated channels: "
                      f"{branches[:4]}{'...' if len(branches) > 4 else ''}")

            yield from iter_object_run_events(
                root_path, run, dt_ns=dt_ns, polarity=polarity,
                branches=branches, entry_stop=entry_stop, step_size=step_size,
                tree_name=tree_name,
            )

    results = run_hits(_events(), args=args, profile=WAVELET_PROFILE,
                       templates=ch_templates, cosine_cutoff_mhz=cosine_cutoff_mhz,
                       sat_threshold=sat_threshold, sink=sink)
    tables = write_deconv_pulse_result_tables(results, out_dir)
    print_deconv_pulse_summary(tables)
    return _veto_report(tables.all_rows,
                        float(getattr(args, "veto_warn_frac", 0.5)),
                        label=f"run {args.runs} (object)")


def process_run(args, diagnostic_sink=None):
    """Per-run peak-finding deconvolution loop (flat format).

    Resolves ``flattened_Run<N>.root``, streams events through
    :func:`pmtdeconv.pipeline.event_source.iter_flat_run`, and feeds
    :func:`pmtdeconv.pipeline.pipeline.run_hits` (wavelet profile). The per-event
    body is identical to the object path; only the reader differs.
    """
    from pmtdeconv.pipeline.diagnostic import noop_sink
    from pmtdeconv.pipeline.event_source import iter_flat_run
    sink = diagnostic_sink if diagnostic_sink is not None else noop_sink()
    dt_ns = args.dt_ns                 # DRS4 sample interval [ns]; per-run via --dt-ns
    # Polarity is applied uniformly in the event source (single flip point).
    # Default +1 is a no-op there, preserving the golden fixtures.
    polarity = int(getattr(args, "polarity", 1))

    entry_stop = args.entry_stop
    step_size = 500
    cosine_cutoff_mhz = float(getattr(args, "cosine_cutoff_mhz", None) or 450.0)
    sat_threshold = args.sat_thresh

    explicit_data_root = getattr(args, "data_root", None)
    if explicit_data_root is not None:
        data_dir = Path(explicit_data_root)
    else:
        data_subdir = getattr(args, "data_subdir", None) or "Na22_lab"
        data_dir = Path(DATA_DIR_DEFAULT) / data_subdir
    explicit_out_dir = getattr(args, "out_dir", None)
    if explicit_out_dir is not None:
        out_dir = Path(explicit_out_dir)
    else:
        out_dir = (OUTPUTS_DIR_DEFAULT
                   / f"deconv_pulse_Na22_lab_cosine_wavelet{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_templates = _legacy_load_channel_templates(
        Path(args.fit_csv) if args.fit_csv else FIT_CSV_DEFAULT)

    print(f"Runs: {args.runs}")
    print(f"Sample interval: dt_ns = {dt_ns}  ({1.0/dt_ns:.3f} GSa/s)")
    print(f"Filter: cosine (cutoff={cosine_cutoff_mhz} MHz)")
    if args.no_denoise:
        print("Wavelet denoising: DISABLED (--no-denoise)")
    else:
        print(f"Wavelet denoising: {args.wavelet}, level={args.wlevel}, "
              f"mode={args.wmode}, k={args.wthresh_k}"
              + (",  ALSO on raw waveform" if args.denoise_raw else ""))
    print(f"Saturation threshold: {sat_threshold} mV")
    print(f"Polarity: {polarity:+d}")
    print(f"Output: {out_dir}")

    def _events():
        for run in args.runs:
            root_path = str(data_dir / f"flattened_Run{run}.root")
            yield from iter_flat_run(
                root_path, run, dt_ns=dt_ns, polarity=polarity,
                branches=None, entry_start=0, entry_stop=entry_stop,
                step_size=step_size,
            )

    results = run_hits(_events(), args=args, profile=WAVELET_PROFILE,
                       templates=ch_templates, cosine_cutoff_mhz=cosine_cutoff_mhz,
                       sat_threshold=sat_threshold, sink=sink)
    tables = write_deconv_pulse_result_tables(results, out_dir)
    print_deconv_pulse_summary(tables, show_full_results=True)
    return _veto_report(tables.all_rows,
                        float(getattr(args, "veto_warn_frac", 0.5)),
                        label=f"run {args.runs}")


def process_run_stft(args, diagnostic_sink=None):
    """STFT-denoiser flat-format peak-finding loop.

    Same event source + shared engine as the wavelet path, selecting
    :data:`~pmtdeconv.pipeline.pipeline.STFT_PROFILE` (relaxed vetoes, loose
    baseline gate, ``stft_denoise``). STFT is flat-only — there is no object
    variant. Byte-identical to the deleted ``process_hits_stft.py`` god-module
    (golden: ``deconv_pulses_stft_run0.csv``).

    STFT knobs come from ``args``: ``stft_nperseg_ns`` (default 20.0),
    ``stft_freq_cutoff_mhz`` (175.0), ``stft_amp_threshold`` (10.0).
    """
    from pmtdeconv.io.templates import load_template
    from pmtdeconv.pipeline.diagnostic import noop_sink
    from pmtdeconv.pipeline.event_source import iter_flat_run
    sink = diagnostic_sink if diagnostic_sink is not None else noop_sink()

    dt_ns = args.dt_ns
    polarity = int(getattr(args, "polarity", 1))
    entry_stop = args.entry_stop
    step_size = 500
    # STFT path historically uses 100 MHz (narrower than the wavelet 450).
    cosine_cutoff_mhz = float(getattr(args, "cosine_cutoff_mhz", None) or 100.0)
    sat_threshold = getattr(args, "sat_thresh", 999.0)
    stft_freq_cutoff_hz = getattr(args, "stft_freq_cutoff_mhz", 175.0) * 1e6
    stft_nperseg_ns = getattr(args, "stft_nperseg_ns", 20.0)
    stft_amp_threshold = getattr(args, "stft_amp_threshold", 10.0)

    explicit_out_dir = getattr(args, "out_dir", None)
    if explicit_out_dir is not None:
        out_dir = Path(explicit_out_dir)
    else:
        out_dir = OUTPUTS_DIR_DEFAULT / "deconv_pulse_Na22_lab_cosine_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_data_root = getattr(args, "data_root", None)
    if explicit_data_root is not None:
        data_dir = Path(explicit_data_root)
    else:
        data_subdir = getattr(args, "data_subdir", None) or "Na22_lab"
        data_dir = Path(DATA_DIR_DEFAULT) / data_subdir

    ch_templates = load_template(
        Path(args.fit_csv) if args.fit_csv else FIT_CSV_DEFAULT,
        strict=False, verbose=True)
    for branch, tmpl in ch_templates.items():
        print(f"  {branch}: τ_rise={tmpl['tau_rise']:.3f} ns, "
              f"τ_fall={tmpl['tau_fall']:.3f} ns, σ={tmpl['sigma']:.3f} ns")

    print(f"Runs: {args.runs}")
    print(f"Sample interval: dt_ns = {dt_ns}  ({1.0/dt_ns:.3f} GSa/s)")
    print(f"Filter: cosine (cutoff={cosine_cutoff_mhz} MHz)")
    print(f"STFT: nperseg={stft_nperseg_ns} ns, freq_cutoff={stft_freq_cutoff_hz:.0e}, "
          f"amp_threshold={stft_amp_threshold}")
    print(f"Saturation threshold: {sat_threshold} mV")
    print(f"Output: {out_dir}")

    def _events():
        for run in args.runs:
            root_path = str(data_dir / f"flattened_Run{run}.root")
            yield from iter_flat_run(
                root_path, run, dt_ns=dt_ns, polarity=polarity,
                branches=None, entry_start=0, entry_stop=entry_stop,
                step_size=step_size,
            )

    results = run_hits(_events(), args=args, profile=STFT_PROFILE,
                       templates=ch_templates, cosine_cutoff_mhz=cosine_cutoff_mhz,
                       sat_threshold=sat_threshold, sink=sink)
    tables = write_deconv_pulse_result_tables(results, out_dir)
    print_deconv_pulse_summary(tables)
    return _veto_report(tables.all_rows,
                        float(getattr(args, "veto_warn_frac", 0.5)),
                        label=f"run {args.runs} (stft)")
