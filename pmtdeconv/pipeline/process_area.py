"""Per-run waveform-summary deconvolution loop (area mode).

Lifted from ``deconvolution/deconv_area_Na22.process_run`` (PR 2 / decision 9);
the byte-identical regression tests (``tests/test_golden_fixture.py``:
``deconv_area_run0``, ``object_deconv_area_run99``, ``deconv_area_fftpad_run0``)
confirm the lift didn't drift.

Step 4 (architecture restructure): the flat and object variants used to inline
the *same* gate→baseline→deconv→denoise→tail loop twice. They now both build an
:class:`~pmtdeconv.pipeline.pipeline.AreaMode` strategy and feed the shared
:func:`~pmtdeconv.pipeline.pipeline.run_area`; the two functions here are thin
I/O readers (flat ``waveforms`` tree vs nested object format). Their signatures
are unchanged so ``commands.area`` and ``commands.tune_cosine`` keep working
(the ``process_run`` shim the plan calls for).
"""
from __future__ import annotations

from pathlib import Path

# Re-exported for backwards compatibility: ``commands.area`` and
# ``commands.tune_cosine`` do ``from ...process_area import process_run,
# FALLBACK_TEMPLATE``. The canonical definition now lives in pipeline.template.
from pmtdeconv.pipeline.pipeline import (  # noqa: F401
    AreaMode,
    FALLBACK_TEMPLATE,
    _next_pow2,
    run_area,
)

STEP = 500    # uproot.iterate chunk size (events), flat format
# Nested-cycle resolution + flat/object waveform iteration now live in
# pmtdeconv.pipeline.event_source (shared with the hits path); the readers
# below are thin adapters that map EventRecords onto AreaEvent tuples.


def process_run_object(root_path, dt_ns, sig_start, sig_end, n_intervals,
                       cosine_cutoff_mhz,
                       wavelet, wlevel, mode, thresh_k,
                       no_trigger_amp,
                       t_peak_min_ns, afterpulse_window_ns, afterpulse_height_ratio,
                       fft_pad, wrap_ratio, wrap_abs_k,
                       entry_stop, templates, branches_keep=None,
                       *, polarity: int = -1, run_num: int = 0,
                       tree_name: str = "Events",
                       step_size: int = 100,
                       measure_width: bool = False,
                       gain=None, pe_band=(0.0, 20.0),
                       in_regime_only: bool = False,
                       entry_keep=None):
    """Object-format area reader for LANL_835/movedBar.

    Same shared prefix + tail as :func:`process_run` (via :func:`run_area`);
    only the I/O differs — nested ``digitizers.waveform[16][1024]`` via
    :func:`pmtdeconv.io.object_format.iter_object_run`, polarity flipped per
    event (LANL convention ``-1``). The object tail historically ran NEITHER
    the absolute-noise-floor left-edge gate NOR the afterpulse veto, so
    ``AreaMode`` is built with both flags off (byte-identical).
    """
    from pmtdeconv.io.object_format import detect_populated_channels
    from pmtdeconv.pipeline.event_source import iter_object_run_events

    if branches_keep:
        branches = list(branches_keep)
    else:
        branches = detect_populated_channels(root_path, tree_name=tree_name)
        print(f"  Auto-detected {len(branches)} populated channels: "
              f"{branches[:4]}{'...' if len(branches) > 4 else ''}")

    area = AreaMode(
        sig_start=sig_start, sig_end=sig_end, n_intervals=n_intervals,
        fft_pad=fft_pad, no_trigger_amp=no_trigger_amp,
        t_peak_min_ns=t_peak_min_ns, afterpulse_window_ns=afterpulse_window_ns,
        afterpulse_height_ratio=afterpulse_height_ratio,
        wrap_ratio=wrap_ratio, wrap_abs_k=wrap_abs_k,
        measure_width=measure_width,
        use_wrap_abs_k=False, use_afterpulse=False,
    )

    print(f"  Reading object-format ROOT {root_path.name} "
          f"(entry_stop={entry_stop or 'all'})")

    def _events():
        # Same reader as the hits path → polarity applied in one place.
        for rec in iter_object_run_events(
            root_path, run_num, dt_ns=dt_ns, polarity=polarity,
            branches=branches, entry_stop=entry_stop, step_size=step_size,
            tree_name=tree_name,
        ):
            if entry_keep is not None and rec.entry not in entry_keep:
                continue
            yield (rec.run, rec.cycle, rec.entry, rec.branch, rec.wave, rec.N)

    df = run_area(_events(), dt_ns=dt_ns, templates=templates,
                  cosine_cutoff_mhz=cosine_cutoff_mhz, wavelet=wavelet,
                  wlevel=wlevel, denoise_mode=mode, thresh_k=thresh_k, area=area,
                  gain=gain, pe_band=pe_band, in_regime_only=in_regime_only)
    print(f"  Total rows : {len(df)}")
    if len(df):
        print(df["reason"].value_counts().to_string())
    return df


def process_run(root_path, dt_ns, sig_start, sig_end, n_intervals,
                cosine_cutoff_mhz,
                wavelet, wlevel, mode, thresh_k,
                no_trigger_amp,
                t_peak_min_ns, afterpulse_window_ns, afterpulse_height_ratio,
                fft_pad, wrap_ratio, wrap_abs_k,
                entry_stop, templates, branches_keep=None,
                *, polarity: int = 1, measure_width: bool = False,
                gain=None, pe_band=(0.0, 20.0),
                in_regime_only: bool = False,
                entry_keep=None):
    """For every trigger, deconvolve each channel and integrate over
    ``[sig_start, sig_end]``. Returns a DataFrame with columns
    ``run, cycle, entry, branch, interval, area, peak_height[, width_ns]``.

    Flat-format area reader. The flat tail runs the full inline vetoes
    (absolute-noise-floor left-edge gate + afterpulse), so ``AreaMode`` is
    built with both flags on.

    Events stream through :func:`pmtdeconv.pipeline.event_source.iter_flat_run`
    — the SAME reader the hits path uses, so polarity is applied in one place
    (``polarity`` defaults +1, a no-op preserving the na22 goldens; negative
    flat datasets now flip correctly here too, closing the 2026-06-10 latent
    bug for the area path).

    ``measure_width`` (default False — keeps the ``waveform`` golden
    byte-identical): when True, add a ``width_ns`` column carrying the FWHM
    (ns) of the reconstructed pulse at each interval's peak. Used by
    ``pmtdeconv tune cosine``.
    """
    from pmtdeconv.pipeline.event_source import iter_flat_run
    run = int(Path(root_path).stem.replace("flattened_Run", ""))

    area = AreaMode(
        sig_start=sig_start, sig_end=sig_end, n_intervals=n_intervals,
        fft_pad=fft_pad, no_trigger_amp=no_trigger_amp,
        t_peak_min_ns=t_peak_min_ns, afterpulse_window_ns=afterpulse_window_ns,
        afterpulse_height_ratio=afterpulse_height_ratio,
        wrap_ratio=wrap_ratio, wrap_abs_k=wrap_abs_k,
        measure_width=measure_width,
        use_wrap_abs_k=True, use_afterpulse=True,
    )

    def _events():
        for rec in iter_flat_run(
            str(root_path), run, dt_ns=dt_ns, polarity=polarity,
            branches=(list(branches_keep) if branches_keep else None),
            entry_stop=entry_stop, step_size=STEP,
        ):
            if entry_keep is not None and rec.entry not in entry_keep:
                continue
            yield (rec.run, rec.cycle, rec.entry, rec.branch, rec.wave, rec.N)

    df = run_area(_events(), dt_ns=dt_ns, templates=templates,
                  cosine_cutoff_mhz=cosine_cutoff_mhz, wavelet=wavelet,
                  wlevel=wlevel, denoise_mode=mode, thresh_k=thresh_k, area=area,
                  gain=gain, pe_band=pe_band, in_regime_only=in_regime_only)
    print(f"  Total rows : {len(df)}")
    if len(df):
        print(df["reason"].value_counts().to_string())
    print(f"  Area computed: {df['area'].notna().sum()}")
    return df
