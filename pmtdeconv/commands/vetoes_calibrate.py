"""``pmtdeconv vetoes calibrate`` — derive a data-driven veto profile (step 5).

Scans a sample of a run's RAW waveforms and derives candidate veto thresholds
from the observed statistics, then prints them for inspection and (with
``--write-profile NAME``) saves the profile to the veto catalog. No silent
per-run auto-calibration — reproducible and reviewable by design; the
high-veto-rate warning (``pmtdeconv hits``) points the user here.

HYBRID criterion (user decision):
  - DATA-DRIVEN (default): quantiles of the run's own raw distributions —
      ``no_trigger_amp`` = ``k × MAD`` above the raw pre-trigger noise floor;
      ``t_peak_min_ns`` / ``t_peak_max_ns`` = [q1, q99] of observed raw signal
      times (events above the new trigger floor).
  - GROUND-TRUTH SCORED (``--truth-csv``): when a truth CSV is supplied, the
      trigger floor is chosen to retain ``--target-eff`` of the TRUE
      (non-saturated) pulses' raw peaks — an ε-constraint consistent with
      ``tune`` (retain ≥ target efficiency, take the largest floor that does) —
      and the time window is [q1, q99] of the TRUE pulse centers, not the noisy
      raw argmax. The achieved efficiency + a purity figure (true-signal / all
      events kept in-window) are reported. Falls back to data-driven (with a
      note) if the truth CSV can't be loaded or matches too few signal rows.

Other thresholds (post-saturation, charge-purity, afterpulse) are inherited
from the base profile in both modes; deriving them needs the deconvolved
waveform and is a tracked follow-up (see :mod:`pmtdeconv.pipeline.calibration`).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.core.dsp import detect_saturation_plateau
from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.pipeline.calibration import ProvenanceRecord
from pmtdeconv.presets import resolve_settings
from pmtdeconv.presets.veto_profiles import VETO_PROFILES, VetoProfile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("vetoes calibrate")
    g.add_argument("--sample-events", type=int, default=500,
                   help="Number of raw waveforms to scan (default 500).")
    g.add_argument("--k-mad", type=float, default=5.0,
                   help="min_raw_amp = k × MAD noise floor (default 5).")
    g.add_argument("--base-denoiser", dest="base_denoiser",
                   choices=("wavelet", "stft", "none"), default="wavelet",
                   help="Base profile to refine (default wavelet → wavelet_hits).")
    g.add_argument("--write-profile", type=str, default=None, metavar="NAME",
                   help="Save the derived profile to the veto catalog under NAME.")
    g.add_argument("--truth-csv", type=str, default=None,
                   help="Ground-truth CSV → scored calibration (trigger floor "
                        "retains --target-eff of true pulses; time window from "
                        "true centers). Absent → data-driven quantiles.")
    g.add_argument("--target-eff", type=float, default=0.99,
                   help="Scored mode: retain at least this fraction of true "
                        "(non-saturated) pulses (default 0.99).")


def _iter_sample_records(settings, n_max: int):
    """Yield up to ``n_max`` raw EventRecords (polarity already applied)."""
    from pmtdeconv.pipeline.event_source import (
        iter_flat_run, iter_object_run_events,
    )
    runs = settings.runs or []
    polarity = int(settings.polarity)
    count = 0

    if settings.format == "object":
        from pmtdeconv.io.object_format import (
            detect_populated_channels, read_dt_ns,
        )
        data_dir = (Path(settings.data_root) if settings.data_root else
                    PROJECT_ROOT.parent.parent.parent / "data" / "835_real")
        for run in runs:
            cands = [p for p in sorted(data_dir.glob(settings.file_glob))
                     if str(run) in p.name]
            if not cands:
                continue
            rp = cands[0]
            dt = (float(settings.dt_ns) if settings.dt_ns != "auto"
                  else read_dt_ns(rp, tree_name=settings.tree or "Events"))
            branches = detect_populated_channels(rp, tree_name=settings.tree or "Events")
            for rec in iter_object_run_events(rp, run, dt_ns=dt,
                                              polarity=polarity, branches=branches,
                                              entry_stop=None):
                yield rec
                count += 1
                if count >= n_max:
                    return
    else:
        data_dir = (Path(settings.data_root) if settings.data_root else
                    PROJECT_ROOT.parent.parent.parent / "data" / "Na22_lab")
        dt = float(settings.dt_ns) if settings.dt_ns != "auto" else 0.5051
        for run in runs:
            rp = str(data_dir / f"flattened_Run{run}.root")
            for rec in iter_flat_run(rp, run, dt_ns=dt, polarity=polarity,
                                     entry_stop=None):
                yield rec
                count += 1
                if count >= n_max:
                    return


def _iter_sample_waves(settings, n_max: int):
    """Yield up to ``n_max`` raw (polarity-applied) waveforms as (wave, dt_ns)."""
    for rec in _iter_sample_records(settings, n_max):
        yield rec.wave, rec.dt_ns


def _collect_raw_stats(settings, n_max: int) -> pd.DataFrame:
    """Per-(event, branch) raw observables for ground-truth scoring:
    ``event, branch, raw_peak, t_raw_ns, max_val`` (pedestal = pre-trigger
    median). Joined against the truth CSV in :func:`_derive_scored`."""
    rows = []
    for rec in _iter_sample_records(settings, n_max):
        wave = rec.wave
        fb = wave[:min(100, len(wave))]
        med = float(np.median(fb))
        rows.append({
            "event": int(rec.entry), "branch": rec.branch,
            "raw_peak": float(np.max(wave) - med),
            "t_raw_ns": float(int(np.argmax(wave)) * rec.dt_ns),
            "max_val": float(np.max(wave)),
        })
    return pd.DataFrame(rows)


def _derive(waves, base: VetoProfile, k_mad: float):
    raw_peaks, t_peaks, noises = [], [], []
    rail = 0
    n = 0
    for wave, dt_ns in waves:
        n += 1
        N = len(wave)
        fb = wave[:min(100, N)]
        med = float(np.median(fb))
        mad = 1.4826 * float(np.median(np.abs(fb - med)))
        noise = mad if mad > 0 else (float(np.std(fb)) or 1.0)
        noises.append(noise)
        raw_peaks.append(float(np.max(wave) - med))
        t_peaks.append(int(np.argmax(wave)) * dt_ns)
        if detect_saturation_plateau(wave, noise)[0]:
            rail += 1

    if n == 0:
        raise SystemExit(
            "vetoes calibrate: no waveforms sampled — check "
            "--dataset / --runs / --data-root.")

    noise_floor = float(np.median(noises))
    min_raw_amp = round(k_mad * noise_floor, 3)
    sig_times = [t for t, p in zip(t_peaks, raw_peaks) if p > min_raw_amp]
    if sig_times:
        t_lo = round(float(np.percentile(sig_times, 1)), 3)
        t_hi = round(float(np.percentile(sig_times, 99)), 3)
    else:
        t_lo, t_hi = base.t_peak_min_ns, base.t_peak_max_ns

    derived = replace(base, name=f"{base.name}_calib",
                      no_trigger_amp=min_raw_amp,
                      t_peak_min_ns=t_lo, t_peak_max_ns=t_hi)
    stats = {
        "sample": n,
        "noise_floor_mad": round(noise_floor, 4),
        "rail_frac": round(rail / n, 4),
        "raw_peak_median": round(float(np.median(raw_peaks)), 3),
        "n_above_floor": len(sig_times),
    }
    return derived, stats


def _describe(base: VetoProfile, derived: VetoProfile, stats: dict) -> None:
    print("\n── derived veto profile (data-driven) ──")
    print(f"  sampled {stats['sample']} waveforms; "
          f"noise floor (MAD) = {stats['noise_floor_mad']}, "
          f"rail fraction = {stats['rail_frac'] * 100:.1f}%, "
          f"{stats['n_above_floor']} above the derived trigger floor")
    for f in ("no_trigger_amp", "t_peak_min_ns", "t_peak_max_ns"):
        print(f"    {f:18s}: {getattr(base, f):>10} → {getattr(derived, f):>10}")
    print("  (other thresholds inherited from the base profile; post-sat / "
          "charge-purity / afterpulse derivation is a tracked follow-up.)")
    if stats["rail_frac"] > 0.3:
        print(f"  NOTE: {stats['rail_frac'] * 100:.0f}% of waveforms hit the "
              f"rail — this run is saturation-dominated; charge is unreliable "
              f"but timing may survive (consider --no-vetoes for a timing pass).")


def _derive_scored(stats_df: pd.DataFrame, truth_df: pd.DataFrame,
                   base: VetoProfile,
                   target_eff: float = 0.99):
    """Ground-truth scored derivation. Returns ``(derived, stats)`` or
    ``(None, None)`` when too few truth-matched signal rows exist (caller
    falls back to data-driven).

    The trigger floor retains ``target_eff`` of the TRUE (non-saturated,
    not observed-clipped) pulses' raw peaks — the largest floor meeting that
    efficiency (ε-constraint, as in ``tune``). The time window is [q1, q99]
    of the TRUE pulse centers. ``purity`` = true-signal / all events kept in
    the window at that floor (the non-signal class = saturated / clipped rows).
    """
    cols = ["event", "branch", "true_amp_mv", "true_center_ns", "saturated"]
    m = stats_df.merge(truth_df[cols], on=["event", "branch"], how="inner")
    if m.empty:
        return None, None
    is_signal = ((~m["saturated"].astype(bool))
                 & (m["true_amp_mv"] > 0.0))
    signal = m[is_signal]
    if len(signal) < 5:
        return None, None

    t_lo = round(float(np.percentile(signal["true_center_ns"], 1)), 3)
    t_hi = round(float(np.percentile(signal["true_center_ns"], 99)), 3)
    # Largest floor retaining >= target_eff of true signal peaks == the
    # (1-target_eff) percentile of the signal raw-peak distribution.
    floor = max(round(float(np.percentile(
        signal["raw_peak"], (1.0 - target_eff) * 100.0)), 3), 0.0)
    achieved_eff = round(float((signal["raw_peak"] >= floor).mean()), 4)

    in_win = (m["t_raw_ns"] >= t_lo) & (m["t_raw_ns"] <= t_hi)
    kept = m[(m["raw_peak"] >= floor) & in_win]
    kept_sig = kept[(~kept["saturated"].astype(bool))
                    & (kept["true_amp_mv"] > 0.0)]
    purity = round(len(kept_sig) / len(kept), 4) if len(kept) else 1.0

    derived = replace(base, name=f"{base.name}_scored",
                      no_trigger_amp=floor,
                      t_peak_min_ns=t_lo, t_peak_max_ns=t_hi)
    stats = {
        "criterion": "ground-truth-scored",
        "matched": int(len(m)),
        "n_signal": int(len(signal)),
        "target_eff": target_eff,
        "achieved_eff": achieved_eff,
        "purity": purity,
        "score_eff_x_purity": round(achieved_eff * purity, 4),
    }
    return derived, stats


def _describe_scored(base: VetoProfile, derived: VetoProfile, stats: dict) -> None:
    print("\n── derived veto profile (ground-truth scored) ──")
    print(f"  matched {stats['matched']} (event,branch) rows; "
          f"{stats['n_signal']} true non-saturated signal pulses")
    print(f"  trigger floor retains {stats['achieved_eff'] * 100:.1f}% of true "
          f"pulses (target {stats['target_eff'] * 100:.0f}%); "
          f"in-window purity = {stats['purity'] * 100:.1f}%  "
          f"(eff×purity = {stats['score_eff_x_purity']})")
    for f in ("no_trigger_amp", "t_peak_min_ns", "t_peak_max_ns"):
        print(f"    {f:18s}: {getattr(base, f):>10} → {getattr(derived, f):>10}")
    print("  (post-sat / charge-purity / afterpulse inherited from base.)")


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    settings, provenance = resolve_settings(args)

    base = VETO_PROFILES.lookup(name=VETO_PROFILES.resolve_name(args.base_denoiser))
    out_dir = settings.out_dir or (
        PROJECT_ROOT / "outputs" / f"veto_calib_{settings.dataset or 'run'}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if maybe_dry_run_exit(subcommand="vetoes calibrate", settings=settings,
                          provenance=provenance, out_dir=out_dir,
                          duration_seconds=time.time() - t0):
        return 0

    criterion = "data-derived"
    truth_df = None
    if args.truth_csv:
        from pmtdeconv.io.truth import TruthSchemaError, load_truth
        try:
            truth_df = load_truth(args.truth_csv, strict=True)
        except (FileNotFoundError, TruthSchemaError) as e:
            print(f"[vetoes calibrate] could not load --truth-csv ({e}); "
                  f"using data-driven quantiles.")
            truth_df = None

    derived = stats = None
    if truth_df is not None:
        stats_df = _collect_raw_stats(settings, args.sample_events)
        derived, stats = _derive_scored(stats_df, truth_df, base,
                                        args.target_eff)
        if derived is None:
            print("[vetoes calibrate] truth matched too few signal rows; "
                  "falling back to data-driven quantiles.")
        else:
            criterion = "ground-truth-scored"
            _describe_scored(base, derived, stats)

    if derived is None:
        waves = _iter_sample_waves(settings, args.sample_events)
        derived, stats = _derive(waves, base, args.k_mad)
        _describe(base, derived, stats)

    written = None
    if args.write_profile:
        name = args.write_profile
        derived = replace(derived, name=name)
        VETO_PROFILES.save(name, derived)
        detail = (f"target_eff={args.target_eff}, "
                  f"achieved_eff={stats.get('achieved_eff')}, "
                  f"purity={stats.get('purity')}"
                  if criterion == "ground-truth-scored"
                  else f"k_mad={args.k_mad}, sample={stats.get('sample')}")
        prov = ProvenanceRecord(
            source=criterion, family="veto", command="vetoes calibrate",
            dataset=settings.dataset, runs=settings.runs, detail=detail)
        prof_path = out_dir / f"veto_profile_{name}.json"
        prof_path.write_text(json.dumps(
            {"profile": asdict(derived), "provenance": prov.to_dict(),
             "stats": stats}, indent=2))
        written = str(prof_path)
        print(f"\n  wrote profile '{name}' → {prof_path}")
        print(f"  (session catalog updated; persistent reload is a step-8 "
              f"follow-up)")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="vetoes calibrate", settings=settings, provenance=provenance,
        inputs=[], outputs=([{"role": "veto_profile", "path": written}]
                            if written else []),
        stats={"base_profile": base.name, "derived": asdict(derived),
               "calib_stats": stats},
        out_path=manifest_path, duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
