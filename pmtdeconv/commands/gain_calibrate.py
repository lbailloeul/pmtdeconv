"""``pmtdeconv gain calibrate`` — fit the 1-PE charge peak, propose ``spe_charge``.

The PE-scale anchor workflow (0–20 PE re-scope, 2026-07-28): per branch, fit
the single-PE charge-quantization peak of an existing hits/area charge CSV
(the proven in-regime observable — vault 17b) and PROPOSE the per-branch 1-PE
charge in pVs as a ready-to-paste card ``spe_charge:`` block. Nothing is
installed silently: the expert pastes the block into the card (or passes
``--write-calibration`` to persist it to the file-backed gain catalog under
``<data_root>/calibrations/gain/<pmt_type>.yaml``, still requiring the card /
CLI to opt in by ``pmt_type``).

An analytic cross-check is printed when the resolved template CSV carries
``median_peak_mV`` + biexp shape columns (first consumer of that column):
the peak-normalized biexp scaled to the median SPE peak height, integrated —
an independent estimate of the same 1-PE charge.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmtdeconv.io.manifest import maybe_dry_run_exit, write_manifest
from pmtdeconv.presets import resolve_settings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("gain calibrate")
    # Renamed from --pulse-csv on 2026-09-21: the file comes from `hits`.
    # (Internal attribute stays `pulse_csv`.)
    from pmtdeconv.cli import add_renamed_alias
    add_renamed_alias(g, "--pulse-csv", "--hits-csv", dest="pulse_csv")
    g.add_argument("--hits-csv", dest="pulse_csv", type=str, default=None,
                   help="Charge CSV to fit (hits.csv from `hits`, or an area CSV). "
                        "Default: auto-resolve the latest hits output for "
                        "--dataset/--runs.")
    g.add_argument("--charge-col", type=str, default=None,
                   help="Charge column to fit (default: 'charge' when present, "
                        "else 'area').")
    g.add_argument("--window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fit window in pVs (applied to every branch). Default: "
                        "auto-locate the 1-PE mode per branch.")
    g.add_argument("--pmt-type", type=str, default=None,
                   help="PMT-type key for --write-calibration (default: the "
                        "dataset's pmt_type).")
    g.add_argument("--write-calibration", action="store_true", default=False,
                   help="Persist the fitted values to the file-backed gain "
                        "catalog (<data_root>/calibrations/gain/<pmt_type>.yaml).")
    g.add_argument("--spectrum-png", action="store_true", default=False,
                   help="Write a per-branch charge-spectrum + fit overlay PNG "
                        "into the output dir (the minimal show-spectrum check).")


def _analytic_spe_charge_pvs(tau_rise: float, tau_fall: float, sigma: float,
                             peak_mv: float) -> float:
    """1-PE charge (pVs) of a biexp SPE scaled to ``peak_mv`` at the apex."""
    from pmtdeconv.core.kernels import gaussian_smeared_biexponential
    dt = 0.05
    t = np.arange(0.0, 600.0, dt)
    g = gaussian_smeared_biexponential(t, 1.0, tau_rise, tau_fall,
                                       t0=10.0, sigma=sigma)
    m = float(g.max())
    if not np.isfinite(m) or m <= 0:
        return float("nan")
    return float(g.sum() * dt * (peak_mv / m))


def _analytic_crosschecks(settings) -> dict:
    """{branch: analytic 1-PE charge (pVs)} from the resolved template CSV's
    ``median_peak_mV`` column, when present. Empty dict otherwise."""
    csv = settings.template_csv or settings.default_template_csv
    if not csv or not Path(csv).exists():
        return {}
    try:
        df = pd.read_csv(csv)
    except Exception:
        return {}
    need = {"median_peak_mV", "tau_rise_ns", "tau_fall_ns", "sigma_ns"}
    if not need.issubset(df.columns):
        return {}
    out = {}
    ch_col = "channel" if "channel" in df.columns else "chan"
    has_det = "det" in df.columns
    for _, row in df.iterrows():
        raw = str(row.get(ch_col, ""))
        if "ch" in raw:
            branch = raw
        elif has_det and pd.notna(row.get("det")):
            branch = f"{row['det']}_ch{int(float(raw))}"
        else:
            continue
        out[branch] = _analytic_spe_charge_pvs(
            float(row["tau_rise_ns"]), float(row["tau_fall_ns"]),
            float(row["sigma_ns"]), float(row["median_peak_mV"]))
    return out


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.fom import fit_1pe_peak
    from pmtdeconv.io.output_paths import resolve_pulse_csv, suffixed_output_dir

    t0 = time.time()
    settings, provenance = resolve_settings(args)

    out_dir = Path(settings.out_dir or suffixed_output_dir(
        "gain_calibrate", settings))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.pulse_csv) if args.pulse_csv else \
        resolve_pulse_csv(settings)
    if csv_path is None or not Path(csv_path).exists():
        raise SystemExit(
            "gain calibrate: no charge CSV found. Run `pmtdeconv hits` first "
            "(ideally on a 1-PE-dominated dataset like cd109), or pass "
            "--pulse-csv <path>.")

    if maybe_dry_run_exit(subcommand="gain calibrate", settings=settings,
                          provenance=provenance, out_dir=out_dir):
        return 0

    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"] == True]           # noqa: E712 (pandas mask)
    charge_col = args.charge_col or ("charge" if "charge" in df.columns
                                     else "area")
    if charge_col not in df.columns:
        raise SystemExit(f"gain calibrate: column {charge_col!r} not in "
                         f"{csv_path} (columns: {list(df.columns)[:12]}...)")
    if "branch" not in df.columns:
        df = df.assign(branch="all")

    window = tuple(args.window) if args.window else None
    fits: dict = {}
    analytic = _analytic_crosschecks(settings)

    print(f"── gain calibrate: {csv_path}  (column: {charge_col}) ──")
    for branch, grp in df.groupby("branch"):
        res = fit_1pe_peak(grp[charge_col].values, window=window)
        fits[str(branch)] = res
        if not res["ok"]:
            print(f"  {branch}: FIT FAILED (n={res['n']}) — no usable 1-PE "
                  f"peak located; supply --window or check the spectrum")
            continue
        q_note = "" if res["resolved"] else \
            "  [warn: pedestal valley NOT resolved — quantization uncertain]"
        xchk = ""
        if str(branch) in analytic and np.isfinite(analytic[str(branch)]):
            a = analytic[str(branch)]
            xchk = (f"  | analytic (median_peak_mV × biexp): {a:.1f} pVs "
                    f"({res['mu'] / a:.2f}× fit)")
        print(f"  {branch}: 1-PE charge = {res['mu']:.1f} pVs  "
              f"σ/μ = {res['sigma_over_mu']:.3f}  n = {res['n']}"
              f"{q_note}{xchk}")

    good = {b: r for b, r in fits.items() if r["ok"]}
    if good:
        print("\nProposed card block (paste into the dataset card):")
        print("spe_charge:")
        for b, r in sorted(good.items()):
            print(f"  {b}: {r['mu']:.1f}")

    if args.spectrum_png and good:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(fits)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.4), squeeze=False)
        for ax, (branch, grp) in zip(axes[0], df.groupby("branch")):
            q = grp[charge_col].values
            q = q[np.isfinite(q) & (q > 0)]
            r = fits[str(branch)]
            hi = float(np.percentile(q, 99.0)) if len(q) else 1.0
            ax.hist(q, bins=120, range=(0, hi), histtype="step", color="k")
            if r["ok"]:
                ax.axvline(r["mu"], color="crimson", lw=1.2,
                           label=f"1-PE μ={r['mu']:.0f} pVs σ/μ={r['sigma_over_mu']:.2f}")
                ax.axvspan(*r["window"], color="crimson", alpha=0.08)
                ax.legend(fontsize=8)
            ax.set_title(str(branch), fontsize=9)
            ax.set_xlabel("charge [pVs]")
            ax.set_yscale("log")
        fig.tight_layout()
        png = out_dir / "gain_spectrum.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"\nSpectrum figure: {png}")

    outputs_meta = []
    written_path = None
    if args.write_calibration:
        from pmtdeconv.pipeline.calibration import ProvenanceRecord
        from pmtdeconv.presets.gain_catalog import (GainCalibration,
                                                    save_gain_yaml)
        pmt_type = args.pmt_type or settings.pmt_type
        if not pmt_type:
            raise SystemExit(
                "gain calibrate --write-calibration needs a PMT-type key: the "
                "dataset has no pmt_type — pass --pmt-type.")
        if not good:
            raise SystemExit("gain calibrate: no successful fits to write.")
        if not settings.data_root:
            raise SystemExit("gain calibrate --write-calibration needs a "
                             "resolvable --data-root for the catalog dir.")
        rec = ProvenanceRecord(
            source="data-derived", family="gain", command="gain calibrate",
            dataset=settings.dataset, runs=settings.runs,
            detail=f"fit_1pe_peak on {csv_path.name} ({charge_col})")
        cal = GainCalibration(
            pmt_type=pmt_type,
            spe_charge_pvs={b: float(r["mu"]) for b, r in good.items()},
            note=f"gain calibrate {settings.dataset or ''} "
                 f"runs={settings.runs}  ({rec.date})",
            version=str(rec.date),
            fit={b: {k: float(v) for k, v in r.items()
                     if isinstance(v, (int, float))}
                 for b, r in good.items()},
        )
        written_path = save_gain_yaml(cal, settings.data_root)
        outputs_meta.append({"role": "gain_calibration",
                             "path": str(written_path)})
        print(f"\nWrote gain calibration: {written_path}")
        print(f"  (resolves for any dataset/card with pmt_type: {pmt_type})")

    manifest_path = (Path(settings.manifest_path)
                     if settings.manifest_path else (out_dir / "manifest.json"))
    write_manifest(
        subcommand="gain calibrate",
        settings=settings,
        provenance=provenance,
        inputs=[{"role": "charge_csv", "path": str(csv_path)}],
        outputs=outputs_meta,
        stats={"fits": {b: {k: (float(v) if isinstance(v, (int, float))
                               else v)
                            for k, v in r.items()}
                        for b, r in fits.items()},
               "charge_col": charge_col,
               "written": str(written_path) if written_path else None},
        out_path=manifest_path,
        duration_seconds=time.time() - t0,
        hash_inputs=settings.hash_inputs,
    )
    print(f"Manifest: {manifest_path}")
    return 0
