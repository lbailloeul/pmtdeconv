"""``pmtdeconv benchmark denoisers`` — the STFT-vs-wavelet verdict harness.

Every arm is a FULL pipeline run under IDENTICAL conditions — the
``benchmark_hits`` neutral veto profile (or the dataset card's explicit
profile), ONE shared cosine cutoff, the same linearized deconvolution — the
only degree of freedom is the denoiser:

    control              — denoiser none (the un-denoised deconvolution)
    wavelet              — production SWT denoiser (shipped defaults)
    stft                 — our element-wise STFT mask at global-median ×
                           amp-threshold (shipped defaults; adapts per waveform)
    stft-fixed-tang2024  — Tang 2411.07620 time-COLUMN selection with a FIXED
                           threshold in the <50 MHz detection band
                           (auto-calibrated; the same cut for every waveform)

The four above are OUR pipeline; they differ only in the denoiser.

    tang2024             — the PAPER'S OWN method end to end (their eq-3.1
                           deconvolution, filters, STFT selection and §4.3
                           recognition; the enhanced FFT+STFT chain), from
                           sim/tang2024_paper.py.
    tang2024-fft         — the paper's plain-FFT chain ("previous" method).

The two tang2024* arms are NOT our pipeline, so they run on the tang2024 suite
only. They are scored by the same code as every other arm — which is what makes
this a head-to-head rather than a replot — and the verdict also carries the
paper's PUBLISHED fig-12/14 values beside them. This replaces the standalone
`sim tang2024-paper` command (retired 2026-09-21). (Arms renamed
2026-09-18: stft-ours -> stft, stft-paper -> stft-fixed-tang2024 — the old
name read as "the paper's result" when it borrows only Tang's threshold rule.)

Suites:
    regimes  — in-regime synthetic truth fixtures (level1-3, noisy,
               pileup_lowpe, pileup_dense) scored per-pulse vs truth
               (`scoring.score_run`, hits/hits-multi auto)
    tang2024 — the Tang MC: charge mix (per-waveform ratio + nonlinearity,
               fig 14) + the two-hit Δt scan (recognition efficiency vs
               separation, fig 12)
    all      — both (default)

Real-data mode (``--real --dataset <ds> --runs <N>``): truth-free proxies —
per-branch 1-PE charge-quantization σ/μ (the PROVEN in-regime observable,
vault 17b) and the paired-channel charge-asymmetry width.

Output: ``verdict.json`` (schema-versioned; per-dataset per-arm metric table
+ head-to-head verdict + full condition provenance) and overlay figures.
This replaces the old ``tune denoiser`` private loop (FWHM-only, own
pedestal, no truth) — the anti-regression test asserts the control arm's
metrics equal a direct pipeline+score_run invocation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 2.0 (2026-09-18): arm keys renamed (stft-ours/stft-paper -> stft/
# stft-fixed-tang2024) — a rename is a major bump under the schema policy.
VERDICT_SCHEMA = "benchmark-denoisers-2.0"

DEFAULT_REGIMES = ["level1", "level2", "level3", "noisy",
                   "pileup_lowpe", "pileup_dense"]
ALL_ARMS = ("control", "wavelet", "stft", "stft-fixed-tang2024",
            "tang2024", "tang2024-fft")
# Arms that are NOT our pipeline: the paper's own reconstruction, one per chain
# it publishes. They carry the Tang kernel, so they only make sense on the Tang
# MC — on regime fixtures or real data they would deconvolve with the wrong
# response, and are skipped there.
TANG_ARM_METHOD = {"tang2024": "enhanced",       # FFT + STFT (their headline)
                   "tang2024-fft": "previous"}   # plain FFT
TANG_ONLY_ARMS = frozenset(TANG_ARM_METHOD)

# Metrics used for the head-to-head verdict: (key, direction) where
# direction "min" = lower is better, "max" = higher, "unity" = closest to 1.
VERDICT_METRICS = [
    ("efficiency", "max"),
    ("timing_sigma_ns", "min"),
    ("charge_scatter", "min"),
    ("charge_ratio_mean", "unity"),
    ("merge_rate", "min"),
    ("split_rate", "min"),
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("benchmark denoisers")
    g.add_argument("--suite", choices=("regimes", "tang2024", "all"),
                   default="all")
    g.add_argument("--arms", nargs="+", choices=ALL_ARMS,
                   default=list(ALL_ARMS),
                   help="Subset of arms to run (default: all four).")
    g.add_argument("--regimes", nargs="+", default=None,
                   help=f"Regime fixtures for the synthetic suite "
                        f"(default: {' '.join(DEFAULT_REGIMES)}).")
    g.add_argument("--n-events", type=int, default=120,
                   help="Events per regime fixture (default 120).")
    g.add_argument("--shared-cutoff-mhz", type=float, default=450.0,
                   help="The ONE cosine cutoff every arm shares on the "
                        "regime suite (default 450 — the stft-fixed-tang2024 arm's "
                        "cutoff, so arms differ ONLY in the denoiser).")
    g.add_argument("--tang-dir", type=str, default=None,
                   help="Existing `sim tang2024` output dir (default: "
                        "generate a small one under --out-dir).")
    g.add_argument("--tang-events", type=int, default=300,
                   help="Mix events when generating the Tang sample.")
    g.add_argument("--tang-per-interval", type=int, default=120,
                   help="Two-hit waveforms per Δt when generating.")
    g.add_argument("--tang-intervals", type=int, nargs="+",
                   default=[2, 3, 4, 5, 6, 8, 10],
                   help="Two-hit separations (ns) for the recognition curve.")
    g.add_argument("--real", action="store_true", default=False,
                   help="Real-data proxy mode: run the arms on "
                        "--dataset/--runs and report 1-PE quantization σ/μ "
                        "+ paired-channel width (no truth needed).")
    g.add_argument("--entry-list", dest="entry_list", type=str, default=None,
                   help="Real mode: forward to every arm (process only the "
                        "selected bunches, e.g. `select multihit` output).")
    g.add_argument("--spe-charge", dest="spe_charge", type=str, nargs="+",
                   default=None, metavar="BR=PVS",
                   help="Real mode: forward the PE anchor to every arm.")
    g.add_argument("--spe-window", dest="spe_window", type=float, nargs=2,
                   default=None, metavar=("LO", "HI"),
                   help="Real mode: expert fit window (pVs) for the per-arm "
                        "1-PE quantization fit (the auto-locator fails on "
                        "long multi-PE tails — same philosophy as "
                        "`gain calibrate --window`).")
    g.add_argument("--seed", type=int, default=42)


# ─── one pipeline run = one arm on one dataset ──────────────────────────

def _arm_cli(arm: str, *, shared_cutoff: Optional[float],
             fixed_thr: Optional[float]) -> List[str]:
    """The CLI fragment that configures one arm. Identical veto profile /
    padding / cutoff across arms — only the denoiser differs."""
    if arm == "control":
        cli = ["--denoiser", "none"]
    elif arm == "wavelet":
        cli = ["--denoiser", "wavelet"]
    elif arm == "stft":
        cli = ["--denoiser", "stft"]
    elif arm == "stft-fixed-tang2024":
        cli = ["--denoiser", "stft", "--stft-threshold-mode", "fixed",
               "--stft-freq-cutoff-mhz", "50",
               "--stft-fixed-threshold", f"{fixed_thr:.8g}"]
    else:                                  # pragma: no cover
        raise ValueError(arm)
    if shared_cutoff is not None:
        cli += ["--cosine-cutoff-mhz", f"{shared_cutoff:g}"]
    return cli


def _run_tang_paper_arm(tang_dir: Path, run: int, out_dir: Path,
                        method: str = "enhanced") -> Path:
    """Run the paper's own method on one Tang MC run; write all_results.csv.

    Every other arm is `hits` with a different denoiser. This one is the
    PAPER'S reconstruction (sim/tang2024_paper.PaperReconstructor; ``method``
    "enhanced" = FFT+STFT, "previous" = plain FFT), which never touches our
    pipeline. It writes the four
    columns the tang scorers read — entry, success, charge, t_peak — so it is
    scored by exactly the same code as our arms, on exactly the same truth.

    Charge scale: the paper chain works in PE (their NormC). The generator
    DEFINES truth as ``true_charge_pVs = n_pe × a_central``, so multiplying the
    paper's PE charge by that same ``a_central`` puts it on the truth's scale
    exactly: the scorer's ratio equals the paper's native ratio, no fudge.

    Per-hit split: each event's total is the paper's own ``event_charge``
    (which unions overlapping integration pads rather than double-counting),
    distributed across hits in proportion to ``hit_charge``. The charge
    scorers only use the per-event sum, so this reproduces the paper's fig-14
    number; ``t_peak`` per hit drives recognition efficiency.

    Assumes the default Tang MC (r5912 kernel, with undershoot) — the same
    calibration convention as the stft-fixed-tang2024 arm.
    """
    import pandas as pd
    from pmtdeconv.sim import tang2024 as sim
    from pmtdeconv.sim.tang2024_paper import (PaperReconstructor, combine,
                                              event_charge, hit_charge,
                                              load_run)
    recon = PaperReconstructor()
    a_central = sim.central_spe_charge_pvs(recon.dt_ns, recon.undershoot,
                                           recon.kernel)
    pad = recon.pad_ns[method]
    waves, _truth = load_run(Path(tang_dir), run)

    rows = []
    for ev, wave in enumerate(waves):
        u = recon.reconstruct(np.asarray(wave, dtype=float), method)
        hits = recon.recognize(u, method)
        if not hits:
            continue
        q_event = event_charge(u, recon.dt_ns, method=method, hits=hits,
                               combine_ns=recon.combine_ns, pad_ns=pad)
        uc = combine(u, recon.dt_ns,
                     width_ns=recon.combine_ns.get(method, 0.0))
        per_hit = np.array([hit_charge(uc, h, recon.dt_ns, pad_ns=pad)
                            for h in hits])
        total = per_hit.sum()
        share = per_hit / total if total > 0 else np.full(len(hits),
                                                          1.0 / len(hits))
        for i, (h, w) in enumerate(zip(hits, share)):
            rows.append({"entry": ev, "branch": sim.BRANCH, "peak_id": i + 1,
                         "success": True, "reason": "",
                         "t_peak": float(h["t_ns"]),
                         "charge": float(q_event * w * a_central)})

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "all_results.csv"
    pd.DataFrame(rows, columns=["entry", "branch", "peak_id", "success",
                                "reason", "t_peak", "charge"]).to_csv(
        csv, index=False)
    return csv


def _run_arm(dataset: str, run: int, arm: str, out_dir: Path, *,
             shared_cutoff: Optional[float], fixed_thr: Optional[float],
             veto_profile: Optional[str] = "benchmark_hits",
             extra: Optional[List[str]] = None) -> Path:
    """In-process `hits` invocation for one (dataset, run, arm). Returns the
    all_results.csv path. Each run writes its own manifest (full provenance).

    The paper arm is dispatched to its own runner: it is not a `hits` call."""
    if arm in TANG_ONLY_ARMS:
        return _run_tang_paper_arm(Path(dataset), run, out_dir,
                                   method=TANG_ARM_METHOD[arm])
    from pmtdeconv.cli import main as cli_main
    argv = ["hits", "--dataset", dataset, "--runs", str(run),
            "--out-dir", str(out_dir), "--log-level", "quiet",
            *_arm_cli(arm, shared_cutoff=shared_cutoff, fixed_thr=fixed_thr)]
    if veto_profile:
        argv += ["--veto-profile", veto_profile]
    argv += list(extra or [])
    rc = cli_main(argv)
    if rc not in (0, None):
        raise RuntimeError(f"arm {arm} on {dataset} run {run} exited {rc}")
    csv = out_dir / "all_results.csv"
    if not csv.exists():
        raise RuntimeError(f"arm {arm} produced no all_results.csv in {out_dir}")
    return csv


# ─── suites ─────────────────────────────────────────────────────────────

def _suite_regimes(args, out_base: Path, arms, fixed_thr) -> Dict:
    from pmtdeconv.commands._tune_shared import ensure_regime_fixture
    from pmtdeconv.io.truth import find_truth_for
    from pmtdeconv.pipeline.scoring import score_run

    regimes = args.regimes or DEFAULT_REGIMES
    results: Dict[str, Dict] = {}
    for regime in regimes:
        fixture = ensure_regime_fixture(regime, 0.5051, args.n_events)
        truth = find_truth_for(fixture)
        results[regime] = {}
        for arm in arms:
            out = out_base / "regimes" / regime / arm
            out.mkdir(parents=True, exist_ok=True)
            # --peak-order tallest: single-pulse truth ⇒ peak_id=1 = the
            # tallest peak (the physics answer; the historical tune-wavelet
            # convention). `first`'s ≥15%-of-tallest gate was tuned for the
            # 100 MHz side-lobe and mis-fires at the shared 450 MHz cutoff.
            # Identical across arms (honored on every denoiser under the
            # explicit profile); multi-pulse regimes score in hits-multi
            # mode, which uses ALL surviving peaks and ignores ordering.
            csv = _run_arm("na22", 0, arm, out,
                           shared_cutoff=args.shared_cutoff_mhz,
                           fixed_thr=fixed_thr,
                           extra=["--data-root", str(fixture.parent),
                                  "--peak-order", "tallest"])
            s = score_run(csv, truth, mode="hits")
            s["csv"] = str(csv)
            results[regime][arm] = s
            print(f"  [{regime:13s}] {arm:10s} eff={s['efficiency']:.2f} "
                  f"tσ={s['timing_sigma_ns']:.2f} ns "
                  f"scatter={s['charge_scatter']*100:.1f}%")
    return results


def _resolving_curve(args, out_base: Path, arms, fixed_thr,
                     tang_dir: Path) -> Dict:
    """Recognition efficiency vs Δt on the Tang two-hit scan (fig 12)."""
    from pmtdeconv.sim.tang2024_validate import recognition_efficiency
    curve: Dict[str, Dict[str, float]] = {arm: {} for arm in arms}
    for d in args.tang_intervals:
        run = 1000 + int(d)
        truth = tang_dir / f"flattened_Run{run}.truth.csv"
        if not truth.exists():
            continue
        for arm in arms:
            out = out_base / "tang2024" / f"scan{d}" / arm
            out.mkdir(parents=True, exist_ok=True)
            csv = _run_arm(str(tang_dir), run, arm, out,
                           shared_cutoff=None,      # card's 450 for every arm
                           fixed_thr=fixed_thr, veto_profile=None)
            curve[arm][str(d)] = recognition_efficiency(csv, truth)
    return curve


def _suite_tang(args, out_base: Path, arms, fixed_thr, rng) -> Dict:
    from pmtdeconv.sim import tang2024 as sim
    from pmtdeconv.sim.tang2024_validate import (nonlinearity_frac,
                                                 nonlinearity_profile,
                                                 per_event_charge_ratio)
    if args.tang_dir:
        tang_dir = Path(args.tang_dir)
    else:
        tang_dir = out_base / "tang_sim"
        if not (tang_dir / "flattened_Run0.root").exists():
            print(f"  generating Tang sample ({args.tang_events} mix events "
                  f"+ scan) …")
            tmpl = sim.write_template_csv(tang_dir)
            sim.write_dataset_card(tang_dir, tmpl)
            waves, truth = sim.make_mix(rng, args.tang_events)
            sim.write_run(tang_dir, 0, waves, truth)
            scans = sim.make_two_hit_scan(
                rng, intervals_ns=tuple(args.tang_intervals),
                per_interval=args.tang_per_interval)
            for d, (w, t) in scans.items():
                sim.write_run(tang_dir, 1000 + d, w, t)

    mix: Dict[str, Dict] = {}
    truth_csv = tang_dir / "flattened_Run0.truth.csv"
    for arm in arms:
        out = out_base / "tang2024" / "mix" / arm
        out.mkdir(parents=True, exist_ok=True)
        csv = _run_arm(str(tang_dir), 0, arm, out, shared_cutoff=None,
                       fixed_thr=fixed_thr, veto_profile=None)
        r = per_event_charge_ratio(csv, truth_csv)
        nl = nonlinearity_frac(nonlinearity_profile(csv, truth_csv),
                               min_bin_n=8)
        mix[arm] = {"charge_ratio_mean": r["mean"],
                    "charge_ratio_std": r["std"], "n_events": r["n"],
                    "nonlinearity_frac": nl, "csv": str(csv)}
        print(f"  [tang mix     ] {arm:10s} ratio={r['mean']:.3f} "
              f"σ={r['std']:.3f} nonlin={nl:.2f}")

    curve = _resolving_curve(args, out_base, arms, fixed_thr, tang_dir)

    # The paper's PUBLISHED numbers beside the measured ones (what the retired
    # `sim tang2024-paper` reported). Caveat carried in the verdict itself: the
    # recognition constants were fitted to this fig-12 curve.
    published = {
        "charge": dict(sim.TANG_PUBLISHED),
        "recognition_vs_dt": {a: {str(k): v for k, v in c.items()}
                              for a, c in sim.TANG_PUBLISHED_RECOGNITION.items()
                              if a in arms},
        "caveat": ("fig-12 values read off the published figure; the paper-"
                   "method recognition constants were calibrated against it, "
                   "so agreement is a consistency check, not validation"),
    }
    for arm in (a for a in arms if a in sim.TANG_PUBLISHED_RECOGNITION):
        pub = sim.TANG_PUBLISHED_RECOGNITION[arm]
        pairs = [f"{d}ns {curve.get(arm, {}).get(str(d), float('nan')):.2f}"
                 f"/{pub[d]:.2f}" for d in sorted(pub)
                 if str(d) in curve.get(arm, {})]
        if pairs:
            print(f"  [vs published ] {arm:12s} measured/published: "
                  + "  ".join(pairs))
    return {"mix": mix, "recognition_vs_dt": curve, "published": published,
            "tang_dir": str(tang_dir)}


def _suite_real(args, settings, out_base: Path, arms, fixed_thr) -> Dict:
    """Truth-free proxies on real data (per arm): per-branch 1-PE
    quantization σ/μ + paired-channel charge-asymmetry MAD-σ."""
    import pandas as pd
    from pmtdeconv.fom import fit_1pe_peak
    results: Dict[str, Dict] = {}
    runs = settings.runs or []
    if not runs:
        raise SystemExit("--real needs --runs")
    forward: list = []
    if settings.data_root:
        forward += ["--data-root", str(settings.data_root)]
    if getattr(args, "entry_list", None):
        forward += ["--entry-list", str(args.entry_list)]
    if getattr(args, "spe_charge", None):
        forward += ["--spe-charge", *args.spe_charge]
    for arm in arms:
        out = out_base / "real" / arm
        out.mkdir(parents=True, exist_ok=True)
        csv = _run_arm(settings.dataset or "na22", runs[0], arm, out,
                       shared_cutoff=None, fixed_thr=fixed_thr, extra=forward)
        df = pd.read_csv(csv)
        ok = df[df["success"] == True]                    # noqa: E712
        per_branch = {}
        spe_window = (tuple(args.spe_window)
                      if getattr(args, "spe_window", None) else None)
        for br, grp in ok.groupby("branch"):
            fit = fit_1pe_peak(grp["charge"].values, window=spe_window)
            per_branch[str(br)] = {
                "spe_sigma_over_mu": fit["sigma_over_mu"],
                "spe_mu_pvs": fit["mu"], "resolved": fit["resolved"],
                "n": fit["n"]}
        # paired-channel asymmetry: events with exactly one success pulse
        # per channel on the first two branches
        asym_sigma = float("nan")
        brs = sorted(ok["branch"].unique())[:2]
        if len(brs) == 2:
            one = ok.groupby(["entry", "branch"]).filter(
                lambda g: len(g) == 1)
            piv = one.pivot_table(index="entry", columns="branch",
                                  values="charge", aggfunc="first")
            piv = piv.dropna(subset=brs)
            if len(piv) > 20:
                a = ((piv[brs[0]] - piv[brs[1]])
                     / (piv[brs[0]] + piv[brs[1]])).to_numpy()
                a = a[np.isfinite(a)]
                if a.size:
                    asym_sigma = float(1.4826 * np.median(
                        np.abs(a - np.median(a))))
        results[arm] = {"per_branch": per_branch,
                        "pair_asym_mad_sigma": asym_sigma,
                        "csv": str(csv)}
        print(f"  [real         ] {arm:10s} " + " ".join(
            f"{b}:σ/μ={v['spe_sigma_over_mu']:.2f}"
            for b, v in per_branch.items()))
    return results


# ─── verdict + figures ──────────────────────────────────────────────────

def _head_to_head(regime_results: Dict) -> Dict:
    """Count per-metric dataset wins for each non-control arm."""
    wins: Dict[str, int] = {}
    detail: Dict[str, Dict[str, str]] = {}
    for regime, per_arm in regime_results.items():
        contenders = {a: m for a, m in per_arm.items() if a != "control"}
        if len(contenders) < 2:
            continue
        detail[regime] = {}
        for key, direction in VERDICT_METRICS:
            vals = {a: m.get(key) for a, m in contenders.items()
                    if m.get(key) is not None
                    and np.isfinite(m.get(key, float("nan")))}
            if len(vals) < 2:
                continue
            if direction == "min":
                best = min(vals, key=vals.get)
            elif direction == "max":
                best = max(vals, key=vals.get)
            else:                                    # unity
                best = min(vals, key=lambda a: abs(vals[a] - 1.0))
            wins[best] = wins.get(best, 0) + 1
            detail[regime][key] = best
    headline = max(wins, key=wins.get) if wins else None
    return {"wins": wins, "per_metric": detail, "headline": headline}


def _figures(out_base: Path, tang: Optional[Dict]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs: List[str] = []
    if tang and tang.get("recognition_vs_dt"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, curve in tang["recognition_vs_dt"].items():
            if not curve:
                continue
            ds = sorted(curve, key=float)
            ax.plot([float(d) for d in ds], [curve[d] for d in ds],
                    marker="o", label=arm)
        ax.set_xlabel("true hit separation Δt [ns]")
        ax.set_ylabel("recognition efficiency (exactly-2, ±2 ns)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Two-hit recognition vs Δt (Tang 2411.07620 fig-12 "
                     "criterion)")
        p = out_base / "efficiency_vs_dt.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        figs.append(str(p))
    return figs


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.io.manifest import write_manifest
    from pmtdeconv.presets import resolve_settings

    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)
    out_base = Path(settings.out_dir or "outputs/benchmark_denoisers")
    out_base.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(args.arms))
    # The paper's method carries the Tang kernel, so it is meaningful only on
    # the Tang MC. Everything else (regimes, real data) runs our arms only.
    pipeline_arms = [a for a in arms if a not in TANG_ONLY_ARMS]
    skipped = [a for a in arms if a in TANG_ONLY_ARMS]
    if skipped and (getattr(args, "real", False)
                    or args.suite in ("regimes", "all")):
        print(f"  note: {', '.join(skipped)} runs on the tang2024 suite only "
              f"(it is the paper's own reconstructor, built for the Tang "
              f"kernel); skipped on regimes/real data.")
    rng = np.random.default_rng(int(args.seed))

    from pmtdeconv.sim.tang2024_validate import paper_stft_fixed_threshold
    fixed_thr = (paper_stft_fixed_threshold()
                 if "stft-fixed-tang2024" in arms else None)

    conditions = {
        "arms": arms,
        "veto_profile": "benchmark_hits (regimes) / card [tang2024]",
        "shared_cutoff_mhz_regimes": args.shared_cutoff_mhz,
        "peak_order_regimes": "tallest",
        "stft_fixed_tang2024_threshold": fixed_thr,
        # The fixed threshold is calibrated for the TANG kernel and dt; on
        # other datasets it does not transfer (typically → no-op ≈ control).
        # Interpret the stft-fixed-tang2024 arm on the tang2024 suite only.
        "stft_fixed_tang2024_note": "fixed threshold calibrated for tang2024 only",
        "tang2024_arms": ("paper's own method: tang2024 = enhanced FFT+STFT, "
                          "tang2024-fft = plain FFT; calibrated recognition "
                          "constants; tang2024 suite only; assumes the default "
                          "Tang MC kernel"),
        "fft_pad": True,
        "seed": int(args.seed),
    }
    verdict: Dict = {"schema": VERDICT_SCHEMA, "conditions": conditions}

    if args.real:
        verdict["real"] = _suite_real(args, settings, out_base,
                                      pipeline_arms, fixed_thr)
    else:
        if args.suite in ("regimes", "all"):
            print("── regime suite ──")
            verdict["regimes"] = _suite_regimes(args, out_base, pipeline_arms,
                                                fixed_thr)
            verdict["head_to_head"] = _head_to_head(verdict["regimes"])
        if args.suite in ("tang2024", "all"):
            print("── tang2024 suite ──")
            verdict["tang2024"] = _suite_tang(args, out_base, arms, fixed_thr,
                                              rng)
        verdict["figures"] = _figures(out_base, verdict.get("tang2024"))

    vpath = out_base / "verdict.json"
    vpath.write_text(json.dumps(verdict, indent=2, default=float))
    print(f"\nverdict: {vpath}")
    if verdict.get("head_to_head", {}).get("headline"):
        h = verdict["head_to_head"]
        print(f"head-to-head wins: {h['wins']}  → headline: {h['headline']}")

    write_manifest(
        subcommand="benchmark denoisers",
        settings=settings, provenance=provenance,
        inputs=[], outputs=[{"role": "verdict_json", "path": str(vpath)}]
        + [{"role": "figure", "path": p} for p in verdict.get("figures", [])],
        stats={"arms": arms, "suite": args.suite,
               "head_to_head": verdict.get("head_to_head")},
        out_path=out_base / "manifest.json",
        duration_seconds=time.time() - t0,
        hash_inputs=False,
    )
    return 0
