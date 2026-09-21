"""``pmtdeconv sim tang2024`` — generate the Tang et al. 2411.07620 MC.

Writes standard ``flattened_Run<N>.root`` files + truth CSVs (with
``pulse_id``/``n_pe``), the central-parameter SAMPLED template CSV, and a
``dataset.yaml`` card carrying the exact analytic ``spe_charge`` anchor — so
the ENTIRE existing pipeline consumes the simulation unchanged:

    pmtdeconv sim tang2024 --out-dir outputs/sim_tang2024
    pmtdeconv hits --dataset outputs/sim_tang2024 --runs 0

Run numbering: the charge mix = Run 0; a two-hit scan writes Run 1000+Δt per
separation (Δt in ns), so the recognition curve is a per-run scoring pass.

Defaults are sized for local work (--n-events 50000); the paper-parity run is
``--n-events 195000`` (+ its 5k 1-PE booster, ratio preserved automatically)
and ``--two-hit-scan --per-interval 10000``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("sim tang2024")
    g.add_argument("--n-events", type=int, default=50000,
                   help="Mix events with n_hits ~ U{0..20} (default 50000; "
                        "paper parity: 195000). A 1-PE booster sample is "
                        "added at the paper's 5k:195k ratio.")
    g.add_argument("--two-hit-scan", action="store_true", default=False,
                   help="Also write the fig-12 two-hit scan: Runs 1000+Δt "
                        "for Δt = 1..10 ns.")
    g.add_argument("--per-interval", type=int, default=10000,
                   help="Two-hit waveforms per Δt (default 10000, = paper).")
    g.add_argument("--intervals-ns", type=int, nargs="+",
                   default=list(range(1, 11)),
                   help="Two-hit separations in ns (default 1..10).")
    g.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default 42) — same seed, same bytes.")
    g.add_argument("--no-adc-quantize", dest="adc_quantize",
                   action="store_false", default=True,
                   help="Skip the 2.2 ADC/mV quantization step.")
    g.add_argument("--skip-mix", action="store_true", default=False,
                   help="Only write the template/card (+ scan if requested).")
    g.add_argument("--no-undershoot", dest="undershoot",
                   action="store_false", default=True,
                   help="Generate WITHOUT the AC-coupling undershoot (an "
                        "R5912/FADC-circuit feature) — closer to our lab "
                        "PMT+DRS4 response; template/card/anchor follow.")
    g.add_argument("--kernel",
                   choices=("r5912", "milliqan_biexp", "milliqan_gs"),
                   default="r5912",
                   help="SPE kernel family: the paper's R5912 model, the "
                        "MilliQan lab biexp, or the MilliQan detector "
                        "template (timewalksim POPT_GS pulse-injection fit); "
                        "the biexp families have no undershoot/reflections "
                        "and 4%% per-hit shape fluctuation.")


def run(args: argparse.Namespace) -> int:
    from pmtdeconv.io.manifest import write_manifest
    from pmtdeconv.presets import resolve_settings
    from pmtdeconv.sim import tang2024 as sim

    t0 = time.time()
    settings, provenance = resolve_settings(args, require_data_root=False)
    out_dir = Path(settings.out_dir or "outputs/sim_tang2024")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    und = bool(args.undershoot)
    kernel = str(args.kernel)
    if kernel != "r5912":
        und = False          # the biexp families have no undershoot
    template_csv = sim.write_template_csv(out_dir, undershoot=und,
                                          kernel=kernel)
    card = sim.write_dataset_card(out_dir, template_csv, undershoot=und,
                                  kernel=kernel)
    print(f"template: {template_csv}  (undershoot={und})")
    print(f"card:     {card}  (spe_charge = "
          f"{sim.central_spe_charge_pvs(undershoot=und, kernel=kernel):.3f} pVs — analytic anchor)")

    outputs_meta = [{"role": "template_csv", "path": str(template_csv)},
                    {"role": "dataset_card", "path": str(card)}]
    stats = {"seed": int(args.seed), "n_events_mix": 0,
             "two_hit_scan": bool(args.two_hit_scan),
             "spe_charge_pvs": sim.central_spe_charge_pvs(undershoot=und,
                                                          kernel=kernel),
             "undershoot": und, "kernel": kernel}

    if not args.skip_mix:
        n = int(args.n_events)
        print(f"generating mix: {n} events (n_hits ~ U{{0..20}}) "
              f"+ {int(round(n * 5000 / 195000))} 1-PE booster …")
        waves, truth = sim.make_mix(rng, n, adc_quantize=args.adc_quantize,
                                    undershoot=und, kernel=kernel)
        sim.write_run(out_dir, 0, waves, truth)
        stats["n_events_mix"] = int(len(waves))
        outputs_meta.append({"role": "root_data",
                             "path": str(out_dir / "flattened_Run0.root")})
        print(f"  wrote Run 0: {len(waves)} waveforms, "
              f"{len(truth)} truth pulses")

    if args.two_hit_scan:
        scans = sim.make_two_hit_scan(
            rng, intervals_ns=tuple(args.intervals_ns),
            per_interval=int(args.per_interval),
            adc_quantize=args.adc_quantize, undershoot=und, kernel=kernel)
        for dt_sep, (waves, truth) in scans.items():
            run_no = 1000 + dt_sep
            sim.write_run(out_dir, run_no, waves, truth)
            print(f"  wrote Run {run_no}: Δt = {dt_sep} ns, "
                  f"{len(waves)} waveforms")
        stats["scan_runs"] = [1000 + d for d in scans]

    write_manifest(
        subcommand="sim tang2024",
        settings=settings, provenance=provenance,
        inputs=[], outputs=outputs_meta, stats=stats,
        out_path=out_dir / "manifest.json",
        duration_seconds=time.time() - t0,
        hash_inputs=False,
    )
    print(f"\nAll outputs in: {out_dir}")
    print(f"Process with:  pmtdeconv hits --dataset {out_dir} --runs 0")
    return 0
