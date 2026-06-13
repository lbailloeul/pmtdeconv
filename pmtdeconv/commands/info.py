"""``pmtdeconv info ...`` — read-only introspection of registries.

Leaves:
- ``info datasets`` — lists every DatasetPreset with key fields.
- ``info presets`` — lists every named ConfigPreset with overrides.
- ``info template <path>`` — describes a CSV template's schema +
  channels.
- ``info artifacts`` — indexes existing output manifests.

No ROOT I/O. No side effects. Pure listing for discoverability so
``pmtdeconv info presets`` works as user-facing documentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pmtdeconv import PROJECT_ROOT
from pmtdeconv.io.templates import load_template
from pmtdeconv.presets.configs import CONFIGS
from pmtdeconv.presets.datasets import DATASETS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="info_topic", required=True)

    # Bug-fix 13 (2026-05-30 final scan): each `info` sub-parser also
    # accepts the same `--log-level` (and other common flags) that the
    # outer `info` parser does. Without this, `pmtdeconv info datasets
    # --log-level quiet` failed with "unrecognized arguments" even
    # though `pmtdeconv info --log-level quiet datasets` worked.
    # We re-import lazily to avoid a circular import on cli.
    from pmtdeconv.cli import add_common_args

    p_datasets = sub.add_parser("datasets",
                                help="List every registered dataset preset")
    add_common_args(p_datasets)
    p_datasets.set_defaults(_info_fn=_info_datasets)

    p_presets = sub.add_parser("presets",
                               help="List every named configuration preset")
    add_common_args(p_presets)
    p_presets.set_defaults(_info_fn=_info_presets)

    p_tmpl = sub.add_parser("template",
                            help="Describe a SPE template CSV")
    p_tmpl.add_argument("path", type=str, help="Path to a template CSV.")
    add_common_args(p_tmpl)
    p_tmpl.set_defaults(_info_fn=_info_template)

    p_off = sub.add_parser("offline-file",
                           help="Inspect a LANL *_offline.root pulse-list file")
    p_off.add_argument("path", type=str, help="Path to a *_offline.root file.")
    p_off.add_argument("--n-sample", type=int, default=50,
                       help="Events sampled for channel detection (default 50).")
    add_common_args(p_off)
    p_off.set_defaults(_info_fn=_info_offline_file)

    p_art = sub.add_parser("artifacts",
                           help="Index existing output manifests")
    p_art.add_argument("--outputs-dir", type=str,
                       default=str(PROJECT_ROOT / "outputs"),
                       help="Outputs directory to scan.")
    p_art.add_argument("--write", nargs="?", const="",
                       default=None,
                       help=("Write JSON index. With no value, writes "
                             "<outputs-dir>/artifact_index.json."))
    p_art.add_argument("--no-orphans", action="store_true",
                       help="Only include directories that contain manifests.")
    p_art.add_argument("--limit", type=int, default=15,
                       help="Rows to print (default 15).")
    add_common_args(p_art)
    p_art.set_defaults(_info_fn=_info_artifacts)


def run(args: argparse.Namespace) -> int:
    return int(args._info_fn(args) or 0)


# ─── leaves ────────────────────────────────────────────────────────

def _info_datasets(args) -> int:
    print(f"{'name':<14}  {'fmt':>6}  {'dt_ns':>6}  {'sat':>6}  {'pol':>3}  "
          f"{'cos_MHz':>8}  paths / templates")
    print("-" * 110)
    for name, ds in DATASETS.items():
        tmpl = (str(ds.default_template_csv.relative_to(
            ds.default_template_csv.parents[2]))
                if ds.default_template_csv else "(none)")
        dt = (f"{ds.dt_ns:.4f}" if isinstance(ds.dt_ns, float)
              else str(ds.dt_ns))
        print(f"{ds.name:<14}  {ds.format:>6}  {dt:>6}  "
              f"{ds.sat_thresh_mv:>6.0f}  {ds.polarity:>+3d}  "
              f"{ds.cosine_cutoff_mhz:>8.1f}")
        print(f"{'':<14}     data_dir: {ds.data_dir}")
        if ds.offline_data_dir is not None:
            print(f"{'':<14}  offline_data_dir: {ds.offline_data_dir}")
            print(f"{'':<14}  offline_glob:     {ds.offline_file_glob}")
        print(f"{'':<14}     template: {tmpl}")
        if ds.notes:
            print(f"{'':<14}     notes:    {ds.notes}")
        print()
    print(f"{len(DATASETS)} dataset preset(s).")
    print()
    print("subcommand applicability:")
    print("  flat-format datasets (na22, cd109, milliqan, lanl_806):")
    print("    template / hits / area / raw-area mqana / raw-area drs / "
          "timewalk / plot")
    print("  object-format datasets (lanl_835, lanl_movedBar):")
    print("    template / hits / area / raw-area mqana / "
          "raw-area offline / timewalk / plot")
    print("    (raw-area drs N/A — use `raw-area offline` for the LANL pulse list)")
    return 0


def _info_presets(args) -> int:
    if not CONFIGS:
        print("(no named configuration presets registered)")
        return 0
    for name, cfg in CONFIGS.items():
        print(f"━━━ {name}")
        print(f"    base_dataset : {cfg.base_dataset}")
        if cfg.references:
            print(f"    references   : {', '.join(cfg.references)}")
        print(f"    description  : {cfg.description}")
        print(f"    overrides    :")
        for k, v in cfg.overrides.items():
            print(f"        {k:<20} = {v!r}")
        print()
    print(f"{len(CONFIGS)} named preset(s).")
    return 0


def _info_offline_file(args) -> int:
    """Summary of a LANL ``*_offline.root`` pulse-list file:
    event count, populated channels, available time fields, sample
    first-event pulse breakdown.
    """
    path = Path(args.path)
    if not path.exists():
        print(f"[error] {path} does not exist.")
        return 1

    from pmtdeconv.io.offline_format import (
        ALLOWED_TIME_FIELDS, count_offline_events,
        detect_offline_channels, iter_offline_run,
    )
    try:
        import uproot
        n_evt = count_offline_events(path)
        chans = detect_offline_channels(path, sample_n=args.n_sample)
    except Exception as e:
        print(f"[error] cannot read {path} as offline format: {e}")
        return 1

    print(f"{path}")
    print(f"  events             : {n_evt}")
    print(f"  populated channels : {len(chans)}")
    print(f"    {', '.join(chans[:8])}"
          f"{', …' if len(chans) > 8 else ''}")

    # Which of the standard time fields exist in this file?
    with uproot.open(path) as f:
        t_key = next((k for k in f.keys() if k.split(';')[0] == 't'), None)
        if t_key is None:
            print("  [warn] no 't' tree found")
            return 1
        keys = set(f[t_key].keys())
    print(f"  time fields present (for --time-field):")
    for tf in ALLOWED_TIME_FIELDS:
        mark = "✓" if tf in keys else "·"
        print(f"    {mark} {tf}")

    # First-event sample
    print(f"  first event sample (aggregated rows):")
    n = 0
    for row in iter_offline_run(path, entry_stop=1):
        print(f"    {row.branch:<10}  "
              f"area_sum={row.area_sum:>9.1f}  "
              f"peak_h={row.peak_height:>7.1f}  "
              f"primary(area={row.primary_area:>8.1f}, "
              f"h={row.primary_height:>7.1f}, "
              f"t_peak={row.primary_t_peak:>8.2f} ns)")
        n += 1
        if n >= 6:
            print(f"    … and {len(chans) - n} more channels")
            break
    return 0


def _info_template(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"[error] {path} does not exist.")
        return 1
    # Use strict=False so we get a description, not an exception
    try:
        tmpl = load_template(path, strict=False, verbose=False)
    except Exception as e:
        print(f"[error] cannot parse {path}: {e}")
        return 1
    if not tmpl:
        print(f"{path}: empty / no recognizable schema.")
        return 1
    print(f"{path}")
    print(f"  {len(tmpl)} channel(s):")
    for branch, params in sorted(tmpl.items()):
        print(f"    {branch:<10}  A={params['A']:.3f}  "
              f"τ_rise={params['tau_rise']:.3f} ns  "
              f"τ_fall={params['tau_fall']:.3f} ns  "
              f"σ={params['sigma']:.3f} ns")
    return 0


def _info_artifacts(args) -> int:
    from pmtdeconv.io.artifacts import (
        DEFAULT_INDEX_NAME, build_artifact_index, write_artifact_index,
    )

    outputs_dir = Path(args.outputs_dir)
    index = build_artifact_index(
        outputs_dir,
        include_orphans=not args.no_orphans,
    )
    summary = index["summary"]

    print(f"{outputs_dir}")
    print(f"  manifests         : {summary['n_manifests']}")
    print(f"  invalid manifests : {summary['n_invalid_manifests']}")
    print(f"  orphan dirs       : {summary['n_orphan_dirs']}")
    print(f"  artifacts indexed : {summary['n_artifacts']}")
    if summary["output_role_counts"]:
        role_bits = [
            f"{role}={count}"
            for role, count in sorted(summary["output_role_counts"].items())
        ]
        print(f"  output roles      : {', '.join(role_bits)}")
    print()

    rows = index["artifacts"][:max(int(args.limit), 0)]
    if rows:
        print(f"{'kind':<14} {'subcommand':<18} {'dataset':<10} "
              f"{'config':<16} {'runs':<16} out_dir")
        print("-" * 120)
        for ent in rows:
            runs = ent.get("runs")
            if isinstance(runs, list):
                runs_s = ",".join(str(r) for r in runs[:4])
                if len(runs) > 4:
                    runs_s += ",..."
            elif runs is None:
                runs_s = ""
            else:
                runs_s = str(runs)
            print(f"{str(ent.get('kind') or ''):<14} "
                  f"{str(ent.get('subcommand') or ''):<18} "
                  f"{str(ent.get('dataset') or ''):<10} "
                  f"{str(ent.get('config') or ''):<16} "
                  f"{runs_s:<16} "
                  f"{Path(str(ent.get('out_dir') or '')).name}")

    if args.write is not None:
        target = (outputs_dir / DEFAULT_INDEX_NAME
                  if args.write == "" else Path(args.write))
        write_artifact_index(index, target)
        print()
        print(f"Wrote: {target}")
    return 0
