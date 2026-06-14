"""Top-level CLI dispatcher for ``pmtdeconv``.

Each subcommand module under ``pmtdeconv/commands/`` exposes two functions:
- ``add_arguments(subparser)`` — register that command's flags
- ``run(args)`` — execute the command from a parsed Namespace

Bug-fix 5 (2026-05-30 scan): **two-stage parsing** so the top-level
``pmtdeconv --help`` and per-command ``pmtdeconv <command> --help`` work in
environments that don't have heavy optional deps (uproot, awkward,
matplotlib). Only when the user drills into a specific *leaf* do we
import that command module; everything above stays import-light.

``cli.main()`` builds the parser tree and dispatches. No business logic
lives here.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import logging
import re
import sys
from typing import Optional, Sequence

from pmtdeconv import __version__


# Two-level subcommand registry. Outer key = level-1 (e.g. "template",
# "raw_area"); inner key = level-2 (e.g. "build", "mqana"). Single-level
# commands use level-2 = None. Module names are relative to
# ``pmtdeconv.commands``.
SUBCOMMANDS = {
    "run": {None: "run_recipe"},
    "template": {
        "build":     "template_build",
        "aggregate": "template_aggregate",
    },
    "hits": {None: "pulse"},
    "area": {None: "waveform"},
    "raw-area": {
        "mqana":   "raw_area_mqana",
        "drs":     "raw_area_drs",
        "offline": "raw_area_offline",
    },
    # Compatibility names retained for existing scripts.
    "pulse": {None: "pulse"},
    "waveform": {None: "waveform"},
    "raw_area": {
        "mqana":   "raw_area_mqana",
        "drs":     "raw_area_drs",
        "offline": "raw_area_offline",
    },
    "timewalk": {
        "run": "timewalk_run",
        "gif": "timewalk_gif",
    },
    "plot": {
        "marginals":    "plot_marginals",
        "energy":       "plot_energy",
        "analyze":      "plot_analyze",
        # "deconv-value" demoted to legacy/ (2026-06-13): the multi-hit + nPE
        # "deconvolution value" figures aren't convincing on synthetic data —
        # restore from legacy/plotting/ + re-add this line once validated on
        # real Na22 data with genuine pile-up.
    },
    # `info` is single-level — its module attaches its own sub-parsers
    # for datasets/presets/template/offline-file.
    "info": {None: "info"},
    "config": {None: "config"},
    "results": {None: "results"},
    # PR 10: regime-driven parameter sweeps with auto-generated presets.
    "tune": {
        "cosine":   "tune_cosine",
        "wavelet":  "tune_wavelet",
        "stft":     "tune_stft",
        "denoiser": "tune_denoiser",
    },
    # Step 5: derive a data-driven veto profile from a run's raw waveforms.
    "vetoes": {
        "calibrate": "vetoes_calibrate",
    },
}


EAGER_SINGLE_LEVEL = {"info", "config", "results"}

DEPRECATED_COMMANDS = {
    "pulse": "hits",
    "waveform": "area",
    "raw_area": "raw-area",
}


# Bug-fix 5: static one-line help strings, so the top-level / verb-level
# `--help` doesn't have to import any command module to render its tree.
LEAF_HELP = {
    ("run",      None):        "Run a complete, named analysis recipe.",
    ("template", "build"):     "Per-(dataset, file, channel) stacked SPE templates.",
    ("template", "aggregate"): "Combine per-file templates into per-channel.",
    ("hits",     None):        "Peak-finding reconstruction (per-hit time + charge).",
    ("area",     None):        "Whole-waveform deconvolved area per trigger.",
    ("raw-area", "mqana"):     "MQAna-style narrow-window raw area.",
    ("raw-area", "drs"):       "Pull precomputed area_<board>_<ch> from DRSprocessing.",
    ("raw-area", "offline"):   "LANL *_offline.root pulse-list area + hit CSVs.",
    ("pulse",    None):        "Deprecated alias for `hits`.",
    ("waveform", None):        "Deprecated alias for `area`.",
    ("raw_area", "mqana"):     "Deprecated alias for `raw-area mqana`.",
    ("raw_area", "drs"):       "Deprecated alias for `raw-area drs`.",
    ("raw_area", "offline"):   "Deprecated alias for `raw-area offline`.",
    ("timewalk", "run"):       "Scan-only timewalk: summary CSV + curve PNG.",
    ("timewalk", "gif"):       "Full animated timewalk visualization.",
    ("plot",     "marginals"): "CH1-vs-CH2 area heatmap + 1D marginals.",
    ("plot",     "energy"):    "Four-method energy histograms (raw/FFT/STFT/wavelet).",
    ("plot",     "analyze"):   "Diagnostic plots from a deconv-pulse CSV (legacy/diagnostic).",
    ("info",     None):        "Read-only introspection of registries.",
    ("config",   None):        "Show and resolve datasets, presets, and overrides.",
    ("results",  None):        "List, inspect, and locate prior CLI results.",
    ("tune",     "cosine"):    "Sweep cosine cutoff per regime, score vs truth.",
    ("tune",     "wavelet"):   "Sweep wavelet config per regime, score vs truth.",
    ("tune",     "stft"):      "Sweep STFT denoiser per regime, score vs truth.",
    ("tune",     "denoiser"):  "STFT-vs-wavelet verdict: per-pulse FWHM vs cutoff, 5 configs.",
    ("vetoes",   "calibrate"): "Derive a data-driven veto profile from raw waveforms.",
}


class CLIArgumentParser(argparse.ArgumentParser):
    """Argument parser that adds a close-match hint for mistyped choices."""

    def error(self, message: str) -> None:
        match = re.search(r"invalid choice: '([^']+)' \(choose from (.+)\)", message)
        if match:
            invalid = match.group(1)
            choices = re.findall(r"'([^']+)'", match.group(2))
            if not choices:
                choices = [
                    choice.strip().strip("'\"")
                    for choice in match.group(2).split(",")
                ]
            suggestion = difflib.get_close_matches(invalid, choices, n=1)
            if suggestion:
                message += f"\nDid you mean {suggestion[0]!r}?"
        super().error(message)


def _parser_description() -> str:
    return (
        "PMT waveform reconstruction and analysis.\n\n"
        "Common workflows:\n"
        "  pmtdeconv run na22-timewalk\n"
        "  pmtdeconv hits --dataset na22 --runs 65213\n"
        "  pmtdeconv area --dataset na22 --runs 65212\n"
        "  pmtdeconv config resolve --preset na22-timewalk\n"
        "  pmtdeconv results list"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand. See the plan §2.1 for the list."""
    g = parser.add_argument_group("common")
    # Derive choices from the registry (2026-06-10): the hardcoded tuple
    # silently drifted from presets/datasets.py when na22_3ch was added.
    from pmtdeconv.presets.datasets import list_datasets
    g.add_argument("--dataset",
                   choices=tuple(list_datasets()),
                   default=argparse.SUPPRESS,
                   help="Dataset preset (resolves dt_ns, tree, branches, "
                        "polarity, default template, sat-thresh, cosine cutoff).")
    g.add_argument("--preset", default=argparse.SUPPRESS,
                   help="Named configuration preset (overlays on top of "
                        "--dataset). See `pmtdeconv config show <preset>`.")
    g.add_argument("--runs", type=int, nargs="+", default=argparse.SUPPRESS,
                   help="Run numbers to process.")
    g.add_argument("--entry-stop", type=int, default=argparse.SUPPRESS,
                   help="Cap entries per file (smoke-test mode).")
    g.add_argument("--branches", type=str, nargs="+",
                   default=argparse.SUPPRESS,
                   help="Restrict to specific branches (e.g. d0_ch1 d0_ch2).")
    g.add_argument("--data-root", type=str, default=argparse.SUPPRESS,
                   help="Override dataset's root data directory.")
    g.add_argument("--tree", type=str, default=argparse.SUPPRESS,
                   help="Override ROOT tree name (waveforms / Events).")
    g.add_argument("--dt-ns", type=str, default=argparse.SUPPRESS,
                   help="Override sample spacing in ns. Pass 'auto' to "
                        "read from object-format files.")
    g.add_argument("--sat-thresh", type=float, default=argparse.SUPPRESS,
                   help="Saturation threshold in mV.")
    g.add_argument("--out-dir", type=str, default=argparse.SUPPRESS,
                   help="Output directory (subcommand-specific default).")
    g.add_argument("--out-suffix", type=str, default=argparse.SUPPRESS,
                   help="Appended to default --out-dir.")
    g.add_argument("--manifest", type=str, default=argparse.SUPPRESS,
                   help="Manifest JSON path (default: <out-dir>/manifest.json).")
    g.add_argument("--hash-inputs", action="store_true",
                   default=argparse.SUPPRESS,
                   help="SHA256 every input file (slow on large ROOT files).")
    g.add_argument("--dry-run", action="store_true",
                   default=argparse.SUPPRESS,
                   help="Resolve flags and write manifest only; skip ROOT I/O.")
    g.add_argument("--log-level", default=argparse.SUPPRESS,
                   choices=("quiet", "info", "debug"))


def _build_root_parser() -> argparse.ArgumentParser:
    """Stage-1 parser: only knows verb + leaf NAMES (and one-line help).

    No command module is imported here. Survives missing optional
    deps like uproot/awkward/matplotlib.
    """
    parser = CLIArgumentParser(
        prog="pmtdeconv",
        description=_parser_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<command>")

    for cmd, leaves in SUBCOMMANDS.items():
        if list(leaves.keys()) == [None]:
            if cmd in EAGER_SINGLE_LEVEL:
                # `info` builds eagerly because it owns its own subparser tree.
                mod = importlib.import_module(f"pmtdeconv.commands.{leaves[None]}")
                cmd_parser = sub.add_parser(cmd,
                    help=LEAF_HELP.get((cmd, None), ""))
                add_common_args(cmd_parser)
                mod.add_arguments(cmd_parser)
                cmd_parser.set_defaults(_run=mod.run, _command_path=cmd)
            else:
                # Stage-1 direct leaf: register only the command name and
                # delay imports until stage 2 so top-level help stays light.
                cmd_parser = sub.add_parser(
                    cmd, help=LEAF_HELP.get((cmd, None), ""),
                    add_help=False)
                cmd_parser.set_defaults(
                    _lazy_module=f"pmtdeconv.commands.{leaves[None]}",
                    _command_path=cmd,
                )
        else:
            cmd_parser = sub.add_parser(cmd, help=f"{cmd} subcommands")
            cmd_sub = cmd_parser.add_subparsers(dest="subcommand", required=True,
                                                metavar="<leaf>")
            for leaf, mod_name in leaves.items():
                # Stage-1: just register the leaf NAME + one-line help.
                # add_arguments is deferred to stage-2 (see main()).
                leaf_parser = cmd_sub.add_parser(
                    leaf, help=LEAF_HELP.get((cmd, leaf), ""),
                    add_help=False)
                # Use parse_known_args at stage 2 so leftover flags
                # like --help get passed to the leaf-specific parser.
                leaf_parser.set_defaults(
                    _lazy_module=f"pmtdeconv.commands.{mod_name}",
                    _command_path=f"{cmd} {leaf}",
                )
    return parser


def _build_leaf_parser(cmd_path: str,
                       mod) -> argparse.ArgumentParser:
    """Stage-2: build a full parser for the SELECTED leaf."""
    leaf_parser = CLIArgumentParser(
        prog=f"pmtdeconv {cmd_path}",
        description=mod.__doc__.splitlines()[0] if mod.__doc__ else "",
    )
    add_common_args(leaf_parser)
    mod.add_arguments(leaf_parser)
    return leaf_parser


def configure_logging(level: str) -> None:
    levels = {"quiet": logging.WARNING, "info": logging.INFO,
              "debug": logging.DEBUG}
    logging.basicConfig(
        level=levels.get(level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Eager parser — imports every command module so that
    ``parse_args(['<leaf>', '--help'])`` works directly. The fast,
    dep-light path used by ``pmtdeconv --help`` lives in :func:`main`
    (see ``_build_root_parser``). Tests rely on this eager form.
    """
    parser = CLIArgumentParser(
        prog="pmtdeconv",
        description=_parser_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<command>")

    for cmd, leaves in SUBCOMMANDS.items():
        if list(leaves.keys()) == [None]:
            mod = importlib.import_module(f"pmtdeconv.commands.{leaves[None]}")
            cmd_parser = sub.add_parser(cmd,
                help=LEAF_HELP.get((cmd, None), ""))
            add_common_args(cmd_parser)
            mod.add_arguments(cmd_parser)
            cmd_parser.set_defaults(_run=mod.run, _command_path=cmd)
        else:
            cmd_parser = sub.add_parser(cmd, help=f"{cmd} subcommands")
            cmd_sub = cmd_parser.add_subparsers(dest="subcommand",
                                                required=True,
                                                metavar="<leaf>")
            for leaf, mod_name in leaves.items():
                mod = importlib.import_module(f"pmtdeconv.commands.{mod_name}")
                leaf_parser = cmd_sub.add_parser(
                    leaf, help=LEAF_HELP.get((cmd, leaf), ""))
                add_common_args(leaf_parser)
                mod.add_arguments(leaf_parser)
                leaf_parser.set_defaults(_run=mod.run,
                                         _command_path=f"{cmd} {leaf}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _build_root_parser()
    args, remaining = root.parse_known_args(argv)

    # Eager-info path: `info` already has its full parser built; nothing
    # extra to do unless leftover args slipped through.
    if hasattr(args, "_run"):
        if remaining:
            raise SystemExit(f"unrecognized arguments: {' '.join(remaining)}")
        configure_logging(getattr(args, "log_level", "info"))
        _warn_if_deprecated(args._command_path)
        return _invoke(args._run, args)

    # Deferred-leaf path: import the chosen module on demand, then re-parse.
    if hasattr(args, "_lazy_module"):
        mod = importlib.import_module(args._lazy_module)
        leaf_parser = _build_leaf_parser(args._command_path, mod)
        leaf_args = leaf_parser.parse_args(remaining)
        # Carry the path forward so commands can introspect if they want
        leaf_args._command_path = args._command_path
        configure_logging(getattr(leaf_args, "log_level", "info"))
        _warn_if_deprecated(args._command_path)
        return _invoke(mod.run, leaf_args)

    raise SystemExit("internal: no _run or _lazy_module was set on parsed args")


def _warn_if_deprecated(command_path: str) -> None:
    command = command_path.split()[0]
    replacement = DEPRECATED_COMMANDS.get(command)
    if replacement:
        suffix = command_path[len(command):]
        print(
            f"warning: `pmtdeconv {command}` is deprecated; "
            f"use `pmtdeconv {replacement}{suffix}`.",
            file=sys.stderr,
        )


def _invoke(run_fn, args: argparse.Namespace) -> int:
    try:
        return int(run_fn(args) or 0)
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise SystemExit(message) from exc
    except RuntimeError as exc:
        # Template resolution failure -> clean guidance, not a traceback
        # (lazy import keeps the two-stage CLI startup fast).
        from pmtdeconv.pipeline.template import TemplateResolutionError
        if isinstance(exc, TemplateResolutionError):
            raise SystemExit(str(exc)) from exc
        raise


if __name__ == "__main__":
    sys.exit(main())
