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
    "hits": {None: "hits"},
    "area": {None: "area"},
    "raw-area": {
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
        # Restored 2026-07-29 (0–20 PE re-scope): the demotion condition —
        # "validated on real data with genuine pile-up" — is met by the
        # `select multihit` + `gain calibrate` path, and the nPE axis is now
        # real photoelectrons when spe_charge is calibrated.
        "deconv-value": "plot_deconv_value",
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
        "fom":      "tune_fom",
    },
    # Step 5: derive a data-driven veto profile from a run's raw waveforms.
    "vetoes": {
        "calibrate": "vetoes_calibrate",
    },
    # 0–20 PE re-scope: fit the 1-PE charge-quantization peak → propose the
    # per-branch `spe_charge` PE-scale anchor (expert applies; never silent).
    "gain": {
        "calibrate": "gain_calibrate",
    },
    # 0–20 PE re-scope: the Tang et al. 2411.07620 MC (the paper that
    # established the 0–20 PE applicability domain) as a standard dataset.
    "sim": {
        "tang2024": "sim_tang2024",
    },
    # 0–20 PE re-scope: THE STFT-vs-wavelet verdict harness — every arm runs
    # the real pipeline under identical conditions (supersedes tune denoiser).
    "benchmark": {
        "denoisers": "benchmark_denoisers",
    },
    # 0–20 PE re-scope: multi-hit bunch selection (vault 11e clump recipe).
    "select": {
        "multihit": "select_multihit",
    },
    # S4: scaffold + validate file-backed dataset cards (no Python edit).
    "dataset": {
        "init": "dataset_init",
        "lint": "dataset_lint",
    },
    # S6: comparative figure-of-merit report (deconv vs raw vs DRS).
    "validate": {None: "validate"},
}


EAGER_SINGLE_LEVEL = {"info", "config", "results"}

# Commands REMOVED 2026-09-14 after their one deprecation cycle. They are no
# longer registered, so argparse would report a bare "invalid choice"; these
# entries turn that into the actual migration instruction instead. Manifests
# written before the removal still store the old names — that translation lives
# separately in ``commands/results.py`` and is unaffected.
REMOVED_COMMANDS = {
    "pulse":    "hits",
    "waveform": "area",
    "raw_area": "raw-area",
}
# Leaves REPLACED by a different command. Unlike the verb renames above these
# are not same-flags renames, so each carries its own date and a note saying
# what actually changed — a migration message must not claim otherwise.
REMOVED_LEAVES = {
    ("tune", "denoiser"): {
        "use": "benchmark denoisers",
        "when": "2026-09-14",
        "note": "Same question (STFT vs wavelet), now scored against truth "
                "with every arm run under identical conditions; see its "
                "--help for the flags.",
    },
    ("sim", "tang2024-paper"): {
        "use": "benchmark denoisers --suite tang2024 --arms tang2024 "
               "tang2024-fft",
        "when": "2026-09-21",
        "note": "The paper's method is now two benchmark arms scored against "
                "the same truth as ours; the published fig-12/14 values are "
                "in verdict.json. Pass --tang-dir <dir> instead of --sim-dir.",
    },
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
    ("timewalk", "run"):       "Scan-only timewalk: summary CSV + curve PNG. [track 2: timing-valid through saturation; charge out of domain]",
    ("timewalk", "gif"):       "Full animated timewalk visualization. [track 2: timing-valid through saturation]",
    ("plot",     "marginals"): "CH1-vs-CH2 area heatmap + 1D marginals.",
    ("plot",     "energy"):    "Four-method energy histograms (raw/FFT/STFT/wavelet).",
    ("plot",     "analyze"):   "Diagnostic plots from a deconv-pulse CSV (legacy/diagnostic).",
    ("info",     None):        "Read-only introspection of registries.",
    ("config",   None):        "Show and resolve datasets, presets, and overrides.",
    ("results",  None):        "List, inspect, and locate prior CLI results.",
    ("tune",     "cosine"):    "Sweep cosine cutoff per regime, score vs truth.",
    ("tune",     "wavelet"):   "Sweep wavelet config per regime, score vs truth.",
    ("tune",     "stft"):      "Sweep STFT denoiser per regime, score vs truth.",
    ("benchmark", "denoisers"): "Head-to-head vs truth: our denoisers + Tang's own method.",
    ("select",   "multihit"):  "Select 0–20 PE multi-hit bunches → reusable event_list.csv.",
    ("plot",     "deconv-value"): "Multi-hit multiplicity + nPE spectrum (real PE when calibrated).",
    ("tune",     "fom"):       "Sweep cutoff on REAL data vs the card FoM (+anti-cheat).",
    ("vetoes",   "calibrate"): "Derive a data-driven veto profile from raw waveforms.",
    ("gain",     "calibrate"): "Fit the 1-PE charge peak → propose spe_charge (PE anchor).",
    ("sim",      "tang2024"):  "Generate the Tang 2411.07620 MC (0–20 PE mix + two-hit scan).",
    ("dataset",  "init"):      "Scaffold a YAML dataset card (no Python edit).",
    ("dataset",  "lint"):      "Validate a dataset card's fields + settings block.",
    ("validate", None):        "Comparative FoM report: deconv vs raw vs DRS (no gate).",
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


class _RenamedFlagAction(argparse.Action):
    """A pre-1.0 spelling kept working after a rename: stores under the NEW
    dest and prints one line pointing at the new name. Hidden from --help, so
    the old spelling keeps existing scripts running without being advertised."""

    def __init__(self, option_strings, dest, new_flag=None, **kwargs):
        self.new_flag = new_flag
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"note: {option_string} is now {self.new_flag} "
              f"(the old spelling works until 1.0).", file=sys.stderr)
        setattr(namespace, self.dest, values)


def add_renamed_alias(parser, old: str, new: str, *, dest: str,
                      nargs=None) -> None:
    """Register ``old`` as a hidden alias of ``new`` (same ``dest``).

    ``default=SUPPRESS`` so the alias never overwrites the real flag's default;
    it only sets ``dest`` when actually passed."""
    kwargs = {"nargs": nargs} if nargs is not None else {}
    parser.add_argument(old, dest=dest, action=_RenamedFlagAction,
                        new_flag=new, help=argparse.SUPPRESS,
                        default=argparse.SUPPRESS, **kwargs)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand. See the plan §2.1 for the list."""
    g = parser.add_argument_group("common")
    # Derive choices from the registry (2026-06-10): the hardcoded tuple
    # silently drifted from presets/datasets.py when na22_3ch was added.
    from pmtdeconv.presets.datasets import list_datasets
    g.add_argument("--dataset",
                   default=argparse.SUPPRESS,
                   metavar="NAME_OR_CARD",
                   help="Dataset: a registry name (" + ", ".join(list_datasets())
                        + ") OR a path to a YAML card / directory with "
                        "dataset.yaml. Resolves dt_ns, tree, branches, "
                        "polarity, default template, cosine cutoff.")
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
    # 2026-09-14: the CLI_TO_SETTINGS entry and every downstream consumer
    # (Detector -> event_source._apply_polarity, area, raw-area mqana) already
    # existed; only this argparse hop was missing, so negative-going data could
    # be handled ONLY via a dataset preset or card. Same file-describing class
    # as --dt-ns/--tree/--branches, so it belongs here in the common group.
    g.add_argument("--polarity", type=int, choices=(1, -1),
                   default=argparse.SUPPRESS,
                   help="Waveform polarity: +1 positive-going (default), -1 "
                        "negative-going (flipped once, in the event source). "
                        "Overrides the dataset/card value.")
    g.add_argument("--out-dir", type=str, default=argparse.SUPPRESS,
                   help="Output directory (subcommand-specific default).")
    g.add_argument("--out-suffix", type=str, default=argparse.SUPPRESS,
                   help="Appended to default --out-dir.")
    g.add_argument("--manifest", type=str, default=argparse.SUPPRESS,
                   help="Manifest JSON path (default: <out-dir>/manifest.json).")
    # Default ON since 2026-09-21. The old reason for opt-in ("slow on large
    # ROOT files") was measured false: SHA256 of a 477 MB run is ~0.2 s,
    # against reconstruction runs of minutes. With a shared, mutable card
    # catalog, hashing is what lets two manifests prove they used the same
    # card, template and data.
    g.add_argument("--hash-inputs", dest="hash_inputs", action="store_true",
                   default=argparse.SUPPRESS,
                   help="SHA256 every input (card, template, data) into the "
                        "manifest. Default: on.")
    g.add_argument("--no-hash-inputs", dest="hash_inputs",
                   action="store_false", default=argparse.SUPPRESS,
                   help="Skip input hashing (the manifest then records paths "
                        "and sizes only).")
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


# Third-party scientific-stack modules pmtdeconv imports lazily. A missing one
# means "you ran from source without `pip install -e .`" — turn the raw
# ModuleNotFoundError traceback into a one-line install instruction.
_THIRD_PARTY_DEPS = {"numpy", "scipy", "pandas", "matplotlib", "uproot",
                     "awkward", "pywt", "PIL", "tqdm"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Tee the console into memory so write_manifest can drop it next to the
    # manifest as run.log. Started here (before parsing and configure_logging)
    # so warnings emitted during resolution are captured too.
    from pmtdeconv.io.run_log import start_capture, stop_capture
    start_capture()
    try:
        return _dispatch(argv)
    except ModuleNotFoundError as exc:
        name = (getattr(exc, "name", "") or "").split(".")[0]
        if name in _THIRD_PARTY_DEPS:
            raise SystemExit(
                f"\nMissing dependency '{name}'. pmtdeconv needs its scientific "
                f"stack installed.\n"
                f"  Install everything in one step (from the bailloeulScripts/ "
                f"directory containing pyproject.toml):\n"
                f"      pip install -e .\n"
                f"  (pulls in numpy, scipy, pandas, matplotlib, uproot, awkward, "
                f"PyWavelets, …)\n"
            ) from exc
        raise
    finally:
        # Always restore: an in-process caller (pytest invoking cli.main())
        # must not be left with a dangling tee on sys.stdout.
        stop_capture()


def _dispatch(argv: Optional[Sequence[str]] = None) -> int:
    _exit_if_removed(argv)
    root = _build_root_parser()
    args, remaining = root.parse_known_args(argv)

    # Eager-info path: `info` already has its full parser built; nothing
    # extra to do unless leftover args slipped through.
    if hasattr(args, "_run"):
        if remaining:
            raise SystemExit(f"unrecognized arguments: {' '.join(remaining)}")
        configure_logging(getattr(args, "log_level", "info"))
        return _invoke(args._run, args)

    # Deferred-leaf path: import the chosen module on demand, then re-parse.
    if hasattr(args, "_lazy_module"):
        mod = importlib.import_module(args._lazy_module)
        leaf_parser = _build_leaf_parser(args._command_path, mod)
        leaf_args = leaf_parser.parse_args(remaining)
        # Carry the path forward so commands can introspect if they want
        leaf_args._command_path = args._command_path
        configure_logging(getattr(leaf_args, "log_level", "info"))
        return _invoke(mod.run, leaf_args)

    raise SystemExit("internal: no _run or _lazy_module was set on parsed args")


def _exit_if_removed(argv: Optional[Sequence[str]]) -> None:
    """Turn a removed command name into its migration instruction.

    Runs BEFORE argparse so the user gets "`pulse` was renamed to `hits`"
    instead of "invalid choice: 'pulse'" — difflib cannot suggest `hits` for
    `pulse`, the names are not similar. Cheap: two dict lookups on argv.
    """
    raw = list(argv if argv is not None else sys.argv[1:])
    # Only the verb/leaf tokens BEFORE the first flag — otherwise a flag's
    # value ("--dataset na22") would be mistaken for a leaf name.
    tokens = []
    for a in raw:
        if a.startswith("-"):
            break
        tokens.append(a)
    if not tokens:
        return
    verb = tokens[0]
    leaf = tokens[1] if len(tokens) > 1 else None

    leaf_info = REMOVED_LEAVES.get((verb, leaf))
    if leaf_info is not None:
        raise SystemExit(
            f"\n`pmtdeconv {verb} {leaf}` was removed on {leaf_info['when']}.\n"
            f"  Use instead:  pmtdeconv {leaf_info['use']}\n"
            f"  {leaf_info['note']}\n")
    renamed = REMOVED_COMMANDS.get(verb)
    if renamed is None:
        return
    target = f"{renamed} {leaf}" if leaf else renamed
    raise SystemExit(
        f"\n`pmtdeconv {verb}` was removed on 2026-09-14 after its "
        f"deprecation cycle.\n"
        f"  Use instead:  pmtdeconv {target}\n"
        f"  Same flags and same output — only the name changed.\n")


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
