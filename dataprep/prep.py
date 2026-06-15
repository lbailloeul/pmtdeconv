"""``dataprep`` — thin orchestrator for the DAQ -> ROOT -> flattened chain.

Three leaves:

  dataprep drs      DRS binary (``Run<N>.dat``) -> ``Run<N>.root``   (wraps
                    drsProcessing/processMultiBoardBinary_v9.py via subprocess,
                    so this CLI never imports ROOT for that step)
  dataprep flatten  ``Run<N>.root`` -> ``flattened_Run<N>.root``     (PyROOT,
                    in-package flatten_waveforms)
  dataprep all      both, into ONE output dir

By convention everything lands in the dataset's data dir
(``<data-root>/<dataset>/``) holding both ``Run<N>.root`` and
``flattened_Run<N>.root`` side by side — the layout ``pmtdeconv`` resolves
(``raw-area drs`` / ``plot energy`` read ``Run<N>.root``; ``hits`` / ``area`` /
``raw-area mqana`` read ``flattened_Run<N>.root``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Vendored drsProcessing entry point (just this script + utils_v9.py, copied
# from https://github.com/mcitron/drsProcessing — see dataprep/drs/NOTICE.md).
# The full upstream clone is NOT required.
_DRS_SCRIPT_DEFAULT = _HERE / "drs" / "processMultiBoardBinary_v9.py"


def _add_drs_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-b", "--binary", required=True,
                   help="Input DRS binary (e.g. Run65213.dat).")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory for Run<N>.root (use the dataset data dir).")
    p.add_argument("--drs-script", default=str(_DRS_SCRIPT_DEFAULT),
                   help="Path to drsProcessing/processMultiBoardBinary_v9.py "
                        "(default: the sibling clone).")
    p.add_argument("-v", "--bias", default=None, help="Bias voltage (-v).")
    p.add_argument("-c", "--csv", default=None, help="CSV path (-c, bias scans).")
    p.add_argument("-x", "--xml", default=None, help="XML config (-x).")


def cmd_drs(args) -> int:
    script = Path(args.drs_script)
    if not script.exists():
        sys.exit(f"DRS processing script not found: {script}\n"
                 "  Expected the vendored copy at dataprep/drs/"
                 "processMultiBoardBinary_v9.py (from "
                 "https://github.com/mcitron/drsProcessing), or pass --drs-script.")
    # Always pass -w (waveform output) — that is the mode that yields the
    # Run<N>.root the deconvolution + flatten steps consume.
    cmd = [sys.executable, str(script), "-b", args.binary, "-o", args.output, "-w"]
    if args.bias is not None:
        cmd += ["-v", str(args.bias)]
    if args.csv:
        cmd += ["-c", args.csv]
    if args.xml:
        cmd += ["-x", args.xml]
    print("→ " + " ".join(cmd))
    return subprocess.call(cmd)


def cmd_flatten(args) -> int:
    try:
        from dataprep.flatten_waveforms import flatten_runs
    except ModuleNotFoundError as exc:
        if (getattr(exc, "name", "") or "") == "ROOT":
            sys.exit("flatten needs PyROOT (`import ROOT`), which is not "
                     "pip-installable.\n  Install it via conda: "
                     "conda install -c conda-forge root")
        raise
    flatten_runs(args.runs, args.input_dir, args.output_dir, treename=args.treename)
    return 0


def cmd_all(args) -> int:
    code = cmd_drs(args)
    if code != 0:
        return code
    # The drs step wrote Run<N>.root into --output; flatten from there into the
    # same dir so raw + flattened sit together.
    args.input_dir = args.output
    args.output_dir = args.output
    return cmd_flatten(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataprep",
        description="DAQ -> ROOT -> flattened prep for the PMT deconvolution pipeline.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<leaf>")

    p_drs = sub.add_parser("drs", help="DRS binary -> Run<N>.root (subprocess).")
    _add_drs_args(p_drs)
    p_drs.set_defaults(_run=cmd_drs)

    p_flat = sub.add_parser("flatten", help="Run<N>.root -> flattened_Run<N>.root (PyROOT).")
    p_flat.add_argument("--runs", nargs="+", type=int, required=True)
    p_flat.add_argument("--input-dir", required=True,
                        help="Directory holding Run<N>.root.")
    p_flat.add_argument("--output-dir", default=None,
                        help="Where flattened_Run<N>.root go (default: --input-dir).")
    p_flat.add_argument("--treename", default="Events")
    p_flat.set_defaults(_run=cmd_flatten)

    p_all = sub.add_parser("all", help="drs then flatten, into one --output dir.")
    _add_drs_args(p_all)
    p_all.add_argument("--runs", nargs="+", type=int, required=True)
    p_all.add_argument("--treename", default="Events")
    p_all.set_defaults(_run=cmd_all)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "flatten" and args.output_dir is None:
        args.output_dir = args.input_dir
    return int(args._run(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
