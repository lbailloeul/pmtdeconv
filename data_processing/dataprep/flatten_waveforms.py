#!/usr/bin/env python3
"""
Flatten processed DRS ROOT files into a waveforms tree with per-channel branches.

Input:  ROOT file with `Events` tree containing branches
          voltages_<board>_<channel>[1024]/D
Output: ROOT file with `waveforms` tree containing branches
          d<di>_ch<channel>[1024]/D
        where <di> is the digitizer index assigned by ordering boards numerically
        and <channel> preserves the original channel number.

This is the canonical home (moved from ``script_factory/flatten_waveforms.py``);
the logic is unchanged. ``flatten_runs`` is the reusable entry point the
``dataprep`` CLI calls; the ``__main__`` block keeps standalone usage.

Usage:
    python -m dataprep.flatten_waveforms --runs 65213 \
        --input-dir <data-root>/Na22_lab --output-dir <data-root>/Na22_lab
"""

import argparse
import re
from pathlib import Path

import numpy as np
import ROOT

N_SAMPLES = 1024
VOLTAGE_BRANCH_RE = re.compile(r"^voltages_(\d+)_(\d+)$")


def discover_channels(tree):
    """Return sorted list of (board, channel) pairs from voltages_ branches."""
    pairs = []
    for b in tree.GetListOfBranches():
        m = VOLTAGE_BRANCH_RE.match(b.GetName())
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    pairs.sort()
    return pairs


def flatten(input_file, output_file, treename="Events"):
    fin = ROOT.TFile.Open(input_file, "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {input_file}")

    tree_in = fin.Get(treename)
    if not tree_in:
        print("Available objects:")
        for key in fin.GetListOfKeys():
            print(f"  {key.GetName()} ({key.GetClassName()})")
        raise RuntimeError(f"Tree '{treename}' not found")

    n_events = tree_in.GetEntries()
    pairs = discover_channels(tree_in)
    if not pairs:
        raise RuntimeError("No voltages_<board>_<channel> branches found")

    boards = sorted({b for b, _ in pairs})
    board_to_di = {b: i for i, b in enumerate(boards)}

    print(f"Input: {input_file}")
    print(f"  events: {n_events}")
    print(f"  channels (board, ch): {pairs}")
    print(f"  board -> digitizer index: {board_to_di}")

    fout = ROOT.TFile(output_file, "RECREATE")
    tree_out = ROOT.TTree("waveforms", "Flattened waveform data")

    # Shared numpy buffers: tree_in writes into them via SetBranchAddress,
    # tree_out reads from the same memory via Branch.
    buffers = {}
    for board, ch in pairs:
        buf = np.zeros(N_SAMPLES, dtype=np.float64)
        buffers[(board, ch)] = buf
        tree_in.SetBranchAddress(f"voltages_{board}_{ch}", buf)

        di = board_to_di[board]
        out_name = f"d{di}_ch{ch}"
        tree_out.Branch(out_name, buf, f"{out_name}[{N_SAMPLES}]/D")

    print(f"  processing {n_events} events ...")
    for i_evt in range(n_events):
        tree_in.GetEntry(i_evt)
        tree_out.Fill()
        if (i_evt + 1) % 10000 == 0:
            print(f"    {i_evt + 1} / {n_events}")

    print(f"  writing {output_file} ...")
    fout.cd()
    tree_out.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    fin.Close()
    print(f"  done: {n_events} events, {len(pairs)} channel branches")


def flatten_runs(runs, input_dir, output_dir, treename="Events"):
    """Flatten ``<input_dir>/Run<N>.root`` -> ``<output_dir>/flattened_Run<N>.root``
    for each run. By the data-root convention input_dir == output_dir (both the
    dataset's data dir), so the raw + flattened files end up side by side."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        in_file = in_dir / f"Run{run}.root"
        out_file = out_dir / f"flattened_Run{run}.root"
        flatten(str(in_file), str(out_file), treename)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flatten processed DRS waveform TTree")
    parser.add_argument("--runs", nargs="+", type=int, default=[65213],
                        help="Run numbers to process")
    parser.add_argument("--input-dir", default=".",
                        help="Directory containing Run<N>.root files")
    parser.add_argument("--output-dir", default=".",
                        help="Directory where flattened_Run<N>.root files are written")
    parser.add_argument("--treename", default="Events",
                        help="Name of input TTree (default: 'Events')")
    args = parser.parse_args(argv)
    flatten_runs(args.runs, args.input_dir, args.output_dir, args.treename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
