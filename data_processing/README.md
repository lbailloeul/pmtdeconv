# pmtdeconv-dataprep — DAQ → ROOT data prep

The **pre-deconvolution** half of the pipeline: turn raw DRS scope binaries into
the ROOT files `pmtdeconv` consumes. It is a **separate package** from
`pmtdeconv` on purpose — it depends on **PyROOT** (`import ROOT`), which is not
pip-installable, whereas `pmtdeconv` stays pure-Python (uproot) and
`pip install -e .`-able. Collaborators who already have ROOT files never need
ROOT installed.

```
DRS binary  ──(drs)──▶  Run<N>.root  ──(flatten)──▶  flattened_Run<N>.root
 Run<N>.dat            area_/vMax_/                 waveforms tree
                       voltages_ branches           (d0_ch1, d0_ch2, …)
```

- **`Run<N>.root`** (raw / DRSprocessing) → read by `pmtdeconv raw-area drs`, `plot energy`.
- **`flattened_Run<N>.root`** (waveforms) → read by `pmtdeconv hits`, `area`, `raw-area mqana`.

Both land **side by side under one data dir** (`<data-root>/<dataset>/`) — the
exact layout `pmtdeconv` resolves (no `processed/` subdir).

## Layout

```
data_processing/
├── pyproject.toml          # this package (pmtdeconv-dataprep)
├── dataprep/               # the CLI + steps
│   ├── prep.py             # `dataprep` console script (drs / flatten / all)
│   ├── flatten_waveforms.py
│   └── drs/                # VENDORED from mcitron/drsProcessing (credit: NOTICE.md)
│       ├── processMultiBoardBinary_v9.py
│       └── utils_v9.py
└── Run<N>.dat              # raw DRS binaries (input)
```

The DRS decode step is **vendored**: only `processMultiBoardBinary_v9.py` and
its one helper `utils_v9.py` are copied (verbatim, with an attribution header)
from https://github.com/mcitron/drsProcessing into `dataprep/drs/` — full
credit + provenance (author, commit, license note) in
[`dataprep/drs/NOTICE.md`](dataprep/drs/NOTICE.md). `dataprep drs` runs the
vendored script via `subprocess`; **the full upstream clone is not required**.
To update, re-copy those two files from upstream (don't hand-edit the vendored
copies).

## Install

ROOT is **not** pip-installable — get it from conda first, then editable-install
this package:

```bash
conda install -c conda-forge root        # provides PyROOT (`import ROOT`)
cd data_processing
pip install -e .                         # numpy + the `dataprep` console script
```

## Usage

```bash
# 1. DRS binary -> Run<N>.root (wraps drsProcessing -w, via subprocess):
dataprep drs -b Run65213.dat -o ../deconvolution/.../data/Na22_lab

# 2. Run<N>.root -> flattened_Run<N>.root (PyROOT):
dataprep flatten --runs 65213 --input-dir <data-root>/Na22_lab

# Or both in one step, into one output dir:
dataprep all -b Run65213.dat -o <data-root>/Na22_lab --runs 65213
```

Point the output at the dataset directory `pmtdeconv` uses (its `--data-root` /
`PMTDECONV_DATA_ROOT`), so the resulting `Run65213.root` +
`flattened_Run65213.root` are immediately picked up by:

```bash
python -m pmtdeconv run na22-timewalk --runs 65213 --data-root <data-root>/Na22_lab
```

## Why separate (not folded into `pmtdeconv`)

- **Dependency split**: `import ROOT` (heavy, conda/system) vs `pmtdeconv`'s
  uproot (pip). Folding it in would force ROOT on every deconv user and undo the
  one-command `pip install -e .` install.
- **Ownership / cadence**: drsProcessing is external; prep runs once per dataset
  (slow, I/O-bound) while deconv runs many times. Clean ROOT-file contract
  between them.
