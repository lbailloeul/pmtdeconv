# bailloeulScripts — pmtdeconv

`pmtdeconv` is a waveform-processing toolkit that recovers precise particle hit
times and charges from smeared photomultiplier tube (PMT) signals. Because a
detector smears signals in a fixed, known way, deconvolution mathematically
reverses this effect — collapsing broad, overlapping voltage pulses back into
sharp, individual hits that ordinary raw integration cannot resolve.

It extracts these hits from digitized DRS4 / object-format / pulse-list ROOT
files using FFT deconvolution against a calibrated single-photoelectron (SPE)
template, with optional STFT or wavelet denoising. Dataset-agnostic: na22,
na22_3ch, cd109, milliqan, lanl_806, lanl_835, lanl_movedBar.

**The single deployable entry point is `pmtdeconv`** — a unified CLI organized
around output granularity (`hits` vs `area`) plus supporting template,
area-source, plotting, timewalk, info, and tune commands.

## Contents

- [Applicability domain — what this toolkit is valid for](#applicability-domain--what-this-toolkit-is-valid-for)
- [Key concepts](#key-concepts)
- [Project structure](#project-structure)
- [Install](#install)
  - [Running the tests](#running-the-tests)
  - [Getting the data](#getting-the-data)
  - [Pointing at your data](#pointing-at-your-data)
- [Quick start](#quick-start)
- [Pipeline overview](#pipeline-overview)
- [Detector parameters (Na22 lab — DRS4)](#detector-parameters-na22-lab--drs4)
- [Data layout](#data-layout)
- [Using pmtdeconv (unified CLI)](#using-pmtdeconv-unified-cli)
  - [Subcommand tree](#subcommand-tree)
  - [Which subcommand do I use?](#which-subcommand-do-i-use)
  - [Settings layering: dataset/card → config → CLI flag](#settings-layering-datasetcard--config--cli-flag)
  - [The shipped dataset cards](#the-shipped-dataset-cards)
  - [Onboarding a new detector with a card](#onboarding-a-new-detector-with-a-card)
  - [Comparing methods + tuning on real data](#comparing-methods--tuning-on-real-data)
  - [Saturation handling](#saturation-handling)
  - [Worked example 1 — Na22 timewalk (the E41473 benchmark)](#worked-example-1--na22-timewalk-the-e41473-benchmark)
  - [Worked example 2 — LANL pulse-list workflow (offline format)](#worked-example-2--lanl-pulse-list-workflow-offline-format)
  - [Worked example 3 — build/refit the Cd109 SPE template](#worked-example-3--buildrefit-the-cd109-spe-template)
  - [Worked example 4 — override defaults / dry-run](#worked-example-4--override-defaults--dry-run)
  - [Worked example 5 — regime-driven parameter sweeps](#worked-example-5--regime-driven-parameter-sweeps)
  - [Worked example 6 — debug a specific event](#worked-example-6--debug-a-specific-event)
  - [Dataset coverage](#dataset-coverage)
  - [Output directory layout](#output-directory-layout)
- [Known limitations & deferred follow-ups (0–20 PE re-scope, 2026-07)](#known-limitations--deferred-follow-ups-020-pe-re-scope-2026-07)
- [Code architecture (for contributors)](#code-architecture-for-contributors)
  - [Dispatch — pmtdeconv/cli.py](#dispatch--pmtdeconvclipy)
  - [Command module contract](#command-module-contract)
  - [Settings resolution — pmtdeconv/presets/__init__.py](#settings-resolution--pmtdeconvpresets__init__py)
  - [Settings → Detector → engine](#settings--detector--engine)
  - [Outputs and provenance](#outputs-and-provenance)
  - [Adding a command, or a flag](#adding-a-command-or-a-flag)
- [Tests](#tests)
- [Dependencies](#dependencies)

## Applicability domain — what this toolkit is valid for

SPE-template deconvolution rests on the linear model **m(t) = SPE ⊛ u(t)**,
which holds only while individual photoelectrons (PEs) and small clusters are
resolvable — the **0–20 PE regime** (Tang et al. 2411.07620; reproduced
in-repo by `pmtdeconv sim tang2024`). Above it, pulses merge into a continuum
and PMT/digitizer nonlinearity breaks the model; in our detectors the 20 PE
boundary roughly coincides with the 950 mV digitizer rail (SPE ≈ 15–26 mV
per tube). The toolkit therefore runs **two tracks**:

- **Track 1 (primary): 0–20 PE deconvolved charge + multi-hit separation.**
  Deconvolved charge is quantitatively meaningful here — the proven
  observable is **1-PE charge quantization σ/μ** (deconv beats raw by
  +33%/+51% on Cd109), *not* photopeak resolution. Workflow: `gain calibrate`
  (fit the 1-PE peak → per-branch `spe_charge` anchor) → `hits` (every row
  carries `npe_raw`/`in_regime`/`npe`; `--in-regime-only` applies the cut) →
  `select multihit` (0–20 PE multi-hit bunches) → `benchmark denoisers`
  (STFT-vs-wavelet under identical conditions) / `plot deconv-value`.
- **Saturated data: raw timing only.** A clipped waveform breaks the SPE-sum
  model — on Na22 Run 65212 its deconvolved charge runs *backwards* with
  energy (Spearman −0.86 vs raw area). Since 2026-10-23 `hits` and `area` drop
  clipped waveforms before deconvolution (reason `saturated`, even under
  `--no-vetoes`). Timewalk on saturated runs uses raw constant-fraction timing
  from `raw-area mqana` (`raw_timing_run<N>.csv`, `t_cfd`, `--cfd-frac`,
  default 0.2), which measured better than STFT there (1.56 vs 1.76 ns).

`in_regime` stays an include-by-default column (`hits --in-regime-only` cuts).

## Key concepts

**Why deconvolve at all?** Standard analysis reads timing off the sloping edge
of a raw pulse (which suffers from amplitude-dependent time-walk) and integrates
a fixed window for charge. By dividing the signal by the SPE footprint,
`pmtdeconv` collapses each pulse into a sharp peak. This inherently suppresses
time-walk and separates hits that arrive only nanoseconds apart.

| Term | Definition |
|---|---|
| **PMT / DRS4** | Photomultiplier tube (the light sensor) and the fast switched-capacitor digitizer (~2 GS/s) that samples the pulses into waveforms. |
| **SPE template** | The measured shape of a single-photoelectron pulse: the baseline "footprint" of one photon in a specific detector. |
| **Kernel** | The SPE template as used in FFT deconvolution — the detector response that is divided out to recover the input (distinct from the cosine low-pass *filter*). |
| **Time-walk** | A timing bias where larger pulses cross a voltage threshold earlier than small ones. |
| **STFT / wavelet denoising** | Techniques to clean the high-frequency noise that deconvolution naturally amplifies (time–frequency masking vs. frequency-band separation). |
| **Veto** | A per-pulse quality cut that drops invalid reconstructions (afterpulses, edge effects, negative charge). |
| **Golden test** | A regression test asserting a command's CSV output is byte-for-byte identical to a frozen reference, so a refactor cannot silently change physics results. |

## Project structure

```
bailloeulScripts/
├── pmtdeconv/                  # Self-contained CLI package. Everything
│   │                             needed to run the CLI lives in here.
│   ├── cli.py, __main__.py     # Argparse tree + two-stage dispatcher
│   ├── core/                   # THE MATH — single source of truth:
│   │                             kernels.py (SPE templates), filters.py
│   │                             (cosine low-pass), dsp.py (FFT deconv,
│   │                             STFT/wavelet denoise, baseline),
│   │                             peaks.py (windows, FWHM, CFD-50, vetoes)
│   ├── commands/               # One module per leaf subcommand
│   ├── pipeline/               # process_{area,hits}, pipeline (the shared
│   │                             per-event engine), detector, denoise,
│   │                             event_source, gain, template*, scoring,
│   │                             calibration (provenance + catalog base),
│   │                             diagnostic sink, timewalk scan
│   ├── plotting/               # scatter2d, marginals, energy, analyze,
│   │                             deconv_value, timewalk_gif_impl, style
│   ├── presets/                # datasets.py + cards.py + configs.py +
│   │                             tuned_configs.py + veto/template/gain
│   │                             catalogs
│   ├── sim/                    # Tang-2024 MC generator + validation
│   └── io/                     # manifest, schemas, root loaders, banner,
│                                 template + truth CSV I/O, output paths
│
├── tests/                      # pytest suite (`python -m pytest tests/`)
│   ├── fixtures/               # synth-ROOT builders, regime generators,
│   │                             golden_spec.py + bless_goldens.py
│   ├── test_pmtdeconv_cli.py   # CLI wiring + preset resolution
│   ├── test_golden_fixture.py  # byte-identical goldens (recipes: fixtures/golden_spec.py)
│   ├── test_flag_plumbing.py   # meta-test: no flag can be silently dropped
│   └── …                       # math, vetoes, gain, PE gate, tune, sim
│
├── outputs/                    # CLI outputs (gitignored)
├── pyproject.toml
├── README.md
└── .gitignore
```

The tracked tree is just **four entries** (`pmtdeconv/`, `tests/`,
`README.md`, `pyproject.toml`) plus the gitignored `outputs/`. `pmtdeconv/` is
**self-contained** — it imports nothing from sibling directories, only from
within its own tree and from the scientific Python stack (numpy, scipy, pandas,
matplotlib, uproot, awkward, PyWavelets), which `pip install -e .` pulls in
automatically (see [Install](#install)).

## Install

`pmtdeconv` needs the scientific Python stack (numpy, scipy, pandas, matplotlib,
uproot, awkward, PyWavelets). **One editable install pulls all of them in** —
you do not pip-install them one by one.

Work in a fresh environment to avoid clashes — create and activate one first
(pick conda **or** venv):

```bash
# --- conda ---
conda create -n pmtdeconv python=3.11 -y
conda activate pmtdeconv

# --- or venv (stdlib) ---
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Then install:

```bash
cd bailloeulScripts                # the dir containing pyproject.toml
pip install -e .                   # runtime deps (numpy/scipy/uproot/PyWavelets/…)
# to also run the test suite:
pip install -e ".[dev]"            # + pytest, pytest-cov
```

After this, `python -m pmtdeconv --help` (and the `pmtdeconv` console script)
work from anywhere. If you see `ModuleNotFoundError: No module named 'uproot'`
(or scipy, pandas, PyWavelets, …), you ran from source without installing — run
the editable install above.

### Running the tests

```bash
pip install -e ".[dev]"                      # once, for pytest
python tests/fixtures/bless_goldens.py       # once, on a fresh checkout
python -m pytest tests/ -q
```

Neither the synthetic ROOT fixtures nor the golden CSVs are committed — both
are gitignored, and no real data is needed for either. The ROOT files are large
and regenerable from fixed RNG seeds; the golden *bytes* depend on the local
numpy/scipy build, so a shared reference set would fail on other machines for
reasons unrelated to the code. `bless_goldens.py` builds any missing ROOT
fixture and regenerates every golden in one step, so each checkout owns its
reference set and byte-identity then guards refactors within it.

Skip the bless step and the ROOT-dependent tests **skip** (each prints what to
run); the math, CLI-wiring, and tune smoke tests still pass with no fixtures at
all. Every regeneration recipe lives as data in `tests/fixtures/golden_spec.py`,
and `test_golden_spec_matches_tests` asserts it covers exactly the goldens the
tests compare against — so the instructions cannot drift from the suite.

### Getting the data

Starting from raw DRS scope binaries (`Run<N>.dat`)? The companion package
`data_processing/` (`pmtdeconv-dataprep`, CLI `dataprep`) turns them into the
`Run<N>.root` and `flattened_Run<N>.root` files below. It needs PyROOT, so it
is installed separately; see `data_processing/README.md`.

The ROOT files are **not** in this repository — they are far too large, and
they are not yours to redistribute. They live in the group's internal share:

> **TODO (Leo):** internal location + access instructions go here.
> e.g. `smb://…` / Globus endpoint / `rsync -av <host>:<path> ~/pmt-data/`

Download once, then point `pmtdeconv` at wherever you put it:

```bash
export PMTDECONV_DATA_ROOT=~/pmt-data      # add to your ~/.zshrc or ~/.bashrc
python -m pmtdeconv info datasets          # confirms the resolved paths exist
```

`info datasets` prints the path each dataset resolved to, so you can check the
layout before a multi-minute run. A run whose file is missing stops with an
error listing every path it searched.

**Verifying you have the same bytes.** Analyses are only comparable if everyone
processed identical inputs. The share carries a `checksums.txt`; verify after
downloading:

```bash
cd ~/pmt-data && shasum -a 256 -c checksums.txt
```

Every run records the SHA256 of every input — card, template, data — in the
manifest (on by default; `--no-hash-inputs` to skip). Two people can compare `manifest.json` and prove they ran the same
data through the same settings — or find exactly where they diverged.


**Your own detector doesn't need any of this.** Point `--data-root` at your own
directory and describe it with a card (see *Onboarding a new detector* below) —
you do not need the group's Na22/Cd109 runs to use the toolkit, and the bundled
SPE templates describe one specific tube, so build your own.

### Pointing at your data

Dataset presets resolve files under a base **data root**. It is found in this
order: the `PMTDECONV_DATA_ROOT` environment variable → the first `data/`
directory found walking up from the package → a legacy fallback. So either:

```bash
export PMTDECONV_DATA_ROOT=/path/to/your/data    # used by every command + `info datasets`
# or, per-invocation, point straight at the directory holding the ROOT files:
python -m pmtdeconv hits --dataset na22 --runs 65213 --data-root /path/to/run/files
```

`info datasets` prints the resolved paths so you can confirm they exist before a
run. (`--data-root` overrides the *per-dataset* directory directly, which is the
simplest option when your files are not laid out as `<root>/Na22_lab/…`.)

**Per-run file layout is flexible.** Within the data root, each run's ROOT file
is located by searching the common layouts, so you do not have to match one
exact path:

- flattened waveforms (`hits`, `area`, `raw-area mqana`): `flattened_Run<N>.root`
  directly under the data root or one level down in a subdirectory.
- DRSprocessing area files (`raw-area drs`): `<root>/processed/Run<N>.root`,
  `<root>/Run<N>.root`, the same one level down, or any `*Run<N>*.root` match.
- You can always pin an exact file with `--file` (or `run --area-file`); it wins
  over the search. `--file` names one file, `--data-root` a directory to search.

**SPE templates are never invented.** A template CSV that is configured but
missing — a typo'd `--template`, a moved file — is a **hard error** in `hits`,
`area` and every `tune` sweep, not a silent substitution:

```
SPE template not found: /path/spe.csv
  (resolved from --template)

Deconvolution divides by the measured single-photoelectron response.
pmtdeconv will not silently substitute a generic kernel: the charges would
look plausible and be wrong.
```

Pass `--allow-generic-fallback` to accept the built-in generic kernel anyway
(timing stays usable; **charges are not calibrated**) — the manifest then
records `source: generic`, never the CSV that was not read. The `na22` and
`cd109` defaults ship inside the package at
`pmtdeconv/data/templates/`, so a fresh clone runs them without building a
template first; build your own per-tube template for any other detector.

**Describing an unregistered file.** A dataset preset or card normally supplies
the file's physical properties, but every one of them is also a flag, so you can
run against an ad-hoc ROOT file with no card at all:

```bash
python -m pmtdeconv hits --data-root /path/to/dir --runs 0 \
    --tree waveforms --branches d0_ch1 --dt-ns 0.5 --polarity -1 \
    --allow-generic-fallback
```

`--polarity -1` flips negative-going pulses (applied exactly once, in the event
source). Without it, negative-going data reconstructs **nothing** — every
waveform fails the `no_trigger` gate. `--allow-generic-fallback` substitutes a
built-in generic SPE shape when no template is configured: fine for exploring
filter settings, not for absolute charge.

If a requested run's file is found nowhere, the command **stops with an error**
that lists every path it searched and the three ways to fix it (`--data-root`,
`PMTDECONV_DATA_ROOT`, `--file`/`--area-file`) — instead of silently producing
empty output. The `run` recipes forward `--data-root` to every step, so a single
`--data-root` (or `PMTDECONV_DATA_ROOT`) covers the whole workflow and a missing
input fails fast at the step that needs it.

## Quick start

```bash
# 0. Install first (see above): cd bailloeulScripts && pip install -e .

# 1. Point at your data (dir holding flattened_Run*.root etc.):
export PMTDECONV_DATA_ROOT=/path/to/your/data    # or pass --data-root on each command
python -m pmtdeconv info datasets                # confirms the resolved paths exist

# 2. Discover the CLI surface:
python -m pmtdeconv --help                       # command tree
python -m pmtdeconv config show na22-timewalk    # inspect a recipe
python -m pmtdeconv <command> --help             # command flag reference

# 3. (Optional) Build the SPE (single-photoelectron) template.
#    This serves as the "baseline footprint" of one photon for a specific detector.
#    Needs Cd109 data. The na22/cd109 defaults ship with the package; a missing
#    template is an error (or pass --allow-generic-fallback for timing only).
python -m pmtdeconv template build --dataset cd109   # → outputs/spe_templates_perfile/spe_templates_perchannel.csv
#    The Na22 preset's default template lives elsewhere, so to USE the one you
#    just built, pass it explicitly: --template outputs/spe_templates_perfile/spe_templates_perchannel.csv

# 4. Run the canonical Na22 timewalk workflow. Its reference window
#    (--a-ref 100-110 nVs) is defined on DRSprocessing areas, so point it at
#    the processed file (MQAna narrow-window areas saturate near 34 nVs on
#    Run 65213 and select nothing — timewalk now says so instead of writing
#    an empty curve):
python -m pmtdeconv run na22-timewalk --area-file /path/to/processed/Run65213.root
python -m pmtdeconv results list

# Tune a denoiser to your detector — --write-presets saves date-stamped
# ConfigPresets and refreshes a stable `-latest-` alias per family:
python -m pmtdeconv tune wavelet --dataset na22 --write-presets
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --preset na22-wavelet-tuned-latest-level3

# Debug an individual event (diagnostic flag):
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --diagnostic-events 5 10 100
```

See the [Using `pmtdeconv` (unified CLI)](#using-pmtdeconv-unified-cli) section
below for worked examples covering every dataset format (flat / object /
offline) and the tune / diagnostic features.

## Pipeline overview

The deconvolution pipeline follows this sequence:

1. **Baseline estimation** — finds the quiet pre-pulse level and noise floor so downstream metrics are measured from true zero. *(adaptive sliding window, MAD-based noise, log-σ acceptance; `core.dsp.baseline_noise_bestwindow`)*
2. **FFT deconvolution** — reverses the PMT's time-smearing, turning broad pulses into sharp spikes. A cosine low-pass filter suppresses high-frequency electronic fuzz. *(`u_rec = IFFT(FFT(y) · filt / FFT(SPE))`)*
3. **Denoising** — deconvolution mathematically amplifies high-frequency noise, which is mitigated via:
   - **STFT** — time–frequency masking (`core.dsp.stft_denoise`). **The `hits` default since 2026-10.**
   - **Wavelet** — frequency-band separation and thresholding (stationary SWT + MAD; `core.dsp.wavelet_denoise`). *Experimental* for `hits` since 2026-10; still the `area` denoiser.

   The 2026-10 denoiser study compared the two (and no denoiser) on the Tang MC
   and on simulations built from real MilliQan and Cd109 noise, each method
   tuned on one sample and scored on a fresh one: STFT reconstructed charge
   best on every test set. On real data the cosine cutoff mattered far more
   than the denoiser (Na22 Run 66810 timing 5.4 ns at 50 MHz vs 10.4 ns at
   450 MHz), and on the Run 65213 timewalk benchmark wavelet left ~2.5 ns
   less residual walk at the same resolution. Details:
   `outputs/denoiser_study_2026-10/` and the vault note.
4. **Peak finding** — locates sharpened hits above an SNR threshold (`scipy.signal.find_peaks`). *(peak-finding scripts only)*
5. **Vetoes** — drop obviously-bad reconstructions: saturated (clipped) waveforms, no-trigger, out-of-time (peak outside the signal window), afterpulse, charge purity *(peak-finding scripts only)*
6. **Charge integration** — sum the area under each hit: trough-walked window per peak (peak-finding) or fixed window (waveform summary)

### Three integration modes

| Command | Per trigger | Peak-find? | Area window |
|---|---|---|---|
| `hits` (STFT default; `--denoiser wavelet` experimental, or `none`) | All peaks above SNR | ✓ | Trough-walked per peak |
| `area` | One row per channel | ✗ | Fixed window (whole-waveform deconv) |
| `raw-area mqana` | One row per channel | ✗ | Fixed window, **no deconv** |

## Detector parameters (Na22 lab — DRS4)

| Parameter | Value |
|---|---|
| Sampling rate | 1.98 GS/s (`dt_ns = 0.5051`) |
| Waveform length | 1024 samples ≈ 517 ns |
| Pulse polarity | Positive-going |
| ADC saturation | ~960 mV (rail) |
| Cosine cutoff | per dataset: na22 75 MHz (below its fs/32 DRS4 lines at 61.9–396 MHz), cd109 50 MHz, MilliQan/LANL 100 MHz (nulls the 100 MHz line of the 50 MHz clock comb); 100 MHz when nothing sets one |
| Denoiser — `hits` | STFT (default since 2026-10); `--denoiser wavelet` is experimental |
| Denoiser — `area` | wavelet `sym8` L5 hard, `thresh_k=2.0` |

Templates are resolved per-channel from the dataset's configured SPE CSV — for
na22 and cd109 this is the calibrated Cd109 stacked template shipped in
`pmtdeconv/data/templates/`. Build your own with `template build --dataset
<name>` (it aggregates per channel automatically), or supply one with
`--template`.

## Data layout

Scripts derive paths dynamically:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # bailloeulScripts/
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DATA_DIR     = PROJECT_ROOT.parent.parent.parent / "data"
```

Expected ROOT files:

```
LANL/data/Na22_lab/
├── flattened_Run65163.root
├── flattened_Run65166.root
└── flattened_Run65213.root
```

## Using `pmtdeconv` (unified CLI)

`pmtdeconv` is a single CLI with git-style subcommands; each subcommand calls
the canonical `pmtdeconv/pipeline/` functions. Every command drops a
`manifest.json` recording the resolved settings, provenance, inputs/outputs,
git SHA, and library versions, plus a `run.log` transcript of everything the
run printed (resolved paths, template lines, warnings, veto rates). Send
someone an output directory and it describes itself: `manifest.json` for what
was configured, `run.log` for what it said. Byte-identical regression is guarded by
deterministic synthetic-ROOT fixtures in four formats (flat, object, DRS,
offline; listed in `tests/fixtures/golden_spec.py`), blessed per checkout — see
[Tests](#tests)).

### Subcommand tree

```
pmtdeconv
├── run {na22-timewalk, na22-energy, build-template} complete workflows
├── dataset  {init, lint}                          scaffold + validate a dataset CARD
├── template {build, aggregate}                 SPE template stacking
├── hits                                       per-hit deconvolution
├── area                                       whole-waveform deconvolution summary
├── raw-area {mqana, drs, offline}              raw-waveform area sources
├── validate                                   comparative FoM report (deconv vs raw vs DRS)
├── gain     {calibrate}                        fit the 1-PE charge peak → spe_charge (PE anchor)
├── select   {multihit}                         0–20 PE multi-hit bunches → event_list.csv
├── sim      {tang2024}                         Tang 2411.07620 MC (0–20 PE mix + two-hit scan)
├── benchmark {denoisers}                       STFT-vs-wavelet verdict (real pipeline, truth metrics)
├── timewalk {run, gif}                         Δt scan + animation (raw CFD `t_cfd` on saturated runs)
├── plot     {marginals, energy, analyze, deconv-value}  diagnostic + deliverable plots
├── info     {datasets, template, offline-file}  inspect a detector or a file
├── config   {show, resolve}                     settings (`show` with no name lists presets)
├── results  {list, show, path, index}           outputs (`index` maintains artifact_index.json)
├── tune     {cosine, wavelet, stft, fom}        parameter sweeps
└── vetoes   {calibrate}                         data-driven veto profile
```

Key flags of the 0–20 PE re-scope (2026-07): `--spe-charge BR=PVS` /
card `spe_charge:` (the PE anchor), `--pe-band LO HI` (default 0 20),
`--in-regime-only`, `--veto-profile NAME` (identical thresholds on any
denoiser — builtins `wavelet_hits`, `stft_hits`, `benchmark_hits`; file-backed
profiles shadow for explicit names), `--entry-list CSV` (re-process selected
bunches), `--no-fft-pad` on `hits` / `area` (reproduce the circular-deconvolution
bytes from before 2026-07-28 for hits, 2026-10-21 for area).

**`--dataset` takes a registry name OR a path to a YAML card** (or a directory
with `dataset.yaml`). A new detector is onboarded with a card — no Python edit —
and a card under `<data_root>/calibrations/datasets/<name>.yaml` *shadows* the
in-code seed of the same name (flagged as `[card]` in the run banner + manifest).
See **Onboarding a new detector** below.

**Why tune?** Because deconvolution amplifies noise, the filter cutoffs are a
balancing act: cut too much and you blur the pulse back out; cut too little and
you get false peaks. `tune` automates finding that sweet spot instead of
guessing by hand.

`tune` runs deterministic sweeps against the seeded
synthetic regime fixtures (`level1`..`level5`, from low SNR to
saturated) and scores each candidate config against ground truth —
charge bias (drift vs the dataset preset's reference config), per-event
scatter, timing σ, and efficiency. Winners are persisted as
auto-generated `ConfigPreset` entries that `area`/`hits`
can pick up via `--preset` with no runtime branching. (`tune fom` does the same
sweep on **real** data against your card's declared figure of merit — see below.)

### Which subcommand do I use?

| I want… | Use |
|---------|-----|
| run the standard Na22 workflow | `run na22-timewalk` |
| **onboard a new detector** (no Python edit) | `dataset init <name>` → edit the card → `dataset lint <card>` |
| build / refit SPE templates | `template build` (aggregates per channel; `template aggregate` re-runs that step) |
| per-trigger whole-waveform deconvolution **summary** | `area` |
| per-event peak-finding **hits with timing** | `hits` (STFT; `--denoiser wavelet` is experimental) |
| MQAna-style **narrow-window raw area** from waveforms | `raw-area mqana` |
| precomputed `area_<board>_<ch>` from a **DRSprocessing flat tree** | `raw-area drs` |
| LANL **pulse-list** (`*_offline.root`) → area + per-channel timing | `raw-area offline` |
| Δt scan against any (area, pulse) CSV pair | `timewalk run` (CSV-only) or `timewalk gif` (animated) |
| ch1-vs-ch2 area heatmap + 1D marginals | `plot marginals` |
| four-method energy spectra (raw / FFT / STFT / wavelet) | `plot energy` |
| charge/time/height diagnostic plots from a hits CSV | `plot analyze --hits-csv <file>` |
| inspect a detector, template, or data file | `info {datasets, template, offline-file}` |
| list every named preset / show one | `config show` / `config show <name>` |
| find prior outputs / maintain the index | `results list` / `results index` |
| inspect resolved settings | `config show`, `config resolve` |
| find a previous output | `results list`, `results show`, `results path` |
| **compare deconv vs raw vs DRS on my FoM** | `validate` (per-channel σ/μ table; report, not a gate) |
| **find best cosine cutoff per regime** (synthetic truth) | `tune cosine` (use `--denoiser none` to isolate the cosine response) |
| **find best cutoff on REAL data vs my FoM** | `tune fom` (+ anti-cheat columns; `--auto-select` to pick) |
| **find best wavelet config per regime** | `tune wavelet` (use `--cosine-cutoff-mhz <N>` to pick the fixed cutoff) |
| **find best STFT config per regime** | `tune stft` |
| **derive a data-driven veto profile** | `vetoes calibrate --write-profile <name>` (persists to the calibrations catalog) |
| **dump intermediates for one or more events** (FFT, denoised, peaks, truth match, PDF) | `hits --diagnostic-events <N> [N …]` |

### Settings layering: dataset/card → config → CLI flag

```text
dataset (seed or CARD)  <  config overrides  <  explicit CLI flag
```

The two flags name two different things and are not interchangeable:
**`--dataset` is the detector** (a registry name or a **card path/dir**), and
**`--preset` is the recipe** (analysis overrides). Passing a preset name to
`--dataset` is an error that points at `--preset`. Combine them to run one
detector's recipe on another detector:

```bash
python -m pmtdeconv hits --dataset lanl_806 --preset na22-timewalk
```

`--preset X` on its own falls back to X's own base dataset. A card can carry
analysis overrides in a `settings:` block, so a single card expresses
everything a dataset + config preset used to. Every run prints a
**provenance banner** (and writes it to the manifest) showing each resolved knob
and *where it came from* (`[card]`/`[dataset]`/`[config]`/`[cli]`):

```text
┌─ pmtdeconv area
│  dataset       na22            [card]
│  cutoff [MHz]  123.0           [card]
│  denoiser      wavelet         [config]
│  fom           photopeak_sigma_over_mu   [card]
└─ pmtdeconv 0.x  git abc12345@leo-waveform-reco
```

```bash
python -m pmtdeconv info datasets        # list datasets + applicable subcommands
python -m pmtdeconv config show na22-timewalk
python -m pmtdeconv config resolve --preset na22-timewalk
python -m pmtdeconv <command> --help     # command flag reference
```

### The shipped dataset cards

Every in-code dataset also ships as a YAML card under
`pmtdeconv/data/datasets/` — the file-backed twin of the Python registry,
generated by `dataset init --from-seed` and pinned by
`tests/test_seed_cards.py` so the two cannot drift.

```bash
ls pmtdeconv/data/datasets/        # na22, na22_3ch, cd109, milliqan, lanl_*
cp pmtdeconv/data/datasets/na22.yaml mydet.yaml   # start from a working example
```

They are machine-independent: data paths use `${PMTDECONV_DATA_ROOT}` (resolved
the same way the in-code presets resolve it) and the bundled SPE templates are
referenced relative to the card, so a card written on one machine works on
another with no edit.

The migration is **additive** — the in-code seeds remain the default
resolution, so nothing about existing runs changes. The cards exist so a new
detector is a YAML file, not a Python edit, and so a card can carry `meta:`
provenance the registry cannot express.

A run records the card it resolved in the manifest `inputs` (and hashes it,
like every input), so two runs from two revisions of the same card are
distinguishable in the record.

### Onboarding a new detector with a card

A dataset **card** is a YAML file describing a detector — the file-backed twin
of the in-code presets. No Python edit is needed:

```bash
python -m pmtdeconv dataset init mydet --out mydet.yaml   # scaffold (or --base na22)
$EDITOR mydet.yaml                                         # fill in the specifics
python -m pmtdeconv dataset lint mydet.yaml               # validate fields + settings
```

A card describes the detector; `meta:` is a free-form block for whatever a
collaborator needs to know about it. pmtdeconv never interprets `meta:` — it
copies it into every manifest, so a shared output directory answers "what
detector is this, and who do I ask?":

```yaml
name: mydet
dt_ns: 0.5
polarity: -1
notes: "Bench setup, Sep 2026. Timing studies only."

meta:
  pmt_serial:   HPK-ZK1234
  source:       Na22
  scintillator: LYSO + CsI
  hv_volts:     1450
  elog:         E41537
  contact:      you@lab.gov
  taken:        2026-09-01
```

Keys pmtdeconv *does* interpret (`polarity`, `dt_ns`, `pmt_type`, `hv_volts`,
`fom`, `spe_charge`, `pe_band`, …) belong at the top level. A top-level key it
does not recognize is reported rather than ignored, both by `dataset lint` and
when the card is loaded for a run:

```
WARN  [mydet] unknown key 'polarty' is ignored — did you mean 'polarity'?
python -m pmtdeconv area --dataset ./mydet.yaml --runs 42 # run it (path form)
```

Drop the card at `<data_root>/calibrations/datasets/mydet.yaml` and it resolves
by **name** (`--dataset mydet`) and *shadows* a same-named seed. Card fields:

```yaml
name: mydet
base: na22                 # optional: inherit a seed/card, override only what differs
data_dir: .                # dir with the ROOT files (default: the card's dir)
file_glob: "flattened_Run*.root"
format: flat               # flat | object
tree: waveforms
branches: [d0_ch1, d0_ch2] # explicit list, or "auto" to discover d0_ch*
dt_ns: 0.5051              # sample spacing [ns], or "auto" (object format)
polarity: 1                # +1 positive-going, -1 negative-going
cosine_cutoff_mhz: 100
spe_band: [5, 30]          # SPE amplitude band [lo, hi] in mV (template build)
fom: photopeak_sigma_over_mu          # figure of merit validate/tune report
template: spe_template.csv            # one shared SPE template, OR ...
templates:                            # ... per-tube templates, one CSV per channel
  d0_ch1: tubes/ch1_spe.csv
  d0_ch2: tubes/ch2_spe.csv
settings:                  # analysis overrides (subsumes a config preset)
  denoiser: wavelet
  wthresh_k: 2.0
  no_vetoes: false
```

File-backed catalogs live under `<data_root>/calibrations/` (`datasets/` for
cards, `veto_profiles/` for versioned veto profiles written by
`vetoes calibrate --write-profile`).

### Comparing methods + tuning on real data

```bash
# Compare deconv vs raw vs DRS on the card's FoM (a report, never a gate):
python -m pmtdeconv validate --dataset ./mydet.yaml \
    --deconv-csv out/deconv/areas_run42.csv \
    --raw-csv    out/raw/area_mqana_run42.csv

# Sweep the cosine cutoff on REAL data vs the FoM, with anti-cheat columns
# (efficiency + charge drift) so a "win" by dropping events / shifting scale
# is visible; --auto-select opts in to picking a winner per channel:
python -m pmtdeconv tune fom --dataset ./mydet.yaml --runs 42 \
    --cutoff-range 50 300 25 --window 4000 6000 --auto-select
```

### Saturation handling

Saturation is detected from the **flat plateau** at the rail (universal — no
per-detector threshold). `hits` and `area` **drop** a clipped waveform before
deconvolution (reason `saturated`; no opt-out), because its deconvolved charge
is not physical. `raw-area mqana` still times it from the raw rising edge
(`raw_timing_run<N>.csv`, column `t_cfd`), which is what timewalk uses on
saturated runs.

### Worked example 1 — Na22 timewalk (the E41473 benchmark)

```bash
python -m pmtdeconv run na22-timewalk --area-file /path/to/processed/Run65213.root

# Or run the individual steps (no deconvolution: Run 65213 is ~86% saturated):
python -m pmtdeconv raw-area mqana --dataset na22 --runs 65213 --out-dir tw/timing
python -m pmtdeconv raw-area drs --dataset na22 --runs 65213 --file /path/to/processed/Run65213.root --out-dir tw/area
python -m pmtdeconv timewalk run --dataset na22 --runs 65213 \
    --area-csv tw/area/area_drs_run65213.csv \
    --hits-csv tw/timing/raw_timing_run65213.csv --time-col t_cfd
```

Timing estimators on Run 65213 (2026-10-21; first 30k events, DRS area axis,
walk-corrected robust σ on the 5262 events every method kept, 68% bootstrap):

| Timing | σ |
|---|---|
| raw CFD-20 (`t_cfd`, the recipe) | 1.56 (1.53–1.59) ns |
| STFT `t_half`, cosine 50 MHz | 1.60 ns |
| STFT `t_half`, cosine 450 MHz (old recipe) | 1.76 (1.72–1.78) ns |
| raw CFD-50 | 1.84 ns |

### Worked example 2 — LANL pulse-list workflow (offline format)

```bash
# Read the LANL *_offline.root pulse list → area + pulse-timing CSVs in one shot
python -m pmtdeconv raw-area offline --dataset lanl_movedBar --runs 868

# Timewalk auto-resolves both CSVs from the offline output dir
python -m pmtdeconv timewalk gif --dataset lanl_movedBar --runs 868 \
    --ref-branch d0_ch04 --scan-branch d0_ch05 \
    --a-ref 1000 10000 --a-scan-start 0 --a-scan-stop 10000
```

`raw-area offline` writes two CSVs per run:

| file | schema | use |
|------|--------|-----|
| `area_offline_run<N>.csv` | AREA_COLUMNS | total charge per channel (sum over pulses); feed timewalk's `--area-csv` |
| `offline_hit_results_run<N>.csv` | HIT_RESULTS_COLUMNS | primary-pulse (`ipulse==0`) with `t_peak` from `--time-field` (default `timeFit_module_calibrated`); feed timewalk's `--hit-csv` |

To inspect a pulse-list file before running: `pmtdeconv info offline-file <path>`.

### Worked example 3 — build/refit the Cd109 SPE template

```bash
python -m pmtdeconv template build --dataset cd109   # per-file fits, then per-channel average
```

It writes `outputs/spe_templates_perfile/spe_templates_perchannel.csv` (plus
the per-file CSV it averages; `--no-aggregate` stops there, and `template
aggregate` re-runs the averaging). The `cd109`/`na22` defaults use the template
shipped in `pmtdeconv/data/templates/`; to use yours, pass it with `--template`
or set `template:` in your dataset card.

### Worked example 4 — override defaults / dry-run

```bash
# Tighter cosine cutoff, no preset
python -m pmtdeconv area --dataset na22 --runs 65213 \
    --cosine-cutoff-mhz 250

# Resolve flags + write manifest, skip ROOT I/O
python -m pmtdeconv hits --dataset na22 --runs 65213 --dry-run
```

The `manifest.json` from a dry-run is enough to confirm the resolved
settings before committing to a multi-minute run.

### Worked example 5 — regime-driven parameter sweeps

The goal: find the filter settings that best recover the true signal. Because we
need a *known* answer to score against, `tune` works on synthetic data where the
truth is exact. A **regime** is one (pulse-amplitude, noise, clipping) operating
point — `level1`..`level5` climb a brightness ladder from small, low-SNR pulses
through the 511 keV-like default (`level3`) up to saturated/clipped pulses
(`level5`) — so a winning config is chosen per condition.

`tune` builds seeded synthetic fixtures (`tests/fixtures/regime/level<N>/`)
at the dataset's `dt_ns`, runs the pipeline at each candidate config,
and scores against the ground-truth CSV that ships next to each
fixture. Three leaves; each writes `recipe.json`,
`sweep_summary.csv`, per-regime `sweep_curves_<regime>.png`, and a
manifest.

**`tune cosine` uses ε-constraint selection** (charge first, width
second — the methodology behind Tang et al. 2411.07620 Fig. 8). The grid
is fine and Nyquist-aware (dense 20–300 MHz @10, coarse above @50, capped
at 0.9·Nyquist for the dataset's `dt_ns`). For each regime it measures the
reconstructed-pulse **FWHM** and the **charge drift** vs the reference
cutoff, then picks the **smallest FWHM** among cutoffs that satisfy *both*
hard constraints:

1. `|charge drift| ≤ --charge-tol` (default 2%; linearity), and
2. `efficiency ≥ --min-efficiency × reference efficiency` (default 0.9;
   keeps the choice below the noise-amplification cliff, where the deconv
   goes unstable and the charge metric — measured on a small veto-selected
   subset — stops being meaningful).

It also emits `width_charge_vs_cutoff_<regime>.png` (FWHM-over-drift, the
Fig.-8-style read-off). Without the efficiency floor, drift alone is too
weak a gate and the sweep rides up into the noise cliff.

```bash
# 1. Sweep cosine cutoff with denoise off (isolates the filter effect,
#    matching the paper's Fig. 8). ε-constraint: min FWHM s.t. charge OK.
python -m pmtdeconv tune cosine --dataset na22 --denoiser none

# 1b. Daya-Bay-strict charge linearity (1%) + a tighter efficiency floor.
python -m pmtdeconv tune cosine --dataset na22 --denoiser none \
    --charge-tol 0.01 --min-efficiency 0.95

# 2. Sweep wavelet config (96 cells × 5 regimes by default).
#    Cosine held at the dataset preset's cutoff; override with --cosine-cutoff-mhz.
python -m pmtdeconv tune wavelet --dataset na22

# 3. Sweep STFT (27 cells × 5 regimes).
python -m pmtdeconv tune stft --dataset na22

# 4. Same as (1) but also writes each regime's winner to tuned_configs.py
#    as an auto-generated ConfigPreset; idempotent across re-runs.
python -m pmtdeconv tune cosine --dataset na22 --denoiser none --write-presets

# 5. Use a written preset (the stable `-latest-` alias tracks the newest winner).
python -m pmtdeconv area --dataset na22 --runs 65213 \
    --preset na22-cosine-tuned-latest-level3

# 6. Narrow grid for fast iteration:
python -m pmtdeconv tune wavelet --dataset na22 --regimes level3 \
    --wavelets sym8 --wlevels 5 --wmodes hard soft --wks 1.0 2.0 \
    --n-events 50
```

Cross-dataset works out of the box — `tune cosine --dataset cd109`
rebuilds the regime fixture at `dt=1.432 ns` and uses cd109's preset
cutoff (50 MHz) as the drift reference. The fixture is cached at
`tests/fixtures/regime/level<N>/.dt_ns` so subsequent runs on the same
dataset reuse it.

When the dataset's reference config is degenerate in a regime (e.g.,
the manually-tuned `wavelet=hard/k=1.0` annihilates the deconv signal
at level1's low SNR), the drift falls back to the per-regime median
ratio with a stdout warning — drift always has a meaningful value,
never `NaN`.

### Worked example 6 — debug a specific event

```bash
# Dump intermediates for events 5, 10, 100 of a real run.
# Writes outputs/.../diagnostic_event<N>__<branch>/ with:
#   raw_waveform.npy, y_sub.npy, cosine_filter.npy, spe_kernel.npy,
#   deconv_pre_denoise.npy, deconv_post_denoise.npy, found_peaks.npy,
#   peaks.json, truth_match.json (if a .truth.csv sits beside the ROOT),
#   diagnostic_event<N>__<branch>.pdf (6-panel summary).
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --diagnostic-events 5 10 100

# Works on the STFT denoiser path too.
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --denoiser stft --diagnostic-events 5
```

Non-diagnostic runs short-circuit the sink at ~1% overhead so byte-
identity of all existing golden CSVs is preserved.

### Dataset coverage

| Dataset | Format | Subcommands |
|---------|--------|-------------|
| `na22`, `na22_3ch`, `cd109`, `milliqan`, `lanl_806` | flat (`flattened_*.root`) | `template`, `hits`, `area`, `raw-area {mqana, drs}`, `timewalk`, `plot` |
| `lanl_835`, `lanl_movedBar` | object (`*_triple.root`) + offline pulse list (`*_triple_offline.root`) | `template`, `hits`, `area`, `raw-area {mqana, offline}`, `timewalk`, `plot` |

`raw-area drs` is N/A on object-format datasets (LANL files have no precomputed
`area_<board>_<ch>` branches) — use `raw-area offline` instead. The CLI hard-fails
with the right pointer if you forget.

### Output directory layout

```
outputs/
├── deconv_area_<dataset>[__<preset>][<suffix>]/
│     ├── areas_run<RUN>.csv                  AREA_COLUMNS
│     ├── deconv_*_ch1_vs_ch2_run<RUN>.pdf
│     └── manifest.json
├── raw_area_mqana_<dataset>[…]/area_mqana_run<RUN>.csv     AREA_COLUMNS
├── raw_area_drs_<dataset>[…]/area_drs_run<RUN>.csv         AREA_COLUMNS
├── raw_area_offline_<dataset>[…]/
│     ├── area_offline_run<RUN>.csv                         AREA_COLUMNS
│     └── offline_hit_results_run<RUN>.csv                  HIT_RESULTS_COLUMNS
├── deconv_pulse_<dataset>_lab_cosine_{wavelet,advanced}[…]/
│     ├── hits.csv                                          HIT_RESULTS_COLUMNS
│     ├── all_results.csv
│     └── manifest.json
├── timewalk[_gif]_<dataset>_run<RUN>[…]/
│     ├── timewalk_summary_run<RUN>.csv       MQAna timewalk schema
│     ├── timewalk_curve_run<RUN>.png
│     └── manifest.json
└── tune_{cosine,wavelet,stft}_<dataset>[…]/
      ├── recipe.json                         per-regime winner + metrics
      ├── sweep_summary.csv                   every (regime, config) row
      ├── sweep_curves_level<N>.png           scatter/drift/efficiency vs axis
      └── manifest.json
└── artifact_index.json                         optional `results index --write`
```

The regime fixtures themselves live under
`tests/fixtures/regime/level<N>/` and are **not committed** — they're
deterministically reproducible from the seeded `make_synth_root.py`
and get built on demand by `tune`. Each fixture has a sibling
`flattened_Run0.truth.csv` capturing the synthetic pulses' true
amplitude, center, charge, σ, and saturation flag for use by the
scoring code (`pmtdeconv/pipeline/scoring.py`) and the diagnostic
sink (`pmtdeconv/pipeline/diagnostic.py`).

All CSV column schemas live in `pmtdeconv/io/schemas.py`; the manifest's
`csv_schema_versions` field records which schema version was used per output.

## Known limitations & deferred follow-ups (0–20 PE re-scope, 2026-07)

Documented so they don't silently evaporate; each is deliberate, not an
oversight. The denoiser internals are *measured by the benchmark* rather than
silently trusted:

- **Denoiser caveats (instruments under test — unchanged on purpose):** the
  SWT universal threshold √(2·ln N) is mis-calibrated for correlated SWT
  coefficients (effectively over-aggressive); our STFT uses a global-median
  noise estimate (record-length dependent) and hard-zeroing that deletes
  sub-threshold few-PE pulses; the cosine filter's derivative discontinuity
  at the cutoff rings (~7% pre-pulse side-lobe — `FIRST_PULSE_MIN_FRAC`).
- **Tang-2024 reproduction**: the paper's own method runs as two
  `benchmark denoisers` arms — `tang2024` (their FFT+STFT chain) and
  `tang2024-fft` (plain FFT) — scored against the same MC truth as our arms,
  with the published fig-12/14 values reported beside them in `verdict.json`:

  ```bash
  pmtdeconv benchmark denoisers --suite tang2024 --arms tang2024 tang2024-fft
  ```

  The paper publishes no code (checked arXiv, JINST 20 P03019, the group's
  2026 follow-up, GitHub/Zenodo) and leaves three recognition constants
  unspecified. **Those were fitted to the published fig-12 curve**, so
  agreement with that curve is a consistency check, not independent
  validation: FFT arm 0/0/16/80/98/97% vs published 0/0/14/80/99/99% at
  3/4/5/6/8/10 ns; FFT+STFT 86/92/94% vs 84/92/96% at 5/6/8 ns; fig-14 charge
  mean 0.995/1.009 vs 1.004, nonlinearity 1.3/1.9% vs ~1%, σ 0.101 vs 0.076.

  **Open: the 3–4 ns corner.** Even with constants fitted toward it, the
  enhanced chain splits 3/40% of 3/4 ns pairs against the published 58/72%.
  The cause is *not* established. Two explanations fit: the published
  algorithm description is incomplete, or our MC is harder than theirs — its
  charge σ is 0.101 vs their 0.076, and every simulated hit carries a 7 ns
  cable-reflection train. The discriminating experiment is to soften the MC
  until σ matches 0.076 and re-measure 3 ns efficiency. Calibrations are
  documented in `pmtdeconv/sim/tang2024_paper.py`.

  Separately, our own arms currently score near zero at 3–6 ns on this MC
  because the veto chain (`no_raw_peak`, `afterpulse_veto`) deletes the
  second hit that the deconvolution *does* find (≥2 peaks in 94% of 5 ns
  events). The head-to-head therefore measures our veto configuration there,
  not our deconvolution.

  The `--stft-threshold-mode fixed` knob on the PIPELINE (renamed from
  `absolute` on 2026-09-18, with `--stft-abs-threshold` →
  `--stft-fixed-threshold`) remains tang-calibrated only; it is what the
  `stft-fixed-tang2024` benchmark arm runs.
- **Deconvolution regularization**: `eps` in the FFT division is a
  divide-by-zero guard, not Tikhonov/Wiener; the cosine filter is the
  de-facto regularizer.
- Baseline gate log-σ anchors are absolute mV constants (profile knob
  `baseline_sigma_scale` scales them; per-detector anchors deferred);
  "ADC"-named fields hold mV; AreaMode carries a second, looser baseline
  gate than the hits profiles (unification deferred).
- `na22_3ch` multi-hit work needs a single-peak sampled kernel + cutoff
  revisit (the production ch2 envelope kernel is timing-track only).
- In-regime re-tuned winners land as opt-in named presets, not new dataset
  defaults; `tune fom` on photopeaks is DEPRECATED as a deconv-charge
  validation target (out of domain).
- `tune wavelet` / `tune stft` still build their own arg namespaces (bypass
  Detector).
- **Open: gates for obviously flawed waveforms (2026-10-24).** The event
  gates (`baseline_quality`, `baseline_failed`) only inspect the first
  200–600 ns, so an electronic fault that starts after the trigger passes.
  Probed with synthetic faults: a post-trigger drift, a +200 mV step and a
  100 mV square block each kept every event with 6–13 fake pulses. Two
  prototyped gates separate them:
  - **tail shift**: |median of the last 60 ns − pedestal|. Drift/step/sine
    faults give 170–270 mV; real Cd109 65267 loses 0.3% at a 50 mV cut.
  - **bipolarity**: −min/max of the baseline-subtracted record. Oscillation,
    ringing and noise give 0.9–1.1, unipolar PMT pulses ~0; Cd109 loses
    0.6–1.4% at 0.8.

  A record-wide sample-noise ratio does *not* work (oscillation is smooth
  sample to sample), and a one-sided square block needs a shape test against
  the SPE template. Before choosing cuts, inspect the real waveforms each gate
  rejects. The ~225 ns slow oscillation on 36% of Cd109 CH2 events is the
  first real target.

## Code architecture (for contributors)

How a command actually runs, end to end. The math itself lives in
`pmtdeconv/core/` — the single source of truth; everything below is the
plumbing that reaches it.

### Dispatch — `pmtdeconv/cli.py`

The whole command tree is one dict, `SUBCOMMANDS`, mapping *verb → leaf →
module name* under `pmtdeconv/commands/`. Dispatch is deliberately
**two-stage**:

- **Stage 1** registers only verb/leaf *names* plus a one-line help string from
  the static `LEAF_HELP` table. It imports no command module, so
  `pmtdeconv --help` works even without uproot/matplotlib installed.
- **Stage 2** imports just the selected leaf, builds its real parser
  (`add_common_args` + the module's `add_arguments`), re-parses the leftover
  argv, and calls `run(args)`.

`info`, `config`, and `results` build eagerly (they own their own sub-parser
trees). A missing scientific-stack import is caught in `main()` and turned into
a `pip install -e .` hint instead of a traceback.

### Command module contract

Every module in `commands/` exposes exactly two functions — no business logic
lives in `cli.py`:

```python
def add_arguments(parser): ...   # register this command's flags
def run(args) -> int: ...        # execute from the parsed Namespace
```

Flags default to `None` so the resolver can distinguish "user passed it" from
"user left it alone".

### Settings resolution — `pmtdeconv/presets/__init__.py`

`run()` starts with `settings, provenance = resolve_settings(args)`, layering
four sources, lowest to highest:

```
Settings default  <  dataset (registry seed or YAML card)  <  config preset  <  CLI flag
```

`Settings` is one flat dataclass holding everything resolvable. CLI flags are
routed through the `CLI_TO_SETTINGS` map, because a few argparse names differ
from their Settings field (`branches` → `branch_pattern`, `manifest` →
`manifest_path`, `spe_charge` → `spe_charge_pvs`). Every assignment records
*which layer won* into `provenance` — that is exactly what `config resolve`
prints and what the run banner and manifest carry.

### Settings → Detector → engine

```
commands/hits.py::run
  └─ Detector.from_settings(settings)          # ONE resolution point:
     │                                         #   template + veto profile + gain
     └─ process_hits.process_run{,_object,_stft}      (format-specific readers)
          └─ event_source.iter_flat_run  →  EventRecord stream
             │                              (polarity applied exactly once, here)
             └─ pipeline.run_hits  →  _process_event per waveform
                  └─ core/{kernels,filters,dsp,peaks}
```

`Detector` is a typed dataclass whose field names mirror what the pipeline
reads off its `args` parameter — it *is* the `args` the engine sees. It
replaced an untyped `SimpleNamespace` round-trip that repeatedly caused
silently-dropped flags.

`_process_event` (in `pipeline/pipeline.py`) is the single shared body behind
all three readers; the reader only chooses the `HitsProfile` (wavelet vs STFT
constants). Per waveform: saturation flag → raw-nPE regime flag → baseline →
cosine-filtered FFT deconvolution → denoise → `find_peaks` → peak ordering →
trough-walked integration windows → vetoes → one row dict per peak. `area`
mirrors this through `run_area` / `AreaMode` in the same module.

Rows are reindexed against `io/schemas.py` (the single source of column truth)
and written as three tables: `hits.csv` (successes),
`all_results.csv` (everything, with a `reason`), and
`deconv_negative_charge.csv`.

### Outputs and provenance

Output directories are computed by `io/output_paths.py` and always carry the
preset name and `--out-suffix`. Downstream commands find prior results by
reconstructing the same path (`resolve_area_csv` / `resolve_pulse_csv`), which
is why `timewalk run --dataset na22 --runs 65213 --preset na22-timewalk` needs
no explicit `--hits-csv`. Every invocation ends with `write_manifest`:
resolved settings, per-field provenance, inputs/outputs (optionally SHA256'd),
git SHA, library versions, duration, and a command-specific `stats` block.

### Adding a command, or a flag

A new command is a module in `commands/` with the two functions above, plus one
line in `SUBCOMMANDS` and one in `LEAF_HELP`.

A new *knob* that presets or cards should be able to set has to land in every
hop — argparse flag → `CLI_TO_SETTINGS` entry → `Settings` field → `Detector`
field → the pipeline's `getattr`. Missing a hop makes the flag silently
no-op, which has happened repeatedly; `tests/test_flag_plumbing.py` is the
meta-test that now catches it.

## Tests

From a clean checkout:

```bash
# 1. Editable install + dev deps (pytest, pytest-cov):
pip install -e .[dev]

# 2. Build the synthetic ROOT fixtures + bless this checkout's goldens.
#    Required once; without it the golden tests fail (they compare against
#    reference bytes that are gitignored, not shipped).
python tests/fixtures/bless_goldens.py

# 3. Run:
python -m pytest tests/                                # full suite
python -m pytest tests/test_pmtdeconv_cli.py -v        # CLI wiring + preset resolution
python -m pytest tests/test_golden_fixture.py -v       # byte-identical fixtures
python -m pytest tests/test_flag_plumbing.py -v        # no CLI flag silently dropped
python -m pytest tests/test_tune_cli.py -v             # tune leaves end-to-end
python -m pytest tests/test_truth_schema.py -v         # regime truth CSV schema
python -m pytest tests/test_vetoes.py -v               # math
```

The full suite should pass before release. The byte-identical goldens cover
four dataset formats: flat (`tests/fixtures/flattened_Run0.root`), object
(`tests/fixtures/MilliQan_Run835.99_triple.root`), DRSprocessing
(`tests/fixtures/processed/Run0.root`), and offline
(`tests/fixtures/MilliQan_Run99.synth_triple_offline.root`).

After a *deliberate* output change, re-bless and inspect the diff:

```bash
python tests/fixtures/bless_goldens.py --force   # then confirm every changed byte
```

Record *why* in `tests/fixtures/GOLDEN_PROVENANCE.md` — the decision trail is
tracked even though the bytes are not. A golden failure you did not intend is a
regression: fix the code rather than blessing over it.

## Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `uproot`, `awkward` (ROOT file I/O)
- `pywavelets` (wavelet denoising — `core.dsp.wavelet_denoise`)

## Credits

- **DRS binary decoding** (`data_processing/dataprep/drs/`): the
  `processMultiBoardBinary_v9.py` and `utils_v9.py` scripts are taken
  verbatim from [drsProcessing](https://github.com/mcitron/drsProcessing)
  (commit `0b1edca`) by Matthew Citron and the drsProcessing contributors,
  and are included with the author's permission. Provenance:
  `data_processing/dataprep/drs/NOTICE.md`.
- **Tang et al. 2024** ([arXiv:2411.07620](https://arxiv.org/abs/2411.07620)):
  the STFT deconvolution method and simulation reproduced by `sim tang2024` and
  the `tang2024` benchmark arms.
