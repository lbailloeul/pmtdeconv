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
│   ├── cli.py, __main__.py     # Argparse tree + dispatcher
│   ├── utils.py                # Shared math (filters, baseline, vetoes,
│   │                             STFT + wavelet denoising)
│   ├── commands/               # One module per leaf subcommand (~26 modules)
│   ├── pipeline/               # process_{area,hits}, deconv, denoise,
│   │                             detector, template_{build,aggregate}_impl,
│   │                             scoring, diagnostic sink, timewalk scan
│   ├── plotting/               # scatter2d, marginals, energy, analyze,
│   │                             timewalk_gif_impl, sweep curves, style
│   ├── presets/                # datasets.py + configs.py + tuned_configs.py
│   └── io/                     # manifest, schemas, root loaders,
│                                 template + truth CSV I/O
│
├── tests/                      # Focused unit tests + byte-id golden fixtures
│   ├── fixtures/               # synth ROOT + truth CSVs + golden CSVs
│   ├── test_pmtdeconv_cli.py
│   ├── test_golden_fixture.py
│   ├── test_tune_cli.py
│   ├── test_truth_schema.py
│   └── test_vetoes.py
│
├── outputs/                    # CLI outputs (gitignored)
├── README.md
└── .gitignore
```

The top-level has just **3 visible entries** (`pmtdeconv/`, `tests/`,
`README.md`) plus the gitignored `outputs/`. `pmtdeconv/` is
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
pip install -e ".[dev]"            # once, for pytest
python -m pytest tests/ -q
```

The byte-identical **golden** tests and the **polarity** tests need synthetic
ROOT fixtures that are *not* committed (they are regenerated, not stored). Build
them once — no real data required:

```bash
python tests/fixtures/make_synth_root.py          # flat waveforms
python tests/fixtures/make_synth_object_root.py    # LANL object format
python tests/fixtures/make_synth_drs_root.py       # DRSprocessing area tree
python tests/fixtures/make_synth_offline_root.py   # LANL offline pulse list
```

Without them those tests **skip** (each prints which script to run); the rest of
the suite still passes. No fixture or external data is needed for the math,
CLI-wiring, or tune smoke tests.

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
- You can always pin an exact file with `--root` / `--area-root` (it wins over
  the search).

If a requested run's file is found nowhere, the command **stops with an error**
that lists every path it searched and the three ways to fix it (`--data-root`,
`PMTDECONV_DATA_ROOT`, `--root`/`--area-root`) — instead of silently producing
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
#    Needs Cd109 data. Without it, deconv falls back to a generic mathematical
#    kernel (with a warning); build it to ensure accurate charge scaling:
python -m pmtdeconv template build --dataset cd109   # → outputs/spe_templates_perfile/spe_templates_perfile.csv
python -m pmtdeconv template aggregate               # → outputs/spe_templates_perfile/spe_templates_perchannel.csv
#    The Na22 preset's default template lives elsewhere, so to USE the one you
#    just built, pass it explicitly: --template outputs/spe_templates_perfile/spe_templates_perchannel.csv

# 4. Run the canonical Na22 timewalk workflow.
#    Flat Na22 data is flattened_Run*.root only (no DRSprocessing Run<N>.root),
#    so use the mqana raw-area source — the default `drs` needs those files:
python -m pmtdeconv run na22-timewalk --raw-source mqana
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

1. **Baseline estimation** — finds the quiet pre-pulse level and noise floor so downstream metrics are measured from true zero. *(adaptive sliding window, MAD-based noise, log-σ acceptance; `utils.baseline_noise_bestwindow`)*
2. **FFT deconvolution** — reverses the PMT's time-smearing, turning broad pulses into sharp spikes. A cosine low-pass filter suppresses high-frequency electronic fuzz. *(`u_rec = IFFT(FFT(y) · filt / FFT(SPE))`)*
3. **Denoising** — deconvolution mathematically amplifies high-frequency noise, which is mitigated via:
   - **STFT** — time–frequency masking (`utils.stft_denoise`)
   - **Wavelet** — frequency-band separation and thresholding (stationary SWT + MAD; `utils.wavelet_denoise`)
4. **Peak finding** — locates sharpened hits above an SNR threshold (`scipy.signal.find_peaks`). *(peak-finding scripts only)*
5. **Vetoes** — drop obviously-bad reconstructions: saturation, no-trigger, out-of-time (peak outside the signal window), post-saturation shift, afterpulse, charge purity *(peak-finding scripts only)*
6. **Charge integration** — sum the area under each hit: trough-walked window per peak (peak-finding) or fixed window (waveform summary)

### Three integration modes

| Command | Per trigger | Peak-find? | Area window |
|---|---|---|---|
| `hits` (`--denoiser wavelet` default, or `stft`) | All peaks above SNR | ✓ | Trough-walked per peak |
| `area` | One row per channel | ✗ | Fixed window (whole-waveform deconv) |
| `raw-area mqana` | One row per channel | ✗ | Fixed window, **no deconv** |

## Detector parameters (Na22 lab — DRS4)

| Parameter | Value |
|---|---|
| Sampling rate | 1.98 GS/s (`dt_ns = 0.5051`) |
| Waveform length | 1024 samples ≈ 517 ns |
| Pulse polarity | Positive-going |
| ADC saturation | ~960 mV (rail) |
| Cosine cutoff | 100 MHz (peak-finding) / 450 MHz (waveform summary wavelet) |
| Wavelet (active) | `sym8` L5 hard, `thresh_k=2.0` |

Templates are resolved per-channel from the dataset's configured SPE CSV — for
na22 this is the calibrated Cd109 stacked template under
`outputs/biexp_fits_Cd109_lab_stacked/` (build it with `template build --dataset
cd109` + `template aggregate`, or supply your own with `--template`).

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
git SHA, and library versions. Byte-identical regression is guarded by
deterministic synthetic-ROOT fixtures (3 flat + 3 object + 2 offline = 8 goldens).

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
├── timewalk {run, gif}                         Δt scan + animation
├── plot     {marginals, energy, analyze}       diagnostic plots
├── info     {datasets, presets, template, offline-file, artifacts}
├── config   {show, resolve}                     configuration inspection
├── results  {list, show, path}                  output discovery
├── tune     {cosine, wavelet, stft, denoiser, fom}  sweeps (synthetic truth + real-data fom)
└── vetoes   {calibrate}                         data-driven veto profile
```

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
| build / refit SPE templates | `template build`, `template aggregate` |
| per-trigger whole-waveform deconvolution **summary** | `area` |
| per-event peak-finding **hits with timing** | `hits` (wavelet) or `hits --denoiser stft` |
| MQAna-style **narrow-window raw area** from waveforms | `raw-area mqana` |
| precomputed `area_<board>_<ch>` from a **DRSprocessing flat tree** | `raw-area drs` |
| LANL **pulse-list** (`*_offline.root`) → area + per-channel timing | `raw-area offline` |
| Δt scan against any (area, pulse) CSV pair | `timewalk run` (CSV-only) or `timewalk gif` (animated) |
| ch1-vs-ch2 area heatmap + 1D marginals | `plot marginals` |
| four-method energy spectra (raw / FFT / STFT / wavelet) | `plot energy` |
| charge/time/height diagnostic plots from a pulse CSV | `plot analyze` |
| inspect a registry, file, or output index | `info {datasets, presets, template, offline-file, artifacts}` |
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

`--dataset` accepts a **registry name**, a **config-preset name** (so
`--dataset na22-timewalk` ≡ `--preset na22-timewalk`), or a **card path/dir**.
A card can carry analysis overrides in a `settings:` block, so a single card
expresses everything a dataset + config preset used to. Every run prints a
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

### Onboarding a new detector with a card

A dataset **card** is a YAML file describing a detector — the file-backed twin
of the in-code presets. No Python edit is needed:

```bash
python -m pmtdeconv dataset init mydet --out mydet.yaml   # scaffold (or --base na22)
$EDITOR mydet.yaml                                         # fill in the specifics
python -m pmtdeconv dataset lint mydet.yaml               # validate fields + settings
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
per-detector threshold). Saturated waveforms are **kept and flagged**
(`saturated` column) rather than dropped, so timing survives; isolate them in the
energy spectra with `plot energy --exclude-saturated`.

### Worked example 1 — Na22 timewalk (the 2026-05 settled config)

```bash
python -m pmtdeconv run na22-timewalk

# Or run the individual steps:
python -m pmtdeconv raw-area drs --dataset na22 --runs 65213
python -m pmtdeconv hits --dataset na22 --runs 65213 --preset na22-timewalk
python -m pmtdeconv timewalk run --dataset na22 --runs 65213 \
    --preset na22-timewalk
```

The preset bakes in the settings that produced σ ≈ 1.33 ns at the
[170, 175] nVs CH2 band (vs ELOG ~1.56 ns) and a 7.74 ns walk length
(vs 13.4 ns raw `t_half`).

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
python -m pmtdeconv template build --dataset cd109   # one dataset at a time
python -m pmtdeconv template aggregate               # weighted avg → per-channel
```

The per-channel CSV `outputs/spe_templates_perfile/spe_templates_perchannel.csv`
is what `--dataset cd109/na22` points at by default.

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
│     ├── deconv_pulse.csv                                  HIT_RESULTS_COLUMNS
│     ├── deconv_pulses.csv                                 compatibility alias
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
└── artifact_index.json                         optional `info artifacts --write`
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

## Tests

From a clean checkout:

```bash
# Editable install + dev deps (pytest, pytest-cov):
pip install -e .[dev]

# Or just the dev deps if pmtdeconv is already on PYTHONPATH:
pip install -r tests/requirements-dev.txt

# Then:
python -m pytest tests/                                # full suite
python -m pytest tests/test_pmtdeconv_cli.py -v        # CLI wiring + preset resolution
python -m pytest tests/test_golden_fixture.py -v       # byte-identical fixtures
python -m pytest tests/test_tune_cli.py -v             # tune leaves end-to-end
python -m pytest tests/test_truth_schema.py -v         # regime truth CSV schema
python -m pytest tests/test_vetoes.py -v               # math
```

The full suite should pass before release. The 8 byte-identical goldens
cover three dataset formats: flat (`tests/fixtures/flattened_Run0.root`),
object (`tests/fixtures/MilliQan_Run835.99_triple.root`), and offline
(`tests/fixtures/MilliQan_Run99.synth_triple_offline.root`).

## Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `uproot`, `awkward` (ROOT file I/O)
- `pywavelets` (only required when calling `utils.wavelet_denoise`)
