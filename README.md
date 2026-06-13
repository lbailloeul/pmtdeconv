# bailloeulScripts — pmtdeconv

PMT-waveform deconvolution toolkit. Extracts hit times and integrated charges
from digitized DRS4 / object-format / pulse-list ROOT files using FFT
deconvolution against a calibrated single-photoelectron (SPE) template, with
optional STFT or wavelet denoising. Dataset-agnostic: na22, cd109, milliqan,
lanl_806, lanl_835, lanl_movedBar.

**The single deployable entry point is `pmtdeconv`** — a unified CLI organized
around output granularity (`hits` vs `area`) plus supporting template,
area-source, plotting, timewalk, info, and tune commands. Pre-CLI analysis
scripts have been moved to `legacy/` (gitignored) and are no longer maintained.

## Project structure

```
bailloeulScripts/
├── pmtdeconv/                  # Self-contained CLI package. Everything
│   │                             needed to run the CLI lives in here.
│   ├── cli.py, __main__.py     # Argparse tree + dispatcher
│   ├── utils.py                # Shared math (filters, baseline, vetoes,
│   │                             STFT + wavelet denoising)
│   ├── commands/               # One module per leaf subcommand (16 total)
│   ├── pipeline/               # process_{area,hits,hits_stft,offline}
│   │                             template_{build,aggregate}_impl,
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
├── legacy/                     # Pre-CLI scripts + the originals of the
│                                 6 modules that got lifted into pmtdeconv/
│                                 (utils.py, aggregate_spe_templates.py,
│                                 build_spe_templates_perfile.py,
│                                 timewalk_data_gif.py, deconv_area_Na22.py,
│                                 analyze_deconv.py). Gitignored.
│
├── CLAUDE.md, README.md
└── .gitignore
```

The top-level has just **4 visible entries** (`pmtdeconv/`, `tests/`,
`CLAUDE.md`, `README.md`) plus two gitignored dirs. `pmtdeconv/` is
pip-installable as-is — no runtime dependencies outside its own tree.

## Quick start

```bash
# Discover the CLI surface:
python -m pmtdeconv --help                       # command tree
python -m pmtdeconv info datasets                # 6 dataset presets
python -m pmtdeconv config show na22-timewalk    # inspect a recipe
python -m pmtdeconv config resolve --preset na22-timewalk
python -m pmtdeconv <command> --help             # command flag reference

# Run the canonical Na22 timewalk workflow:
python -m pmtdeconv run na22-timewalk
python -m pmtdeconv results list

# Tune the denoiser to your detector (PR 10):
python -m pmtdeconv tune wavelet --dataset na22 --write-presets
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --preset na22-wavelet-tuned-2026-05-29-level3

# Debug an individual event (PR 10 diagnostic flag):
python -m pmtdeconv hits --dataset na22 --runs 65213 \
    --diagnostic-events 5 10 100
```

See the [Using `pmtdeconv` (unified CLI)](#using-pmtdeconv-unified-cli) section
below for worked examples covering all six dataset formats and the new tune /
diagnostic features.

## Pipeline overview

The deconvolution pipeline follows this sequence:

1. **Baseline estimation** — adaptive sliding window, MAD-based noise, log-σ acceptance threshold (`utils.baseline_noise_bestwindow`)
2. **FFT deconvolution** with cosine low-pass filter — `u_rec = IFFT(FFT(y) · filt / FFT(SPE))`
3. **Denoising** (one of):
   - **STFT** — Gaussian-windowed time-frequency mask (`utils.stft_denoise`)
   - **Wavelet** — stationary (SWT) decomposition + MAD-thresholding (`utils.wavelet_denoise`)
4. **Peak finding** — `scipy.signal.find_peaks` with SNR threshold *(peak-finding scripts only)*
5. **Vetoes** — saturation, no-trigger, out-of-time (peak outside the signal window), post-saturation shift, afterpulse, charge purity *(peak-finding scripts only)*
6. **Charge integration** — trough-walked window per peak (peak-finding) or fixed window (waveform summary)

### Three integration modes

| Script | Per trigger | Peak-find? | Area window |
|---|---|---|---|
| `cosine_advanced_convolve_at_scale_Na22.py` | All peaks above SNR | ✓ | Trough-walked per peak |
| `wavelets_cosine_deconvolution_Na22.py` | All peaks above SNR | ✓ | Trough-walked per peak |
| `deconv_area_Na22.py` | One row per channel | ✗ | Fixed sample window |
| `raw_area_Na22.py` | One row per channel | ✗ | Fixed sample window, **no deconv** |

## Detector parameters (Na22 lab — DRS4)

| Parameter | Value |
|---|---|
| Sampling rate | 1.98 GS/s (`dt_ns = 0.5051`) |
| Waveform length | 1024 samples ≈ 517 ns |
| Pulse polarity | Positive-going |
| ADC saturation | ~960 mV (rail) |
| Cosine cutoff | 100 MHz (peak-finding) / 450 MHz (waveform summary wavelet) |
| Wavelet (active) | `sym8` L5 hard, `thresh_k=2.0` |

Templates are loaded per-channel from `outputs/biexp_fits_Na22_lab/biexp_fit_results.csv`.

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

The legacy per-stage scripts are consolidated behind a single
`pmtdeconv` CLI with git-style subcommands. The math is unchanged —
each subcommand calls the canonical `pmtdeconv/pipeline/` functions,
and the legacy scripts re-export the same names so existing invocations
keep working. Every command drops a `manifest.json` recording the
resolved settings, provenance, inputs/outputs, git SHA, and library
versions. Byte-identical regression is guarded by deterministic
synthetic-ROOT fixtures (3 flat + 3 object + 2 offline = 8 goldens).

### Subcommand tree

```
pmtdeconv
├── run {na22-timewalk, na22-energy, build-template} complete workflows
├── template {build, aggregate}                 SPE template stacking
├── hits                                       per-hit deconvolution
├── area                                       whole-waveform deconvolution summary
├── raw-area {mqana, drs, offline}              raw-waveform area sources
├── timewalk {run, gif}                         Δt scan + animation
├── plot     {marginals, energy, analyze}       diagnostic plots
├── info     {datasets, presets, template, offline-file, artifacts}
├── config   {show, resolve}                     configuration inspection
├── results  {list, show, path}                  output discovery
└── tune     {cosine, wavelet, stft}            regime-driven parameter sweeps (PR 10)
```

The older `pulse`, `waveform`, and `raw_area` spellings remain available as
deprecated aliases for existing scripts.

`tune` runs deterministic sweeps against the seeded
synthetic regime fixtures (`level1`..`level5`, from low SNR to
saturated) and scores each candidate config against ground truth —
charge bias (drift vs the dataset preset's reference config), per-event
scatter, timing σ, and efficiency. Winners are persisted as
auto-generated `ConfigPreset` entries that `area`/`hits`
can pick up via `--preset` with no runtime branching.

### Which subcommand do I use?

| I want… | Use |
|---------|-----|
| run the standard Na22 workflow | `run na22-timewalk` |
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
| **find best cosine cutoff per regime** | `tune cosine` (use `--denoiser none` to isolate the cosine response) |
| **find best wavelet config per regime** | `tune wavelet` (use `--cosine-cutoff-mhz <N>` to pick the fixed cutoff) |
| **find best STFT config per regime** | `tune stft` |
| **dump intermediates for one or more events** (FFT, denoised, peaks, truth match, PDF) | `hits --diagnostic-events <N> [N …]` |

### Settings layering: `--dataset` → `--preset` → CLI flag

```text
dataset default  <  named config preset  <  explicit CLI flag
```

```bash
python -m pmtdeconv info datasets        # list 6 datasets + applicable subcommands
python -m pmtdeconv config show na22-timewalk
python -m pmtdeconv config resolve --preset na22-timewalk
python -m pmtdeconv <command> --help     # command flag reference
```

### Worked example 1 — Na22 timewalk (the 2026-05 settled config)

```bash
python -m pmtdeconv run na22-timewalk

# Equivalent low-level commands remain available:
python -m pmtdeconv raw-area drs --dataset na22 --runs 65213
python -m pmtdeconv hits --dataset na22 --runs 65213 --preset na22-timewalk
python -m pmtdeconv timewalk run --dataset na22 --runs 65213 \
    --preset na22-timewalk
```

The preset baked in the settings that produced σ ≈ 1.33 ns at the
[170, 175] nVs CH2 band (vs ELOG ~1.56 ns) and a 7.74 ns walk length
(vs 13.4 ns raw `t_half`). See
`.claude/plans/iterative-chasing-sunbeam.md` for the rationale.

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
    --cosine-cutoff-mhz 250 --sat-thresh 920

# Resolve flags + write manifest, skip ROOT I/O
python -m pmtdeconv hits --dataset na22 --runs 65213 --dry-run
```

The `manifest.json` from a dry-run is enough to confirm the resolved
settings before committing to a multi-minute run.

### Worked example 5 — regime-driven parameter sweeps (PR 10)

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

# 5. Use a written preset.
python -m pmtdeconv area --dataset na22 --runs 65213 \
    --preset na22-cosine-tuned-2026-05-29-level3

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
| `na22`, `cd109`, `milliqan`, `lanl_806` | flat (`flattened_*.root`) | `template`, `pulse`, `waveform`, `raw_area {mqana, drs}`, `timewalk`, `plot` |
| `lanl_835`, `lanl_movedBar` | object (`*_triple.root`) + offline pulse list (`*_triple_offline.root`) | `template`, `pulse`, `waveform`, `raw_area {mqana, offline}`, `timewalk`, `plot` |

`raw_area drs` is N/A on object-format datasets (LANL files have no precomputed
`area_<board>_<ch>` branches) — use `raw_area offline` instead. The CLI hard-fails
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
└── tune_{cosine,wavelet,stft}_<dataset>[…]/    (PR 10)
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
