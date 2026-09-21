# Golden fixture provenance

Decision trail for the byte-identical golden CSVs in `golden/`.

> **Why this file sits here and not in `golden/`:** that directory is
> gitignored (goldens are blessed per checkout — see
> `tests/fixtures/bless_goldens.py`), so a log kept inside it would never
> ship and would vanish with the first `rm -rf golden/`. The *bytes* are
> local; this *decision trail* is shared history and is tracked.

The
`filecmp.cmp(shallow=False)` gate in `tests/test_golden_fixture.py` fails loudly on
*any* byte drift. When a change to a golden is **deliberate** (a provable equivalence
or an intended improvement), regenerate the golden, add an entry here, and bump the
matching version in `pmtdeconv/io/schemas.py`. Accidental drift then still fails; a
deliberate change is a reviewable two-file diff (bytes + this log).

Each entry: golden file · date · git SHA · command · reason.

---

## 2026-07-28 — RE-BLESS 8 goldens: PE-scale schema append (npe_raw / in_regime / npe)

- **Files:** the 4 hits goldens + `deconv_area_run0.csv`,
  `deconv_area_fftpad_run0.csv`, `object_deconv_area_run99.csv`,
  `offline_hit_results_run99.csv`
- **Reason:** 0–20 PE re-scope step 4. Every row now carries `npe_raw`
  (deconv-independent mqana-style raw charge / spe_charge) and `in_regime`
  (npe_raw within the card `pe_band`, default 0–20 — the deconvolution
  applicability domain); hits peak rows add `npe` = charge / spe_charge.
  Include-by-default (mirrors the `saturated` flag); `--in-regime-only`
  applies the cut with `reason="out_of_pe_range"`. Columns appended at the
  END → no schema version bump per the bump policy.
- **Audit (2026-07-28):** for all 8 files, the pre-existing columns are
  cell-identical to the previous goldens and the appended columns are ENTIRELY
  BLANK — the fixture datasets have no `spe_charge` calibration, so the
  re-bless is trivially reviewable (append-only, no value drift).
- mqana/drs area goldens untouched (AREA_COLUMNS itself unchanged; the new
  columns ride the same off-schema append mechanism as `saturated`).

## 2026-07-28 — RE-BLESS all 4 hits goldens: linear (zero-padded) deconvolution

- **Files:** `deconv_pulses_run0.csv`, `deconv_pulses_vetoon_run0.csv`,
  `deconv_pulses_stft_run0.csv`, `object_deconv_pulses_run99.csv`
- **git SHA (before change):** 03afb8d (+ uncommitted 0–20 PE re-scope steps 1–2)
- **Commands:** the standard four regeneration commands from
  `test_golden_fixture.py`'s docstring (defaults now pad).
- **Reason:** The hits path used BARE (circular) FFT deconvolution — late-pulse
  tails wrapped into early time, a genuine charge/timing artifact the code
  comments falsely denied ("linear (zero-padded) deconvolution"). Step 2 of the
  0–20 PE re-scope adopts zero-padding to `next_pow2(2N)` (mirroring the area
  path's `fft_pad`), default ON; `hits --no-fft-pad` byte-reproduces the old
  goldens exactly (verified 2026-07-28).
- **Evidence:** level3 truth fixture (120 events, `--peak-order first`):
  efficiency 0.787 → **0.938** at unchanged timing σ (0.395→0.398 ns) and
  charge scatter (2.09→2.10%); ratio_median 1.011→1.016. Circular wraparound
  was corrupting/vetoing real pulses. On the legacy bare-Gaussian golden,
  primary-peak median charge drift 0.048%, median t_peak shift 0.000 ns;
  >1% drift confined to noise-dominated windows. Full accuracy-floor suite
  (timing, t_half, saturation-robust, pileup separation) green at UNCHANGED
  tolerances before re-bless. `u_rec_noise` drops (e.g. 0.348→0.259 on the
  object fixture) — the wraparound contribution to the baseline window is gone,
  so more small true peaks clear the SNR threshold (598→673 rows on the flat
  no-vetoes golden).
- **Schema:** unchanged (values-only re-bless; columns identical, no version bump).
- Also in this step: `raw-area mqana` flat path saturation OR→per-channel
  (matches the object path). `area_mqana_run0.csv` byte-identical — the flat
  fixture has no plateau-saturated events, so no re-bless needed.

## 2026-06-12 — add `deconv_pulses_vetoon_run0.csv` (new golden)

- **File:** `deconv_pulses_vetoon_run0.csv`
- **git SHA:** c966254
- **Command:**
  `python -m pmtdeconv pulse --dataset na22 --vetoes --data-root tests/fixtures --runs 0`
  (golden = the produced `all_results.csv`)
- **Reason:** Step 0 of the architecture restructure. The existing `deconv_pulses_run0.csv`
  golden runs the `na22-timewalk-winning-2026-05` preset (`no_vetoes=True`), leaving the
  per-peak veto chain byte-untested on the flat path — exactly the code the
  EventSource/Pipeline collapse (steps 1–3) rewrites. This golden captures every peak +
  its veto `reason` (101 of 598 survive; no_raw_peak / negative_charge / out_of_time fire),
  so the collapse must preserve every veto decision byte-for-byte. New file, no prior bytes.

## 2026-06-12 — add `deconv_pulses_stft_run0.csv` (new golden)

- **File:** `deconv_pulses_stft_run0.csv`
- **git SHA:** c966254
- **Command:**
  `python -m pmtdeconv pulse --denoiser stft --dataset na22 --data-root tests/fixtures --runs 0`
  (golden = the produced `all_results.csv`)
- **Reason:** Step 2 of the architecture restructure. The STFT path
  (`process_hits_stft.py`) had NO golden, so it could not be folded into the unified
  `_process_event` safely (silent-deletion trap). This golden locks the STFT-specific
  behavior — relaxed veto constants (afterpulse_height_ratio=100, min_charge_purity=0),
  the loose baseline gate, the 1.0×-noise raw-peak cut — so folding STFT into
  `_process_event` (via `STFT_PROFILE`) and deleting `process_hits_stft.py` is provably
  byte-identical. New file, no prior bytes.

## 2026-06-13 — add `deconv_area_fftpad_run0.csv` (new golden)

- **File:** `deconv_area_fftpad_run0.csv`
- **git SHA:** c966254
- **Command:**
  `python -m pmtdeconv waveform --dataset na22 --fft-pad --data-root tests/fixtures --runs 0`
  (golden = the produced `areas_run0.csv`)
- **Reason:** Step 4 of the architecture restructure rewrites the area path behind an
  `AreaMode` strategy, including the optional FFT zero-padding (`fft_pad` → next_pow2(2N))
  kernel/filter variant — which had NO golden (the `deconv_area_run0` golden runs with
  `fft_pad=False`). This locks the padded path so the refactor is provably byte-identical.
  Output differs from `deconv_area_run0.csv` (padding changes the deconvolution), confirming
  the path is genuinely exercised. New file, no prior bytes.

## 2026-06-13 — add `area_drs_run0.csv` (new golden) + DRS fixture

- **File:** `area_drs_run0.csv` (fixture: `tests/fixtures/processed/Run0.root`)
- **git SHA:** c966254
- **Command:** `python tests/fixtures/make_synth_drs_root.py` then
  `python -m pmtdeconv raw-area drs --dataset na22 --root tests/fixtures/processed/Run0.root --runs 0`
- **Reason:** Step 8 test-gap fill. The `raw-area drs` reader (precomputed
  `area_<board>_<ch>` / `vMax_<board>_<ch>` branches from a DRSprocessing `Events`
  tree) had ZERO coverage — no fixture in that format existed. Added a deterministic
  generator (`make_synth_drs_root.py`, fixed seed, explicit TTree via `mktree` so it's
  stable across uproot ≥5.7) + this byte golden. New file, no prior bytes. (Timewalk got a
  smoke+content test instead of a byte golden — per-bin scipy `curve_fit` would make
  committed bytes scipy-version-fragile; see `tests/test_timewalk.py`.)
