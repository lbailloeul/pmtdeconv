# Bundled SPE templates

Calibrated single-photoelectron templates that ship **with the package**, so a
fresh clone or `pip install` can run `hits` / `area` on the `na22` and `cd109`
datasets without first rebuilding a template.

They were previously referenced from `outputs/biexp_fits_Cd109_lab_stacked/`,
which is gitignored — so on any machine but the author's the two flagship
datasets pointed at files that did not exist (2026-09-18 fix).

| File | What it is |
|------|------------|
| `spe_template_per_channel_unlocked.csv` | Cd109 Run 65267 stacked per-channel biexp fit, `tau_rise` unlocked. The canonical template for `na22` and `cd109`. |
| `spe_template_unlocked_na22fmt.csv` | The same fit in the na22-format schema the na22 loader expects. |

Regenerate with `pmtdeconv template build --dataset cd109` +
`pmtdeconv template aggregate`; these are copies of that output, frozen so the
defaults are reproducible. Build your own per-tube template for a new detector
rather than reusing these — they describe one specific PMT.
