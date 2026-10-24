# Vendored code — drsProcessing

The files in this directory are **vendored verbatim** (only an attribution
header was prepended) from:

- **Repository:** https://github.com/mcitron/drsProcessing
- **Remote:** `git@github.com:mcitron/drsProcessing.git`
- **Commit:** `0b1edca`
- **Author / credit:** Matthew Citron (`mcitron`) and the drsProcessing contributors
- **Vendored:** 2026-06-14, into `pmtdeconv-dataprep`

## Files

| file | role |
|------|------|
| `processMultiBoardBinary_v9.py` | DRS scope binary → ROOT (the `-w` waveform mode); the DAQ decode step. |
| `utils_v9.py` | its only local dependency (helper functions). |

`utilsMultipeak_v9.py`, `processMultiBoardBinary_scans.py` (bias-scan variant)
and the C++ `process.cc` from upstream are **not** vendored — they are not
imported by `processMultiBoardBinary_v9.py` in the fixed-bias `-w` workflow this
package uses.

## Why vendored (not a submodule of the whole clone)

Only this one script + its single helper are needed, so we copy just those
rather than depend on the entire upstream repo. **All credit for this code
belongs to the drsProcessing authors.**

## License

No `LICENSE` file was present in the upstream repository at the retrieved
commit. These files are redistributed here with the author's permission, to run
the documented DRS-binary→ROOT step.

## Updating

Do not hand-edit the vendored files. To update, re-copy from upstream and
re-prepend the attribution header (see how `dataprep/drs/` was created).
