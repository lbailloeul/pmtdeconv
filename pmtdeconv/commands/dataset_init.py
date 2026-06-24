"""``pmtdeconv dataset init <name>`` — scaffold a YAML dataset card.

Writes a starter card so a new detector is configured without a Python edit.
With ``--base`` it emits a minimal inheriting card; otherwise a fully commented
standalone card. Validate the result with ``pmtdeconv dataset lint <path>``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

_STANDALONE = """\
# Dataset card for {name} — file-backed twin of a DatasetPreset.
# Validate with:  pmtdeconv dataset lint {path}
name: {name}
data_dir: .                    # dir holding the ROOT files (relative to this card)
file_glob: "flattened_Run*.root"
format: flat                   # flat | object
tree: waveforms
branches: [d0_ch1, d0_ch2]     # explicit list, or "auto" to discover d0_ch*
dt_ns: 0.5051                  # sample spacing [ns], or "auto" (object format)
polarity: 1                    # +1 positive-going, -1 negative-going
cosine_cutoff_mhz: 100
# template: spe_template.csv   # per-tube SPE calibration CSV (optional)
spe_band: [5, 30]              # SPE amplitude band [lo, hi] in mV (template build)
# fom: photopeak_sigma_over_mu # figure of merit `validate`/`tune` report
# pmt_type: milliqan_lab
# hv_volts: {{d0_ch1: 1500, d0_ch2: 1500}}
# settings:                    # analysis overrides (subsumes a config preset)
#   denoiser: wavelet
#   wthresh_k: 2.0
#   no_vetoes: false
notes: ""
"""

_INHERIT = """\
# Dataset card for {name} — inherits {base}; override only what differs.
# Validate with:  pmtdeconv dataset lint {path}
name: {name}
base: {base}
# cosine_cutoff_mhz: 100
# spe_band: [5, 30]
# settings:
#   wthresh_k: 2.0
notes: ""
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="Dataset name (card file stem).")
    parser.add_argument("--base", default=None,
                        help="Inherit from this dataset/card: emits a minimal "
                             "card with `base:` instead of the full template.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: "
                             "<data_root>/calibrations/datasets/<name>.yaml).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing card.")


def _default_out(name: str) -> Path:
    from pmtdeconv.presets.datasets import _resolve_data_root
    return _resolve_data_root() / "calibrations" / "datasets" / f"{name}.yaml"


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser() if args.out else _default_out(args.name)
    if out.exists() and not args.force:
        print(f"refusing to overwrite existing card {out} (use --force)")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmpl = _INHERIT if args.base else _STANDALONE
    out.write_text(tmpl.format(name=args.name, base=args.base, path=out))
    print(f"wrote dataset card {out}")
    print(f"  edit it, then validate: pmtdeconv dataset lint {out}")
    return 0
