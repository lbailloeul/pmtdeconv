"""``pmtdeconv dataset init <name>`` — scaffold a YAML dataset card.

Writes a starter card so a new detector is configured without a Python edit.
With ``--base`` it emits a minimal inheriting card; otherwise a fully commented
standalone card. Validate the result with ``pmtdeconv dataset lint <path>``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

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
    parser.add_argument("--from-seed", dest="from_seed", action="store_true",
                        help="Serialize the in-code dataset preset of this "
                             "NAME into a complete card (the migration path "
                             "from the Python registry to file-backed cards). "
                             "Paths under the data root are written as "
                             "${PMTDECONV_DATA_ROOT}/... so the card is "
                             "shareable, not pinned to this machine.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing card.")


def _seed_to_card(name: str, out: Optional[Path] = None) -> str:
    """Serialize an in-code :class:`DatasetPreset` into card YAML.

    Faithful, not decorative: the emitted card must resolve back to the same
    preset field-for-field (``tests/test_seed_cards.py`` asserts it for every
    registry entry). Only fields the seed actually sets are written, so the
    card stays readable.

    Paths under the resolved data root become ``${PMTDECONV_DATA_ROOT}/...``:
    the seeds compute their data root per machine, and a card full of one
    author's absolute paths would not survive being shared.
    """
    import yaml
    from pmtdeconv.presets.datasets import DATASETS, _resolve_data_root

    if name not in DATASETS:
        raise SystemExit(
            f"--from-seed: no in-code dataset named {name!r}. "
            f"Known: {', '.join(DATASETS)}")
    ds = DATASETS[name]
    data_root = _resolve_data_root()

    def _portable(path) -> str:
        """Machine-independent spelling of a path, best effort.

        Data files -> ``${PMTDECONV_DATA_ROOT}/...`` (resolved per machine).
        Anything sitting near the card itself (the bundled SPE templates, which
        live one directory over) -> a path relative to the card, since card
        paths resolve against the card's own directory. Only a path that is
        neither stays absolute.
        """
        p = Path(str(path))
        try:
            return "${PMTDECONV_DATA_ROOT}/" + str(p.relative_to(data_root))
        except ValueError:
            pass
        if out is not None:
            import os as _os
            rel = _os.path.relpath(p, out.parent)
            # Accept only a short hop (../templates/x.csv), never a long
            # ../../../ climb that would be more fragile than an absolute path.
            if not rel.startswith(".." + _os.sep + ".."):
                return rel
        return str(p)

    card: dict = {"name": ds.name, "data_dir": _portable(ds.data_dir)}
    if ds.file_glob:
        card["file_glob"] = ds.file_glob
    if ds.exclude_glob:
        card["exclude_glob"] = ds.exclude_glob
    card["format"] = ds.format
    card["tree"] = ds.tree
    card["branches"] = (list(ds.branch_pattern)
                        if isinstance(ds.branch_pattern, list)
                        else ds.branch_pattern)
    card["dt_ns"] = ds.dt_ns
    card["polarity"] = ds.polarity
    card["cosine_cutoff_mhz"] = ds.cosine_cutoff_mhz
    card["spe_band"] = list(ds.spe_band_mv)
    if ds.default_template_csv:
        card["template"] = _portable(ds.default_template_csv)
    if ds.default_template_csv_na22fmt:
        card["template_na22fmt"] = _portable(ds.default_template_csv_na22fmt)
    if ds.template_csv_by_channel:
        card["templates"] = {b: _portable(p)
                             for b, p in ds.template_csv_by_channel.items()}
    if ds.offline_data_dir:
        card["offline_data_dir"] = _portable(ds.offline_data_dir)
    if ds.offline_file_glob:
        card["offline_file_glob"] = ds.offline_file_glob
    if ds.pmt_type:
        card["pmt_type"] = ds.pmt_type
    if ds.hv_volts:
        card["hv_volts"] = dict(ds.hv_volts)
    if ds.fom:
        card["fom"] = ds.fom
    if ds.spe_charge_pvs:
        card["spe_charge"] = dict(ds.spe_charge_pvs)
    if ds.pe_band:
        card["pe_band"] = list(ds.pe_band)
    if ds.notes:
        card["notes"] = ds.notes
    if ds.meta:
        card["meta"] = dict(ds.meta)

    header = (
        f"# Dataset card for {ds.name} — generated from the in-code preset by\n"
        f"#   pmtdeconv dataset init {ds.name} --from-seed\n"
        f"#\n"
        f"# Copy this to describe your OWN detector: edit data_dir, dt_ns,\n"
        f"# polarity and branches, drop `template` (build your own per tube),\n"
        f"# then:  pmtdeconv dataset lint <your card>\n"
        f"#\n"
        f"# ${{PMTDECONV_DATA_ROOT}} resolves the same way the in-code presets\n"
        f"# do: $PMTDECONV_DATA_ROOT, else the nearest data/ directory.\n"
        f"#\n"
        f"# `meta:` is free-form provenance — tube serial, source, ELOG ref,\n"
        f"# contact. pmtdeconv never interprets it; it copies it into every\n"
        f"# manifest so a shared output directory says what produced it.\n"
    )
    return header + yaml.safe_dump(card, sort_keys=False, allow_unicode=True)


def _default_out(name: str) -> Path:
    from pmtdeconv.presets.datasets import _resolve_data_root
    return _resolve_data_root() / "calibrations" / "datasets" / f"{name}.yaml"


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser() if args.out else _default_out(args.name)
    if out.exists() and not args.force:
        print(f"refusing to overwrite existing card {out} (use --force)")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "from_seed", False):
        out.write_text(_seed_to_card(args.name, out))
    else:
        tmpl = _INHERIT if args.base else _STANDALONE
        out.write_text(tmpl.format(name=args.name, base=args.base, path=out))
    print(f"wrote dataset card {out}")
    print(f"  edit it, then validate: pmtdeconv dataset lint {out}")
    return 0
