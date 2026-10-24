"""Dataset presets — the canonical registry of per-detector defaults.

Each entry encodes everything the CLI needs to know about a dataset:
data path, ROOT tree+branch convention, sample spacing, saturation
threshold, pulse polarity, FFT cosine-cutoff, and the recommended SPE
template file. Subcommands consume these via
:func:`pmtdeconv.presets.resolve_settings`.

This is the single source of truth — values previously scattered across
``build_spe_templates_perfile.DATASETS``, ``raw_area_Na22.DATA_DIR``,
the various ``deconvolution/*Na22.py`` scripts, and ``utils.py``'s
``LANL_TEMPLATE`` / ``MILLIQAN_TEMPLATE`` constants.

Cross-references:
- ``cd109-na22-sampling-rates`` memory: 65267 dt=1.432, 65212/65213 dt=0.5051
- ``CLAUDE.md``: "Use sat-thresh 960 for runs where CH1 frequently saturates at 930"
- ``project_cd109_spe_template`` memory: Cd109 stacked unlocked template
- ELOG E41473 (UCSB): per-HV-setting Na22 saturation; LANL polarity = -1
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pmtdeconv import PACKAGE_DIR, PROJECT_ROOT


# Path conventions: per-dataset files live under a base data root in
# ``<DATA_DIR>/<dataset_dir>/``. The base is resolved portably (the old fixed
# ``PROJECT_ROOT.parent.parent.parent / "data"`` guess only worked at one
# checkout depth — it printed nonexistent paths on other machines):
#
#   1. ``PMTDECONV_DATA_ROOT`` environment variable (authoritative override), then
#   2. the first existing ``data/`` directory walking up from the package
#      (handles the common "repo and data/ side by side" layout), then
#   3. the legacy fixed-depth fallback (kept so existing checkouts + CI are
#      byte-identical).
#
# ``--data-root`` still overrides the *per-dataset* directory directly for
# ad-hoc / flat layouts.

# A candidate ``data/`` dir is only auto-selected if it actually looks like a
# data root (contains a known per-dataset subdir) — so a stray empty ``data/``
# higher in the tree can't shadow the real one.
_DATA_MARKERS = ("Na22_lab", "Cd109_lab", "1000_MQ_real",
                 "806_real", "835_real", "movedBar")


def _resolve_data_root(start: Optional[Path] = None) -> Path:
    """Resolve the base data root: ``PMTDECONV_DATA_ROOT`` env → nearest
    marker-bearing ``data/`` walking up from ``start`` → legacy fallback.
    ``start`` defaults to the package dir; it is a parameter only for testing."""
    env = os.environ.get("PMTDECONV_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    root = start or PROJECT_ROOT
    # bounded discovery: package dir + a few parents, requiring a real marker
    for base in [root, *list(root.parents)[:4]]:
        cand = base / "data"
        if cand.is_dir() and any((cand / m).is_dir() for m in _DATA_MARKERS):
            return cand
    return root.parent.parent.parent / "data"   # legacy fallback


DATA_DIR = _resolve_data_root()

# Cd109 stacked-unlocked SPE template — the "winning" template settled
# during the 2026-05 session and now the default for Na22 deconv.
#
# Two CSVs exist for the same fit (different column conventions):
# - per-channel "stacked" schema used by ``deconv_area_Na22.load_templates``
# - per-event "na22fmt" schema used by
#   ``wavelets_cosine_deconvolution_Na22.load_channel_templates``
# We expose the stacked CSV here (the canonical one) and the na22fmt
# alias so each legacy loader sees the format it expects until the
# loaders are unified in PR 2.
#
# These live INSIDE the package (2026-09-18). They previously pointed into
# ``outputs/``, which is gitignored — so on any machine but the author's the
# na22 and cd109 defaults referenced files that did not exist, and (before the
# existence check in pipeline/template.py) the run silently fell back to the
# generic kernel while the manifest still named the missing CSV. Bundling them
# means a fresh clone or `pip install` can run those datasets as shipped.
_BUNDLED_TEMPLATES = PACKAGE_DIR / "data" / "templates"
CD109_STACKED_UNLOCKED_CSV = (
    _BUNDLED_TEMPLATES / "spe_template_per_channel_unlocked.csv"
)
CD109_STACKED_UNLOCKED_NA22FMT_CSV = (
    _BUNDLED_TEMPLATES / "spe_template_unlocked_na22fmt.csv"
)


@dataclass(frozen=True)
class DatasetPreset:
    """Frozen per-dataset configuration. Loaded once, never mutated.

    Override fields by passing the corresponding CLI flag (the CLI
    constructs a fresh ``Settings`` rather than mutating this).
    """
    name: str
    data_dir: Path
    file_glob: str
    exclude_glob: Optional[str]
    format: Literal["flat", "object"]
    tree: str
    # `branch_pattern` is either an explicit list (e.g. ["d0_ch1", "d0_ch2"])
    # or the sentinel "auto" — meaning "discover branches matching d0_ch*
    # at load time".
    branch_pattern: Union[List[str], Literal["auto"]]
    # `dt_ns` is either a fixed sample spacing in ns or "auto" — meaning
    # "read from the object-format `digitizers.nanosecondsPerSample` field".
    dt_ns: Union[float, Literal["auto"]]
    polarity: int                       # +1 or -1
    cosine_cutoff_mhz: float
    default_template_csv: Optional[Path]
    spe_band_mv: Tuple[float, float]
    notes: str = ""
    # Optional alternate-schema CSV consumed by the legacy
    # wavelets_cosine_deconvolution_Na22.load_channel_templates (per-event
    # det/chan schema). When None, ``pmtdeconv pulse`` falls back to the
    # canonical ``default_template_csv``.
    default_template_csv_na22fmt: Optional[Path] = None
    # PR 8: directory containing LANL ``*_offline.root`` pulse-list
    # files. For object-format datasets, this is the canonical
    # DRSprocessing-equivalent output. May differ from ``data_dir``
    # (the waveform-file dir); e.g. lanl_835 waveforms in
    # ``data/835_real/`` but offline files in ``data/835/``.
    offline_data_dir: Optional[Path] = None
    # PR 8: glob for the offline files (file_glob's `_offline.root`
    # cousin). Defaults set in the per-dataset entries below.
    offline_file_glob: Optional[str] = None
    # Step 3 (architecture restructure): per-detector PMT metadata used by the
    # template resolver + manifest. `pmt_type` is a PROVISIONAL label keying the
    # template catalog (presets/template_catalog.py) when no per-channel CSV is
    # configured — it does NOT alter the kernel shape, only which catalogued
    # template a metadata-only run resolves to. `hv_volts` is recorded for
    # charge-scale context + the manifest ONLY (it never changes the kernel;
    # user decision: shape-table only). Both default to None (unknown).
    pmt_type: Optional[str] = None
    hv_volts: Optional[Dict[str, float]] = None
    # S6: the figure of merit this dataset is judged on (a name in
    # pmtdeconv.fom.FOMS, e.g. "photopeak_sigma_over_mu"). `validate` and the
    # real-data `tune` report/optimize THIS expert-chosen number; the tool never
    # picks it. None = no declared FoM (validate then needs --fom).
    fom: Optional[str] = None
    # S9: per-channel SPE templates — {branch: csv_path}. Lets each PMT tube use
    # its OWN transferable SPE artifact (e.g. d0_ch1 -> tubeA.csv, d0_ch2 ->
    # tubeB.csv) instead of one shared template. When set it supersedes
    # default_template_csv for the listed channels. None = single-CSV behavior.
    template_csv_by_channel: Optional[Dict[str, Path]] = None
    # 0–20 PE re-scope (2026-07-28): per-branch 1-PE charge in pVs (mV·ns) —
    # the PE-scale anchor. {branch: charge}; nPE = deconvolved charge / this.
    # EXPERT-SET (card key `spe_charge`), proposed by `gain calibrate` from a
    # fit of the 1-PE charge-quantization peak; None = uncalibrated (nPE
    # columns stay blank, `--in-regime-only` refuses to run).
    spe_charge_pvs: Optional[Dict[str, float]] = None
    # Free-form provenance a card may carry: tube serial, source, geometry,
    # ELOG reference, who took the data. pmtdeconv never interprets any of it —
    # it exists so a shared card answers "what detector is this, and who do I
    # ask about it?". Copied onto Settings, so it lands in every manifest.
    meta: Optional[Dict[str, Any]] = None
    # The deconvolution applicability window in nPE (card key `pe_band`).
    # Default = the literature-established 0–20 PE linear-model domain
    # (Tang et al. 2411.07620; vault 17b).
    pe_band: Tuple[float, float] = (0.0, 20.0)


DATASETS: Dict[str, DatasetPreset] = {
    "na22": DatasetPreset(
        name="na22",
        data_dir=DATA_DIR / "Na22_lab",
        file_glob="flattened_Run*.root",
        exclude_glob=None,
        format="flat",
        tree="waveforms",
        branch_pattern=["d0_ch1", "d0_ch2"],
        dt_ns=0.5051,
        polarity=+1,
        # 75 MHz since 2026-10-23 (was the legacy 450): the DRS4 records carry
        # fs/32-harmonic lines at 61.9-396 MHz (~0.7 mV rms through 450 MHz);
        # 75 passes only 61.9 at 0.27 gain. Real 66810 timing 5.54 ns @75 vs
        # 10.4 @450; 50 MHz times 0.1 ns better but separates pile-up worse.
        cosine_cutoff_mhz=75.0,
        default_template_csv=CD109_STACKED_UNLOCKED_CSV,
        default_template_csv_na22fmt=CD109_STACKED_UNLOCKED_NA22FMT_CSV,
        spe_band_mv=(50.0, 200.0),
        pmt_type="milliqan_lab",
        hv_volts={"d0_ch1": 1450.0, "d0_ch2": 1450.0},  # ~1.45 kV (65212/65213)
        notes=("Na22 lab DRS4 (2-channel mode, ~1.98 GS/s). "
               "Default template = Cd109 stacked unlocked."),
    ),
    "na22_3ch": DatasetPreset(
        name="na22_3ch",
        data_dir=DATA_DIR / "Na22_lab",
        file_glob="flattened_Run*.root",
        exclude_glob=None,
        format="flat",
        tree="waveforms",
        branch_pattern=["d0_ch1", "d0_ch2", "d0_ch3"],
        # Runs 66727-29 (2026-06-10): measured 1.9818 GHz → 0.5045844 ns.
        # Override per run with --dt-ns (66727: 0.5045841, 66728:
        # 0.5045843, 66729: 0.5045848).
        dt_ns=0.5045844,
        polarity=-1,                  # NEGATIVE-going pulses in this setup
        # 100 MHz (2026-06-10 diagnosis): 450 inherited from na22 produced
        # a ringing comb (raw waveforms ring; tallest-peak timing lottery).
        # With the single-peak kernels, 100 MHz gives non-negative deconv,
        # physical peak multiplicities, and stable onset timing; 200+ MHz
        # re-fragments. See vault note 14d + cutoff_sweep_na22_3ch.png.
        cosine_cutoff_mhz=100.0,
        default_template_csv=None,    # build per-run: template build --dataset na22_3ch
        spe_band_mv=(50.0, 200.0),
        # ELOG E41537 — HV/source/detector VARY per run (hv_volts below = 66727):
        #   66727             : Na22,  LYSO+CsI,   HPK PMT,  HV1 -1900 / HV2 +1300
        #   66808/66809/66810 : Ba133, CeBr+Bar2,  7725 PMT, HV1 -1400 / HV2 -2000
        pmt_type="na22_3ch_bipolar",
        hv_volts={"d0_ch1": -1900.0, "d0_ch2": 1300.0},
        notes=("Recent 3-channel bipolar DRS4 runs (66727+, Range Center 0, "
               "negative-going). SOURCE / DETECTOR / PMT VARY PER RUN — check "
               "ELOG E41537 before use: 66727 = Na22 (LYSO+CsI, HPK PMT); "
               "66808/66809/66810 = Ba133 (CeBr+Bar2, 7725 PMT). This is a "
               "DIFFERENT tube than the Cd109/Na22 (E41473, non-7725) template, "
               "so templates are built per run (default_template_csv=None). "
               "TRACK-2 / TIMING ONLY for multi-hit work: the production ch2 "
               "template is the all-stack ENVELOPE kernel (cable-reflection "
               "echoes baked in) that deliberately collapses a clump into ONE "
               "spike — optimal Δt, NO intra-clump PE separation. Multi-hit / "
               "0–20 PE charge analysis here needs a single-peak (sampled) "
               "kernel + a cutoff revisit (vault 14d; deferred follow-up)."),
    ),
    "cd109": DatasetPreset(
        name="cd109",
        data_dir=DATA_DIR / "Cd109_lab",
        file_glob="flattened_Run*.root",
        exclude_glob=None,
        format="flat",
        tree="waveforms",
        branch_pattern=["d0_ch1", "d0_ch2"],
        dt_ns=1.432,
        polarity=+1,
        cosine_cutoff_mhz=50.0,    # conditioning cliff above ~65 MHz
        default_template_csv=CD109_STACKED_UNLOCKED_CSV,
        spe_band_mv=(5.0, 25.0),
        pmt_type="milliqan_lab",   # source of the catalogued stacked template
        notes=("Cd109 22-keV X-ray source, 4-channel mode (~700 MS/s). "
               "Source of the stacked SPE template. "
               "Cosine cutoff capped at 50 MHz per the conditioning-cliff finding."),
    ),
    "milliqan": DatasetPreset(
        name="milliqan",
        data_dir=DATA_DIR / "1000_MQ_real",
        file_glob="flattened_waveform_Run*.root",
        exclude_glob=None,
        format="flat",
        tree="waveforms",
        branch_pattern="auto",
        dt_ns=1.25,
        polarity=+1,
        cosine_cutoff_mhz=100.0,
        default_template_csv=None,    # falls back to MILLIQAN_TEMPLATE
        spe_band_mv=(5.0, 30.0),
        pmt_type="milliqan",
        notes="MilliQan multi-channel digitizer. Per-run channel set auto-detected.",
    ),
    "lanl_806": DatasetPreset(
        name="lanl_806",
        data_dir=DATA_DIR / "806_real",
        file_glob="flattened_waveform_*.root",
        exclude_glob=None,
        format="flat",
        tree="waveforms",
        branch_pattern="auto",
        dt_ns=1.25,
        polarity=-1,                  # NEGATIVE-going LANL pulses
        cosine_cutoff_mhz=100.0,
        default_template_csv=None,    # falls back to LANL_TEMPLATE
        spe_band_mv=(5.0, 25.0),
        pmt_type="lanl_fast",
        notes="LANL Run 806 (fast PMT). Polarity = -1.",
    ),
    "lanl_835": DatasetPreset(
        name="lanl_835",
        data_dir=DATA_DIR / "835_real",
        file_glob="MilliQan_Run835.*_triple.root",
        exclude_glob=None,
        format="object",
        tree="Events",
        branch_pattern="auto",
        dt_ns="auto",                 # read from digitizers.nanosecondsPerSample
        polarity=-1,
        cosine_cutoff_mhz=100.0,
        default_template_csv=None,
        spe_band_mv=(5.0, 25.0),
        # LANL_835 offline pulse-list files live in a SEPARATE
        # directory from the waveform files (data/835/ vs data/835_real/).
        offline_data_dir=DATA_DIR / "835",
        offline_file_glob="MilliQan_Run835.*_triple_offline.root",
        pmt_type="lanl_fast",
        notes="LANL Run 835 (object format, nested per-event arrays).",
    ),
    "lanl_movedBar": DatasetPreset(
        name="lanl_movedBar",
        data_dir=DATA_DIR / "movedBar",
        file_glob="MilliQan_Run868.*_triple.root",
        exclude_glob="*_offline.root",
        format="object",
        tree="Events",
        branch_pattern="auto",
        dt_ns="auto",
        polarity=-1,
        cosine_cutoff_mhz=100.0,
        default_template_csv=None,
        spe_band_mv=(5.0, 25.0),
        # movedBar offline files live alongside the waveform files,
        # distinguished by the ``_offline.root`` suffix.
        offline_data_dir=DATA_DIR / "movedBar",
        offline_file_glob="MilliQan_Run868.*_triple_offline.root",
        pmt_type="lanl_fast",
        notes="LANL moved-bar geometry (object format). Excludes *_offline.root.",
    ),
}


def list_datasets() -> List[str]:
    """Return registered dataset names in registry order."""
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetPreset:
    """Return the DatasetPreset for ``name`` or raise ``KeyError``."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset preset {name!r}. "
                       f"Known: {', '.join(DATASETS)}")
    return DATASETS[name]


def _registry_card_path(name: str) -> Optional[Path]:
    """Return the file-backed card for ``name`` under
    ``<data_root>/calibrations/datasets/<name>.{yaml,yml}`` if present, else None.

    Resolved FRESH via :func:`_resolve_data_root` (honors ``PMTDECONV_DATA_ROOT``)
    so a shared calibrations directory can shadow the in-code seeds without a
    re-import.
    """
    base = _resolve_data_root() / "calibrations" / "datasets"
    for ext in (".yaml", ".yml"):
        cand = base / f"{name}{ext}"
        if cand.exists():
            return cand
    return None


def resolve_dataset_with_source(
        name_or_path: str) -> Tuple[DatasetPreset, str, dict]:
    """Resolve a ``--dataset`` value to ``(DatasetPreset, source, overrides)``.

    Resolution order: card PATH → registry card under
    ``<data_root>/calibrations/datasets/`` → SHIPPED card under
    ``pmtdeconv/data/datasets/`` → in-code seed.

    ``source`` is ``"card"`` for a user-supplied card (a path, or a registry
    card shadowing the default) and ``"dataset"`` for the shipped default —
    whether that came from the shipped card or the in-code seed. So ``[card]``
    in the banner/manifest keeps meaning "something user-supplied overrides the
    default", and is never silent. ``overrides`` is the card's ``settings:``
    block (``{}`` for a seed).

    Shipped cards resolve since 2026-09-21: before, ``--dataset na22`` ignored
    ``pmtdeconv/data/datasets/na22.yaml`` entirely, so editing it did nothing.
    They round-trip exactly to their seeds (``tests/test_seed_cards.py``), so
    resolved settings — and every output byte — are unchanged.
    """
    from pmtdeconv.presets.cards import (
        is_card_reference, load_card, read_card_settings)
    if is_card_reference(name_or_path):
        return (load_card(name_or_path), "card",
                read_card_settings(name_or_path))
    card = _registry_card_path(name_or_path)
    if card is not None:
        return load_card(card), "card", read_card_settings(card)
    shipped = _shipped_card_path(name_or_path)
    if shipped is not None:
        return load_card(shipped), "dataset", read_card_settings(shipped)
    return get_dataset(name_or_path), "dataset", {}


def _shipped_card_path(name: str) -> Optional[Path]:
    """The card shipped inside the package for ``name``, if any."""
    base = PACKAGE_DIR / "data" / "datasets"
    for ext in (".yaml", ".yml"):
        cand = base / f"{name}{ext}"
        if cand.exists():
            return cand
    return None


def card_path_for(name_or_path: str) -> Optional[Path]:
    """The card FILE a ``--dataset`` value resolves to, or None for a seed.

    The manifest records it as an input so a run says which card produced it
    (and ``--hash-inputs`` fingerprints it). Without this, two runs from two
    revisions of the same card were indistinguishable in the record.
    """
    from pmtdeconv.presets.cards import _resolve_card_path, is_card_reference
    if is_card_reference(name_or_path):
        try:
            return _resolve_card_path(name_or_path)
        except (FileNotFoundError, OSError):
            return None
    return (_registry_card_path(name_or_path)
            or _shipped_card_path(name_or_path))


def resolve_dataset(name_or_path: str) -> DatasetPreset:
    """Resolve a ``--dataset`` value to a :class:`DatasetPreset` (card path,
    registry card, or in-code seed). See :func:`resolve_dataset_with_source`."""
    return resolve_dataset_with_source(name_or_path)[0]
