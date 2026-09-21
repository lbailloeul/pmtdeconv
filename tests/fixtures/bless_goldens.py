#!/usr/bin/env python3
"""Build the synthetic ROOT fixtures and bless all byte-identity goldens.

This is the one command a fresh checkout runs before ``pytest``::

    python tests/fixtures/bless_goldens.py

Neither the synthetic ROOT inputs nor the golden CSVs are committed (both are
gitignored): the ROOT files are large and regenerable from fixed RNG seeds,
and the golden bytes depend on the local numpy/scipy build. So each checkout
blesses its own reference set once; from then on the golden suite guards
against *drift within that checkout* — which is what it is for.

What it does, per :mod:`tests.fixtures.golden_spec`:

1. Runs any ``make_synth_*.py`` whose ROOT fixture is missing (``--rebuild``
   forces all of them).
2. Runs each of the 12 registered ``pmtdeconv`` commands into a temp dir.
3. Copies the produced CSV to ``tests/fixtures/golden/<golden name>``.

Flags
-----
``--rebuild``   rebuild the ROOT fixtures even if they already exist
``--only PAT``  bless only goldens whose filename contains PAT (repeatable)
``--dry-run``   print what would run, touch nothing
``--force``     re-bless goldens that already exist (default: skip them)

**When to re-bless.** After a *deliberate* output change (schema bump,
defaults shift): re-run with ``--force``, then diff the result and confirm
every changed byte is intended. Record *why* in
``tests/fixtures/GOLDEN_PROVENANCE.md`` — that log is tracked, unlike the
bytes. A golden failure you did not intend is a regression: fix the code, do
not bless over it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_spec import CASES, FIXTURE_BUILDERS   # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent
TESTS_DIR = FIXTURE_DIR.parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = FIXTURE_DIR / "golden"
PYBIN = sys.executable

PLACEHOLDERS = {
    "{FIXTURES}": str(FIXTURE_DIR),
    "{DRS_ROOT}": str(FIXTURE_DIR / "processed" / "Run0.root"),
    "{OFFLINE_ROOT}": str(FIXTURE_DIR
                          / "MilliQan_Run99.synth_triple_offline.root"),
}


def _resolve(argv):
    return [PLACEHOLDERS.get(a, a) for a in argv]


def build_fixtures(rebuild: bool, dry_run: bool) -> set:
    """Run each make_synth_*.py whose ROOT fixture is missing. Returns the
    set of fixture keys that are available afterwards."""
    available = set()
    for key, (rel_path, script) in FIXTURE_BUILDERS.items():
        target = FIXTURE_DIR / rel_path
        if target.exists() and not rebuild:
            print(f"  [have]  {key:8} {rel_path}")
            available.add(key)
            continue
        print(f"  [build] {key:8} {script} -> {rel_path}")
        if dry_run:
            available.add(key)
            continue
        result = subprocess.run([PYBIN, str(FIXTURE_DIR / script)],
                                cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED ({result.returncode}):\n{result.stderr}",
                  file=sys.stderr)
            continue
        if target.exists():
            available.add(key)
        else:
            print(f"    {script} ran but {rel_path} is still missing",
                  file=sys.stderr)
    return available


def bless(case, tmp_root: Path, dry_run: bool) -> bool:
    """Run one case and copy its output into the golden dir."""
    out_dir = tmp_root / case.golden.replace(".csv", "")
    argv = _resolve(case.argv)
    cmd = [PYBIN, "-m", "pmtdeconv", *argv,
           "--out-dir", str(out_dir), "--log-level", "quiet"]
    if dry_run:
        print(f"    would run: pmtdeconv {' '.join(argv)}")
        return True
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})\n{result.stderr}",
              file=sys.stderr)
        return False
    produced = out_dir / case.produced
    if not produced.exists():
        print(f"    command succeeded but {case.produced} was not written",
              file=sys.stderr)
        return False
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, GOLDEN_DIR / case.golden)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild ROOT fixtures even if they exist")
    ap.add_argument("--only", action="append", default=[], metavar="PAT",
                    help="bless only goldens whose name contains PAT")
    ap.add_argument("--force", action="store_true",
                    help="re-bless goldens that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    args = ap.parse_args(argv)

    print("Synthetic ROOT fixtures:")
    available = build_fixtures(args.rebuild, args.dry_run)

    cases = CASES
    if args.only:
        cases = [c for c in cases
                 if any(pat in c.golden for pat in args.only)]

    print(f"\nGoldens -> {GOLDEN_DIR}")
    blessed = skipped = failed = 0
    for case in cases:
        if case.fixture not in available:
            print(f"  [skip]  {case.golden}  "
                  f"({case.fixture} fixture unavailable)")
            skipped += 1
            continue
        if (GOLDEN_DIR / case.golden).exists() and not args.force:
            print(f"  [keep]  {case.golden}  (exists; --force to re-bless)")
            skipped += 1
            continue
        print(f"  [bless] {case.golden}  — {case.note}")
        with tempfile.TemporaryDirectory() as tmp:
            ok = bless(case, Path(tmp), args.dry_run)
        blessed += ok
        failed += (not ok)

    print(f"\n{blessed} blessed, {skipped} skipped, {failed} failed "
          f"({len(CASES)} registered).")
    if failed:
        return 1
    print("\nNow run:  python -m pytest tests/test_golden_fixture.py -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
