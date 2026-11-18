#!/usr/bin/env python3
"""Losslessly extract existing QOBLib .qs.xz archives into a QUBO dataset.

Run from anywhere after placing this script in QOBLIB/my_code/:
    python3 my_code/extract_qubos.py --family labs --dry-run --limit 5
    python3 my_code/extract_qubos.py --family labs
    python3 my_code/extract_qubos.py --all
"""

import argparse
import lzma
import os
import shutil
import tempfile
from pathlib import Path


# Verified from the local QOBLib directory inventory (commit 2b400f43).
FAMILIES = {
    "marketsplit": "01-marketsplit/models/binary_unconstrained/qs_files",
    "labs": "02-labs/models/quadratic_unconstrained/qs_files",
    "portfolio": "06-portfolio/models/unconstrained_quadratic_optimization/qs_files",
    "independentset": "07-independentset/models/binary_unconstrained/qs_files",
}
CHUNK_SIZE = 1024 * 1024


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--family", choices=FAMILIES, help="Extract one family")
    target.add_argument("--all", action="store_true", help="Extract all four families")
    parser.add_argument("--dry-run", action="store_true", help="Preview; do not write files")
    parser.add_argument("--limit", type=int, help="Process at most N archives (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Print each archive's status")
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="QOBLIB root (default: parent of this script's my_code directory)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Output directory (default: <repo-root>/extracted_qb)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def destination_for(archive: Path, source_root: Path, output_root: Path, family: str) -> Path:
    """Retain Portfolio subfolders; give every .qs its own instance directory."""
    relative = archive.relative_to(source_root)
    instance = archive.name.removesuffix(".qs.xz")
    return output_root / family / relative.parent / instance / f"{instance}.qs"


def content_matches(archive: Path, destination: Path) -> bool:
    """Compare an existing extraction to the complete decompressed archive."""
    with lzma.open(archive, "rb") as source, destination.open("rb") as existing:
        while True:
            data = source.read(CHUNK_SIZE)
            if not data:
                return existing.read(1) == b""
            if existing.read(len(data)) != data:
                return False


def extract_without_overwriting(archive: Path, destination: Path) -> None:
    """Fully decompress to a temporary file, then publish without replacing a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            with lzma.open(archive, "rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        # A hard link is atomic and fails if the destination already exists.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = arguments()
    repo_root = args.repo_root.expanduser().resolve()
    output_root = (args.output_root or repo_root / "extracted_qb").expanduser().resolve()
    selected = list(FAMILIES) if args.all else [args.family]

    print(f"Source repository: {repo_root}")
    print(f"Output dataset:    {output_root}")
    print(f"Mode:              {'DRY RUN' if args.dry_run else 'EXTRACT'}")

    archives = []
    errors = 0
    for family in selected:
        source_root = repo_root / FAMILIES[family]
        if not source_root.is_dir():
            print(f"[ERROR] Missing source directory: {source_root}")
            errors += 1
            continue
        found = sorted(source_root.rglob("*.qs.xz"))
        print(f"Discovered {len(found)} archives in {family}")
        archives.extend((family, source_root, archive) for archive in found)

    if args.limit is not None:
        archives = archives[:args.limit]
    counts = {"planned": 0, "extracted": 0, "verified_existing": 0, "conflict": 0}

    for family, source_root, archive in archives:
        destination = destination_for(archive, source_root, output_root, family)
        try:
            if destination.exists():
                if not destination.is_file() or not content_matches(archive, destination):
                    counts["conflict"] += 1
                    print(f"[CONFLICT] Existing output differs: {destination}")
                else:
                    counts["verified_existing"] += 1
                    if args.verbose or args.dry_run:
                        print(f"[VERIFIED] {destination}")
            elif args.dry_run:
                counts["planned"] += 1
                print(f"[WOULD EXTRACT] {archive.relative_to(repo_root)} -> {destination.relative_to(output_root)}")
            else:
                extract_without_overwriting(archive, destination)
                counts["extracted"] += 1
                if args.verbose:
                    print(f"[EXTRACTED] {destination}")
        except (OSError, EOFError, lzma.LZMAError) as exc:
            errors += 1
            print(f"[ERROR] {archive}: {exc}")

    print("\nSummary:")
    print(f"  Archives selected: {len(archives)}")
    for key, value in counts.items():
        print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    print(f"  Errors: {errors}")
    return 1 if errors or counts["conflict"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
