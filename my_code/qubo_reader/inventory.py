"""Build per-instance QOBLib .qs metadata inventories (no semantic assumptions).

From repository root:
    python3 -m my_code.qubo_reader.inventory --family labs --limit 5 --dry-run
    python3 -m my_code.qubo_reader.inventory --family labs
    python3 -m my_code.qubo_reader.inventory --all

Writes CSV into extracted_qb/_reports/, leaving source files unchanged.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

from .qs import QSFormatError, parse_qs

FAMILIES = ("marketsplit", "labs", "portfolio", "independentset")
FIELDS = (
    "family", "instance_id", "relative_file", "num_variables",
    "num_stored_terms", "num_diagonal_terms", "num_interactions",
    "graph_density", "offset", "offset_status", "declared_direction",
    "stored_direction", "direction_relation", "coefficient_convention",
    "structural_validation",
)


def discover(root: Path, families: tuple[str, ...]):
    """Yield (family, .qs path) from a given dataset root; preserves nested paths."""
    for family in families:
        family_dir = root / family
        if not family_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {family_dir}")
        yield from ((family, p) for p in sorted(family_dir.rglob("*.qs")))


def inventory_row(family: str, source: Path, root: Path):
    q = parse_qs(source)
    return {
        "family": family,
        # Relative path is the collision-safe dataset identity.
        "instance_id": source.relative_to(root / family).with_suffix("").as_posix(),
        "relative_file": source.relative_to(root).as_posix(),
        "num_variables": q.num_variables,
        "num_stored_terms": q.declared_nonzero_count,
        "num_diagonal_terms": q.num_diagonal_terms,
        "num_interactions": q.num_interactions,
        "graph_density": f"{q.graph_density:.10f}",
        "offset": "" if q.offset is None else str(q.offset),
        "offset_status": "unknown" if q.offset is None else "explicit",
        "declared_direction": q.declared_direction,
        "stored_direction": "unknown",
        "direction_relation": "unknown",
        "coefficient_convention": "unverified",
        "structural_validation": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--family", choices=FAMILIES)
    select.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("extracted_qb"))
    parser.add_argument("--output", type=Path, help="Override the CSV destination")
    parser.add_argument("--limit", type=int, help="Only the first N instances (testing)")
    parser.add_argument("--dry-run", action="store_true", help="Show summaries without writing CSV")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    root = args.dataset.resolve()
    families = FAMILIES if args.all else (args.family,)
    output = args.output or (root / "_reports" / ("QOBLib_instances_info.csv" if args.all else f"QOBLib_{args.family.upper()}_family_info.csv"))
    chosen = list(discover(root, families))
    if args.limit:
        chosen = chosen[:args.limit]
    print(f"Dataset: {root}")
    print(f"Selected: {len(chosen)} .qs files")

    rows, failures = [], 0
    for family, source in chosen:
        try:
            rows.append(inventory_row(family, source, root))
        except (OSError, UnicodeError, QSFormatError) as exc:
            failures += 1
            print(f"[ERROR] {source.relative_to(root)}: {exc}")
    print(f"Parsed: {len(rows)}; errors: {failures}")
    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        if not family_rows:
            continue
        ns = [r["num_variables"] for r in family_rows]
        print(f"  {family}: {len(ns)} instances; dimensions {min(ns)}..{max(ns)} (median {median(ns):g})")
    if args.dry_run:
        for row in rows[:5]:
            print(f"  [PREVIEW] {row['relative_file']}: n={row['num_variables']} edges={row['num_interactions']} offset={row['offset'] or 'unknown'}")
        print("DRY RUN: No CSV written")
    elif rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Do not publish an incomplete inventory if any instance failed validation.
        if failures:
            print("[NOT WRITTEN] Inventory incomplete; inspect errors first.")
        else:
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
