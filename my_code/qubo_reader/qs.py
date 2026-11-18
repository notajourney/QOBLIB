"""Read Zimpl .qs files without choosing a QUBO matrix/solver convention.

Parsed entries are the ORIGINAL stored triplets (1-based indices). We do not
symmetrize, double coefficients, invert signs, or assume an optimization sense.

Usage, from QOBLIB repository root:
    python3 -m my_code.qubo_reader.qs extracted_qb/labs/labs006/labs006.qs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re


class QSFormatError(ValueError):
    """File is not a valid .qs instance under the expected Zimpl format."""


@dataclass(frozen=True)
class QSTerm:
    i: int  # 1-based index, exactly as stored
    j: int
    coefficient: Decimal


@dataclass(frozen=True)
class QSInstance:
    source: Path
    num_variables: int
    declared_nonzero_count: int
    offset: Decimal | None
    declared_direction: str  # From source comments only; NOT the stored QUBO direction
    terms: tuple[QSTerm, ...]
    # Raw preamble comments, retained for provenance (not interpreted).
    comments: tuple[str, ...] = ()

    @property
    def num_interactions(self) -> int:
        """Nonzero off-diagonal pairs, assuming each stored pair appears once."""
        return sum(t.i != t.j for t in self.terms)

    @property
    def num_diagonal_terms(self) -> int:
        return sum(t.i == t.j for t in self.terms)

    @property
    def graph_density(self) -> float:
        n = self.num_variables
        return (2 * self.num_interactions / (n * (n - 1))) if n > 1 else 0.0


_OFFSET = re.compile(r"^#\s*ObjectiveOffset\s+(\S+)\s*$", re.IGNORECASE)
_HEADER = re.compile(r"^#\s*Vars\s+Non-zeros\s*$", re.IGNORECASE)
_DIRECTION = re.compile(r"^#\s*(minimize|maximize)\b", re.IGNORECASE)


def _decimal(raw: str, path: Path, line: int) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise QSFormatError(f"{path}:{line}: invalid coefficient/offset {raw!r}") from exc
    if not value.is_finite():
        raise QSFormatError(f"{path}:{line}: nonfinite coefficient/offset {raw!r}")
    return value


def parse_qs(path: str | Path) -> QSInstance:
    """Parse one .qs text file; validate row counts, indices and stored triangle.

    Deliberately excludes semantic claims (objective direction, off-diagonal
    polynomial scaling, optimality). The source model comment may report its
    DECLARED direction; a stored representation's direction is not inferred.
    """
    path = Path(path)
    state = "preamble"
    direction = "unknown"
    offset = None
    n = count = None
    terms: list[QSTerm] = []
    comments: list[str] = []
    seen: set[tuple[int, int]] = set()

    with path.open("r", encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if state == "preamble":
                if line.startswith("#"):
                    comments.append(line)
                mo = _OFFSET.match(line)
                if mo:
                    if offset is not None:
                        raise QSFormatError(f"{path}:{lineno}: duplicate ObjectiveOffset")
                    offset = _decimal(mo.group(1), path, lineno)
                    continue
                if _HEADER.match(line):
                    state = "dimensions"
                    continue
                md = _DIRECTION.match(line)
                if md:
                    found = md.group(1).lower()
                    if direction != "unknown" and direction != found:
                        raise QSFormatError(f"{path}:{lineno}: conflicting directions in comments")
                    direction = found
                elif not line.startswith("#"):
                    raise QSFormatError(f"{path}:{lineno}: unexpected content before header: {line!r}")
            elif state == "dimensions":
                if line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) != 2:
                    raise QSFormatError(f"{path}:{lineno}: expected '<Vars> <Non-zeros>'")
                try:
                    n, count = map(int, tokens)
                except ValueError as exc:
                    raise QSFormatError(f"{path}:{lineno}: invalid header counts") from exc
                if n < 1 or count < 0:
                    raise QSFormatError(f"{path}:{lineno}: invalid dimension/term count")
                state = "terms"
            else:
                if line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) != 3:
                    raise QSFormatError(f"{path}:{lineno}: expected 'i j coefficient'")
                try:
                    i, j = int(tokens[0]), int(tokens[1])
                except ValueError as exc:
                    raise QSFormatError(f"{path}:{lineno}: invalid variable indices") from exc
                if not (1 <= i <= j <= n):
                    raise QSFormatError(f"{path}:{lineno}: expected 1 <= i <= j <= {n}, got ({i},{j})")
                if (i, j) in seen:
                    raise QSFormatError(f"{path}:{lineno}: duplicate coefficient pair ({i},{j})")
                seen.add((i, j))
                value = _decimal(tokens[2], path, lineno)
                if value == 0:
                    raise QSFormatError(f"{path}:{lineno}: zero coefficient in nonzero list")
                terms.append(QSTerm(i, j, value))
                if len(terms) > count:
                    raise QSFormatError(f"{path}:{lineno}: more entries than declared ({count})")

    if state != "terms":
        raise QSFormatError(f"{path}: missing 'Vars Non-zeros' header/dimensions")
    if len(terms) != count:
        raise QSFormatError(f"{path}: declared {count} terms but parsed {len(terms)}")
    return QSInstance(path, n, count, offset, direction, tuple(terms), tuple(comments))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path)
    args = cli.parse_args()
    item = parse_qs(args.file)
    print(f"source: {item.source}")
    print(f"num_variables: {item.num_variables}")
    print(f"stored_nonzero_terms: {item.declared_nonzero_count}")
    print(f"num_interactions: {item.num_interactions}")
    print(f"num_diagonal_terms: {item.num_diagonal_terms}")
    print(f"graph_density: {item.graph_density:.6f}")
    print(f"offset: {item.offset if item.offset is not None else 'unknown'}")
    print(f"declared_direction_from_comments: {item.declared_direction}")
    print("stored_qubo_direction: unknown (not inferred)")
    print("coefficient_convention: unverified (not inferred)")
    print("structural_validation: passed")


if __name__ == "__main__":
    main()
