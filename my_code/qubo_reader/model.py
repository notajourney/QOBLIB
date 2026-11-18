"""
Layer 2: immutable, source-preserving sparse QUBO records.

This module intentionally does NOT normalize off-diagonal coefficients, reverse
optimization sense, infer source-model/encoded-objective equivalence, or evaluate
energy. Those operations belong to explicit interpretation/adapters (layer 3).

Examples from the QOBLIB repository root:

    from my_code.qubo_reader.model import load_qs
    q = load_qs('extracted_qb/labs/labs006/labs006.qs', dataset_root='extracted_qb')
    assert q.num_variables == 21 and q.num_interactions == 117
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .qs import QSInstance, parse_qs


@dataclass(frozen=True, slots=True)
class QuboTerm:
    """A *stored* triplet. No implicit doubling, symmetrization or sign change."""
    i: int
    j: int
    coefficient: Decimal


@dataclass(frozen=True, slots=True)
class QuboInstance:
    """Sparse raw QUBO data + provenance, independent of a particular solver.

    Indices are 1-based and include all declared variables, even if no term
    mentions a variable. `raw_terms` are entries from the source, not polynomial
    coefficients. In particular, `raw_off_diagonal` is NOT a dimod quadratic dict.

    Data fields are immutable. Derived views are freshly computed, so no cached
    statistics can become inconsistent with the coefficients.
    """
    instance_id: str
    source_path: Path
    source_format: str
    family: str
    num_variables: int
    declared_nonzero_count: int
    raw_terms: tuple[QuboTerm, ...]
    offset: Decimal | None
    declared_direction: str
    source_comments: tuple[str, ...] = ()
    # Intentionally unresolved until interpretation and independent validation.
    stored_direction: str = "unknown"
    direction_relation: str = "unknown"
    coefficient_convention: str = "unverified"
    structural_validation: str = "passed"
    index_base: int = 1

    def __post_init__(self) -> None:
        if self.index_base != 1:
            raise ValueError("This v0.2 representation requires 1-based variable indices")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.num_variables < 1:
            raise ValueError("num_variables must be positive")
        if self.declared_nonzero_count != len(self.raw_terms):
            raise ValueError("declared_nonzero_count differs from raw_terms length")
        if self.declared_direction not in ("minimize", "maximize", "unknown"):
            raise ValueError("invalid declared_direction")
        pairs: set[tuple[int, int]] = set()
        for t in self.raw_terms:
            if not 1 <= t.i <= t.j <= self.num_variables:
                raise ValueError(f"term ({t.i}, {t.j}) violates 1-based upper triangle")
            if not t.coefficient.is_finite() or t.coefficient == 0:
                raise ValueError(f"term ({t.i}, {t.j}) has invalid coefficient")
            if (t.i, t.j) in pairs:
                raise ValueError(f"duplicate pair ({t.i}, {t.j})")
            pairs.add((t.i, t.j))
        if self.offset is not None and not self.offset.is_finite():
            raise ValueError("nonfinite offset")

    @property
    def variables(self) -> range:
        """All declared variable labels, including labels absent from terms."""
        return range(1, self.num_variables + 1)

    @property
    def num_stored_terms(self) -> int:
        return len(self.raw_terms)

    @property
    def num_diagonal_terms(self) -> int:
        return sum(t.i == t.j for t in self.raw_terms)

    @property
    def num_interactions(self) -> int:
        return sum(t.i < t.j for t in self.raw_terms)

    @property
    def graph_density(self) -> float:
        n = self.num_variables
        return 2 * self.num_interactions / (n * (n - 1)) if n > 1 else 0.0

    @property
    def raw_diagonal(self) -> dict[int, Decimal]:
        """Original stored diagonal coefficients (absent variables omitted)."""
        return {t.i: t.coefficient for t in self.raw_terms if t.i == t.j}

    @property
    def raw_off_diagonal(self) -> dict[tuple[int, int], Decimal]:
        """Original stored off-diagonal entries, NOT normalized polynomial biases."""
        return {(t.i, t.j): t.coefficient for t in self.raw_terms if t.i < t.j}


def from_qs(parsed: QSInstance, *, family: str = "unknown", instance_id: str | None = None) -> QuboInstance:
    """Adapt an already parsed .qs file into the source-preserving core model."""
    return QuboInstance(
        instance_id=instance_id if instance_id is not None else parsed.source.stem,
        source_path=parsed.source,
        source_format="zimpl_qs",
        family=family,
        num_variables=parsed.num_variables,
        declared_nonzero_count=parsed.declared_nonzero_count,
        raw_terms=tuple(QuboTerm(t.i, t.j, t.coefficient) for t in parsed.terms),
        offset=parsed.offset,
        declared_direction=parsed.declared_direction,
        source_comments=parsed.comments,
    )


def load_qs(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    family: str | None = None,
    instance_id: str | None = None,
) -> QuboInstance:
    """Parse one .qs via v0.1's parser and adapt it to QuboInstance.

    With dataset_root, family and family-relative instance_id are inferred from
    the relative file path, matching the existing inventory CSV. With no root,
    the caller may supply family/id; otherwise family='unknown', id=file stem.
    Parsing does not traverse directories or write any files.
    """
    parsed = parse_qs(path)
    resolved_family = family or "unknown"
    resolved_id = instance_id
    if dataset_root is not None:
        root = Path(dataset_root).resolve()
        try:
            relative = parsed.source.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f".qs path {parsed.source} is outside dataset_root {root}") from exc
        if len(relative.parts) < 2:
            raise ValueError("Expected dataset_root/<family>/<instance>.qs layout")
        detected_family = relative.parts[0]
        if family is not None and family != detected_family:
            raise ValueError(f"family={family!r} disagrees with path family={detected_family!r}")
        resolved_family = detected_family
        if resolved_id is None:
            resolved_id = Path(*relative.parts[1:]).with_suffix("").as_posix()
    return from_qs(parsed, family=resolved_family, instance_id=resolved_id)
