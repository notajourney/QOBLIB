"""
Layer 3a: source-preserving interaction-graph adapters.

These functions use only the *support* of stored off-diagonal QUBO terms.
They do NOT infer the matrix convention, multiply coefficients by two,
invert signs, or interpret the encoded objective. Graphs are meant for
minor embedding, not as energy-equivalent QUBO models.

Indices remain 1-based. All declared variables, including isolated ones,
are retained when creating a NetworkX graph.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TypeAlias

from .model import QuboInstance

Edge: TypeAlias = tuple[int, int]
RawWeightedEdge: TypeAlias = tuple[int, int, Decimal]


def to_edge_list(
    instance: QuboInstance, *, include_coefficients: bool = False
) -> tuple[Edge, ...] | tuple[RawWeightedEdge, ...]:
    """
    Get source-order, 1-based interaction edges (i, j), i < j.

    If include_coefficients=True, append the **raw stored** coefficient,
    NOT the polynomial coefficient or a solver-ready coupling.

    The edge list alone cannot represent isolated variables; pair it with
    instance.num_variables/instance.variables or use to_networkx().
    """
    if include_coefficients:
        return tuple(
            (t.i, t.j, t.coefficient) for t in instance.raw_terms if t.i < t.j
        )
    return tuple((t.i, t.j) for t in instance.raw_terms if t.i < t.j)


def to_interaction_dict(instance: QuboInstance) -> dict[Edge, Decimal]:
    """
    Return a fresh {(i, j): raw_coefficient} mapping, i < j, 1-based.

    These are source matrix entries, not normalized BQM quadratic biases.
    """
    return instance.raw_off_diagonal


def to_networkx(instance: QuboInstance):
    """
    Build an undirected NX graph for *topology/embedding* studies.

    - All declared nodes 1..n are retained, including isolated nodes.
    - Every stored off-diagonal pair becomes exactly one edge.
    - Edge 'raw_coefficient' is the stored entry; NOT a QUBO polynomial bias.
    - Node 'raw_diagonal_coefficient' is set only for stored diagonal entries.
    - The conventional NX edge attribute 'weight' is deliberately NOT set:
      consumers must not mistake raw entries for normalized couplings.

    NetworkX is an optional dependency, imported only when this is called.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "to_networkx() requires NetworkX; install it with 'python3 -m pip install networkx'"
        ) from exc

    graph = nx.Graph()
    graph.add_nodes_from(instance.variables)
    for term in instance.raw_terms:
        if term.i == term.j:
            graph.nodes[term.i]["raw_diagonal_coefficient"] = term.coefficient
        else:
            graph.add_edge(term.i, term.j, raw_coefficient=term.coefficient)

    graph.graph.update(
        instance_id=instance.instance_id,
        family=instance.family,
        source_path=str(instance.source_path),
        num_variables=instance.num_variables,
        offset=instance.offset,
        index_base=instance.index_base,
        declared_direction=instance.declared_direction,
        stored_direction=instance.stored_direction,
        coefficient_convention=instance.coefficient_convention,
        edge_coefficient_semantics="raw_stored_off_diagonal_entry",
    )
    return graph
