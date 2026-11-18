"""Composable, source-preserving QUBO reader.

Layer 1: .qs.parse_qs() -- raw format parsing.
Layer 2: .model.load_qs() -- canonical sparse data + provenance.
Layer 3: future explicit interpretation and solver adapters.
"""

from .model import QuboInstance, QuboTerm, from_qs, load_qs

__all__ = ["QuboInstance", "QuboTerm", "from_qs", "load_qs"]
