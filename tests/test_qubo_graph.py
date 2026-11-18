"""Run: python3 -m unittest discover -s tests -p 'test_qubo_*.py' -v"""

from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from my_code.qubo_reader.graph import (
    to_edge_list, to_interaction_dict, to_networkx
)
from my_code.qubo_reader.model import load_qs


class GraphAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def load(self, data: str):
        file = Path(self.temp.name) / "q.qs"
        file.write_text(data, encoding="utf-8")
        return load_qs(file)

    def test_sparse_raw_edges_and_no_factor_two(self):
        inst = self.load(
            "# ObjectiveOffset 2\n# Vars Non-zeros\n4 4\n"
            "1 1 -7\n1 3 11.3\n2 3 -4\n4 4 5\n"
        )
        self.assertEqual(to_edge_list(inst), ((1, 3), (2, 3)))
        self.assertEqual(
            to_edge_list(inst, include_coefficients=True),
            ((1, 3, Decimal("11.3")), (2, 3, Decimal("-4"))),
        )
        self.assertEqual(
            to_interaction_dict(inst),
            {(1, 3): Decimal("11.3"), (2, 3): Decimal("-4")},
        )
        self.assertEqual(inst.offset, Decimal("2"))
        self.assertEqual(inst.coefficient_convention, "unverified")

    def test_interaction_dictionary_does_not_modify_instance(self):
        inst = self.load("# Vars Non-zeros\n2 1\n1 2 3\n")
        interactions = to_interaction_dict(inst)
        interactions[(1, 2)] = Decimal("20")
        self.assertEqual(inst.raw_off_diagonal, {(1, 2): Decimal("3")})

    def test_networkx_includes_all_isolated_nodes(self):
        inst = self.load("# Vars Non-zeros\n5 2\n1 3 2\n2 2 -9\n")
        graph = to_networkx(inst)
        self.assertEqual(sorted(graph.nodes), [1, 2, 3, 4, 5])
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(graph[1][3]["raw_coefficient"], Decimal("2"))
        self.assertNotIn("weight", graph[1][3])
        self.assertEqual(graph.nodes[2]["raw_diagonal_coefficient"], Decimal("-9"))
        self.assertEqual(set(graph.degree([4, 5])), {(4, 0), (5, 0)})
        self.assertEqual(graph.graph["edge_coefficient_semantics"],
                         "raw_stored_off_diagonal_entry")

    def test_mis_source_maximize_is_not_inferred_as_energy_maximize(self):
        inst = self.load("# maximize obj: ...\n# Vars Non-zeros\n2 3\n1 1 -1\n2 2 -1\n1 2 1\n")
        graph = to_networkx(inst)
        self.assertEqual(graph.graph["declared_direction"], "maximize")
        self.assertEqual(graph.graph["stored_direction"], "unknown")
        self.assertEqual(graph[1][2]["raw_coefficient"], Decimal("1"))
        self.assertEqual(graph.nodes[1]["raw_diagonal_coefficient"], Decimal("-1"))

    def test_dimension_with_no_terms(self):
        inst = self.load("# Vars Non-zeros\n3 0\n")
        self.assertEqual(to_edge_list(inst), ())
        self.assertEqual(to_interaction_dict(inst), {})
        graph = to_networkx(inst)
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 0)

    def test_graph_counts_match_canonical_metrics(self):
        inst = self.load("# Vars Non-zeros\n4 4\n1 1 3\n1 2 2\n1 4 9\n2 3 -5\n")
        graph = to_networkx(inst)
        self.assertEqual(graph.number_of_nodes(), inst.num_variables)
        self.assertEqual(graph.number_of_edges(), inst.num_interactions)
        self.assertEqual(len(to_edge_list(inst)), inst.num_interactions)


if __name__ == "__main__":
    unittest.main()
