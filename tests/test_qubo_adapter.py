
import itertools
import tempfile
import unittest
from pathlib import Path

import dimod

from my_code.qubo_reader import load_qs
from my_code.qubo_reader.conventions import (
    interpret_qubo,
    convert_qubo,
    evaluate_interpreted,
)
from my_code.qubo_reader.adapters import to_dimod_bqm
from decimal import Decimal
from my_code.qubo_reader.adapters import (to_dimod_bqm,
                                          to_networkx_graph,
                                          to_objective_function,
                                          to_mqlib_file)




class TestDimodAdapter(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        path = Path(self.temp.name) / "toy.qs"
        path.write_text(
            "# ObjectiveOffset 7\n"
            "# Vars Non-zeros\n"
            "4 2\n"
            "1 1 -2\n"
            "1 2 3\n",
            encoding="utf-8",
        )

        self.raw = load_qs(path)

    def test_creates_bqm_and_preserves_variables(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        bqm = to_dimod_bqm(converted)

        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(set(bqm.variables), {1, 2, 3, 4})
        self.assertEqual(bqm.linear[1], -2)
        self.assertEqual(bqm.linear[3], 0)
        self.assertEqual(bqm.quadratic[1, 2], 3)
        self.assertEqual(bqm.offset, 7)

        # The BQM must evaluate the same objective as our model.
        for bits in itertools.product((0, 1), repeat=4):
            sample = dict(enumerate(bits, start=1))
            expected = evaluate_interpreted(interpreted, bits)
            self.assertEqual(bqm.energy(sample), float(expected))

    def test_symmetric_source_is_not_double_counted(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="symmetric",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        bqm = to_dimod_bqm(converted)

        # Source coefficient 3 became 6 during interpretation.
        # The adapter must preserve 6, not multiply it again.
        self.assertEqual(bqm.quadratic[1, 2], 6)

    def test_rejects_incompatible_conventions(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )

        symmetric = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="minimize",
        )
        with self.assertRaises(ValueError):
            to_dimod_bqm(symmetric)

        maximizing = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="maximize",
        )
        with self.assertRaises(ValueError):
            to_dimod_bqm(maximizing)



#-------------------------nx graph tests------------------------------#
    def test_networkx_graph_without_metadata(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        graph = to_networkx_graph(converted)

        self.assertEqual(set(graph.nodes), {1, 2, 3, 4})
        self.assertEqual(set(graph.edges), {(1, 2)})

        # No QUBO attributes when metadata is disabled.
        self.assertEqual(graph.graph, {})
        self.assertEqual(dict(graph.nodes[1]), {})
        self.assertEqual(dict(graph.edges[1, 2]), {})

    def test_networkx_graph_with_metadata(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        graph = to_networkx_graph(
            converted, include_metadata=True
        )

        self.assertEqual(
            graph.nodes[1]["linear_coefficient"], Decimal("-2")
        )
        self.assertEqual(
            graph.edges[1, 2]["quadratic_coefficient"], Decimal("3")
        )
        self.assertEqual(graph.graph["offset"], Decimal("7"))
        self.assertEqual(graph.graph["convention"], "upper_triangular")
        self.assertEqual(graph.graph["direction"], "minimize")

        # No diagonal term for variable 3.
        self.assertNotIn("linear_coefficient", graph.nodes[3])

    def test_networkx_does_not_change_conventions(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="symmetric",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="maximize",
        )

        graph = to_networkx_graph(
            converted, include_metadata=True
        )

        # Adapter preserves exactly the converted coefficients.
        self.assertEqual(
            graph.edges[1, 2]["quadratic_coefficient"], Decimal("-3")
        )
        self.assertEqual(graph.graph["offset"], Decimal("-7"))
        self.assertEqual(graph.graph["convention"], "symmetric")
        self.assertEqual(graph.graph["direction"], "maximize")

#------------------------objective func tests----------------------------#

    def test_objective_function_energy_equivalence(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="symmetric",
            source_direction="minimize",
        )

        # Test both target conventions and both directions.
        for convention in ("symmetric", "upper_triangular"):
            for direction in ("minimize", "maximize"):
                converted = convert_qubo(
                    interpreted,
                    target_convention=convention,
                    target_direction=direction,
                )

                f = to_objective_function(converted)

                # Compare energies for every binary assignment.
                for bits in itertools.product((0, 1), repeat=4):
                    expected = (
                        converted.energy_sign
                        * evaluate_interpreted(interpreted, bits)
                    )
                    self.assertEqual(f(bits), expected)

    def test_objective_function_accepts_dictionary(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        f = to_objective_function(converted)

        # E = 7 - 2*x1 + 3*x1*x2
        assignment = {1: 1, 2: 1, 3: 0, 4: 0}

        self.assertEqual(f(assignment), 8)
        self.assertEqual(f([1, 1, 0, 0]), 8)

    def test_objective_function_rejects_invalid_assignments(self):
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="minimize",
        )

        f = to_objective_function(converted)

        with self.assertRaises(ValueError):
            f([1, 0, 1])  # Missing variable

        with self.assertRaises(ValueError):
            f([1, 0, 2, 0])  # Not binary

        with self.assertRaises(ValueError):
            f({1: 1, 2: 0})  # Missing dictionary keys

        with self.assertRaises(ValueError):
            f([True, 0, 1, 0])  # Booleans are not accepted

#------------------------temp file tests----------------------------------#

    def test_mqlib_file_content(self):
        """
        Check file format, coefficients, and declared dimensions.

        """
        interpreted = interpret_qubo(
            self.raw, # this is the raw toy qubo defined in setUp
            source_convention="upper_triangular",
            source_direction="minimize",
        ) # we interpret the raw data as upper & minize
        converted = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="maximize",
        ) # converting the interpterted qubo into the adapter convention requirement (symm & max)

        with to_mqlib_file(converted) as path: # the convertor takes a 'ConvertedQubo' object!
            lines = path.read_text(encoding="utf-8").splitlines()

        # Source: E = 7 - 2*x1 + 3*x1*x2
        # Target: -E, symmetric matrix convention.
        self.assertEqual(lines, [
            "4 2",
            "1 1 2",
            "1 2 -1.5",
        ])

        # Variables 3 and 4 still exist through the header.
        # The offset is preserved in ConvertedQubo, not the file.
        self.assertEqual(converted.offset, Decimal("-7"))


    def test_mqlib_file_exists_only_inside_with(self):
        """Check the lifetime of the temporary file and directory."""
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="maximize",
        )

        with to_mqlib_file(converted) as path:
            self.assertTrue(path.is_file())

            directory = path.parent
            self.assertTrue(directory.is_dir())

        # The context manager has finished.
        self.assertFalse(path.exists())
        self.assertFalse(directory.exists())


    def test_mqlib_file_cleanup_when_caller_raises(self):
        """Check cleanup when an error occurs inside the with block."""
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )
        converted = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="maximize",
        )

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with to_mqlib_file(converted) as path:
                self.assertTrue(path.is_file())

                # Simulate a failure during the caller's work.
                raise RuntimeError("test failure")

        # Cleanup must happen even after an exception.
        self.assertFalse(path.exists())


    def test_mqlib_file_rejects_incompatible_input(self):
        """Reject invalid types and incompatible target conventions."""
        interpreted = interpret_qubo(
            self.raw,
            source_convention="upper_triangular",
            source_direction="minimize",
        )

        wrong_matrix = convert_qubo(
            interpreted,
            target_convention="upper_triangular",
            target_direction="maximize",
        )

        wrong_direction = convert_qubo(
            interpreted,
            target_convention="symmetric",
            target_direction="minimize",
        )

        for invalid in (wrong_matrix, wrong_direction):
            with self.subTest(
                convention=invalid.convention,
                direction=invalid.direction,
            ):
                with self.assertRaises(ValueError):
                    with to_mqlib_file(invalid):
                        pass

        # A raw QuboInstance is not a valid adapter input.
        with self.assertRaises(TypeError):
            with to_mqlib_file(self.raw):
                pass







if __name__ == "__main__":
    unittest.main()
