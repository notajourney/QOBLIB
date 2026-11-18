"""Tests for the simple interpretation + convention-conversion layer.

Run from the QOBLIB repository root:
    python3 -m unittest discover -s tests -p 'test_qubo_*.py' -v
"""

import itertools
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from my_code.qubo_reader import load_qs
from my_code.qubo_reader.conventions import (
    interpret_qubo,
    convert_qubo,
    evaluate_interpreted,
)


class ConventionTests(unittest.TestCase):
    def setUp(self):
        """Input: none; Output: temp directory; Role: isolate sample files."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def raw(self, body):
        """Input: sample QS file text; Output: raw QuboInstance."""
        path = Path(self.temp.name) / "toy.qs"
        path.write_text(body, encoding="utf-8")
        return load_qs(path)

    def target_energy(self, converted, bits):
        """Independent test evaluator of target stored matrix convention."""
        energy = converted.offset
        for (i, j), coeff in converted.coefficients.items():
            multiplier = 2 if i != j and converted.convention == "symmetric" else 1
            energy += coeff * multiplier * bits[i - 1] * bits[j - 1]
        return energy

    def test_two_interpretations_of_exactly_the_same_raw_input(self):
        raw = self.raw("# ObjectiveOffset 5\n# Vars Non-zeros\n3 2\n1 1 -1\n1 2 3\n")
        symmetric = interpret_qubo(raw, source_convention="symmetric", source_direction="minimize")
        triangular = interpret_qubo(raw, source_convention="upper_triangular", source_direction="minimize")

        self.assertEqual(symmetric.quadratic[(1, 2)], Decimal(6))
        self.assertEqual(triangular.quadratic[(1, 2)], Decimal(3))
        self.assertEqual(evaluate_interpreted(symmetric, [1, 1, 0]), Decimal(10))
        self.assertEqual(evaluate_interpreted(triangular, [1, 1, 0]), Decimal(7))
        self.assertEqual(raw.raw_off_diagonal[(1, 2)], Decimal(3))

    def test_all_sixteen_source_target_combinations(self):
        raw = self.raw("# ObjectiveOffset 11.25\n# Vars Non-zeros\n4 4\n1 1 -7\n1 3 11.3\n2 3 -4\n4 4 5\n")
        choices = itertools.product(
            ("symmetric", "upper_triangular"),
            ("minimize", "maximize"),
            ("symmetric", "upper_triangular"),
            ("minimize", "maximize"),
        )
        for source_matrix, source_direction, target_matrix, target_direction in choices:
            with self.subTest(source=(source_matrix, source_direction),
                              target=(target_matrix, target_direction)):
                interpreted = interpret_qubo(raw,
                    source_convention=source_matrix, source_direction=source_direction)
                converted = convert_qubo(interpreted,
                    target_convention=target_matrix, target_direction=target_direction)
                sign = 1 if source_direction == target_direction else -1
                for bits in itertools.product((0, 1), repeat=4):
                    expected = sign * evaluate_interpreted(interpreted, bits)
                    self.assertEqual(self.target_energy(converted, bits), expected)

    def test_specific_symmetric_min_to_triangular_max(self):
        raw = self.raw("# ObjectiveOffset 3\n# Vars Non-zeros\n2 3\n1 1 -1\n1 2 1\n2 2 -1\n")
        interpreted = interpret_qubo(raw, source_convention="symmetric", source_direction="minimize")
        converted = convert_qubo(interpreted,
            target_convention="upper_triangular", target_direction="maximize")
        self.assertEqual(converted.coefficients, {
            (1, 1): Decimal(1), (1, 2): Decimal(-2), (2, 2): Decimal(1)})
        self.assertEqual(converted.offset, Decimal(-3))
        self.assertEqual(converted.energy_sign, -1)
        self.assertEqual(self.target_energy(converted, [1, 1]), Decimal(-3))

    def test_comment_is_not_assumed_to_be_stored_direction(self):
        raw = self.raw("# maximize obj: original\n# ObjectiveOffset 0\n# Vars Non-zeros\n2 2\n1 1 -1\n1 2 1\n")
        interpreted = interpret_qubo(raw, source_convention="symmetric",
            source_direction="minimize", assumption_note="MIS example")
        self.assertEqual(raw.declared_direction, "maximize")
        self.assertEqual(interpreted.direction, "minimize")
        self.assertEqual(interpreted.assumption_note, "MIS example")

    def test_offset_must_be_explicit_when_source_is_missing(self):
        raw = self.raw("# Vars Non-zeros\n2 1\n1 2 5\n")
        with self.assertRaisesRegex(ValueError, "offset missing"):
            interpret_qubo(raw, source_convention="symmetric", source_direction="minimize")
        interpreted = interpret_qubo(raw, source_convention="symmetric",
            source_direction="minimize", offset_override="0")
        self.assertEqual(evaluate_interpreted(interpreted, [1, 1]), Decimal(10))
        self.assertIsNone(raw.offset)

    def test_invalid_inputs_rejected(self):
        raw = self.raw("# ObjectiveOffset 0\n# Vars Non-zeros\n2 1\n1 2 3\n")
        with self.assertRaises(ValueError):
            interpret_qubo(raw, source_convention="unknown", source_direction="minimize")
        with self.assertRaises(TypeError):
            interpret_qubo(raw, source_convention="symmetric", source_direction="minimize", offset_override=0.1)
        interpreted = interpret_qubo(raw, source_convention="symmetric", source_direction="maximize")
        with self.assertRaises(ValueError):
            convert_qubo(interpreted, target_convention="symmetric", target_direction="unknown")
        with self.assertRaises(ValueError):
            evaluate_interpreted(interpreted, [1])
        with self.assertRaises(ValueError):
            evaluate_interpreted(interpreted, [1, 2])

    def test_absent_diagonal_and_isolated_variable_keep_dimension(self):
        raw = self.raw("# ObjectiveOffset 0\n# Vars Non-zeros\n4 1\n1 2 5\n")
        interpreted = interpret_qubo(raw, source_convention="upper_triangular",
                                     source_direction="minimize")
        converted = convert_qubo(interpreted, target_convention="symmetric",
                                 target_direction="minimize")
        self.assertEqual(converted.num_variables, 4)
        self.assertEqual(converted.coefficients, {(1, 2): Decimal("2.5")})
        self.assertEqual(evaluate_interpreted(interpreted, [1, 1, 0, 0]), Decimal(5))


if __name__ == "__main__":
    unittest.main()
