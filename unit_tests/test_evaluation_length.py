import unittest

from cdrseqgen.methods.mlm.length import resolve_generation_length_policy


class EvaluationLengthModeTests(unittest.TestCase):
    def test_modes_resolve_without_silent_bound_ignoring(self):
        self.assertIsNone(resolve_generation_length_policy("fixed_native"))
        self.assertEqual(resolve_generation_length_policy("inferred").mode, "inferred")
        bounded = resolve_generation_length_policy("bounded", min_length=11, max_length=15)
        self.assertEqual(bounded.mode, "bounded")
        self.assertEqual((bounded.constraints.minimum, bounded.constraints.maximum), (11, 15))

    def test_contradictory_modes_fail(self):
        with self.assertRaisesRegex(ValueError, "inferred mode"):
            resolve_generation_length_policy("inferred", min_length=11)
        with self.assertRaisesRegex(ValueError, "fixed_native mode"):
            resolve_generation_length_policy("fixed_native", max_length=15)
        with self.assertRaisesRegex(ValueError, "bounded mode"):
            resolve_generation_length_policy("bounded")
        with self.assertRaisesRegex(ValueError, "equal bounds"):
            resolve_generation_length_policy("bounded", min_length=12, max_length=12)


if __name__ == "__main__":
    unittest.main()
