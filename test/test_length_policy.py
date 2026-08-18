import unittest

from cdrseqgen.methods.mlm.length import (
    LengthConstraints,
    native_mask_length,
    resolve_length_policy,
)


class LengthPolicyTest(unittest.TestCase):
    def test_equal_bounds_are_an_explicit_fixed_length(self):
        policy = resolve_length_policy(min_length=12, max_length=12)
        self.assertEqual(policy.mode, "fixed")
        self.assertEqual(policy.length, 12)
        self.assertEqual(policy.source, "explicit_bounds")

    def test_native_length_preserves_current_evaluation_policy(self):
        decision = native_mask_length(9)
        self.assertEqual(decision.length, 9)
        self.assertEqual(decision.source, "native_mask_count")

    def test_native_truth_takes_precedence_when_request_agrees(self):
        policy = resolve_length_policy(
            native_length=11, use_native_length=True, min_length=11, max_length=11
        )
        self.assertEqual(policy.length, 11)
        self.assertEqual(policy.source, "native_truth")

    def test_deprecated_target_length_remains_supported(self):
        policy = resolve_length_policy(target_length=10)
        self.assertEqual(policy.length, 10)
        self.assertEqual(policy.source, "legacy_target_length")

    def test_ranges_are_general_policies_with_inclusive_bounds(self):
        policies = [
            resolve_length_policy(min_length=11, max_length=15),
            resolve_length_policy(min_length=11),
            resolve_length_policy(max_length=15),
        ]
        self.assertTrue(all(policy.mode == "bounded" for policy in policies))
        self.assertTrue(all(policy.requires_predictor for policy in policies))
        self.assertEqual(policies[0].constraints.as_dict(), {"minimum": 11, "maximum": 15})
        self.assertTrue(policies[0].constraints.contains(11))
        self.assertTrue(policies[0].constraints.contains(15))
        self.assertFalse(policies[0].constraints.contains(16))

    def test_no_bounds_is_inferred_and_native_truth_does_not_leak(self):
        policy = resolve_length_policy(native_length=12)
        self.assertEqual(policy.mode, "inferred")
        self.assertTrue(policy.requires_predictor)
        fixed = resolve_length_policy(native_length=12, use_native_length=True)
        self.assertEqual(fixed.length, 12)
        self.assertEqual(fixed.source, "native_truth")

    def test_use_native_length_without_a_native_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a native length"):
            resolve_length_policy(use_native_length=True)

    def test_conflicting_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_length_policy(target_length=12, min_length=12, max_length=12)
        with self.assertRaisesRegex(ValueError, "conflicts with native"):
            resolve_length_policy(
                native_length=10, use_native_length=True, min_length=12, max_length=12
            )

    def test_invalid_constraints_are_rejected(self):
        for kwargs in (
            {"min_length": 0, "max_length": 0},
            {"min_length": 15, "max_length": 11},
            {"min_length": True, "max_length": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LengthConstraints(
                        minimum=kwargs["min_length"], maximum=kwargs["max_length"]
                    )


if __name__ == "__main__":
    unittest.main()
