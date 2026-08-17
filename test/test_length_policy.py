import unittest

from cdrseqgen.methods.mlm.length import (
    LengthConstraints,
    UnsupportedLengthPolicy,
    native_mask_length,
    resolve_fixed_length,
    resolve_length_policy,
)


class LengthPolicyTest(unittest.TestCase):
    def test_equal_bounds_are_an_explicit_fixed_length(self):
        decision = resolve_fixed_length(min_length=12, max_length=12)
        self.assertEqual(decision.length, 12)
        self.assertEqual(
            decision.as_dict(),
            {
                "mode": "fixed",
                "source": "explicit_bounds",
                "minimum": 12,
                "maximum": 12,
                "selected_length": 12,
            },
        )

    def test_native_length_preserves_current_evaluation_policy(self):
        decision = native_mask_length(9)
        self.assertEqual(decision.length, 9)
        self.assertEqual(decision.source, "native_mask_count")

    def test_native_truth_takes_precedence_when_request_agrees(self):
        decision = resolve_fixed_length(
            native_length=11, min_length=11, max_length=11
        )
        self.assertEqual(decision.length, 11)
        self.assertEqual(decision.source, "native_truth")

    def test_deprecated_target_length_remains_supported(self):
        decision = resolve_fixed_length(target_length=10)
        self.assertEqual(decision.length, 10)
        self.assertEqual(decision.source, "legacy_target_length")

    def test_legacy_default_remains_supported(self):
        decision = resolve_fixed_length(fallback_length=10)
        self.assertEqual(decision.length, 10)
        self.assertEqual(decision.source, "legacy_default")

    def test_real_ranges_are_reserved_for_stage_three(self):
        for kwargs in (
            {"min_length": 11, "max_length": 15},
            {"min_length": 11},
            {"max_length": 15},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    UnsupportedLengthPolicy, "variable-length generation"
                ):
                    resolve_fixed_length(**kwargs)

    def test_ranges_are_general_policies_with_inclusive_bounds(self):
        policies = [
            resolve_length_policy(min_length=11, max_length=15),
            resolve_length_policy(min_length=11),
            resolve_length_policy(max_length=15),
        ]
        self.assertTrue(all(policy.mode == "bounded" for policy in policies))
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

    def test_missing_length_without_fallback_is_not_inferred_yet(self):
        with self.assertRaisesRegex(UnsupportedLengthPolicy, "inferred-length"):
            resolve_fixed_length()

    def test_conflicting_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_fixed_length(
                target_length=12, min_length=12, max_length=12
            )
        with self.assertRaisesRegex(ValueError, "conflicts with native"):
            resolve_fixed_length(native_length=10, min_length=12, max_length=12)

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
