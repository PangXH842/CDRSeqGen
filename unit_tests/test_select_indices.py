import random
import unittest

from cdrseqgen.methods.mlm.evaluation import bucket_of, select_indices


class SelectIndicesTests(unittest.TestCase):
    """Guards the draw that decides which frameworks every generative number is computed on.

    An under-populated length bucket cannot fill its quota, and without redistribution the draw
    silently returns fewer than k examples. That is not hypothetical: eight runs of the 07-29 batch
    had their generative metrics computed on ~half the intended sample before it was fixed, and the
    failure is invisible in the output because the summary just reports a smaller n.
    """

    def _skewed_lengths(self):
        # 749 short loops, 1 medium, 0 long — the SD2 H1 shape that caused the original incident
        return [5] * 749 + [14]

    def test_a_starved_bucket_does_not_shrink_the_draw(self):
        picked = select_indices(self._skewed_lengths(), 150, True, random.Random(42))
        self.assertEqual(len(picked), 150)

    def test_stratified_draw_is_still_without_replacement(self):
        picked = select_indices(self._skewed_lengths(), 150, True, random.Random(42))
        self.assertEqual(len(set(picked)), len(picked))
        self.assertTrue(all(0 <= i < 750 for i in picked))

    def test_stratification_still_over_represents_the_rare_bucket(self):
        """The point of stratifying: the lone medium loop must survive a 150-of-750 draw."""
        lengths = self._skewed_lengths()
        picked = select_indices(lengths, 150, True, random.Random(42))
        self.assertIn("medium", {bucket_of(lengths[i]) for i in picked})

    def test_balanced_data_fills_every_bucket(self):
        lengths = [5] * 100 + [14] * 100 + [25] * 100
        picked = select_indices(lengths, 30, True, random.Random(0))
        self.assertEqual(len(picked), 30)
        self.assertEqual({bucket_of(lengths[i]) for i in picked}, {"short", "medium", "long"})

    def test_unstratified_draw_returns_exactly_k(self):
        picked = select_indices([5] * 749 + [14], 150, False, random.Random(42))
        self.assertEqual(len(picked), 150)
        self.assertEqual(len(set(picked)), 150)

    def test_requesting_everything_returns_every_index_in_order(self):
        lengths = [5, 14, 25, 9]
        for k in (-1, 4, 99):
            with self.subTest(k=k):
                self.assertEqual(select_indices(lengths, k, True, random.Random(1)), [0, 1, 2, 3])

    def test_selection_is_reproducible_for_a_seed(self):
        lengths = self._skewed_lengths()
        first = select_indices(lengths, 150, True, random.Random(7))
        second = select_indices(lengths, 150, True, random.Random(7))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
