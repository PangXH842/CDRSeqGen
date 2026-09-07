"""The causal decoder: the properties that make autoregressive training meaningful.

Causality is the one that matters most. If position t can attend to t+1 the model sees the answer it
is meant to predict, training collapses into copying, and every downstream number is meaningless
while looking entirely healthy. It gets tested on both attention paths, because SDPA cannot combine
`is_causal=True` with an explicit mask — padded batches take a different branch, and that branch is
the one training actually uses.
"""
import unittest

import torch

from cdrseqgen.methods.ar.models.transformer import CausalAntibodyModel, build_model


def _model(**kw):
    torch.manual_seed(0)
    return build_model(vocab_size=27, hidden_size=64, num_layers=2, num_attention_heads=4,
                       **kw).eval()


class CausalityTests(unittest.TestCase):
    def test_a_later_token_cannot_change_an_earlier_prediction(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        with torch.no_grad():
            base = model(ids)["logits"]
            for t in (4, 7, 11):
                alt = ids.clone()
                alt[0, t] = (alt[0, t] + 3) % 20 + 5
                out = model(alt)["logits"]
                self.assertTrue(torch.allclose(base[0, :t], out[0, :t], atol=1e-6),
                                f"perturbing t={t} changed an earlier position")
                self.assertFalse(torch.allclose(base[0, t:], out[0, t:], atol=1e-6),
                                 f"perturbing t={t} changed nothing at or after it")

    def test_causality_also_holds_on_the_padded_branch(self):
        # SDPA takes a different code path when an explicit mask is present, and that is the branch
        # every real training batch uses.
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        mask = torch.ones_like(ids)
        mask[0, 9:] = 0
        with torch.no_grad():
            base = model(ids, attention_mask=mask)["logits"]
            alt = ids.clone()
            alt[0, 6] = (alt[0, 6] + 5) % 20 + 5
            out = model(alt, attention_mask=mask)["logits"]
        self.assertTrue(torch.allclose(base[0, :6], out[0, :6], atol=1e-6))

    def test_padded_positions_cannot_influence_real_ones(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        mask = torch.ones_like(ids)
        mask[0, 9:] = 0
        with torch.no_grad():
            base = model(ids, attention_mask=mask)["logits"]
            alt = ids.clone()
            alt[0, 10] = (alt[0, 10] + 7) % 20 + 5       # inside the padded region
            out = model(alt, attention_mask=mask)["logits"]
        self.assertTrue(torch.allclose(base[0, :9], out[0, :9], atol=1e-6))


class LossWiringTests(unittest.TestCase):
    def test_initial_loss_matches_a_uniform_guess_over_the_vocab(self):
        # ln(27) ~ 3.296. A wildly different value means the shift or the ignore_index is wrong.
        model = _model()
        ids = torch.randint(5, 25, (2, 16))
        labels = ids.clone()
        with torch.no_grad():
            loss = model(ids, labels=labels)["loss"].item()
        self.assertAlmostEqual(loss, torch.log(torch.tensor(27.0)).item(), delta=0.25)

    def test_the_next_token_shift_drops_no_supervision(self):
        # Positions 0-1 are the supplied prompt; positions 2-4 are the answer.
        # Compare with its three next-token log probabilities, excluding prompt supervision.
        model = _model()
        ids = torch.tensor([[5, 6, 7, 8, 9]])
        labels = torch.tensor([[-100, -100, 7, 8, 9]])
        with torch.no_grad():
            output = model(ids, labels=labels)
        log_probs = torch.log_softmax(output["logits"][0], dim=-1)
        expected = -(log_probs[1, 7] + log_probs[2, 8] + log_probs[3, 9]) / 3
        torch.testing.assert_close(output["loss"], expected)

    def test_a_fully_ignored_batch_produces_no_finite_loss_demand(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 8))
        labels = torch.full_like(ids, -100)
        loss = model(ids, labels=labels)["loss"]
        self.assertTrue(torch.isnan(loss))   # CrossEntropyLoss over zero targets; caller must guard


class ArchitectureTests(unittest.TestCase):
    def test_heads_must_divide_the_width(self):
        with self.assertRaises(ValueError):
            build_model(hidden_size=100, num_attention_heads=8)

    def test_ffn_defaults_to_four_times_width(self):
        # Same rule as the masked model, so an architecture sweep is comparable across methods.
        self.assertEqual(build_model(hidden_size=96, num_attention_heads=8).config.intermediate_size,
                         4 * 96)

    def test_rotary_positions_mean_no_learned_position_table(self):
        # Findings §8: learned absolute positions are why a prepended partner chain shifts every
        # position and locks a checkpoint to its context regime. Causal prompts vary in length by
        # construction, so there must be no absolute table to shift.
        model = build_model()
        self.assertFalse(any("pos_embedding" in name for name, _ in model.named_parameters()))
        self.assertIsInstance(model, CausalAntibodyModel)




if __name__ == "__main__":
    unittest.main()
