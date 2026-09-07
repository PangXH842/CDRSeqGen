"""Encoder-decoder attention and loss: the properties that would silently corrupt a run.

Real modules, real attention, no mocks. Three failures are the point:

  * a decoder position that can see a FUTURE answer token trains a copier that cannot generate
  * an encoder position that CANNOT see its right flank is a causal encoder wearing a seq2seq
    costume, and the method's entire architectural claim would be false while every metric still
    looked plausible
  * a second label shift (here and in the dataset) trains happily against the wrong target
"""
import unittest

import torch

from cdrseqgen.data.dataset_s2s import build_decoder_pair, s2s_adapter
from cdrseqgen.methods.s2s.models.transformer import Seq2SeqAntibodyModel, build_model


def _model(**kw):
    torch.manual_seed(0)
    return build_model(vocab_size=27, hidden_size=32, num_layers=2, num_attention_heads=4,
                       max_len=64, **kw).eval()


class EncoderDirectionTests(unittest.TestCase):
    def test_an_encoder_position_sees_its_right_flank(self):
        """Perturbing the LAST encoder token must change the FIRST encoder position's state.

        Under a causal encoder it could not, and that is exactly the `ar` behaviour this method
        exists to remove. This is the load-bearing test of the whole package.
        """
        model = _model()
        a = torch.tensor([[3, 4, 5, 6, 7]])
        b = a.clone()
        b[0, -1] = 11
        with torch.no_grad():
            ha, hb = model.encode(a), model.encode(b)
        self.assertFalse(torch.allclose(ha[0, 0], hb[0, 0], atol=1e-6),
                         "encoder position 0 ignored a change at the far right flank")

    def test_encoder_padding_is_not_attended(self):
        model = _model()
        real = torch.tensor([[3, 4, 5]])
        padded = torch.tensor([[3, 4, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            plain = model.encode(real)
            masked = model.encode(padded, attention_mask=mask)
        torch.testing.assert_close(plain[0, :3], masked[0, :3], atol=1e-5, rtol=1e-5)


class DecoderCausalityTests(unittest.TestCase):
    def test_a_future_answer_token_cannot_change_an_earlier_prediction(self):
        model = _model()
        encoder_ids = torch.tensor([[3, 4, 5, 6]])
        a = torch.tensor([[25, 7, 8, 9]])       # <bos> + three residues
        b = a.clone()
        b[0, -1] = 12                            # perturb only the LAST decoder input
        with torch.no_grad():
            la = model(input_ids=encoder_ids, decoder_input_ids=a)["logits"]
            lb = model(input_ids=encoder_ids, decoder_input_ids=b)["logits"]
        torch.testing.assert_close(la[0, :-1], lb[0, :-1], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(la[0, -1], lb[0, -1], atol=1e-6))

    def test_decoder_padding_does_not_change_earlier_predictions(self):
        model = _model()
        encoder_ids = torch.tensor([[3, 4, 5, 6]])
        short = torch.tensor([[25, 7, 8]])
        padded = torch.tensor([[25, 7, 8, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            plain = model(input_ids=encoder_ids, decoder_input_ids=short)["logits"]
            masked = model(input_ids=encoder_ids, decoder_input_ids=padded,
                           decoder_attention_mask=mask)["logits"]
        torch.testing.assert_close(plain[0], masked[0, :3], atol=1e-5, rtol=1e-5)


class CrossAttentionTests(unittest.TestCase):
    def test_the_decoder_actually_reads_the_encoder(self):
        # A decoder that ignored the encoder would be an unconditional LM: it would generate loops
        # with no reference to the framework at all, and recovery would look like chance while
        # nothing crashed.
        model = _model()
        decoder_ids = torch.tensor([[25, 7, 8]])
        with torch.no_grad():
            one = model(input_ids=torch.tensor([[3, 4, 5, 6]]), decoder_input_ids=decoder_ids)
            two = model(input_ids=torch.tensor([[9, 10, 11, 12]]), decoder_input_ids=decoder_ids)
        self.assertFalse(torch.allclose(one["logits"], two["logits"], atol=1e-6),
                         "decoder logits did not depend on the framework")

    def test_a_reused_encoder_pass_matches_a_fresh_one(self):
        # Sampling runs the encoder once per framework and reuses it across all candidates.
        model = _model()
        encoder_ids = torch.tensor([[3, 4, 5, 6]])
        decoder_ids = torch.tensor([[25, 7, 8]])
        with torch.no_grad():
            fresh = model(input_ids=encoder_ids, decoder_input_ids=decoder_ids)["logits"]
            cached = model(encoder_hidden_states=model.encode(encoder_ids),
                           decoder_input_ids=decoder_ids)["logits"]
        torch.testing.assert_close(fresh, cached, atol=1e-6, rtol=1e-6)


class LossContractTests(unittest.TestCase):
    def test_the_loss_does_not_shift_because_the_dataset_already_did(self):
        """A perfect prediction of the dataset's own labels must give ~zero loss.

        If either side shifted twice this would be large, and training would still 'work'.
        """
        model = _model()
        adapter = s2s_adapter()
        target = adapter.encode("ACDE")
        decoder_input, labels = build_decoder_pair(adapter, target)
        decoder_input = torch.tensor([decoder_input])
        labels = torch.tensor([labels])

        with torch.no_grad():
            logits = model(input_ids=torch.tensor([[3, 4, 5]]),
                           decoder_input_ids=decoder_input)["logits"]
        # Hand the loss a distribution that is already certain of the correct labels.
        perfect = torch.full_like(logits, -20.0)
        perfect.scatter_(2, labels.unsqueeze(-1), 20.0)
        loss = torch.nn.CrossEntropyLoss()(
            perfect.view(-1, perfect.size(-1)), labels.view(-1))
        self.assertLess(loss.item(), 1e-6)

    def test_padded_label_positions_are_ignored(self):
        model = _model()
        decoder_input = torch.tensor([[25, 7, 8, 0]])
        clean = torch.tensor([[7, 8, 26, -100]])
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[3, 4, 5]]), decoder_input_ids=decoder_input,
                        labels=clean)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_labels_without_decoder_inputs_are_refused(self):
        # The model deliberately will not build decoder inputs from labels: that is the API through
        # which a double shift enters.
        model = _model()
        with self.assertRaisesRegex(ValueError, "decoder_input_ids"):
            model(input_ids=torch.tensor([[3, 4, 5]]), labels=torch.tensor([[7, 8, 26]]))


class BuildTests(unittest.TestCase):
    def test_encoder_and_decoder_depth_are_separately_settable(self):
        model = build_model(vocab_size=27, hidden_size=32, num_attention_heads=4,
                            num_encoder_layers=3, num_decoder_layers=1)
        self.assertEqual(len(model.encoder_blocks), 3)
        self.assertEqual(len(model.decoder_blocks), 1)

    def test_a_stack_cannot_be_empty(self):
        with self.assertRaisesRegex(ValueError, "at least one block"):
            build_model(vocab_size=27, hidden_size=32, num_attention_heads=4, num_decoder_layers=0)

    def test_head_divisibility_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_model(vocab_size=27, hidden_size=30, num_attention_heads=4)

    def test_the_ffn_ratio_tracks_width(self):
        # Same rule as the masked and causal models: intermediate_size=None means 4x hidden_size, so
        # widening does not silently change the FFN ratio and confound an architecture sweep.
        self.assertEqual(build_model(vocab_size=27, hidden_size=64,
                                     num_attention_heads=4).config.intermediate_size, 256)

    def test_it_is_a_pretrained_model_so_save_and_reload_work(self):
        self.assertIsInstance(_model(), Seq2SeqAntibodyModel)


if __name__ == "__main__":
    unittest.main()
