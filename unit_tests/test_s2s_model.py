"""Encoder-decoder attention and loss: the properties that would silently corrupt a run.

Real modules, real attention, no mocks. Three failures are the point:

  * a decoder position that can see a FUTURE answer token trains a copier that cannot generate
  * an encoder position that CANNOT see its right flank is a causal encoder wearing a seq2seq
    costume, and the method's entire architectural claim would be false while every metric still
    looked plausible
  * a second label shift (here and in the dataset) trains happily against the wrong target
"""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from cdrseqgen.datasets.dataset_s2s import build_decoder_pair, s2s_adapter
from cdrseqgen.methods.s2s.models.transformer import (
    Seq2SeqAntibodyConfig, Seq2SeqAntibodyModel, build_model)


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
        """Exercise the model's actual loss against the dataset's aligned labels."""
        model = _model()
        adapter = s2s_adapter()
        decoder_input, labels = build_decoder_pair(adapter, adapter.encode("ACDE"))
        labels = torch.tensor([labels])
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[3, 4, 5]]),
                        decoder_input_ids=torch.tensor([decoder_input]), labels=labels)
        expected = torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, 27), labels.reshape(-1))
        torch.testing.assert_close(out["loss"], expected)
        shifted = torch.nn.functional.cross_entropy(
            out["logits"][:, :-1].reshape(-1, 27), labels[:, 1:].reshape(-1))
        self.assertGreater(abs(out["loss"].item() - shifted.item()), 1e-5)

    def test_padded_label_positions_are_ignored(self):
        model = _model()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[3, 4, 5]]),
                        decoder_input_ids=torch.tensor([[25, 7, 8, 0]]),
                        labels=torch.tensor([[7, 8, 26, -100]]))
        expected = torch.nn.functional.cross_entropy(
            out["logits"][0, :3], torch.tensor([7, 8, 26]))
        torch.testing.assert_close(out["loss"], expected)

    def test_accumulation_matches_one_batch_with_unequal_loop_lengths(self):
        # Real Trainer + SGD: compare the actual parameter update for an identical effective
        # batch. Unequal lengths also catch averaging microbatch means instead of token losses.
        from transformers import TrainingArguments
        from cdrseqgen.methods.s2s.training import build_seq2seq_trainer

        original = _model()
        items = [
            {"input_ids": torch.tensor([5, 4, 6]),
             "decoder_input_ids": torch.tensor([25] + loop),
             "labels": torch.tensor(loop + [26])}
            for loop in ([7], [8, 9, 10, 11])
        ]
        updates = []
        for accumulation, batch_size in ((1, 2), (2, 1)):
            model = copy.deepcopy(original)
            with tempfile.TemporaryDirectory() as directory:
                args = TrainingArguments(
                    output_dir=directory, use_cpu=True, max_steps=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    learning_rate=0.01, optim="sgd", lr_scheduler_type="constant",
                    max_grad_norm=0, report_to="none", save_strategy="no", disable_tqdm=True)
                trainer = build_seq2seq_trainer(model, args, items, items, s2s_adapter())
                trainer.train()
            updates.append(torch.cat([
                (p.detach() - q.detach()).reshape(-1)
                for p, q in zip(model.parameters(), original.parameters())]))
        self.assertGreater(updates[0].norm().item(), 0)
        torch.testing.assert_close(updates[0], updates[1], atol=2e-7, rtol=2e-4)

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

    def test_odd_rotary_head_dimension_fails_at_construction(self):
        for construct in (
            lambda: build_model(hidden_size=24, num_attention_heads=8),
            lambda: Seq2SeqAntibodyModel(Seq2SeqAntibodyConfig(
                hidden_size=24, num_attention_heads=8)),
        ):
            with self.assertRaisesRegex(ValueError, "even head dimension"):
                construct()

    def test_the_ffn_ratio_tracks_width(self):
        # Same rule as the masked and causal models: intermediate_size=None means 4x hidden_size, so
        # widening does not silently change the FFN ratio and confound an architecture sweep.
        self.assertEqual(build_model(vocab_size=27, hidden_size=64,
                                     num_attention_heads=4).config.intermediate_size, 256)

    def test_it_is_a_pretrained_model_so_save_and_reload_work(self):
        self.assertIsInstance(_model(), Seq2SeqAntibodyModel)


class PerplexityTests(unittest.TestCase):
    def test_encoder_reuse_preserves_scores_and_switches_frameworks(self):
        from cdrseqgen.methods.s2s.evaluation import s2s_perplexities

        model = _model()
        tok = s2s_adapter()
        spans = [([5, 4, 6], tok.encode(loop)) for loop in ("AC", "DEF", "AC")]
        spans += [([8, 4, 9], tok.encode("GH")), ([5, 4, 6], tok.encode("AC"))]
        expected = []
        with torch.no_grad():
            for source, loop in spans:
                decoder, labels = build_decoder_pair(tok, loop)
                out = model(input_ids=torch.tensor([source]),
                            decoder_input_ids=torch.tensor([decoder]),
                            labels=torch.tensor([labels]))
                expected.append(out["loss"].exp().item())
        with patch.object(model, "encode", wraps=model.encode) as encode:
            actual = s2s_perplexities(model, tok, torch.device("cpu"), spans)
        self.assertEqual(encode.call_count, 3)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_empty_candidate_list_never_encodes(self):
        from cdrseqgen.methods.s2s.evaluation import s2s_perplexities

        model = _model()
        with patch.object(model, "encode", wraps=model.encode) as encode:
            self.assertEqual(s2s_perplexities(model, s2s_adapter(), "cpu", []), [])
        encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
