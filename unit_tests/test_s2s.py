"""Tests merged from: test_s2s_diagnostics.py, test_s2s_model.py, test_s2s_plm.py, test_s2s_sampling.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments
from experiments.spec import suite_trials, spec_trials
from cdrseqgen.datasets.dataset_s2s import s2s_collate_fn
from cdrseqgen.methods.s2s.training import ConditioningProbe
from cdrseqgen.methods.s2s.models.transformer import build_model
import copy
from unittest.mock import patch
from cdrseqgen.datasets.dataset_s2s import build_decoder_pair, s2s_adapter
from cdrseqgen.methods.s2s.models.transformer import (
    Seq2SeqAntibodyConfig,
    Seq2SeqAntibodyModel,
    build_model,
)
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    EsmConfig,
    EsmForMaskedLM,
    EsmTokenizer,
    TrainingArguments,
)
from cdrseqgen.datasets.dataset_s2s import (
    Seq2SeqInfillingDataset,
    build_encoder_input,
    s2s_adapter,
    s2s_collate_fn,
)
from cdrseqgen.methods.s2s.backbones import resolve
from cdrseqgen.methods.s2s.evaluation import perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops
from cdrseqgen.methods.s2s.training import build_trainer
from types import SimpleNamespace
from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS,
    Seq2SeqInfillingDataset,
    build_encoder_input,
    s2s_adapter,
)


# ====================================================================================================
# from test_s2s_diagnostics.py
# ====================================================================================================

class DiagnosticsTests(unittest.TestCase):
    def test_suite_controls_and_schedule(self):
        trials, _ = spec_trials('configs/experiments/20260914_igbert_diagnostic.json', {})
        self.assertEqual([(t['config']['learning_rate'], t['config']['freeze_encoder']) for t in trials],
                         [(3e-5, False), (1e-5, False), (3e-5, True)])
        for t in trials:
            c = t['config']
            self.assertEqual(c['num_train_epochs'], 10)
            self.assertEqual(c['diagnostic_stop_epoch'], 3)
            self.assertIsNone(c['save_total_limit'])
            self.assertFalse(c['run_generative_eval'])
        self.assertEqual(len(suite_trials(['s2s_igbert', 's2s_igbert'], {})), 2)
        with self.assertRaises(ValueError):
            spec_trials('configs/experiments/20260914_igbert_diagnostic.json', {'FREEZE_ENCODER': 'true'})

    def test_real_trainer_stops_without_shortening_schedule(self):
        torch.set_num_threads(1)
        rows = [dict(input_ids=torch.tensor([5+i,4,8]), decoder_input_ids=torch.tensor([25,6,7]),
                     labels=torch.tensor([6,7,26])) for i in range(4)]
        model = build_model(hidden_size=16, num_layers=1, num_attention_heads=2)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(output_dir=tmp, num_train_epochs=4,
                per_device_train_batch_size=2, per_device_eval_batch_size=2,
                report_to='none', use_cpu=True, save_strategy='epoch', eval_strategy='epoch',
                disable_tqdm=True)
            callback = ConditioningProbe(rows, s2s_collate_fn, interval=1, stop_epoch=1)
            trainer = Trainer(model=model,args=args,train_dataset=rows,eval_dataset=rows,
                              data_collator=s2s_collate_fn,callbacks=[callback])
            trainer.train()
            self.assertEqual(trainer.state.max_steps,8)
            self.assertEqual(trainer.state.global_step,2)
            self.assertTrue((Path(tmp)/'checkpoint-2').is_dir())
            logs = [json.loads(s) for s in (Path(tmp)/'conditioning_probe.jsonl').read_text().splitlines()]
            self.assertEqual(logs[0]['step'],0)
            self.assertTrue(all(r['finite'] for r in logs))
            self.assertGreater(logs[0]['swap_mean_abs_logit_change'],0)
            model.train()
            rng=torch.random.get_rng_state().clone()
            callback.measure(args,trainer.state,model)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))


# ====================================================================================================
# from test_s2s_model.py
# Encoder-decoder attention and loss: the properties that would silently corrupt a run.
#
# Real modules, real attention, no mocks. Three failures are the point:
#
#   * a decoder position that can see a FUTURE answer token trains a copier that cannot generate
#   * an encoder position that CANNOT see its right flank is a causal encoder wearing a seq2seq
#     costume, and the method's entire architectural claim would be false while every metric still
#     looked plausible
#   * a second label shift (here and in the dataset) trains happily against the wrong target
# ====================================================================================================

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
        from cdrseqgen.methods.s2s.training import build_trainer

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
                trainer = build_trainer(model, args, items, items, s2s_adapter())
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
        from cdrseqgen.methods.s2s.evaluation import perplexities

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
            actual = perplexities(model, tok, torch.device("cpu"), spans)
        self.assertEqual(encode.call_count, 3)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_empty_candidate_list_never_encodes(self):
        from cdrseqgen.methods.s2s.evaluation import perplexities

        model = _model()
        with patch.object(model, "encode", wraps=model.encode) as encode:
            self.assertEqual(perplexities(model, s2s_adapter(), "cpu", []), [])
        encode.assert_not_called()


# ====================================================================================================
# from test_s2s_plm.py
# Real tiny ESM/BERT checkpoints exercise PLM weights, two vocabularies, and offline reload.
# ====================================================================================================

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
FRAMEWORK = dict(pt1="ACDE", pt2="WY", target="GGG", target_length=3,
                 partner_chain="KK", antigen=None, max_len=32)

class PLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.sources = {}
        for family in ("esm2", "protein_bert"):
            path = cls.root / ("tiny-esm2" if family == "esm2" else "tiny-prot_bert")
            path.mkdir()
            if family == "esm2":
                vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *RESIDUES, "<mask>"]
                (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
                tokenizer = EsmTokenizer(vocab_file=str(path / "vocab.txt"))
                config = EsmConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                   num_attention_heads=4, intermediate_size=48, pad_token_id=1,
                                   mask_token_id=len(vocab)-1, position_embedding_type="rotary",
                                   token_dropout=True, hidden_dropout_prob=0.1)
                model = EsmForMaskedLM(config)
            else:
                # Different pad/residue ids from both ESM and the decoder are intentional.
                vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", *RESIDUES]
                (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
                tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
                config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                    num_attention_heads=4, intermediate_size=48, pad_token_id=3,
                                    max_position_embeddings=64, hidden_dropout_prob=0.1)
                model = BertForMaskedLM(config)
            # A distinctive checkpoint value detects accidentally reinitializing the loaded encoder.
            with torch.no_grad():
                model.get_input_embeddings().weight[5].fill_(0.314)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            cls.sources[family] = path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make(self, family="esm2", **kwargs):
        source = str(self.sources[family])
        backbone = resolve(source)
        tokenizer = backbone.build_tokenizer(source)
        model = backbone.load_training_model(source, hidden_size=16, num_attention_heads=4,
                                              num_decoder_layers=1, max_len=32, **kwargs)
        return model, tokenizer

    def test_native_encoder_wrapping_and_decoder_vocabulary_are_independent(self):
        for family in self.sources:
            with self.subTest(family=family):
                model, tokenizer = self.make(family)
                encoder = tokenizer.encoder_tokenizer
                ids, _ = build_encoder_input(tokenizer, FRAMEWORK)
                expected = ([encoder.cls_id] + encoder.encode("KK") + [encoder.chain_sep_id]
                            + encoder.encode("ACDE") + [encoder.mask_id]
                            + encoder.encode("WY") + [encoder.end_id])
                self.assertEqual(ids, expected)
                self.assertEqual(tokenizer.encode(RESIDUES), s2s_adapter().encode(RESIDUES))
                self.assertEqual(model.config.vocab_size, 27)
                self.assertEqual(len(build_encoder_input(tokenizer, dict(FRAMEWORK, max_len=3))[0]), 3)
                with self.assertRaisesRegex(ValueError, "wrappers"):
                    build_encoder_input(tokenizer, dict(FRAMEWORK, max_len=2))
                self.assertEqual(ids, build_encoder_input(tokenizer, dict(
                    FRAMEWORK, target="A" * 80, target_length=80))[0])

    def test_dataset_labels_use_decoder_ids_and_collator_uses_both_padding_ids(self):
        model, tok = self.make()
        chain = dict(fwr1="AC", cdr1="DE", fwr2="FG", cdr2="HI", fwr3="KL",
                     cdr3="MNP", fwr4="QR", full_aa="ACDEFGHIKLMNPQR")
        path = self.root / "data.json"
        path.write_text(json.dumps([dict(heavy=chain, light=chain, split="train")]))
        dataset = Seq2SeqInfillingDataset(path, tok, max_len=32, target_region="h3")
        item = dataset[0]
        self.assertEqual(item["labels"].tolist(), tok.encode("MNP") + [tok.eos_id])
        self.assertEqual(item["input_ids"].tolist(), build_encoder_input(tok, dataset.framework_parts(0))[0])
        short = {k: v[:1] for k, v in item.items() if torch.is_tensor(v)}
        batch = s2s_collate_fn([item, short], pad_token_id=tok.pad_id,
                               encoder_pad_token_id=tok.encoder_tokenizer.pad_id)
        self.assertEqual(batch["input_ids"][1, -1].item(), 1)
        self.assertEqual(batch["decoder_input_ids"][1, -1].item(), 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["attention_mask"][1].sum().item(), 1)
        self.assertTrue(torch.isfinite(model(**batch)["loss"]))

    def test_checkpoint_weights_survive_construction_and_random_control_differs(self):
        for family in self.sources:
            with self.subTest(family=family):
                model, _ = self.make(family)
                random_model, _ = self.make(family, encoder_init="random")
                torch.testing.assert_close(model.encoder.get_input_embeddings().weight[5],
                                           torch.full((24,), 0.314))
                self.assertFalse(torch.equal(model.encoder.get_input_embeddings().weight,
                                             random_model.encoder.get_input_embeddings().weight))
                self.assertEqual([p.shape for p in model.parameters()],
                                 [p.shape for p in random_model.parameters()])

    def test_gradients_and_dropout_follow_freeze_control(self):
        for family in self.sources:
            for freeze in (False, True):
                with self.subTest(family=family, freeze=freeze):
                    model, tok = self.make(family, freeze_encoder=freeze)
                    model.train()
                    self.assertEqual(model.encoder.training, not freeze)
                    self.assertTrue(model.decoder_blocks.training)
                    ids, _ = build_encoder_input(tok, FRAMEWORK)
                    out = model(input_ids=torch.tensor([ids]), decoder_input_ids=torch.tensor([[25, 5]]),
                                labels=torch.tensor([[5, 26]]))
                    out["loss"].backward()
                    self.assertGreater(model.encoder_projection.weight.grad.norm().item(), 0)
                    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
                    self.assertEqual(bool(encoder_grads), not freeze)
                    if encoder_grads:
                        self.assertGreater(sum(g.abs().sum().item() for g in encoder_grads), 0)

    def test_padding_loss_alignment_and_cached_encoding(self):
        for family in self.sources:
            with self.subTest(family=family):
                model, tok = self.make(family)
                model.eval()
                ids, _ = build_encoder_input(tok, FRAMEWORK)
                source = torch.tensor([ids])
                padded = torch.tensor([ids + [tok.encoder_tokenizer.pad_id] * 3])
                mask = torch.tensor([[1] * len(ids) + [0] * 3])
                decoder = torch.tensor([[25, 5, 6]])
                gold = torch.tensor([[5, 6, 26]])
                with torch.no_grad():
                    plain = model(input_ids=source, decoder_input_ids=decoder, labels=gold)
                    encoded = model.encode(padded, mask)
                    cached = model(encoder_hidden_states=encoded, attention_mask=mask,
                                   decoder_input_ids=decoder)
                    implicit = model.encode(padded)
                    changed = model(input_ids=source, decoder_input_ids=torch.tensor([[25, 5, 9]]))
                torch.testing.assert_close(plain["logits"], cached["logits"], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(encoded, implicit)
                torch.testing.assert_close(plain["logits"][:, :2], changed["logits"][:, :2])
                torch.testing.assert_close(plain["loss"], torch.nn.functional.cross_entropy(
                    plain["logits"].reshape(-1, 27), gold.reshape(-1)))
                actual = perplexities(model, tok, "cpu", [(ids, [5, 6])])
                self.assertAlmostEqual(actual[0], plain["loss"].exp().item(), places=4)

    def test_save_reload_is_exact_and_uses_only_local_checkpoint_assets(self):
        for family in self.sources:
            with self.subTest(family=family):
                model, tok = self.make(family, freeze_encoder=True)
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    tok.save_pretrained(directory)
                    backbone = resolve(str(self.sources[family]))
                    with patch('transformers.EsmModel.from_pretrained', side_effect=AssertionError), \
                         patch('transformers.BertModel.from_pretrained', side_effect=AssertionError):
                        restored = backbone.load_pretrained_model(directory)
                    # The original id need not exist once the saved tokenizer is available.
                    reloaded_tok = backbone.build_tokenizer("missing/original", checkpoint_path=directory)
                    self.assertEqual(build_encoder_input(tok, FRAMEWORK),
                                     build_encoder_input(reloaded_tok, FRAMEWORK))
                    for key, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
                    restored.train()
                    self.assertFalse(restored.encoder.training)
                    self.assertEqual(restored.config.encoder_family, family)
                    # Real sampler, forced toward EOS so this test never relies on random completion.
                    with torch.no_grad():
                        restored.lm_head.weight.zero_()
                        restored.decoder_norm.weight.zero_()
                        restored.decoder_norm.bias.fill_(1)
                        restored.lm_head.weight[tok.eos_id].fill_(1)
                    ids, _ = build_encoder_input(reloaded_tok, FRAMEWORK)
                    with patch.object(restored, 'encode', wraps=restored.encode) as encode:
                        draws = sample_loops(restored, reloaded_tok, ids, device='cpu',
                                             temperature=0, max_new_tokens=3)
                    self.assertEqual(draws, [dict(loop='', length=0, complete=True)])
                    self.assertEqual(encode.call_count, 1)

    def test_real_trainer_updates_decoder_and_preserves_frozen_encoder(self):
        model, tok = self.make(freeze_encoder=True)
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        decoder_before = model.lm_head.weight.detach().clone()
        ids, _ = build_encoder_input(tok, FRAMEWORK)
        items = [dict(input_ids=torch.tensor(ids), decoder_input_ids=torch.tensor([25, 5, 6]),
                      labels=torch.tensor([5, 6, 26]))]
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                     learning_rate=0.01, report_to="none", save_strategy="no",
                                     disable_tqdm=True)
            trainer = build_trainer(model, args, items, items, tok)
            trainer.train()
        self.assertFalse(torch.equal(model.lm_head.weight, decoder_before))
        for key, value in before.items():
            torch.testing.assert_close(model.encoder.state_dict()[key], value, atol=0, rtol=0)

    def test_incomplete_pretrained_weights_are_rejected(self):
        from safetensors.torch import load_file, save_file
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'broken-esm2'
            shutil.copytree(self.sources['esm2'], source)
            weights = load_file(source / 'model.safetensors')
            key = next(k for k in weights if 'word_embeddings' in k)
            weights.pop(key)
            # The MLM embedding is tied to lm_head.decoder, which HF can use as a fallback.
            weights.pop('lm_head.decoder.weight', None)
            save_file(weights, source / 'model.safetensors', metadata={'format': 'pt'})
            with self.assertRaisesRegex(ValueError, 'did not load completely'):
                resolve(str(source)).load_training_model(str(source), max_len=32)

    def test_invalid_encoder_settings_fail_before_loading_weights(self):
        for kwargs, message in ((dict(encoder_init="typo"), "encoder_init"),
                                (dict(num_encoder_layers=2), "fixed")):
            with self.assertRaisesRegex(ValueError, message):
                self.make(**kwargs)
        with self.assertRaisesRegex(ValueError, "position limit"):
            resolve(str(self.sources['protein_bert'])).load_training_model(
                str(self.sources['protein_bert']), max_len=65)
        with self.assertRaisesRegex(ValueError, "own adapter"):
            resolve("Rostlab/prot_t5_xl_uniref50")


# ====================================================================================================
# from test_s2s_sampling.py
# Seq2seq decoding: the sampler contract, and its agreement with training.
#
# The failure that matters most is silent: a sampler that reads a different encoder input than
# training built produces a model meeting frameworks it never saw, and nothing crashes — that is the
# 2026-09-02 prompt-divergence bug, which cost a batch and was invisible until <eos> stopped arriving.
# ====================================================================================================

S2S_SAMPLING_FRAMEWORK = dict(pt1="ACDEFGHIK", pt2="LMNPQRST", partner_chain=None, antigen=None,
                 max_len=64, target="WYVWY", target_length=5)

class StubModel:
    """A model with scripted next-token behaviour, so decode control flow is testable exactly."""

    def __init__(self, script, vocab_size=27, hidden=4):
        self.script = list(script)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.calls = 0
        self.encode_calls = 0

    def eval(self):
        return self

    def encode(self, input_ids, attention_mask=None):
        self.encode_calls += 1
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden)

    def __call__(self, *, decoder_input_ids, encoder_hidden_states=None, attention_mask=None,
                 **_kw):
        step = decoder_input_ids.shape[1] - 1
        rows = decoder_input_ids.shape[0]
        logits = torch.full((rows, decoder_input_ids.shape[1], self.vocab_size), -20.0)
        token = self.script[min(step, len(self.script) - 1)]
        logits[:, -1, token] = 20.0
        self.calls += 1
        return {"logits": logits, "loss": None}


class DecodeControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.tok = s2s_adapter()
        self.encoder_ids, _ = build_encoder_input(self.tok, dict(S2S_SAMPLING_FRAMEWORK))

    def test_decoding_stops_at_eos_and_the_terminator_is_not_a_residue(self):
        residues = self.tok.encode("WYV")
        model = StubModel(residues + [self.tok.eos_id])
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                               temperature=0)
        self.assertEqual(draw["loop"], "WYV")
        self.assertEqual(draw["length"], 3)
        self.assertTrue(draw["complete"])

    def test_a_draw_that_never_terminates_is_reported_incomplete_not_truncated(self):
        # A cut-off loop is not a length the model chose, and scoring it as one would make an
        # unhealthy <eos> mechanism look like a merely short-loop model.
        model = StubModel(self.tok.encode("W"))
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                               temperature=0, max_new_tokens=5)
        self.assertFalse(draw["complete"])
        self.assertEqual(draw["length"], 5)

    def test_greedy_returns_exactly_one_candidate(self):
        # Matches the masked and causal samplers: --temperature 0 means one deterministic answer,
        # so a table cannot silently average 20 identical greedy draws as if they were a sample.
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        draws = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                             temperature=0, num_samples=20)
        self.assertEqual(len(draws), 1)

    def test_sampling_returns_the_requested_number_of_candidates(self):
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        draws = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                             temperature=1.0, num_samples=7,
                             generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(draws), 7)

    def test_the_encoder_runs_once_for_all_candidates(self):
        # Every candidate must condition on byte-identical framework states, so between-candidate
        # variation is sampling noise rather than a re-encoded prompt. It is also the speed win.
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                     temperature=1.0, num_samples=12,
                     generator=torch.Generator().manual_seed(0))
        self.assertEqual(model.encode_calls, 1)

    def test_non_terminator_special_tokens_cannot_appear_in_a_loop(self):
        # Left unbanned the model may emit <mask>/<sep>/<pad> mid-loop and have it scored as a
        # residue. Scripted to always want <mask>, the sampler must never produce it.
        model = StubModel([self.tok.mask_id])
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                              temperature=0, max_new_tokens=4)
        self.assertNotIn("<mask>", draw["loop"])
        for token in ("<pad>", "<sep>", "<bos>", "<eos>"):
            self.assertNotIn(token, draw["loop"])

    def test_the_decode_budget_is_validated_before_the_model_is_touched(self):
        model = StubModel([self.tok.eos_id])
        with self.assertRaises(ValueError):
            sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                         max_new_tokens=S2S_DECODE_STEPS + 1)
        self.assertEqual(model.calls, 0)


class TrainingGenerationAgreementTests(unittest.TestCase):
    """The encoder input the sampler reads must be the one training built. Byte-identical."""

    def test_generation_rebuilds_the_training_encoder_input_exactly(self):
        import json
        import os
        import tempfile

        chain = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
                 "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
        record = {"heavy": dict(chain, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
                  "light": dict(chain, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
                  "antigen": {"full_aa": "WWWW"}, "split": "train"}
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([record], handle)
        handle.close()
        try:
            adapter = s2s_adapter()
            dataset = Seq2SeqInfillingDataset(handle.name, adapter, max_len=64, target_region="h3")
            trained = dataset[0]["input_ids"].tolist()
            # Generation path: the framework only, with no access to the native loop or its length.
            framework = dataset.framework_parts(0)
            generated, _ = build_encoder_input(adapter, framework)
            self.assertEqual(trained, generated)
        finally:
            os.unlink(handle.name)


if __name__ == "__main__":
    unittest.main()
