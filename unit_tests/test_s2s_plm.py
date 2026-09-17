"""Real tiny ESM/BERT checkpoints exercise PLM weights, two vocabularies, and offline reload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer, EsmConfig,
                          EsmForMaskedLM, EsmTokenizer, TrainingArguments)

from cdrseqgen.datasets.dataset_s2s import (Seq2SeqInfillingDataset, build_encoder_input,
                                       s2s_adapter, s2s_collate_fn)
from cdrseqgen.methods.s2s.backbones import resolve
from cdrseqgen.methods.s2s.evaluation import s2s_perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops
from cdrseqgen.methods.s2s.training import build_seq2seq_trainer

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
                actual = s2s_perplexities(model, tok, "cpu", [(ids, [5, 6])])
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
            trainer = build_seq2seq_trainer(model, args, items, items, tok)
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


if __name__ == '__main__':
    unittest.main()
