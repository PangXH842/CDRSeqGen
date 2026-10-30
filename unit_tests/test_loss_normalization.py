"""AR and S2S must give the same gradient whether a batch is split into accumulation steps or not.

Both models take ``**kwargs`` in ``forward``, which makes the Trainer stop dividing the loss by
``gradient_accumulation_steps`` and hand over ``num_items_in_batch`` instead. A model that ignores
that count and returns its microbatch mean ends up with a gradient k times too large at k steps; AR
did exactly that until the loss learned to use the count. The Trainer's own ``training_step`` is
driven here, with real model code, so the test covers the division the Trainer does or skips.
"""
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import Trainer, TrainingArguments

from cdrseqgen.methods.ar.models.transformer import build_model as build_causal
from cdrseqgen.methods.ar.training import build_trainer as build_causal_trainer
from cdrseqgen.methods.s2s.models.transformer import build_model as build_seq2seq
from cdrseqgen.methods.s2s.training import build_trainer as build_seq2seq_trainer


def _trainer(model):
    with tempfile.TemporaryDirectory() as directory:
        return Trainer(model=model, args=TrainingArguments(
            output_dir=directory, use_cpu=True, report_to="none"))


def _gradient(model, trainer, parts, num_items):
    model.zero_grad(set_to_none=True)
    trainer.current_gradient_accumulation_steps = len(parts)
    for part in parts:
        trainer.training_step(model, dict(part), num_items_in_batch=num_items)
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def _causal_batch():
    ids = torch.randint(3, 20, (4, 10))
    labels = ids.clone()
    labels[:, :6] = -100       # the prompt is not supervised
    labels[0, 8:] = -100       # rows with different numbers of supervised tokens
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def _seq2seq_batch():
    encoder = torch.randint(3, 20, (4, 8))
    decoder = torch.randint(3, 20, (4, 6))
    labels = decoder.clone()
    labels[0, 3:] = -100
    labels[1, 5:] = -100
    return dict(input_ids=encoder, attention_mask=torch.ones_like(encoder),
                decoder_input_ids=decoder, labels=labels)


class AccumulationTests(unittest.TestCase):
    def check(self, model, batch):
        trainer = _trainer(model)
        half = lambda rows: {key: value[rows] for key, value in batch.items()}
        items = int(batch["labels"].ne(-100).sum())
        whole = _gradient(model, trainer, [batch], items)
        split = _gradient(model, trainer, [half(slice(0, 2)), half(slice(2, 4))], items)
        self.assertTrue(torch.allclose(split, whole, atol=1e-6),
                        f"accumulated gradient is {(split.norm() / whole.norm()).item():.3f}x the "
                        "single-batch gradient")

    def test_causal_gradient_does_not_depend_on_the_accumulation_split(self):
        torch.manual_seed(0)
        self.check(build_causal(hidden_size=32, num_layers=1, num_attention_heads=4),
                   _causal_batch())

    def test_seq2seq_gradient_does_not_depend_on_the_accumulation_split(self):
        torch.manual_seed(0)
        self.check(build_seq2seq(hidden_size=32, num_layers=1, num_attention_heads=4),
                   _seq2seq_batch())


def _aligned_cross_entropy(shift, scale=1.0):
    """The loss a selectable objective would supply: align, sum, divide by the window's count."""
    def loss_func(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"]
        if shift:
            logits, labels = logits[:, :-1], labels[:, 1:]
        total = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        count = (labels.ne(-100).sum() if num_items_in_batch is None
                 else torch.as_tensor(num_items_in_batch))
        return scale * total / count
    return loss_func


def _method_trainer(build, model, compute_loss_func):
    with tempfile.TemporaryDirectory() as directory:
        args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none")
        return build(model, args, None, None, SimpleNamespace(pad_id=0),
                     compute_loss_func=compute_loss_func)


class LossSeamTests(unittest.TestCase):
    """build_trainer(compute_loss_func=...) is where a selectable AR/S2S loss would plug in."""

    CASES = (("causal", build_causal_trainer, build_causal, _causal_batch, True),
             ("seq2seq", build_seq2seq_trainer, build_seq2seq, _seq2seq_batch, False))

    def test_a_function_that_aligns_and_normalises_reproduces_the_models_own_loss(self):
        for name, build_trainer, build_model, make_batch, shift in self.CASES:
            with self.subTest(method=name):
                torch.manual_seed(0)
                model = build_model(hidden_size=32, num_layers=1, num_attention_heads=4)
                batch = make_batch()
                items = int(batch["labels"].ne(-100).sum())
                half = lambda rows: {key: value[rows] for key, value in batch.items()}
                own = _method_trainer(build_trainer, model, None)
                custom = _method_trainer(build_trainer, model, _aligned_cross_entropy(shift))
                self.assertIsNone(own.compute_loss_func)
                reference = _gradient(model, own, [batch], items)
                whole = _gradient(model, custom, [batch], items)
                split = _gradient(model, custom, [half(slice(0, 2)), half(slice(2, 4))], items)
                self.assertTrue(torch.allclose(whole, reference, atol=1e-6))
                self.assertTrue(torch.allclose(split, reference, atol=1e-6))
                model.eval()
                with torch.no_grad():  # evaluation passes no count
                    self.assertAlmostEqual(custom.compute_loss(model, dict(batch)).item(),
                                           own.compute_loss(model, dict(batch)).item(), places=6)

    def test_the_supplied_function_replaces_the_models_loss(self):
        for name, build_trainer, build_model, make_batch, shift in self.CASES:
            with self.subTest(method=name):
                torch.manual_seed(0)
                model = build_model(hidden_size=32, num_layers=1, num_attention_heads=4)
                batch = make_batch()
                items = int(batch["labels"].ne(-100).sum())
                own = _gradient(model, _method_trainer(build_trainer, model, None), [batch], items)
                doubled = _gradient(model, _method_trainer(
                    build_trainer, model, _aligned_cross_entropy(shift, scale=2.0)), [batch], items)
                self.assertTrue(torch.allclose(doubled, 2 * own, atol=1e-6))


class DirectCallTests(unittest.TestCase):
    def test_without_a_count_the_loss_is_the_plain_token_mean(self):
        torch.manual_seed(0)
        model = build_causal(hidden_size=32, num_layers=1, num_attention_heads=4).eval()
        batch = _causal_batch()
        with torch.no_grad():
            out = model(**batch)
            expected = torch.nn.functional.cross_entropy(
                out["logits"][:, :-1].reshape(-1, out["logits"].size(-1)),
                batch["labels"][:, 1:].reshape(-1))
        self.assertAlmostEqual(out["loss"].item(), expected.item(), places=6)

    def test_a_count_of_the_whole_batch_gives_the_same_loss_as_no_count(self):
        """At one accumulation step the Trainer passes the batch's own token count, so training
        must see exactly the loss that evaluation and direct calls see."""
        torch.manual_seed(0)
        for build, batch in ((build_causal, _causal_batch()), (build_seq2seq, _seq2seq_batch())):
            model = build(hidden_size=32, num_layers=1, num_attention_heads=4).eval()
            items = int(batch["labels"].ne(-100).sum())
            with torch.no_grad():
                plain = model(**batch)["loss"]
                counted = model(**batch, num_items_in_batch=items)["loss"]
            with self.subTest(model=type(model).__name__):
                self.assertAlmostEqual(counted.item(), plain.item(), places=6)


if __name__ == "__main__":
    unittest.main()
