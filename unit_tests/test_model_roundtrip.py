"""Real CPU models: processed data -> padded batch -> update -> checkpoint -> identical logits.

No pretrained downloads or mocked model/optimizer/serializer. This is a plumbing check, not evidence
that a one-step model learns antibody biology. Full training and external PLMs need separate runs.
"""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from cdrseqgen.data.dataset_ar import CausalInfillingDataset, causal_collate_fn
from cdrseqgen.data.dataset_mlm import AntibodyInfillingDataset, collate_fn
from cdrseqgen.methods.ar.backbones import resolve as resolve_ar
from cdrseqgen.methods.mlm.backbones import resolve as resolve_mlm


class CheckpointRoundTripTests(unittest.TestCase):
    def check_roundtrip(self, resolve, dataset_class, collate):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            root = Path(directory)
            records = []
            for loop in ('GGG', 'GSGGG'):
                chain = dict(fwr1='AAAA', cdr1='CC', fwr2='DDDD', cdr2='EE',
                             fwr3='FFFF', cdr3=loop, fwr4='HHHH')
                chain['full_aa'] = ''.join(chain.values())
                records.append({'heavy': chain, 'light': chain, 'split': 'train'})
            path = root / 'data.json'
            path.write_text(json.dumps(records))
            backbone = resolve('scratch')
            tokenizer = backbone.build_tokenizer('scratch')
            model = backbone.load_training_model('scratch', vocab_size=tokenizer.vocab_size,
                                                hidden_size=32, num_layers=1, num_attention_heads=4)
            dataset = dataset_class(str(path), tokenizer, target_region='h3')
            collated = collate([dataset[i] for i in range(len(dataset))], pad_token_id=tokenizer.pad_id)
            batch = {k: collated[k] for k in ('input_ids', 'attention_mask', 'labels')}
            self.assertTrue((batch['attention_mask'] == 0).any(), 'fixture must exercise padding')
            initial = {k: v.detach().clone() for k, v in model.named_parameters()}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            model.train()
            loss = model(**batch)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.named_parameters()),
                            'an optimizer step must change learned weights')
            model.eval()
            with torch.no_grad():
                expected = model(**batch)['logits']
            checkpoint = root / 'checkpoint'
            model.save_pretrained(checkpoint)
            # A fresh RNG state must not influence a correctly restored checkpoint.
            torch.manual_seed(999)
            restored = backbone.load_pretrained_model(str(checkpoint))
            with torch.no_grad():
                actual = restored(**batch)['logits']
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_masked_model_updates_and_reloads(self):
        self.check_roundtrip(resolve_mlm, AntibodyInfillingDataset, collate_fn)

    def test_causal_model_updates_and_reloads(self):
        self.check_roundtrip(resolve_ar, CausalInfillingDataset, causal_collate_fn)


if __name__ == '__main__':
    unittest.main()
