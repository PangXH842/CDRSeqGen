import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments
from cdrseqgen.experiments import suite_trials, spec_trials
from cdrseqgen.datasets.dataset_s2s import s2s_collate_fn
from cdrseqgen.methods.s2s.training import ConditioningProbe
from cdrseqgen.methods.s2s.models.transformer import build_model


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

if __name__=='__main__':unittest.main()
