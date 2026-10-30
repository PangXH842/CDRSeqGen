"""Opt-in fixed-validation conditioning measurements, separate from checkpoint selection."""
import json
import random
from pathlib import Path

import torch
from transformers import TrainerCallback


class ConditioningProbe(TrainerCallback):
    def __init__(self, dataset, collate, interval=250, stop_epoch=0):
        self.interval = interval
        self.stop_epoch = stop_epoch
        indices = random.Random(42).sample(range(len(dataset)), min(16, len(dataset)))
        if interval and len(indices) < 2:
            raise ValueError("Conditioning probes require at least two validation examples")
        self.indices = indices
        self.batch = collate([dataset[i] for i in indices]) if interval else None

    def measure(self, args, state, model):
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in self.batch.items()}
        was_training = model.training
        # Even future stochastic probe operations must not alter the training RNG stream.
        devices = [device.index] if device.type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                memory = model.encode(batch['input_ids'], batch['attention_mask'])
                valid = batch['attention_mask'].bool()
                original = model(**batch, encoder_hidden_states=memory)['logits']
                swapped = dict(batch, attention_mask=batch['attention_mask'].roll(1, 0))
                changed = model(**swapped, encoder_hidden_states=memory.roll(1, 0))['logits']
                targets = batch['labels'].ne(-100)
                row = dict(step=state.global_step, epoch=state.epoch,
                           validation_subset_indices=self.indices,
                           projected_token_std=memory[valid].float().std(0).mean().item(),
                           finite=bool(torch.isfinite(memory[valid]).all() and torch.isfinite(original[targets]).all()),
                           swap_mean_abs_logit_change=(original-changed).abs()[targets].mean().item(),
                           swap_argmax_change_rate=(original.argmax(-1)[targets]
                                                    != changed.argmax(-1)[targets]).float().mean().item())
        finally:
            model.train(was_training)
        with (Path(args.output_dir) / 'conditioning_probe.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.interval and state.is_world_process_zero:
            self.measure(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.interval and state.global_step % self.interval == 0 and state.is_world_process_zero:
            self.measure(args, state, model)
        if self.stop_epoch and state.epoch >= self.stop_epoch:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.interval and state.is_world_process_zero:
            self.measure(args, state, model)
