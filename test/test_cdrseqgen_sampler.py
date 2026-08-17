import unittest
from types import SimpleNamespace

import torch

from benchmark.adapters.cdrseqgen import CDRSeqGenSampler


class CDRSeqGenSamplerTests(unittest.TestCase):
    def test_fixed_path_passes_loaded_model_to_shared_decoder(self):
        sampler = CDRSeqGenSampler.__new__(CDRSeqGenSampler)
        sampler.torch = torch
        sampler.device = torch.device("cpu")
        sampler.track = "fixed_native_length"
        sampler.model = object()
        sampler.tokenizer = SimpleNamespace(decode=lambda ids: "AC")
        sampler.input_ids = [1, 2, 3]
        sampler.mask_positions = [1, 2]
        captured = {}

        def fake_decoder(**kwargs):
            captured.update(kwargs)
            return [[5, 6]]

        sampler.sample_masked_span = fake_decoder
        samples = sampler.draw_many(
            num_samples=1,
            decoding="independent",
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            length_temperature=1.0,
            seed=42,
        )

        self.assertIs(captured["model"], sampler.model)
        self.assertEqual(samples, ["AC"])


if __name__ == "__main__":
    unittest.main()
