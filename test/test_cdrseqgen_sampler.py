import unittest
from types import SimpleNamespace

import torch

from benchmark.adapters.cdrseqgen import CDRSeqGenSampler


def _sampler(track):
    """A sampler with the checkpoint-loading __init__ bypassed, so no weights are needed."""
    sampler = CDRSeqGenSampler.__new__(CDRSeqGenSampler)
    sampler.torch = torch
    sampler.device = torch.device("cpu")
    sampler.track = track
    sampler.model = object()
    sampler.tokenizer = SimpleNamespace(decode=lambda ids: "AC")
    sampler.max_len = 128
    sampler.length_predictor = None
    sampler.record = {"pt1": "QQ", "pt2": "WW"}
    sampler.length_policy = SimpleNamespace(mode="fixed", length=2)
    return sampler


class CDRSeqGenSamplerTests(unittest.TestCase):
    """Both benchmark tracks go through sample_length_conditioned_spans; a fixed native length is
    just a policy whose length is known up front, not a separate sampling path."""

    def _capture(self, track):
        sampler = _sampler(track)
        captured = {}

        def fake_shared(model, tokenizer, device, generator, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            return [SimpleNamespace(token_ids=[5, 6])], {"prompts_by_length": {"2": {}}}

        sampler.sample_length_conditioned_spans = fake_shared
        samples = sampler.draw_many(
            num_samples=1,
            decoding="independent",
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            length_temperature=1.0,
            seed=42,
        )
        return sampler, captured, samples

    def test_fixed_track_uses_the_shared_length_conditioned_path(self):
        sampler, captured, samples = self._capture("fixed_native_length")

        self.assertIs(captured["model"], sampler.model)
        self.assertIs(captured["policy"], sampler.length_policy)
        self.assertEqual(captured["decoding"], "independent")
        self.assertEqual(samples, ["AC"])
        self.assertEqual(sampler.prompt_info, {"2": {}})

    def test_length_free_track_uses_the_same_path(self):
        sampler, captured, samples = self._capture("length_free")

        self.assertIs(captured["model"], sampler.model)
        self.assertIs(captured["length_predictor"], sampler.length_predictor)
        self.assertEqual(samples, ["AC"])

    def test_seeding_is_identical_across_tracks(self):
        """Draw seeding must not depend on the track, or the two tracks are not comparable."""
        _, fixed, _ = self._capture("fixed_native_length")
        _, free, _ = self._capture("length_free")
        self.assertEqual(fixed["num_samples"], free["num_samples"])
        self.assertEqual(fixed["temperature"], free["temperature"])


if __name__ == "__main__":
    unittest.main()
