"""IgCraft adapter accounting, capacity guards and the flank-region export, with a stub sampler.

No IgCraft package, checkpoint or GPU is needed: the adapter imports the model only inside IgCraftSampler.
"""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark._common import stable_example_seed
from benchmark.adapters.igcraft import (
    HIDDEN_CDR3, HIDDEN_LIGHT, HIDDEN_REGIONS, InvalidGeneration, REGIONS, capacity_problems, config_capacities,
    extract_loop, load_flank_table, run_example)
from benchmark.export_regions import FLANK_REGIONS, build, flank_regions

# The released model's slot sizes (igcraft-model/.hydra/config.yaml).
CAPACITIES = dict(vh=dict(fwr1=30, cdr1=15, fwr2=20, cdr2=15, fwr3=42, cdr3=30, fwr4=12),
                  vl=dict(fwr1=26, cdr1=12, fwr2=18, cdr2=8, fwr3=38, cdr3=14, fwr4=11))
FLANKS = dict(fwr1='QVQLV', cdr1='GYTF', fwr2='MHWVR', cdr2='INPN', fwr3='NYAQKFQ', fwr4='WGQGTLVTVSS')


def manifest_record(example_id='oas:7:example', native='AAA'):
    pt1 = ''.join(FLANKS[name] for name in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3'))
    return dict(example_id=example_id, parent_sequence=pt1 + native + FLANKS['fwr4'], pt1=pt1, native=native,
                pt2=FLANKS['fwr4'], native_length=len(native), infill_range=[len(pt1), len(pt1) + len(native)],
                source_index=3, sequence_id='seq-3')


class StubSampler:
    """Returns scripted slots (or raises scripted exceptions) one call at a time."""

    def __init__(self, *calls):
        self.calls, self.counts, self.seed, self.prepared = iter(calls), [], None, None

    def set_seed(self, seed):
        self.seed = seed

    def prepare(self, regions):
        self.prepared = regions

    def draw(self, count):
        self.counts.append(count)
        result = next(self.calls)
        if isinstance(result, Exception):
            raise result
        return result


def run(sampler, **overrides):
    options = dict(num_samples=3, batch_size=3, base_seed=42, max_errors=5, maximum_length=30)
    return run_example(manifest_record(), FLANKS, sampler, **{**options, **overrides})


class DecodingTests(unittest.TestCase):
    def test_pads_are_stripped_like_igcrafts_own_decoder(self):
        self.assertEqual(extract_loop('ARDY' + '-' * 26), 'ARDY')
        self.assertEqual(extract_loop('AR-DY---'), 'ARDY')

    def test_all_pad_slot_is_an_invalid_generation(self):
        with self.assertRaises(InvalidGeneration) as caught:
            extract_loop('-' * 30)
        self.assertEqual(caught.exception.code, 'empty_generation')

    def test_non_residue_characters_are_rejected(self):
        with self.assertRaises(InvalidGeneration) as caught:
            extract_loop('AR*Y')
        self.assertEqual(caught.exception.code, 'invalid_amino_acid')


class CapacityTests(unittest.TestCase):
    def test_fitting_framework_and_placeholders_pass(self):
        self.assertEqual(capacity_problems(FLANKS, CAPACITIES), [])
        self.assertLessEqual(len(HIDDEN_CDR3), CAPACITIES['vh']['cdr3'])
        self.assertTrue(all(len(HIDDEN_LIGHT[name]) <= CAPACITIES['vl'][name] for name in REGIONS))

    def test_overflowing_region_is_named_instead_of_misaligning_the_encoding(self):
        problems = capacity_problems({**FLANKS, 'fwr4': 'W' * 13}, CAPACITIES)
        self.assertEqual(problems, ['fwr4 has 13 residues for 12 slots'])

    def test_a_smaller_model_would_reject_the_placeholders_too(self):
        tight = json.loads(json.dumps(CAPACITIES))
        tight['vl']['fwr1'] = 5
        tight['vh']['cdr3'] = 3
        problems = capacity_problems(FLANKS, tight)
        self.assertIn('hidden CDR3 placeholder overflows its slot', problems)
        self.assertIn('hidden light fwr1 placeholder overflows its slot', problems)

    def test_capacities_come_from_the_model_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('model:\n  datamodule:\n    vh_tokenizer:\n      region_lengths: {fwr1: 30, cdr3: 30}\n'
                              '    vl_tokenizer:\n      region_lengths: {fwr1: 26}\n')
            self.assertEqual(config_capacities(config), dict(vh=dict(fwr1=30, cdr3=30), vl=dict(fwr1=26)))

    def test_hidden_regions_cover_cdr3_and_every_light_region(self):
        self.assertEqual(HIDDEN_REGIONS, ['H-cdr3', *(f'L-{name}' for name in REGIONS)])
        self.assertEqual(len(HIDDEN_REGIONS), 8)


class RunExampleTests(unittest.TestCase):
    def test_every_valid_loop_is_kept_at_its_own_length_with_pad_diagnostics(self):
        sampler = StubSampler(['ARDY' + '-' * 26, 'AAA' + '-' * 27, 'AR-DY' + '-' * 25])
        result = run(sampler)
        self.assertEqual(result['samples'], ['ARDY', 'AAA', 'ARDY'])
        self.assertEqual((result['attempts'], result['valid_attempts'], result['invalid_attempts']), (3, 3, 0))
        self.assertEqual((result['length_matches'], result['length_mismatches']), (1, 2))
        self.assertEqual([g['interior_gap'] for g in result['raw_generations']], [False, False, True])
        self.assertEqual(result['raw_generations'][0]['full_sequence'], manifest_record()['pt1'] + 'ARDY' + FLANKS['fwr4'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['length_policy'], dict(mode='inferred', source='igcraft_pad_tokens', minimum=None, maximum=30))

    def test_empty_slot_is_a_failed_attempt_charged_to_the_budget(self):
        result = run(StubSampler(['-' * 30, 'AAA' + '-' * 27, 'AAAA' + '-' * 26]))
        self.assertEqual((result['attempts'], result['valid_attempts'], result['invalid_attempts']), (3, 2, 1))
        self.assertEqual([f['code'] for f in result['failures']], ['empty_generation'])
        self.assertFalse(result['complete'])

    def test_attempts_are_drawn_in_batches_and_seeded_per_example(self):
        sampler = StubSampler(['AAA'] * 2, ['AAA'] * 2, ['AAA'])
        result = run(sampler, num_samples=5, batch_size=2)
        self.assertEqual(sampler.counts, [2, 2, 1])
        self.assertEqual(result['attempts'], 5)
        self.assertEqual(sampler.seed, stable_example_seed(42, 'oas:7:example'))
        self.assertEqual(result['seed'], sampler.seed)

    def test_sampler_is_given_flank_regions_only(self):
        sampler = StubSampler(['AAA'] * 3)
        run(sampler)
        self.assertEqual(set(sampler.prepared), set(FLANK_REGIONS))

    def test_runtime_failure_costs_every_attempt_in_its_batch(self):
        sampler = StubSampler(RuntimeError('out of memory'), ['AAA', 'AAA'])
        result = run(sampler, num_samples=4, batch_size=2)
        self.assertEqual((result['attempts'], result['valid_attempts'], result['invalid_attempts']), (4, 2, 2))
        self.assertEqual([f['code'] for f in result['failures']], ['generation_exception'] * 2)
        self.assertIn('out of memory', result['failures'][0]['detail'])

    def test_consecutive_failures_stop_the_example_and_say_so(self):
        sampler = StubSampler(RuntimeError('boom'))
        result = run(sampler, num_samples=6, batch_size=2, max_errors=2)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['failures'][-1]['code'], 'max_consecutive_errors')
        self.assertFalse(result['complete'])


class FlankRegionTests(unittest.TestCase):
    def corpus_and_manifest(self):
        record = manifest_record()
        entry = dict(sequence_id='seq-3', heavy=dict(**FLANKS, cdr3='AAA'))
        manifest = dict(benchmark_id='b', source=dict(processed_file='c.json', processed_file_sha256='abc'),
                        records=[dict(record, source_index=0)])
        return [entry], manifest

    def test_export_holds_flanks_only_and_nothing_derived_from_the_loop(self):
        corpus, manifest = self.corpus_and_manifest()
        table = build(manifest, corpus, 'abc')
        self.assertEqual(table['records'], {'oas:7:example': FLANKS})
        self.assertEqual(table['regions'], list(FLANK_REGIONS))
        self.assertNotIn('AAA', json.dumps(table))  # the native loop is nowhere in the table

    def test_export_refuses_a_different_corpus_or_entry(self):
        corpus, manifest = self.corpus_and_manifest()
        with self.assertRaisesRegex(ValueError, 'Corpus differs'):
            build(manifest, corpus, 'other')
        corpus[0]['sequence_id'] = 'seq-9'
        with self.assertRaisesRegex(ValueError, 'different corpus entry'):
            build(manifest, corpus, 'abc')

    def test_regions_must_rebuild_pt1_and_pt2(self):
        record = manifest_record()
        self.assertEqual(flank_regions(record, FLANKS), FLANKS)
        with self.assertRaisesRegex(ValueError, 'do not rebuild'):
            flank_regions(record, {**FLANKS, 'cdr1': 'GYTW'})
        with self.assertRaisesRegex(ValueError, 'do not rebuild'):
            flank_regions(record, {**FLANKS, 'fwr4': 'WGQG'})

    def test_adapter_rejects_an_unknown_table_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'regions.json'
            path.write_text(json.dumps(dict(schema_version='other')))
            with self.assertRaisesRegex(ValueError, 'unsupported flank-region table'):
                load_flank_table(path)


if __name__ == '__main__':
    unittest.main()
