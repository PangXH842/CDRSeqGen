"""Tests merged from: test_benchmark_externals.py, test_igcraft_adapter.py, test_iglm_adapter.py, test_retrieval_adapter.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
from benchmark import blind_check
from benchmark._common import execution_source_identity, validate_manifest_record
from benchmark.analysis.stats import planned_pairs
from benchmark.adapters.registry import EXTERNALS, External, plan_externals, recipe_externals
from benchmark.run import (
    ROOT,
    build_plan,
    environment_identity,
    execute,
    freeze_external_inputs,
    generation_commands,
    validate_frozen_plan,
)
from benchmark._common import stable_example_seed
from benchmark.adapters.igcraft import (
    HIDDEN_CDR3,
    HIDDEN_LIGHT,
    HIDDEN_REGIONS,
    InvalidGeneration,
    REGIONS,
    capacity_problems,
    config_capacities,
    extract_loop,
    load_flank_table,
    run_example,
)
from benchmark.manifest import FLANK_REGIONS, build_regions as build, flank_regions
from benchmark.adapters.iglm import (
    InvalidGeneration as IglmAdapterInvalidGeneration,
    assert_resume_compatible,
    extract_infilled_loop,
    run_example as iglm_adapter_run_example,
    stable_example_seed as iglm_adapter_stable_example_seed,
)
import copy
import hashlib
import subprocess
from benchmark.adapters.retrieval import (
    KEYS,
    flank_key,
    run_example as retrieval_adapter_run_example,
    training_groups,
)
from benchmark.adapters.registry import EXTERNALS
from benchmark.scorer import score_payload, validate_adapter_record


# ====================================================================================================
# from test_benchmark_externals.py
# External-generator registry, the runner's use of it, multi-external pairing and the blind check.
#
# No model runs: adapters are stubs, interpreters are the test's own, and plans are built by hand.
# ====================================================================================================

SAMPLING = dict(num_samples=20, temperature=1., top_p=.9, top_k=0, length_temperature=1., seed=42, max_new_tokens=97)
IGLM_OPTIONS = dict(model_name='IgLM', max_length=256)
FAKE = External(script='benchmark/adapters/fake.py', conda_env='fake',
                arguments=lambda output, sampling, options: ['--output', str(output), '--flag', options['flag']])

class RegistryTests(unittest.TestCase):
    def test_legacy_iglm_key_is_one_external(self):
        self.assertEqual(recipe_externals({'iglm': IGLM_OPTIONS}),
                         [dict(label='iglm', adapter='iglm', options=IGLM_OPTIONS)])

    def test_declared_externals_are_validated(self):
        entry = dict(label='iglm', adapter='iglm', options=IGLM_OPTIONS)
        self.assertEqual(recipe_externals({'externals': [entry]}), [entry])
        self.assertEqual(recipe_externals({}), [])
        for recipe, message in [({'externals': [entry, entry]}, 'unique'),
                                ({'externals': [dict(entry, label='empirical')]}, 'reserved'),
                                ({'externals': [dict(entry, adapter='nope')]}, 'Unknown external adapter'),
                                ({'externals': [entry], 'iglm': IGLM_OPTIONS}, 'both')]:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                recipe_externals(recipe)

    def test_plan_reader_accepts_both_layouts(self):
        legacy = dict(iglm_python='/envs/iglm/python', iglm=IGLM_OPTIONS)
        self.assertEqual(plan_externals(legacy), [dict(label='iglm', adapter='iglm', python='/envs/iglm/python', options=IGLM_OPTIONS)])
        current = dict(externals=[dict(label='x', adapter='iglm', python='p', options={})])
        self.assertIs(plan_externals(current), current['externals'])

    def test_iglm_rejects_a_prompt_over_its_token_budget(self):
        record = dict(example_id='e', pt1='A' * 100, pt2='B' * 11)
        EXTERNALS['iglm'].check_record(record, SAMPLING, IGLM_OPTIONS)
        with self.assertRaisesRegex(ValueError, 'needs 212 IgLM tokens'):
            EXTERNALS['iglm'].check_record(record, SAMPLING, dict(IGLM_OPTIONS, max_length=211))


IGCRAFT_OPTIONS = dict(checkpoint='../IgCraft/final.ckpt', config='../IgCraft/config.yaml', flank_regions='regions.json',
                       hf_revision='rev', sha256=dict(checkpoint='ck', config='cf', flank_regions='fr'),
                       num_steps=20, num_particles=32, temperature=0.1, batch_size=20, compile=True)


class IgCraftRegistryTests(unittest.TestCase):
    def test_adapter_arguments_carry_the_model_controls_and_pins(self):
        arguments = EXTERNALS['igcraft'].arguments(Path('/out/igcraft.json'), SAMPLING, IGCRAFT_OPTIONS)
        value = lambda flag: arguments[arguments.index(flag) + 1]
        self.assertEqual(value('--output'), '/out/igcraft.json')
        self.assertEqual((value('--num-steps'), value('--num-particles'), value('--igcraft-temperature'),
                          value('--batch-size'), value('--max-errors')), ('20', '32', '0.1', '20', '20'))
        self.assertEqual((value('--checkpoint-sha256'), value('--hf-revision')), ('ck', 'rev'))
        self.assertNotIn('--no-compile', arguments)
        uncompiled = EXTERNALS['igcraft'].arguments(Path('o'), SAMPLING, dict(IGCRAFT_OPTIONS, compile=False))
        self.assertIn('--no-compile', uncompiled)

    def test_pins_are_optional(self):
        bare = {k: v for k, v in IGCRAFT_OPTIONS.items() if k not in ('sha256', 'hf_revision')}
        arguments = EXTERNALS['igcraft'].arguments(Path('o'), SAMPLING, bare)
        self.assertNotIn('--checkpoint-sha256', arguments)
        self.assertEqual(EXTERNALS['igcraft'].inputs(bare), {'regions.json': None, '../IgCraft/config.yaml': None,
                                                             '../IgCraft/final.ckpt': None})

    def test_inputs_map_each_file_to_its_pin(self):
        self.assertEqual(EXTERNALS['igcraft'].inputs(IGCRAFT_OPTIONS),
                         {'regions.json': 'fr', '../IgCraft/config.yaml': 'cf', '../IgCraft/final.ckpt': 'ck'})

    def test_record_check_needs_a_table_entry_that_rebuilds_pt1_and_pt2(self):
        record = dict(example_id='e', pt1='AB', pt2='CD')
        with tempfile.TemporaryDirectory() as directory:
            table, config = Path(directory) / 'regions.json', Path(directory) / 'config.yaml'
            config.write_text('model:\n  datamodule:\n    vh_tokenizer:\n      region_lengths: '
                              '{fwr1: 30, cdr1: 15, fwr2: 20, cdr2: 15, fwr3: 42, cdr3: 30, fwr4: 12}\n'
                              '    vl_tokenizer:\n      region_lengths: '
                              '{fwr1: 26, cdr1: 12, fwr2: 18, cdr2: 8, fwr3: 38, cdr3: 14, fwr4: 11}\n')
            options = dict(flank_regions=str(table), config=str(config))
            good = dict(fwr1='A', cdr1='', fwr2='', cdr2='', fwr3='B', fwr4='CD')
            long_tail = dict(record, pt2='C' * 13)
            for entries, message, framework in [({'e': good}, None, record),
                                                ({}, 'absent from the flank-region table', record),
                                                ({'e': dict(good, fwr4='XX')}, 'do not rebuild', record),
                                                ({'e': dict(good, fwr4='C' * 13)}, 'does not fit IgCraft', long_tail)]:
                table.write_text(json.dumps(dict(records=entries)))
                import benchmark.adapters.registry as module
                module._read_json.cache_clear()
                module._igcraft_capacities.cache_clear()
                with self.subTest(message):
                    if message is None:
                        EXTERNALS['igcraft'].check_record(framework, SAMPLING, options)
                    else:
                        with self.assertRaisesRegex(ValueError, message):
                            EXTERNALS['igcraft'].check_record(framework, SAMPLING, options)


class FreezeInputsTests(unittest.TestCase):
    def test_inputs_are_hashed_into_the_plan_and_pins_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.bin'
            path.write_bytes(b'abc')
            digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
            entry = dict(label='fake_a', adapter='fake', options=dict(pin=digest))
            fake = External(script='x', conda_env='x', arguments=lambda *a: [],
                            inputs=lambda options: {str(path): options['pin']})
            frozen = {}
            with patch.dict(EXTERNALS, {'fake': fake}):
                freeze_external_inputs(entry, frozen)
                self.assertEqual(frozen, {str(path): digest})
                freeze_external_inputs(dict(entry, options=dict(pin=None)), {})
                with self.assertRaisesRegex(ValueError, 'not its pinned'):
                    freeze_external_inputs(dict(entry, options=dict(pin='0' * 64)), {})


class RunnerTests(unittest.TestCase):
    def plan(self, **overrides):
        return dict(cdr_python='cdr-python', manifest='inputs.json', sampling=SAMPLING, arms=[
            dict(label='mlm', method='mlm', checkpoint='mlm', decoding='independent')], **overrides)

    def test_legacy_plan_builds_the_command_it_always_did(self):
        plan = self.plan(iglm_python='iglm-python', iglm=IGLM_OPTIONS)
        output = Path('/tmp/example-only')
        label, path, command = generation_commands(plan, output)[-1]
        self.assertEqual((label, path), ('iglm', output / 'generations/iglm.json'))
        self.assertEqual(command, [
            'iglm-python', str(ROOT / 'benchmark/adapters/iglm.py'),
            '--manifest', str(ROOT / 'inputs.json'), '--num-samples', '20', '--temperature', '1.0', '--top-p', '0.9',
            '--seed', '42', '--device', 'cuda', '--output', str(path), '--model-name', 'IgLM',
            '--length-policy', 'natural', '--max-attempts', '20', '--max-errors', '20',
            '--max-length', '256', '--max-new-tokens', '97'])

    def test_every_listed_external_is_launched_with_its_own_interpreter(self):
        externals = [dict(label='iglm', adapter='iglm', python='iglm-python', options=IGLM_OPTIONS),
                     dict(label='fake_a', adapter='fake', python='fake-python', options=dict(flag='a'))]
        with patch.dict(EXTERNALS, {'fake': FAKE}):
            commands = generation_commands(self.plan(externals=externals), Path('/out'))
        self.assertEqual([label for label, _, _ in commands], ['mlm', 'iglm', 'fake_a'])
        fake = commands[-1]
        self.assertEqual(fake[1], Path('/out/generations/fake_a.json'))
        self.assertEqual(fake[2][:2], ['fake-python', str(ROOT / 'benchmark/adapters/fake.py')])
        self.assertEqual(fake[2][-4:], ['--output', str(fake[1]), '--flag', 'a'])
        self.assertEqual(fake[2][fake[2].index('--num-samples') + 1], '20')

    def test_comparison_figure_needs_two_runs(self):
        one = dict(label='fake_a', adapter='fake', python='fake-python', options=dict(flag='a'))
        for externals, plots in [([one], 0), ([one, dict(one, label='fake_b')], 1)]:
            with self.subTest(runs=len(externals)), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(EXTERNALS, {'fake': FAKE}), patch('benchmark.run.validate_frozen_plan'), \
                    patch('benchmark.run.subprocess.run') as run:
                plan = self.plan(externals=externals, frameworks=1, judge=dict(name='judge', max_len=384))
                plan['arms'] = []
                execute(plan, Path(directory) / 'batch', '0')
            scripts = [Path(call.args[0][1]).name for call in run.call_args_list]
            self.assertEqual(scripts.count('plot_comparison.py'), plots)
            self.assertEqual(scripts.count('scorer.py'), 1)
            self.assertEqual(scripts.count('fake.py'), len(externals))

    def test_missing_package_is_recorded_as_null(self):
        identity = environment_identity(sys.executable, ('no-such-distribution',))
        self.assertEqual(identity['packages'], {'no-such-distribution': None})
        self.assertEqual(identity['python'], sys.version)

    def test_resume_compares_the_package_names_the_plan_stored(self):
        stored = environment_identity(sys.executable, ('no-such-distribution',))
        plan = {'execution_source': execution_source_identity(), 'frozen_files': {},
                'environments': {sys.executable: stored}}
        validate_frozen_plan(plan)
        stored['packages']['no-such-distribution'] = '1.0'
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            validate_frozen_plan(plan)

    def test_plan_rejects_bad_labels_and_stray_interpreters_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            def recipe(**content):
                path = Path(directory) / 'recipe.json'
                path.write_text(json.dumps(dict(sampling=SAMPLING, **content)))
                return path
            iglm = dict(iglm=IGLM_OPTIONS)
            here = {'iglm': Path(sys.executable)}
            for content, interpreters, message in [
                    (dict(arms=[dict(label='iglm')], **iglm), here, 'unique'),
                    (dict(arms=[dict(label='empirical')]), {}, 'non-reserved'),
                    (dict(arms=[]), {}, 'at least one'),
                    (iglm, {**here, 'ghost': Path('x')}, 'does not name'),
                    (iglm, {'iglm': Path('/no/such/python')}, 'No interpreter for external .iglm.*--external-python iglm=PATH')]:
                with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                    build_plan(recipe(**content), 'full', Path(sys.executable), interpreters)

    def test_plan_supports_test_split_and_arbitrary_cohort_size_with_value_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            processed = tmp / 'data/processed'
            processed.mkdir(parents=True)
            corpus = processed / 'processed_test.json'
            corpus.write_text('[]')
            from benchmark._common import sha256_file
            corpus_hash = sha256_file(corpus)
            records = [
                dict(example_id=f'oas:1:s{i}', parent_sequence='AAAACCCCCGGGG',
                     pt1='AAAA', native='CCCCC', pt2='GGGG', native_length=5, infill_range=[4, 9])
                for i in range(3)
            ]
            manifest = dict(
                task=dict(track='length_free'),
                selection=dict(split='test'),
                source=dict(processed_file='processed_test.json', processed_file_sha256=corpus_hash),
                records=records,
            )
            mpath = tmp / 'manifest.json'
            mpath.write_text(json.dumps(manifest))
            mpath.with_suffix('.json.sha256').write_text(f'{sha256_file(mpath)}  manifest.json\n')
            rpath = tmp / 'recipe.json'
            rpath.write_text(json.dumps(dict(
                name='test_cohort',
                split='test',
                expected_frameworks=dict(full=3),
                manifest='manifest.json',
                smoke_manifest='manifest.json',
                arms=[],
                externals=[dict(label='fake', adapter='fake', options={})],
                sampling=dict(num_samples=2, smoke_num_samples=1, temperature=1.0, top_p=0.9, top_k=0,
                              length_temperature=1.0, max_new_tokens=97, seed=42),
                judge=dict(name='Exscientia/IgBert', max_len=384),
                endpoint='edit_similarity_mean',
                limits=[],
            )))
            fake = External(script='benchmark/adapters/fake.py', conda_env='cdrprj',
                            arguments=lambda *a: [], packages=())
            with patch('benchmark.run.ROOT', tmp), patch.dict(EXTERNALS, {'fake': fake}):
                plan = build_plan(rpath, 'full', Path(sys.executable), {'fake': Path(sys.executable)})
                self.assertEqual(plan['frameworks'], 3)
                self.assertEqual(plan['requested_attempts'], 6)
                # Bad split raises ValueError (not AssertionError)
                manifest['selection']['split'] = 'train'
                mpath.write_text(json.dumps(manifest))
                mpath.with_suffix('.json.sha256').write_text(f'{sha256_file(mpath)}  manifest.json\n')
                with self.assertRaisesRegex(ValueError, "must be 'validation' or 'test'"):
                    build_plan(rpath, 'full', Path(sys.executable), {'fake': Path(sys.executable)})


class PairTests(unittest.TestCase):
    LOCAL = ['mlm_s42', 'mlm_s43', 'ar_s42', 'ar_s43', 's2s_s42', 's2s_s43']

    def test_default_family_is_unchanged(self):
        pairs = planned_pairs(self.LOCAL + ['iglm', 'empirical'])
        self.assertEqual(len(pairs), 13)
        self.assertEqual(pairs[:6], [(label, 'iglm') for label in self.LOCAL])
        self.assertEqual(pairs.count(('iglm', 'empirical')), 1)

    def test_each_external_adds_its_own_comparisons(self):
        pairs = planned_pairs(self.LOCAL + ['iglm', 'igcraft', 'empirical'], externals=('iglm', 'igcraft'))
        self.assertEqual(len(pairs), 6 + 6 + 8)
        self.assertIn(('ar_s43', 'igcraft'), pairs)
        self.assertIn(('igcraft', 'empirical'), pairs)
        self.assertNotIn(('iglm', 'igcraft'), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_references_form_a_ladder_beneath_every_system(self):
        labels = self.LOCAL + ['iglm', 'igcraft', 'empirical', 'retrieval']
        pairs = planned_pairs(labels, externals=('iglm', 'igcraft'), references=('empirical', 'retrieval'))
        self.assertEqual(len(pairs), 12 + 9 + 8)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertIn(('retrieval', 'empirical'), pairs)
        self.assertNotIn(('empirical', 'retrieval'), pairs)
        self.assertEqual({a for a, b in pairs if b == 'retrieval'}, set(self.LOCAL) | {'iglm', 'igcraft'})
        self.assertNotIn(('iglm', 'igcraft'), pairs)
        self.assertEqual(planned_pairs(labels[:-1], externals=('iglm', 'igcraft')), planned_pairs(
            labels[:-1], externals=('iglm', 'igcraft'), references=('empirical',)))

    def test_one_label_cannot_be_an_external_and_a_reference(self):
        with self.assertRaisesRegex(ValueError, 'no label in both roles'):
            planned_pairs(['a', 'iglm', 'empirical'], externals=('iglm',), references=('empirical', 'iglm'))

    def test_missing_labels_are_rejected(self):
        for labels, externals in [(['a', 'empirical'], ('iglm',)), (['a', 'iglm'], ('iglm',)),
                                  (['a', 'iglm', 'empirical'], ()), (['a', 'a', 'iglm', 'empirical'], ('iglm',))]:
            with self.subTest(labels=labels, externals=externals), self.assertRaises(ValueError):
                planned_pairs(labels, externals)


STUB = textwrap.dedent('''
    import argparse, hashlib, json, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest'); parser.add_argument('--output'); parser.add_argument('--reads', default='flanks')
    args = parser.parse_args()
    records = []
    for r in json.load(open(args.manifest))['records']:
        basis = r['pt1'] + r['pt2']
        if args.reads == 'native': basis += r['native']
        if args.reads == 'index': basis += str(r['source_index'])
        if args.reads == 'clock': basis += str(time.time_ns())
        digest = hashlib.sha256(basis.encode()).hexdigest()
        loop = ''.join('ACDEFGHIKLMNPQRSTVWY'[int(c, 16)] for c in digest[:8])
        records.append(dict(example_id=r['example_id'], samples=[loop], attempts=1, valid_attempts=1, invalid_attempts=0,
                            raw_generations=[dict(attempt=1, loop=loop, length=len(loop), length_match=len(loop) == r['native_length'])],
                            failures=[]))
    json.dump(dict(records=records), open(args.output, 'w'))
''')


def manifest(track='length_free', count=3):
    records = []
    for i in range(count):
        pt1, native, pt2 = 'QVQL' + 'KV' * (i + 1), 'ARDY' + 'G' * i, 'WGQGTLVTVSS'
        records.append(dict(example_id=f'e{i}', source_index=100 + i, sequence_id=f's{i}', pt1=pt1, native=native, pt2=pt2,
                            native_length=len(native), infill_range=[len(pt1), len(pt1) + len(native)],
                            parent_sequence=pt1 + native + pt2))
    return dict(task=dict(track=track), records=records)


class BlindCheckTests(unittest.TestCase):
    def verdict(self, reads, **options):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / 'stub.py'
            stub.write_text(STUB)
            command = [sys.executable, str(stub), '--manifest', '{manifest}', '--output', '{output}', '--reads', reads]
            return blind_check.check(options.pop('manifest', manifest()), command, Path(directory) / 'work', **options)

    def test_perturbation_keeps_flanks_and_replaces_everything_hidden(self):
        original = manifest()
        blind = blind_check.perturb_manifest(original)
        for before, after in zip(original['records'], blind['records']):
            validate_manifest_record(after)
            self.assertEqual((before['pt1'], before['pt2']), (after['pt1'], after['pt2']))
            self.assertNotEqual(before['native_length'], after['native_length'])
            self.assertEqual(before['example_id'], after['example_id'])
        self.assertEqual([r['source_index'] for r in blind['records']], [101, 102, 100])
        self.assertEqual(blind_check.perturb_manifest(original), blind)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            blind_check.perturb_manifest(dict(original, records=original['records'][:1]))

    def test_adapter_reading_only_flanks_passes(self):
        report = self.verdict('flanks')
        self.assertEqual((report['verdict'], report['control_differs'], report['hidden_field_differs']), ('pass', [], []))

    def test_adapter_reading_the_native_loop_is_caught(self):
        report = self.verdict('native')
        self.assertEqual(report['verdict'], 'leak')
        self.assertEqual(report['hidden_field_differs'], ['e0', 'e1', 'e2'])

    def test_adapter_looking_the_answer_up_by_index_is_caught(self):
        self.assertEqual(self.verdict('index')['verdict'], 'leak')

    def test_irreproducible_adapter_is_inconclusive_not_a_pass(self):
        self.assertEqual(self.verdict('clock')['verdict'], 'inconclusive')

    def test_control_can_be_skipped(self):
        report = self.verdict('flanks', control=False)
        self.assertEqual((report['verdict'], report['control_differs']), ('pass', None))

    def test_unusable_requests_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'length_free'):
            self.verdict('flanks', manifest=manifest('fixed_native_length'))
        with self.assertRaisesRegex(ValueError, 'placeholders'):
            blind_check.check(manifest(), ['python', 'adapter.py'], tempfile.gettempdir())
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.verdict('flanks', limit=1)


# ====================================================================================================
# from test_igcraft_adapter.py
# IgCraft adapter accounting, capacity guards and the flank-region export, with a stub sampler.
#
# No IgCraft package, checkpoint or GPU is needed: the adapter imports the model only inside IgCraftSampler.
# ====================================================================================================

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


# ====================================================================================================
# from test_iglm_adapter.py
# ====================================================================================================

def iglm_adapter_manifest_record():
    return {
        "example_id": "oas:7:example",
        "parent_sequence": "QQAAAWW",
        "pt1": "QQ",
        "native": "AAA",
        "pt2": "WW",
        "native_length": 3,
        "infill_range": [2, 5],
        "chain_token": "[HEAVY]",
        "species_token": "[HUMAN]",
    }


class FakeSampler:
    def __init__(self, draws):
        self.draws = iter(draws)
        self.seed = None
        self.prepared = None

    def set_seed(self, seed):
        self.seed = seed

    def prepare(self, record):
        self.prepared = record["example_id"]

    def draw(self, **_kwargs):
        value = next(self.draws)
        if isinstance(value, Exception):
            raise value
        return value


class IgLMAdapterTests(unittest.TestCase):
    def test_extracts_loop_only_when_both_flanks_are_preserved(self):
        self.assertEqual(extract_infilled_loop("QQCCCWW", "QQ", "WW"), "CCC")
        with self.assertRaisesRegex(IglmAdapterInvalidGeneration, "pt1"):
            extract_infilled_loop("RRCCCWW", "QQ", "WW")
        with self.assertRaisesRegex(IglmAdapterInvalidGeneration, "pt2"):
            extract_infilled_loop("QQCCCVV", "QQ", "WW")

    def test_attempt_accounting_preserves_variable_lengths_and_duplicates(self):
        sampler = FakeSampler(["QQAAAAWW", "QQCCCWW", "QQCCCWW"])
        result = iglm_adapter_run_example(
            iglm_adapter_manifest_record(),
            sampler,
            num_samples=2,
            max_attempts=5,
            temperature=1.0,
            top_p=0.9,
            base_seed=42,
            max_errors=3,
        )

        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["valid_attempts"], 3)
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["samples"], ["CCC", "CCC"])
        self.assertEqual([item["loop"] for item in result["raw_generations"]], ["AAAA", "CCC", "CCC"])
        self.assertTrue(result["complete"])
        self.assertEqual(sampler.prepared, iglm_adapter_manifest_record()["example_id"])

    def test_invalid_draws_are_recorded_and_bounded(self):
        sampler = FakeSampler(["RRCCCWW", "RRCCCWW", "QQCCCWW"])
        result = iglm_adapter_run_example(
            iglm_adapter_manifest_record(),
            sampler,
            num_samples=1,
            max_attempts=5,
            temperature=1.0,
            top_p=0.9,
            base_seed=42,
            max_errors=2,
        )

        self.assertEqual(result["attempts"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["invalid_attempts"], 2)
        self.assertEqual(result["failures"][-1]["code"], "max_consecutive_errors")

    def test_natural_policy_retains_variable_lengths_without_filtering(self):
        sampler = FakeSampler(["QQAAAAWW", "QQCCCWW"])
        sampler.length_policy = "natural"
        result = iglm_adapter_run_example(
            iglm_adapter_manifest_record(), sampler,
            num_samples=2, max_attempts=2, temperature=1.0, top_p=0.9,
            base_seed=42, max_errors=3,
        )
        self.assertEqual(result["samples"], ["AAAA", "CCC"])
        self.assertEqual(result["length_matches"], 1)
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["length_policy"]["mode"], "inferred")

    def test_example_seed_is_stable_and_id_specific(self):
        self.assertEqual(iglm_adapter_stable_example_seed(42, "a"), iglm_adapter_stable_example_seed(42, "a"))
        self.assertNotEqual(iglm_adapter_stable_example_seed(42, "a"), iglm_adapter_stable_example_seed(42, "b"))

    def test_resume_rejects_changed_sampling_parameters(self):
        expected = {
            "schema_version": "v1",
            "adapter": "iglm",
            "benchmark_id": "b",
            "manifest": {"sha256": "x"},
            "model": {"name": "IgLM"},
            "sampling": {"temperature": 1.0},
        }
        assert_resume_compatible(dict(expected), expected)
        changed = {**expected, "sampling": {"temperature": 0.8}}
        with self.assertRaisesRegex(ValueError, "sampling"):
            assert_resume_compatible(changed, expected)


# ====================================================================================================
# from test_retrieval_adapter.py
# Flank-keyed retrieval reference: keys, training-only pools, fallback, hidden-field independence, CLI and scoring contract.
# ====================================================================================================

RETRIEVAL_ADAPTER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RETRIEVAL_ADAPTER_ROOT / 'benchmark/adapters/retrieval.py'

def heavy(fwr3, fwr4, cdr3):
    return dict(fwr1='QVQL', cdr1='GYTF', fwr2='MHWVR', cdr2='INPN', fwr3=fwr3, cdr3=cdr3, fwr4=fwr4)


# Two J groups sharing a V tail, one with a different tail, plus loops that must never be returned.
CORPUS = [
    dict(split='train', heavy=heavy('NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC', 'WGQGTLVTVSS', 'ARAAAAAA')),
    dict(split='train', heavy=heavy('NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC', 'WGQGTLVTVSS', 'ARCCCCCC')),
    dict(split='train', heavy=heavy('NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC', 'WGKGTTVTVSS', 'ARDDDDDDDD')),
    dict(split='train', heavy=heavy('SYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC', 'WGQGTLVTVSS', 'ARGGGGGGGGGG')),
    dict(split='validation', heavy=heavy('NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC', 'WGQGTLVTVSS', 'ARVALIDATION')),
    dict(split='test', heavy=heavy('NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYC', 'WGQGTLVTVSS', 'ARTESTTEST')),
]
TRAIN_LOOPS = {'ARAAAAAA', 'ARCCCCCC', 'ARDDDDDDDD', 'ARGGGGGGGGGG'}


def framework(example_id='f1', pt1=None, pt2='WGQGTLVTVSS', native='ARNATIVE'):
    h = CORPUS[0]['heavy']
    pt1 = pt1 or h['fwr1'] + h['cdr1'] + h['fwr2'] + h['cdr2'] + h['fwr3']
    return dict(example_id=example_id, source_index=5, sequence_id='s5', pt1=pt1, native=native, pt2=pt2,
                native_length=len(native), infill_range=[len(pt1), len(pt1) + len(native)], parent_sequence=pt1 + native + pt2)


class KeyTests(unittest.TestCase):
    def test_primary_key_is_the_documented_v_j_proxy(self):
        self.assertEqual(flank_key('fr3tail12+fr4head6', 'A' * 30 + 'VYYCARDY', 'WGQGTLVTVSS'), ('A' * 4 + 'VYYCARDY', 'WGQGTL'))
        self.assertEqual(flank_key('fr4', 'xx', 'WGQGTLVTVSS'), 'WGQGTLVTVSS')
        self.assertEqual(flank_key('fr3tail20+fr4', 'B' * 30, 'WG'), ('B' * 20, 'WG'))
        self.assertEqual(list(KEYS)[0], 'fr3tail12+fr4head6')


class GroupTests(unittest.TestCase):
    def test_only_training_loops_are_indexed_and_grouped_by_flank_key(self):
        groups, pool = training_groups(CORPUS, 'fr3tail12+fr4head6')
        self.assertEqual({loop for _, loop in pool}, TRAIN_LOOPS)
        self.assertEqual(len(groups), 3)
        self.assertEqual(sorted(len(group) for group in groups.values()), [1, 1, 2])
        self.assertTrue(all(CORPUS[index]['split'] == 'train' for index, _ in pool))

    def test_bad_corpora_fail_instead_of_being_skipped(self):
        for corpus, message in [([dict(heavy=heavy('A' * 12, 'WGQGTL', 'ARAA'))], 'explicit supported split'),
                                ([dict(split='train', heavy=heavy('A' * 12, 'WGQGTL', 'ARXA'))], 'invalid training H3'),
                                ([dict(split='train', heavy=heavy('A' * 12, 'WGQGTL', ''))], 'invalid training H3'),
                                ([dict(split='test', heavy=heavy('A' * 12, 'WGQGTL', 'ARAA'))], 'no training H3')]:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                training_groups(corpus, 'fr4')


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.groups, self.pool = training_groups(CORPUS, 'fr3tail12+fr4head6')

    def draw(self, record, **options):
        return retrieval_adapter_run_example(record, self.groups, self.pool, key='fr3tail12+fr4head6', **{'num_samples': 20, 'base_seed': 42, **options})

    def test_draws_come_from_the_framework_s_own_group(self):
        result = self.draw(framework())
        self.assertEqual(result['pool'], 'flank_group')
        self.assertEqual(result['pool_size'], 2)
        self.assertLessEqual(set(result['samples']), {'ARAAAAAA', 'ARCCCCCC'})
        self.assertEqual(len(result['samples']), 20)

    def test_an_unseen_key_falls_back_to_the_unconditional_pool_and_says_so(self):
        result = self.draw(framework(pt2='WGXXXXXXXXX'))
        self.assertEqual((result['pool'], result['pool_size']), ('unconditional_fallback', 4))
        self.assertLessEqual(set(result['samples']), TRAIN_LOOPS)

    def test_validation_and_test_loops_are_never_returned(self):
        seen = set()
        for number in range(40):
            seen |= set(self.draw(framework(f'f{number}'))['samples'])
        self.assertTrue(seen <= TRAIN_LOOPS and not seen & {'ARVALIDATION', 'ARTESTTEST'})

    def test_output_is_deterministic_per_framework_and_seed(self):
        a, b = self.draw(framework()), self.draw(framework())
        self.assertEqual(a['samples'], b['samples'])
        self.assertNotEqual(a['samples'], self.draw(framework(), base_seed=43)['samples'])
        self.assertNotEqual(a['samples'], self.draw(framework('f2'))['samples'])

    def test_hidden_fields_do_not_reach_the_draws(self):
        hidden = framework(native='ARCHANGED')
        hidden.update(source_index=99, sequence_id='other', parent_sequence=hidden['pt1'] + 'ARCHANGED' + hidden['pt2'])
        self.assertEqual(self.draw(framework())['samples'], self.draw(hidden)['samples'])

    def test_accounting_satisfies_the_adapter_contract(self):
        record = framework(native='ARAAAAAA')
        result = self.draw(record)
        validate_adapter_record(result, record, track='length_free')
        self.assertEqual(result['length_matches'] + result['length_mismatches'], result['valid_attempts'])
        self.assertTrue(all(g['training_source_index'] in {0, 1} for g in result['raw_generations']))
        self.assertEqual(result['raw_generations'][0]['full_sequence'], record['pt1'] + result['samples'][0] + record['pt2'])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.corpus = self.dir / 'corpus.json'
        self.corpus.write_text(json.dumps(CORPUS))
        digest = hashlib.sha256(self.corpus.read_bytes()).hexdigest()
        records = [framework('f1'), framework('f2', pt2='WGKGTTVTVSS', native='ARNATIVEX'), framework('f3')]
        self.manifest_data = dict(benchmark_id='b', schema_version='m', task=dict(track='length_free'),
                                  source=dict(processed_file='corpus.json', processed_file_sha256=digest), records=records)
        self.manifest = self.dir / 'manifest.json'
        self.manifest.write_text(json.dumps(self.manifest_data))

    def run_cli(self, output, *extra, manifest=None):
        return subprocess.run([sys.executable, str(SCRIPT), '--manifest', str(manifest or self.manifest), '--corpus', str(self.corpus),
                               '--output', str(output), *extra], capture_output=True, text=True, cwd=RETRIEVAL_ADAPTER_ROOT, timeout=60)

    def test_cli_writes_a_payload_the_scorer_accepts(self):
        output = self.dir / 'out.json'
        result = self.run_cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(output.read_text())
        self.assertEqual((payload['adapter'], payload['model']['parameters_network'], payload['model']['neural']), ('retrieval', 0, False))
        self.assertEqual((payload['sampling']['temperature'], payload['sampling']['sampling_controls_applied']), (1.0, False))
        self.assertEqual(payload['summary']['total_valid_attempts'], 60)
        self.assertTrue(payload['finished_at'])

        class NoJudge:
            def score(self, record, samples):
                return [2.0] * len(samples)
        by_id = {r['example_id']: r for r in self.manifest_data['records']}
        scored = score_payload('retrieval', output, payload, by_id, 2, NoJudge())
        self.assertEqual(len(scored['records']), 3)

    def test_an_interrupted_run_resumes_to_the_identical_result(self):
        output = self.dir / 'out.json'
        self.assertEqual(self.run_cli(output).returncode, 0)
        complete = json.loads(output.read_text())
        interrupted = copy.deepcopy(complete)
        interrupted['records'] = interrupted['records'][:2]
        interrupted['finished_at'] = None
        interrupted.pop('summary')
        output.write_text(json.dumps(interrupted))
        resumed = self.run_cli(output, '--resume')
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn('skip', resumed.stdout)
        again = json.loads(output.read_text())
        self.assertEqual([r['samples'] for r in again['records']], [r['samples'] for r in complete['records']])
        self.assertEqual(again['summary']['records_completed'], 3)

    def test_resume_refuses_changed_settings_existing_output_and_a_different_corpus(self):
        output = self.dir / 'out.json'
        self.assertEqual(self.run_cli(output, '--limit', '2').returncode, 0)
        refused = {'no --resume': self.run_cli(output), 'different key': self.run_cli(output, '--resume', '--limit', '2', '--key', 'fr4'),
                   'different limit': self.run_cli(output, '--resume')}
        for reason, result in refused.items():
            self.assertNotEqual(result.returncode, 0, reason)
        self.assertIn('incompatible for resume', refused['different key'].stderr)
        self.corpus.write_text(json.dumps(CORPUS[:5]))
        rejected = self.run_cli(self.dir / 'other.json')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('corpus differs', rejected.stderr)

    def test_it_passes_the_information_budget_test(self):
        command = [sys.executable, str(SCRIPT), '--manifest', '{manifest}', '--output', '{output}', '--corpus', str(self.corpus), '--num-samples', '5']
        report = blind_check.check(self.manifest_data, command, self.dir / 'blind')
        self.assertEqual(report['verdict'], 'pass')

    def test_registry_builds_the_command_and_pins_the_corpus(self):
        options = dict(corpus='data/x.json', key='fr4', sha256='abc')
        spec = EXTERNALS['retrieval']
        self.assertEqual(spec.arguments(Path('/o.json'), {}, options), ['--output', '/o.json', '--corpus', 'data/x.json', '--key', 'fr4'])
        self.assertEqual(spec.inputs(options), {'data/x.json': 'abc'})
        spec.check_record(framework(), {}, options)
        with self.assertRaisesRegex(ValueError, 'unknown retrieval key'):
            spec.check_record(framework(), {}, dict(options, key='nope'))
        self.assertEqual(spec.packages, ())


if __name__ == "__main__":
    unittest.main()
