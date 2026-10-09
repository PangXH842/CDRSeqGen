"""External-generator registry, the runner's use of it, multi-external pairing and the blind check.

No model runs: adapters are stubs, interpreters are the test's own, and plans are built by hand.
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
from benchmark.analysis import planned_pairs
from benchmark.externals import EXTERNALS, External, plan_externals, recipe_externals
from benchmark.run import (
    ROOT, build_plan, environment_identity, execute, freeze_external_inputs, generation_commands, validate_frozen_plan)

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
                import benchmark.externals as module
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


if __name__ == '__main__':
    unittest.main()
