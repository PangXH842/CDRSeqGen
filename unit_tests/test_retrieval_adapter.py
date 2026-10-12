"""Flank-keyed retrieval reference: keys, training-only pools, fallback, hidden-field independence, CLI and scoring contract."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark import blind_check
from benchmark.adapters.retrieval import KEYS, flank_key, run_example, training_groups
from benchmark.externals import EXTERNALS
from benchmark.scorer import score_payload, validate_adapter_record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'benchmark/adapters/retrieval.py'


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
        return run_example(record, self.groups, self.pool, key='fr3tail12+fr4head6', **{'num_samples': 20, 'base_seed': 42, **options})

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
                               '--output', str(output), *extra], capture_output=True, text=True, cwd=ROOT, timeout=60)

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


if __name__ == '__main__':
    unittest.main()
