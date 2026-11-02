"""The weekend queue's recipe generator and spec writer, on synthetic batches and the committed templates."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from benchmark.arms_recipe import ROOT, batch_arms, build
sys.path.insert(0, str(ROOT / 'scripts'))
import queue_specs

FIELDS = ['seed', 'status', 'run_dir', 'trial_id']


def make_batch(directory, trials, rows):
    """trials: [(suffix, method, seed, model_type)], rows: [(trial index, status, run dir name)] in file order."""
    directory.mkdir(parents=True)
    plan = dict(trials=[dict(suffix=s, trial_id=f'id{i}', config=dict(method=m, seed=seed, model_type=t))
                        for i, (s, m, seed, t) in enumerate(trials)])
    (directory / 'plan.json').write_text(json.dumps(plan))
    with (directory / 'runs.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter='\t')
        writer.writeheader()
        for index, status, run in rows:
            writer.writerow(dict(seed=trials[index][2], status=status, run_dir=str(ROOT / 'results/runs' / run), trial_id=f'id{index}'))
    return directory


class ArmsRecipeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_completed_scratch_trials_become_labelled_arms(self):
        batch = make_batch(self.root / 'b', [('mlm_scratch_no_partner_s44', 'mlm', 44, 'scratch'),
                                             ('ar_scratch_no_partner_L_s42', 'ar', 42, 'scratch')],
                           [(0, 'completed', 'b/mlm44'), (1, 'completed', 'b/ar')])
        self.assertEqual(batch_arms(batch), [
            dict(label='mlm_s44', method='mlm', training_seed=44, checkpoint='results/runs/b/mlm44'),
            dict(label='ar_L_s42', method='ar', training_seed=42, checkpoint='results/runs/b/ar')])

    def test_a_retry_row_replaces_the_failed_one(self):
        batch = make_batch(self.root / 'b', [('ar_scratch_no_partner_s42', 'ar', 42, 'scratch')],
                           [(0, 'failed', 'b/ar'), (0, 'completed', 'b/ar_2')])
        self.assertEqual(batch_arms(batch)[0]['checkpoint'], 'results/runs/b/ar_2')

    def test_unfinished_and_pretrained_trials_stop_the_script_unless_waived(self):
        batch = make_batch(self.root / 'b', [('a_scratch_no_partner_s1', 'ar', 1, 'scratch'), ('c_esm2', 'mlm', 1, 'esm2'),
                                             ('d_scratch_no_partner_s2', 'ar', 2, 'scratch')],
                           [(0, 'completed', 'b/a'), (1, 'completed', 'b/c'), (2, 'running', 'b/d')])
        with self.assertRaisesRegex(ValueError, 'pretrained mlm model'):    # pretrained AR/S2S are served now
            batch_arms(batch)
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            batch_arms(batch, skip_pretrained=True)
        self.assertEqual([a['label'] for a in batch_arms(batch, skip_pretrained=True, skip_incomplete=True)], ['a_s1'])

    def test_recipe_keeps_the_reference_protocol_and_rejects_duplicate_labels(self):
        batch = make_batch(self.root / 'b', [('mlm_scratch_no_partner_s44', 'mlm', 44, 'scratch')], [(0, 'completed', 'b/m')])
        recipe = build('queue', [batch])
        reference = json.loads((ROOT / 'benchmark/recipes/oas4_h3_length_free.json').read_text())
        for key in ('manifest', 'smoke_manifest', 'sampling', 'judge', 'endpoint'):
            self.assertEqual(recipe[key], reference[key])
        self.assertNotIn('iglm', recipe)
        self.assertEqual(len(recipe['limits']), len(reference['limits']) + 1)
        with self.assertRaisesRegex(ValueError, 'unique labels'):
            build('queue', [batch, batch])
        with self.assertRaisesRegex(ValueError, 'unique labels'):
            build('queue', [])


class QueueSpecTests(unittest.TestCase):
    def test_twelve_specs_with_the_intended_trials(self):
        specs = queue_specs.specs()
        self.assertEqual(len(specs), 12)
        for method in queue_specs.METHODS:
            self.assertEqual([t['name'] for t in specs[f'seeds_{method}']['trials']],
                             [f'{method}_scratch_no_partner_s{s}' for s in queue_specs.SEEDS])
            for k in (1, 2):
                trials = specs[f'repeat_{method}_r{k}']['trials']
                self.assertEqual((len(trials), trials[0]['overrides']['seed']), (1, 42))
            capacity = specs[f'capacity_{method}']['trials']
            self.assertEqual(len(capacity), 6)
            self.assertEqual(len({t['name'] for t in capacity}), 6)

    def test_trials_change_only_the_seed_or_the_size(self):
        for stem, spec in queue_specs.specs().items():
            method = stem.split('_')[1]  # seeds_<m>, repeat_<m>_r<k>, capacity_<m>
            _, source = queue_specs.template(method)
            for trial in spec['trials']:
                changed = {k for k in set(trial['overrides']) | set(source['overrides'])
                           if trial['overrides'].get(k) != source['overrides'].get(k)}
                self.assertLessEqual(changed, {'seed', 'data_seed', 'hidden_size', 'num_layers',
                                               'num_encoder_layers', 'num_decoder_layers'}, (stem, trial['name']))
                self.assertEqual(trial.get('base_config'), source.get('base_config'))

    def test_s2s_depth_is_set_on_both_sides_and_other_methods_leave_it_alone(self):
        specs = queue_specs.specs()
        xl = next(t for t in specs['capacity_s2s']['trials'] if '_XL_' in t['name'])['overrides']
        self.assertEqual((xl['num_encoder_layers'], xl['num_decoder_layers'], xl['num_layers']), (4, 4, 8))
        for method in ('mlm', 'ar'):
            self.assertTrue(all('num_encoder_layers' not in t['overrides'] for t in specs[f'capacity_{method}']['trials']))

    def test_committed_specs_match_the_generator(self):
        for stem, spec in queue_specs.specs().items():
            self.assertEqual(json.loads((queue_specs.OUT / f'{stem}.json').read_text()), spec, stem)


if __name__ == '__main__':
    unittest.main()
