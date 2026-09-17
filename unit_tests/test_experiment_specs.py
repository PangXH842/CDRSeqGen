"""Specs must merge predictably and feed the existing plan identity/resume machinery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cdrseqgen.experiments import build_plan, spec_trials, SUITES


class ExperimentSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'base.json'
        self.base.write_text(json.dumps(dict(method='s2s', model_type='scratch',
            data_path='original.json', learning_rate=1e-4, seed=42)))
        self.path = self.root / 'spec.json'
        self.spec = dict(name='probe', base_config=str(self.base),
            overrides=dict(learning_rate=3e-5), trials=[dict(name='first'),
                dict(name='second', overrides=dict(learning_rate=1e-5))])

    def write(self):
        self.path.write_text(json.dumps(self.spec))
        return str(self.path)

    def test_merge_and_corpus_override(self):
        trials, name = spec_trials(self.write(), {'DATA_PATH': 'relocated.json'})
        self.assertEqual(name, 'probe')
        self.assertEqual([t['config']['learning_rate'] for t in trials], [3e-5,1e-5])
        self.assertTrue(all(t['config']['data_path']=='relocated.json' for t in trials))
        self.assertNotIn('s2s_igbert_diagnostic', SUITES)
        other = self.root / 'other.json'
        other.write_text(json.dumps(dict(method='ar',model_type='scratch',data_path='x.json')))
        self.spec['trials'][1]['base_config'] = str(other)
        self.assertEqual(spec_trials(self.write(), {})[0][1]['config']['method'],'ar')

    def test_invalid_specs_fail_early(self):
        original = json.loads(json.dumps(self.spec))
        cases = [dict(trials=[]), dict(trials=[dict(name='../bad')]),
                 dict(trials=[dict(name='same'),dict(name='same')]),
                 dict(trials=[dict(name='a'),dict(name='b')]),
                 dict(overrides=[]), dict(unknown=True), dict(name=None),
                 dict(trials=[dict(name='a',overrides=[])]),
                 dict(trials=[dict(name='a',typo=True)])]
        for change in cases:
            with self.subTest(change=change):
                self.spec = {**original, **change}
                with self.assertRaises(ValueError): spec_trials(self.write(),{})
        self.spec = original
        for key in ('SEED','FREEZE_ENCODER','ENCODER_INIT'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                spec_trials(self.write(),{key:'42'})

    def test_plan_identity_and_shared_execution_format(self):
        data=self.root/'data.json'
        data.write_text(json.dumps([dict(split='train'),dict(split='validation')]))
        self.spec['overrides'].update(data_path=str(data),eval_split='validation')
        with patch('cdrseqgen.experiments.source_identity',return_value={'source_sha256':'fixed'}):
            plan, devices = build_plan(['0,1','spec',self.write()],{})
            self.assertEqual(plan['mode'],'suite')
            self.assertEqual(devices,['0','1'])
            self.spec['trials'][1]['overrides']['learning_rate']=2e-5
            changed,_=build_plan(['2','spec',self.write()],{})
        self.assertEqual(plan['trials'][0]['trial_id'],changed['trials'][0]['trial_id'])
        self.assertNotEqual(plan['trials'][1]['trial_id'],changed['trials'][1]['trial_id'])
        with self.assertRaises(ValueError):build_plan(['0','spec'],{})

    def test_identity_ignores_where_the_corpus_lives(self):
        """Same bytes at two paths = same experiment. data_path is provenance, not identity."""
        payload = json.dumps([dict(split='train'), dict(split='validation')])
        first, second = self.root/'a'/'corpus.json', self.root/'b'/'corpus.json'
        for path in (first, second):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        seen, plans = [], []
        with patch('cdrseqgen.experiments.source_identity', return_value={'source_sha256': 'fixed'}):
            for path in (first, second):
                self.spec['overrides'].update(data_path=str(path), eval_split='validation')
                plan, _ = build_plan(['0', 'spec', self.write()], {})
                seen.append([t['trial_id'] for t in plan['trials']])
                plans.append(plan)
        self.assertEqual(seen[0], seen[1], "moving the corpus must not change trial identity")
        self.assertEqual([t['setting_id'] for t in plans[0]['trials']],
                         [t['setting_id'] for t in plans[1]['trials']])
        # Location is still recorded, just not fingerprinted.
        self.assertEqual({t['config']['data_path'] for t in plans[1]['trials']},
                         {str(second.resolve())})
        # A genuinely different corpus must still separate.
        (self.root/'c').mkdir(exist_ok=True)
        third = self.root/'c'/'corpus.json'
        third.write_text(json.dumps([dict(split='train'), dict(split='validation'),
                                     dict(split='validation')]))
        with patch('cdrseqgen.experiments.source_identity', return_value={'source_sha256': 'fixed'}):
            self.spec['overrides'].update(data_path=str(third), eval_split='validation')
            other, _ = build_plan(['0', 'spec', self.write()], {})
        self.assertNotEqual(seen[0], [t['trial_id'] for t in other['trials']])

if __name__=='__main__':unittest.main()
