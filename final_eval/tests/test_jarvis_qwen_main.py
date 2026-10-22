"""Qwen main wiring, frozen pilot guard and accounting; no network/real keys."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import pandas as pd

from final_eval import jarvis_main as main
from final_eval.inference.client import generation_config
from final_eval.tests import test_jarvis_main as main_tests
from final_eval.utils.artifacts import atomic_csv, atomic_json, digest, file_hash


class QwenMainTests(unittest.TestCase):
    def setUp(self):
        self.fixture = main_tests.MainApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config_path,self.pilot_path,self.output = main.stage_paths('qwen')
        self.pilot_cfg = {**self.fixture.cfg,'name':'qwen-3.8-27b-fp8','max_tokens':8192,
                          'reasoning_effort':'low','request_budget':300}
        atomic_json(self.config_path.parent/'target.qwen.pilot8192.json',self.pilot_cfg)
        comp = pd.read_csv(main.INPUT,keep_default_na=False)
        comp = comp[comp.uid.isin([f'q{i}' for i in range(16)])].copy()
        comp['uid'] = 'p' + comp.uid
        comp['base_group_id'] = 'pilot'
        atomic_csv(comp,main.PILOT_INPUT)
        rows = [dict(uid=t['uid'],eval_model=self.pilot_cfg['name'],method=t['method'],target_ratio=t['target_ratio'],
                     response_status='ok',correct=True,parse_status='ok') for t in main.evaluate.compression_tasks(comp)]
        path = self.pilot_path.parent/'responses.csv'
        atomic_csv(pd.DataFrame(rows),path)
        atomic_json(str(path)+'.manifest.json',{'config':generation_config(self.pilot_cfg)})
        atomic_json(self.pilot_path,{'passed':True,'target_config_hash':digest(generation_config(self.pilot_cfg)),
            'responses_sha256':file_hash(path),'compression_sha256':file_hash(main.PILOT_INPUT),'pilot_base_groups':['pilot']})

    def test_preflight_reads_no_keys_and_does_not_change_gptoss_config(self):
        before = main.CONFIG.read_bytes()
        with patch.object(main,'dotenv_values',side_effect=AssertionError('key read')), \
                patch.object(main.evaluate,'make_client',side_effect=AssertionError('client created')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(True,'qwen'),0)
        self.assertEqual(main.CONFIG.read_bytes(),before)
        self.assertFalse(self.output.exists())
        cfg = json.loads(self.config_path.read_text())
        self.assertEqual(generation_config(cfg),generation_config(self.pilot_cfg))
        self.assertEqual(cfg['request_budget'],3000)

    def test_all2400_wiring_isolated_outputs_and_qwen_prices(self):
        old = main.OUTPUT/'responses.csv'
        old.parent.mkdir(parents=True)
        old.write_text('preserved GPT-OSS responses\n')
        def fake(args):
            self.assertEqual(args.pilot_check,str(self.pilot_path))
            self.assertEqual(args.input_csv,str(main.INPUT))
            self.assertEqual(args.model_config,str(self.config_path))
            self.assertIsNone(args.limit)
            cfg = json.loads(self.config_path.read_text())
            self.assertEqual(cfg['max_tokens'],8192)
            self.assertEqual(cfg['reasoning_effort'],'low')
            comp = pd.read_csv(main.INPUT,keep_default_na=False)
            rows = [dict(uid=t['uid'],eval_model=cfg['name'],method=t['method'],target_ratio=t['target_ratio'],
                response_status='ok',correct=True,parse_status='ok',reused_generation=False,
                generation_usage=json.dumps({'prompt_tokens':10,'completion_tokens':5}))
                for t in main.evaluate.compression_tasks(comp)]
            atomic_csv(pd.DataFrame(rows),args.output_csv)
            atomic_json(str(args.output_csv)+'.usage.json',{'attempts':2400})
        with patch.dict(os.environ,{},clear=False),patch.object(main.evaluate,'run_eval',side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(model='qwen'),0)
        report = json.loads((self.output/'check.json').read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(report['original_questions'],160)
        self.assertAlmostEqual(report['returned_usage_cost_estimate_inr'],(24000*28.431+12000*222.709)/1e6)
        self.assertEqual(old.read_text(),'preserved GPT-OSS responses\n')

    def test_modified_pilot_answers_rejected_before_key_read(self):
        with (self.pilot_path.parent/'responses.csv').open('a') as stream:
            stream.write('\n')
        with patch.object(main,'dotenv_values',side_effect=AssertionError('key read')):
            with self.assertRaisesRegex(ValueError,'response provenance'):
                main.run(model='qwen')

    def test_qwen_cumulative_attempt_limit_stops_without_http(self):
        main.prepare_qwen_main()
        cfg = json.loads(self.config_path.read_text())
        atomic_json(self.output/'responses.csv.usage.json',
                    {'config_hash':digest(generation_config(cfg)),'attempts':3000})
        with patch.dict(os.environ,{},clear=False), \
                patch('final_eval.inference.client.requests.post',side_effect=AssertionError('HTTP after cap')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(model='qwen'),1)
        report = json.loads((self.output/'check.json').read_text())
        self.assertEqual(report['http_attempts'],3000)
        self.assertFalse(report['grid_complete'])


if __name__ == '__main__':
    unittest.main()
