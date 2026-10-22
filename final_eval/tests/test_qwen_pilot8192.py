"""Uniform8192 pilot guard, isolation and resume with synthetic/mock data."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

from final_eval import jarvis_pilot as pilot
from final_eval import jarvis_qwen_cap as cap
from final_eval.compressor.evaluate import PROMPT_TEMPLATE, compression_tasks
from final_eval.tests import test_jarvis_qwen_cap as cap_tests


class Qwen8192PilotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = cap_tests.QwenCapTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        result,count,summary = self.fixture.execute(self.fixture.reply)
        self.assertEqual((result,count),(0,3))

    def test_preflight_checks_calibration_reads_no_keys_and_preserves_old_pilot(self):
        original = (cap.PRIOR / 'responses.csv').read_bytes()
        with patch.object(pilot,'dotenv_values',side_effect=AssertionError('key read')), \
                patch.object(pilot.evaluate,'make_client',side_effect=AssertionError('client created')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(True,'qwen',8192),0)
        cfg = json.loads((cap.CONFIG.parent / 'target.qwen.pilot8192.json').read_text())
        self.assertEqual(cfg['max_tokens'],8192)
        self.assertEqual(cfg['request_budget'],300)
        self.assertEqual((cap.PRIOR / 'responses.csv').read_bytes(),original)
        self.assertFalse((cap.PRIOR.parent / 'qwen_8192').exists())

    def test_full_uniform_pilot_and_resume_use_only_new_settings_and_outputs(self):
        original = (cap.PRIOR / 'responses.csv').read_bytes()
        def reply(url,**kwargs):
            cfg = kwargs['json']
            self.assertEqual(cfg['max_tokens'],8192)
            self.assertEqual(cfg['reasoning_effort'],'low')
            query = cfg['messages'][0]['content']
            answer = self.fixture.fixture.answers.get(query,'999')
            return Mock(status_code=200,json=lambda:{'model':'qwen-3.8-27b-fp8',
                'choices':[{'message':{'content':'Final Answer: '+answer},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10}})
        with patch.dict(os.environ,{},clear=False), contextlib.redirect_stdout(io.StringIO()), \
                patch('final_eval.inference.client.requests.post',side_effect=reply) as http, \
                patch('final_eval.inference.client.time.sleep'):
            self.assertEqual(pilot.run(model='qwen',qwen_output_cap=8192),0)
            attempts = http.call_count
            self.assertEqual(attempts,238)
            self.assertEqual(pilot.run(model='qwen',qwen_output_cap=8192),0)
            self.assertEqual(http.call_count,attempts)
        output = cap.PRIOR.parent / 'qwen_8192'
        self.assertTrue(json.loads((output / 'check.json').read_text())['passed'])
        usage = json.loads((output / 'usage_summary.json').read_text())
        self.assertEqual(usage['returned_usage_records'],238)
        self.assertFalse(usage['main_started'])
        self.assertEqual((cap.PRIOR / 'responses.csv').read_bytes(),original)

    def test_ineligible_diagnostic_blocks_before_credentials(self):
        path = cap.DIRECTORY / 'summary.json'
        summary = json.loads(path.read_text())
        summary['eligible_for_uniform_8192_pilot'] = False
        path.write_text(json.dumps(summary))
        with patch.object(pilot,'dotenv_values',side_effect=AssertionError('key read')):
            with self.assertRaisesRegex(ValueError,'eligible Qwen8192'):
                pilot.run(model='qwen',qwen_output_cap=8192)

    def test_false_summary_cannot_override_truncated_saved_diagnostic(self):
        path = cap.DIRECTORY / 'responses.json'
        saved = json.loads(path.read_text())
        first = next(iter(saved['responses'].values()))
        first['status'] = 'truncated'
        path.write_text(json.dumps(saved))
        with patch.object(pilot,'dotenv_values',side_effect=AssertionError('key read')):
            with self.assertRaisesRegex(ValueError,'completion/control gate'):
                pilot.run(model='qwen',qwen_output_cap=8192)


if __name__ == '__main__':
    unittest.main()
