"""Offline Qwen cap check with synthetic pilot files and mocked HTTP."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from final_eval import jarvis_qwen_cap as cap
from final_eval.compressor.evaluate import PROMPT_TEMPLATE, compression_tasks
from final_eval.inference.client import generation_config
from final_eval.tests import test_jarvis_pilot as pilot_tests
from final_eval.utils.artifacts import atomic_json, digest, file_hash


class QwenCapTests(unittest.TestCase):
    def setUp(self):
        self.fixture = pilot_tests.PilotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.prepare_qwen()
        patcher = patch.object(cap, 'ROOT', self.fixture.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        comp = pd.read_csv(cap.jarvis_pilot.INPUT, keep_default_na=False)
        comp.loc[comp.uid.eq('q1'), 'source_dataset'] = 'other_math'
        mask = comp.uid.eq('q0') & comp.method.isin(['ours', 'grouped_hybrid', 'grouped_signal']) & comp.target_ratio.eq(.5)
        comp.loc[mask, 'compressed_prompt'] = 'failed compressed query'
        comp.to_csv(cap.jarvis_pilot.INPUT, index=False)
        atomic_json('outputs/imports/llmlingua_verified_import_20261021.json',
                    {'compressed_sha256':file_hash(cap.jarvis_pilot.INPUT)})
        cap.jarvis_pilot.prepare_qwen_config()
        cfg = json.loads(cap.CONFIG.read_text())
        rows = []
        for task in compression_tasks(comp):
            failed = task['compressed_prompt'] == 'failed compressed query'
            rows.append({k:task[k] for k in ('uid', 'method', 'target_ratio')})
            rows[-1].update(response_status='truncated' if failed else 'ok', correct='' if failed else True,
                            query_hash=digest([generation_config(cfg),PROMPT_TEMPLATE.format(question=task['compressed_prompt'])]))
        cap.PRIOR.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(cap.PRIOR / 'responses.csv', index=False)
        atomic_json(cap.PRIOR / 'check.json', {'passed':False,
            'responses_sha256':file_hash(cap.PRIOR / 'responses.csv'),
            'target_config_hash':digest(generation_config(cfg))})

    def reply(self, url, **kwargs):
        payload = kwargs['json']
        self.assertEqual(payload['max_tokens'],8192)
        self.assertEqual(payload['reasoning_effort'],'low')
        prompt = payload['messages'][0]['content']
        return Mock(status_code=200,json=lambda:{'model':'qwen-3.8-27b-fp8','choices':[{
            'message':{'content':'Final Answer: ' + self.fixture.answers.get(prompt,'999')},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':20,'completion_tokens':10}})

    def execute(self, callback):
        with patch.dict(os.environ,{},clear=False), contextlib.redirect_stdout(io.StringIO()), \
                patch('final_eval.inference.client.requests.post',side_effect=callback) as http, \
                patch('final_eval.inference.client.time.sleep'):
            result = cap.run()
        return result,http.call_count,json.loads((cap.DIRECTORY / 'summary.json').read_text())

    def test_dry_run_never_reads_keys_or_initializes_client(self):
        with patch.object(cap,'dotenv_values',side_effect=AssertionError('key read')), \
                patch.object(cap,'HostedClient',side_effect=AssertionError('client created')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cap.run(True),0)
        self.assertFalse(cap.DIRECTORY.exists())

    def test_three_cases_resume_preserve_prior_and_allow_incorrect_compressed(self):
        before = (cap.PRIOR / 'responses.csv').read_bytes()
        result,count,summary = self.execute(self.reply)
        self.assertEqual((result,count),(0,3))
        self.assertTrue(summary['eligible_for_uniform_8192_pilot'])
        self.assertFalse(summary['full_pilot_passed'])
        self.assertFalse(summary['main_started'])
        self.assertAlmostEqual(summary['returned_usage_cost_estimate_inr'],(60*28.431+30*222.709)/1e6)
        result,count,summary = self.execute(lambda *a,**k:self.fail('repeat HTTP'))
        self.assertEqual((result,count),(0,0))
        self.assertEqual((cap.PRIOR / 'responses.csv').read_bytes(),before)

    def test_truncation_is_retained_and_does_not_trigger_repeat_requests(self):
        def reply(url,**kwargs):
            response = self.reply(url,**kwargs)
            data = response.json()
            if 'failed compressed query' in kwargs['json']['messages'][0]['content']:
                data['choices'][0]['message']['content'] = ''
                data['choices'][0]['finish_reason'] = 'length'
            return Mock(status_code=200,json=lambda:data)
        result,count,summary = self.execute(reply)
        self.assertEqual((result,count),(1,3))
        self.assertFalse(summary['eligible_for_uniform_8192_pilot'])
        self.assertEqual(summary['response_statuses']['truncated'],1)
        result,count,summary = self.execute(lambda *a,**k:self.fail('retry unchanged truncation'))
        self.assertEqual(count,0)

    def test_attempt_ceiling_persists_across_restarts(self):
        cfg = cap.prepare()['config']
        atomic_json(cap.DIRECTORY / 'responses.json.usage.json',
                    {'config_hash':digest(generation_config(cfg)), 'attempts':4})
        result,count,summary = self.execute(lambda *a,**k:self.fail('HTTP after ceiling'))
        self.assertEqual((result,count),(1,0))
        self.assertEqual(summary['http_attempts'],4)
        self.assertFalse(summary['eligible_for_uniform_8192_pilot'])

    def test_changed_prior_rejected_before_credentials(self):
        with (cap.PRIOR / 'responses.csv').open('a') as stream:
            stream.write('\n')
        with patch.object(cap,'dotenv_values',side_effect=AssertionError('key read')):
            with self.assertRaisesRegex(ValueError,'unchanged failed'):
                cap.run()


if __name__ == '__main__':
    unittest.main()
