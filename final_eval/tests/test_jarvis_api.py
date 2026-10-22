"""Offline checks for the bounded hosted-access helper; no real credentials."""
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from final_eval import jarvis_api
from final_eval.inference.client import ModelResponse


class JarvisAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / 'test.env'
        self.env.write_text('JARVISLABS_API_KEY=fake-test-only\nJARVISLABS_BASE_URL=https://models.jarvislabs.net/v1\nJARVISLABS_DEEPSEEK_MODEL=exact-test-model\n', encoding='utf-8')
        self.config = jarvis_api.configurations(self.env, ['deepseek'])[0]['deepseek']

    def test_reject_key_destination_and_missing_model(self):
        with self.assertRaisesRegex(ValueError, 'exact Token Factory API ID'):
            jarvis_api.configurations(self.env, ['gptoss'])
        self.env.write_text(self.env.read_text().replace('models.jarvislabs.net', 'example.org'))
        with self.assertRaisesRegex(ValueError, 'only to models.jarvislabs.net'):
            jarvis_api.configurations(self.env, ['deepseek'])

    def test_qwen_low_reasoning_wire_resume_and_confirmed_rates(self):
        with self.env.open('a') as stream:
            stream.write('JARVISLABS_QWEN_MODEL=qwen-3.8-27b-fp8\n')
        cfg = jarvis_api.configurations(self.env, ['qwen'])[0]['qwen']
        self.assertEqual(cfg['reasoning_effort'], 'low')
        self.assertNotIn('reasoning_effort', self.config)
        replies = [Mock(status_code=200, json=lambda answer=answer: {
            'model': 'qwen-3.8-27b-fp8',
            'choices': [{'message': {'content': 'Final Answer: ' + answer}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20},
        }) for answer in ('5', '7')]
        with patch.dict(os.environ, {'JARVISLABS_API_KEY': 'fake-test-only'}), \
                patch('final_eval.inference.client.requests.post', side_effect=replies) as http, \
                patch('final_eval.inference.client.time.sleep'):
            self.assertTrue(jarvis_api.access_check('qwen', cfg, self.root))
            self.assertTrue(jarvis_api.access_check('qwen', cfg, self.root))
        self.assertEqual(http.call_count, 2)
        for call in http.call_args_list:
            self.assertEqual(call.kwargs['json']['reasoning_effort'], 'low')
            self.assertEqual(call.kwargs['json']['max_tokens'], 4096)
        summary = json.loads((self.root/'access.qwen.json.summary.json').read_text())
        self.assertAlmostEqual(summary['response_usage_cost_estimate_inr'], (20*28.431+40*222.709)/1e6)
        manifest = json.loads((self.root/'access.qwen.json.manifest.json').read_text())
        self.assertEqual(manifest['config']['reasoning_effort'], 'low')
        self.assertNotIn('fake-test-only', ''.join(p.read_text() for p in self.root.glob('*.json')))

    def test_cached_resume_makes_no_calls_and_records_no_key(self):
        replies = [ModelResponse(text='Final Answer: 5', usage={'prompt_tokens': 10, 'completion_tokens': 20}),
                   ModelResponse(text='Final Answer: 7', usage={'prompt_tokens': 10, 'completion_tokens': 20})]
        with patch.object(jarvis_api.HostedClient, 'generate', side_effect=replies) as generate:
            self.assertTrue(jarvis_api.access_check('deepseek', self.config, self.root))
            self.assertTrue(jarvis_api.access_check('deepseek', self.config, self.root))
            self.assertEqual(generate.call_count, 2)
        self.assertNotIn('fake-test-only', ''.join(p.read_text() for p in self.root.glob('*.json')))
        summary = json.loads((self.root/'access.deepseek.json.summary.json').read_text())
        self.assertAlmostEqual(summary['response_usage_cost_estimate_inr'], (20*13+40*25)/1e6)

    def test_auth_or_truncation_stops_without_second_prompt(self):
        for status in ('auth_failed', 'truncated'):
            directory = self.root/status
            with patch.object(jarvis_api.HostedClient, 'generate', return_value=ModelResponse(status=status, text='Final Answer: 5')) as generate:
                self.assertFalse(jarvis_api.access_check('deepseek', self.config, directory))
                self.assertEqual(generate.call_count, 1)

    def test_http_ceiling_persists_and_auth_stops(self):
        with patch.dict(os.environ, {'JARVISLABS_API_KEY': 'fake-test-only'}):
            with patch('final_eval.inference.client.requests.post', return_value=Mock(status_code=401)) as post:
                self.assertFalse(jarvis_api.access_check('deepseek', self.config, self.root/'auth'))
                self.assertFalse(jarvis_api.access_check('deepseek', self.config, self.root/'auth'))
                self.assertEqual(post.call_count, 1)
            with patch('final_eval.inference.client.time.sleep'), patch('final_eval.inference.client.requests.post', return_value=Mock(status_code=429)) as post:
                for _ in range(3):
                    self.assertFalse(jarvis_api.access_check('deepseek', self.config, self.root/'limited', retry_failed=True))
                self.assertEqual(post.call_count, 4)
                usage = json.loads((self.root/'limited/access.deepseek.json.usage.json').read_text())
                self.assertEqual(usage['attempts'], 4)

    def test_changed_model_cannot_reuse_saved_responses(self):
        with patch.object(jarvis_api.HostedClient, 'generate', return_value=ModelResponse(status='auth_failed')):
            jarvis_api.access_check('deepseek', self.config, self.root)
        with self.assertRaisesRegex(ValueError, 'Configuration/input mismatch'):
            jarvis_api.access_check('deepseek', {**self.config, 'name': 'other-model'}, self.root)

    def test_lock_blocks_overlap_and_cleans_up(self):
        with jarvis_api.exclusive_check(self.root):
            with self.assertRaisesRegex(ValueError, 'another check may be running'):
                with jarvis_api.exclusive_check(self.root):
                    self.fail('Should reject overlap')
        self.assertFalse((self.root/'access_check.lock').exists())


if __name__ == '__main__':
    unittest.main()
