"""Regress real upstream window/cache bugs with synthetic losses, no weights."""
import unittest
from unittest.mock import Mock

import torch
from final_eval.compressor.compress import BaselineRunner
from final_eval.tests.fixtures.llmlingua_022_window import NativeFixture


class FakeModernCache:
    def __iter__(self):
        yield (None, None, 'new cache metadata')


class WindowRegressionTests(unittest.TestCase):
    def fixture(self, length):
        native = NativeFixture()
        native.device = 'cpu'
        native.max_position_embeddings = 2048
        native.get_token_length = lambda text, *args: len(text.split())
        def tokenize(text, **kwargs):
            ids = torch.tensor([[int(t) for t in text.split()]])
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
        native.tokenizer = tokenize
        self.calls = 0
        def ppl(text, granularity, input_ids, attention_mask, **kwargs):
            self.calls += 1
            end = kwargs.get('end') or input_ids.shape[1]
            return input_ids[0, 1:end].float(), FakeModernCache()
        native.get_ppl = ppl
        return native, ' '.join(map(str, range(1, length + 1)))

    def test_reproduces_short_input_budget_floor(self):
        native, text = self.fixture(31)
        ids, _ = native.iterative_compress_prompt([text], 1, iterative_size=50, dynamic_ratio=[0.])
        self.assertEqual(ids.shape[1], 29)

    def test_reproduces_modern_cache_unpacking_error(self):
        native, text = self.fixture(123)
        with self.assertRaisesRegex(ValueError, 'too many values to unpack'):
            native.iterative_compress_prompt([text], 90, iterative_size=50, dynamic_ratio=[0.])

    def test_whole_question_window_handles_both_without_cache_reuse(self):
        for length in (20, 31, 49, 50, 70, 123, 500):
            for target in (1, int(length*.5), int(length*.75)):
                with self.subTest(length=length, target=target):
                    native, text = self.fixture(length)
                    ids, _ = native.iterative_compress_prompt([text], target, iterative_size=length, dynamic_ratio=[0.])
                    self.assertLessEqual(ids.shape[1], target)
                    self.assertGreater(ids.shape[1], 0)
                    self.assertEqual(self.calls, 1)

    def test_adapter_passes_native_whole_question_size_and_preserves_text(self):
        compressor = Mock(max_position_embeddings=2048)
        compressor.get_token_length.return_value = 123
        compressor.compress_prompt.return_value = {'compressed_prompt': 'native unchanged output'}
        runner = BaselineRunner('cpu', 'fixture', 'unused')
        runner.objects['llmlingua'] = compressor
        self.assertEqual(runner.run('llmlingua', 'source', .5, 20), 'native unchanged output')
        compressor.compress_prompt.assert_called_once_with('source', rate=.5, target_token=20, iterative_size=123)

    def test_context_overflow_fails_without_truncation(self):
        compressor = Mock(max_position_embeddings=2048)
        compressor.get_token_length.return_value = 2049
        runner = BaselineRunner('cpu', 'fixture', 'unused')
        runner.objects['llmlingua'] = compressor
        with self.assertRaisesRegex(ValueError, 'exceeds the model context'):
            runner.run('llmlingua', 'source', .5, 20)
        compressor.compress_prompt.assert_not_called()

    def test_llmlingua2_settings_are_unchanged(self):
        compressor = Mock()
        compressor.compress_prompt.return_value = {'compressed_prompt': 'official output'}
        runner = BaselineRunner('cpu', 'unused', 'fixture')
        runner.objects['llmlingua2'] = compressor
        self.assertEqual(runner.run('llmlingua2', 'source', .5, 20), 'official output')
        compressor.compress_prompt.assert_called_once_with('source', rate=.5, target_token=20)


if __name__ == '__main__':
    unittest.main()
