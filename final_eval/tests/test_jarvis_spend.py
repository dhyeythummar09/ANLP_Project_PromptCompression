"""Cost allowance, uncertain retry charges and interruption persistence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from final_eval.inference.client import ModelResponse
from final_eval.jarvis_spend import TokenCostGuard


class CostGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'cost.json'
        self.cfg = {'name':'qwen','max_tokens':8192,'max_attempts':2}
        self.client = Mock(attempts=0)

    def guard(self,limit=5):
        return TokenCostGuard(self.client,self.cfg,self.path,limit,28.431,222.709)

    def test_known_usage_releases_unused_reservation_and_resumes(self):
        def reply(prompt):
            self.client.attempts += 1
            return ModelResponse(usage={'prompt_tokens':100,'completion_tokens':1000})
        self.client.generate.side_effect = reply
        self.guard().generate('hello')
        saved=json.loads(self.path.read_text())
        self.assertAlmostEqual(saved['accounted_inr'],(100*28.431+1000*222.709)/1e6)
        self.assertEqual(saved['unsettled_reservations'],[])
        self.guard().generate('hello again')
        self.assertEqual(self.client.generate.call_count,2)

    def test_too_small_allowance_stops_before_http(self):
        response = self.guard(limit=1).generate('hello')
        self.assertEqual(response.status,'request_budget_exhausted')
        self.client.generate.assert_not_called()

    def test_interruption_reservation_survives_restart(self):
        self.client.generate.side_effect=RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            self.guard().generate('hello')
        self.assertGreater(json.loads(self.path.read_text())['accounted_inr'],3)
        response=self.guard().generate('hello')
        self.assertEqual(response.status,'request_budget_exhausted')
        self.assertEqual(self.client.generate.call_count,1)

    def test_unknown_retry_retains_worst_case_attempt_charge(self):
        def reply(prompt):
            self.client.attempts += 2
            return ModelResponse(usage={'prompt_tokens':100,'completion_tokens':1000})
        self.client.generate.side_effect=reply
        self.guard().generate('hello')
        saved=json.loads(self.path.read_text())
        self.assertEqual(saved['unknown_attempts'],1)
        self.assertGreater(saved['accounted_inr'],2)
        self.assertEqual(saved['unsettled_reservations'],[])


if __name__ == '__main__':
    unittest.main()
