"""Bounded fallback check integration; synthetic data and mocked HTTP only."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from final_eval import jarvis_reader_check as reader
from final_eval.tests import test_jarvis_calibrate as calibration_tests
from final_eval.utils.artifacts import atomic_json, digest
from final_eval.inference.client import generation_config


class ReaderCheckTests(unittest.TestCase):
    def setUp(self):
        self.fixture = calibration_tests.CalibrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        patcher = patch.object(reader,"ROOT",self.root)
        patcher.start(); self.addCleanup(patcher.stop)
        plan = calibration_tests.calibration.prepare()
        atomic_json(reader.DEEPSEEK / "plan.json",plan)
        atomic_json(reader.DEEPSEEK / "summary.json",{"completed":True,"provisional_recommendation":None})
        cfg = dict(plan["configs"]["low_8192"])
        cfg.pop("thinking"); cfg.pop("reasoning_effort")
        cfg.update(name="test-gptoss",max_tokens=4096)
        atomic_json("outputs/final_eval/jarvis/target.gptoss.json",cfg)
        atomic_json("outputs/final_eval/jarvis/access.gptoss.json.summary.json",{"passed":True})
        atomic_json("outputs/final_eval/jarvis/access.gptoss.json.manifest.json",{"config":generation_config(cfg)})

    def reply(self,url,**kw):
        question=kw["json"]["messages"][0]["content"]
        text="Final Answer: "+self.fixture.fixture.answers.get(question,"999")
        return Mock(status_code=200,json=lambda:{"model":"test-gptoss","choices":[{
            "message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10}})

    def execute(self,callback):
        with patch.dict(os.environ,{},clear=False), contextlib.redirect_stdout(io.StringIO()), \
                patch("final_eval.inference.client.requests.post",side_effect=callback) as http, \
                patch("final_eval.inference.client.time.sleep"):
            result=reader.run()
        return result,http.call_count,json.loads((reader.DIRECTORY/"summary.json").read_text())

    def test_complete_then_resume_without_repeating_http(self):
        result,count,summary=self.execute(self.reply)
        self.assertEqual((result,count),(0,6))
        self.assertTrue(summary["eligible_for_full_pilot"])
        self.assertFalse(summary["full_pilot_passed"])
        result,count,summary=self.execute(lambda *a,**k:self.fail("repeat request"))
        self.assertEqual((result,count),(0,0))
        self.assertEqual(summary["http_attempts"],6)

    def test_existing_attempt_limit_blocks_requests(self):
        cfg=reader.prepare()["config"]
        atomic_json(reader.DIRECTORY/"responses.json.usage.json",{"config_hash":digest(generation_config(cfg)),"attempts":8})
        result,count,summary=self.execute(lambda *a,**k:self.fail("request over cap"))
        self.assertEqual((result,count),(1,0))
        self.assertEqual(summary["http_attempts"],8)

    def test_auth_failure_stops_and_is_not_retried_on_resume(self):
        result,count,summary=self.execute(lambda *a,**k:Mock(status_code=401))
        self.assertEqual((result,count),(1,1))
        result,count,summary=self.execute(lambda *a,**k:self.fail("repeat auth failure"))
        self.assertEqual(count,0)

    def test_dry_run_reads_no_key_or_network(self):
        with patch.object(reader,"dotenv_values",side_effect=AssertionError("key read")), \
                patch("final_eval.inference.client.requests.post",side_effect=AssertionError("HTTP")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reader.run(True),0)
        self.assertFalse(reader.DIRECTORY.exists())
