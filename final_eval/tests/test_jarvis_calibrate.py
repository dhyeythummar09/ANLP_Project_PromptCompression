"""Offline HTTP integration tests. No keys, model weights or network access."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from final_eval import jarvis_calibrate as calibration
from final_eval.tests import test_jarvis_pilot as pilot_tests
from final_eval.inference.client import HostedClient, generation_config, public_config
from final_eval.utils.artifacts import file_hash, digest


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = pilot_tests.PilotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.root_patch = patch.object(calibration, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        comp = pd.read_csv(calibration.jarvis_pilot.INPUT, keep_default_na=False)
        prior = comp[["uid", "method", "target_ratio"]].copy()
        prior["response_status"] = "truncated"
        path = calibration.jarvis_pilot.OUTPUT / "responses.csv"
        path.parent.mkdir(parents=True)
        prior.to_csv(path, index=False)
        (path.parent / "check.json").write_text(json.dumps({"passed":False,
            "response_statuses":{"truncated":224}, "responses_sha256":file_hash(path)}))

    def output(self, data):
        return Mock(status_code=200, json=lambda:data)

    def reply(self, url, **kwargs):
        cfg = kwargs["json"]
        prompt = cfg["messages"][0]["content"]
        off = cfg.get("thinking", {}).get("type") == "disabled"
        return self.output(dict(model="test-model", choices=[dict(message={
            "content":"Final Answer: " + self.fixture.answers.get(prompt, "999"),
            "reasoning_content":"" if off else "short reasoning"}, finish_reason="stop")],
            usage={"prompt_tokens":100, "completion_tokens":20,
                   "completion_tokens_details":{"reasoning_tokens":0 if off else 10}}))

    def run_with_http(self, callback):
        with patch.dict(os.environ, {}, clear=False), \
                patch("final_eval.inference.client.requests.post", side_effect=callback) as post, \
                patch("final_eval.inference.client.time.sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            result = calibration.run()
        summary = json.loads((calibration.OUTPUT / "summary.json").read_text())
        return result, post.call_count, summary

    def test_dry_run_no_credentials_or_network(self):
        with patch.object(calibration, "dotenv_values", side_effect=AssertionError("credentials read")), \
                patch("final_eval.inference.client.requests.post", side_effect=AssertionError("HTTP")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(calibration.run(True), 0)
        self.assertFalse(calibration.OUTPUT.exists())

    def test_payload_controls_full_run_and_resume_preserve_answers(self):
        result, count, summary = self.run_with_http(self.reply)
        self.assertEqual((result, count), (0,12))
        self.assertEqual(summary["provisional_recommendation"], "low_8192")
        self.assertTrue(summary["variants"]["no_thinking_4096"]["thinking_disabled_observed"])
        before = (calibration.OUTPUT / "low_8192.json").read_bytes()
        result, count, summary = self.run_with_http(lambda *a, **k:self.fail("repeat HTTP"))
        self.assertEqual((result,count), (0,0))
        self.assertEqual(summary["total_http_attempts"],12)
        self.assertEqual(before,(calibration.OUTPUT / "low_8192.json").read_bytes())

    def test_ignored_disabled_control_not_recommended(self):
        def ignored(url, **kw):
            result = self.reply(url, **kw)
            data = result.json()
            if kw["json"]["thinking"]["type"] == "disabled":
                data["usage"]["completion_tokens_details"]["reasoning_tokens"] = 15
            else:
                data["choices"][0]["finish_reason"] = "length"
            return self.output(data)
        result,count,summary = self.run_with_http(ignored)
        self.assertEqual((result,count),(1,12))
        self.assertIsNone(summary["provisional_recommendation"])

    def test_unknown_parameter_rejection_stops_setting_after_one_attempt(self):
        result,count,summary = self.run_with_http(lambda *a,**kw:Mock(status_code=400))
        self.assertEqual((result,count),(1,2))
        self.assertEqual(summary["total_http_attempts"],2)
        result,count,summary = self.run_with_http(lambda *a,**kw:self.fail("retry rejected controls"))
        self.assertEqual(count,0)

    def test_null_reasoning_metadata_does_not_claim_disable_was_honored(self):
        def absent(url, **kw):
            data = self.reply(url, **kw).json()
            data["usage"]["completion_tokens_details"] = None
            return self.output(data)
        result,count,summary = self.run_with_http(absent)
        self.assertEqual((result,count),(0,12))
        self.assertFalse(summary["variants"]["no_thinking_4096"]["thinking_disabled_observed"])

    def test_cumulative_total_cap_including_retries_and_resumptions(self):
        result,count,summary = self.run_with_http(lambda *a,**kw:Mock(status_code=429))
        self.assertEqual((result,count),(1,4))
        self.assertLessEqual(summary["total_http_attempts"],16)
        # Simulate a previous run already consuming each entire per-setting cap.
        plan = calibration.prepare()
        for name,cfg in plan["configs"].items():
            path = calibration.OUTPUT / f"{name}.json"
            path.write_text('{"responses":{}}')
            Path(str(path)+".usage.json").write_text(json.dumps({
                "config_hash":digest(generation_config(cfg)),"attempts":8}))
        result,count,summary = self.run_with_http(lambda *a,**kw:self.fail("HTTP after limit"))
        self.assertEqual(count,0)
        self.assertEqual(summary["total_http_attempts"],16)

    def test_default_wire_body_unchanged_and_reasoning_changes_identity(self):
        cfg = dict(name="model",base_url="https://example.invalid/v1",sleep_seconds=0)
        with patch("final_eval.inference.client.requests.post", side_effect=self.reply) as post:
            HostedClient(cfg).generate("question")
            self.assertEqual(set(post.call_args.kwargs["json"]),{"model","messages","temperature","max_tokens"})
            changed = {**cfg,"thinking":{"type":"enabled"},"reasoning_effort":"low"}
            HostedClient(changed).generate("question")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["thinking"],{"type":"enabled"})
            self.assertEqual(payload["reasoning_effort"],"low")
        self.assertNotEqual(generation_config(cfg),generation_config(changed))
        for invalid in ({"thinking":{"type":"disabled","api_key":"bad"}}, {"reasoning_effort":["low"]}):
            with self.assertRaises(ValueError):
                public_config({**cfg,**invalid})


if __name__ == "__main__":
    unittest.main()
