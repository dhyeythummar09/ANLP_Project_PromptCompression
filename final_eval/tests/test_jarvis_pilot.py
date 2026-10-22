"""Offline integration checks for hosted pilot, caching and its safety boundary."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from final_eval import jarvis_pilot as pilot
from final_eval.inference.client import generation_config
from final_eval.utils.artifacts import file_hash, digest


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Synthetic fixture; no local data, keys or returned ZIPs required.
        rows = []
        self.answers = {}
        for i in range(16):
            prompt = f"What is {i} plus 1?"
            self.answers[pilot.evaluate.PROMPT_TEMPLATE.format(question=prompt)] = str(i + 1)
            for method in pilot.METHODS:
                for ratio in (0.5, 0.75):
                    rows.append(dict(uid=f"q{i}", example_id=f"q{i}", source_dataset="math",
                        base_group_id=f"g{i}", original_prompt=prompt,
                        ground_truth_answer=str(i + 1), method=method, target_ratio=ratio,
                        actual_ratio=ratio, compressed_prompt=f"{i}+1 {method} {ratio}",
                        status="ok", budget_compliant=True))
        target = self.root / pilot.INPUT
        target.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(target, index=False)
        audit = self.root / "outputs/imports/llmlingua_verified_import_20261021.json"
        audit.parent.mkdir(parents=True)
        audit.write_text(json.dumps({"compressed_sha256": file_hash(target)}))
        cfg = dict(backend="hosted", provider="openai-compatible", name="test-model",
            base_url="https://models.jarvislabs.net/v1", api_key_env="JARVISLABS_API_KEY",
            max_tokens=4096, temperature=0, sleep_seconds=2, timeout=180,
            max_attempts=2, request_budget=300)
        config_path = self.root / pilot.CONFIG
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(cfg))
        (config_path.parent / "access.deepseek.json.summary.json").write_text('{"passed":true}')
        (config_path.parent / "access.deepseek.json.manifest.json").write_text(json.dumps({"config":generation_config(cfg)}))
        (self.root / "final_eval").mkdir()
        (self.root / "final_eval/jarvis.env").write_text("JARVISLABS_API_KEY=fake-test-only\n")
        os.chdir(self.root)
        self.root_patch = patch.object(pilot, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_dry_run_never_reads_credentials_or_initializes_client(self):
        with patch.object(pilot, "dotenv_values", side_effect=AssertionError("key read")), \
                patch.object(pilot.evaluate, "make_client", side_effect=AssertionError("client created")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(True), 0)
        self.assertFalse((self.root / pilot.OUTPUT).exists())

    def test_complete_mocked_http_run_then_resume_without_more_requests(self):
        def reply(url, **kwargs):
            self.assertEqual(url, "https://models.jarvislabs.net/v1/chat/completions")
            prompt = kwargs["json"]["messages"][0]["content"]
            return Mock(status_code=200, json=lambda: dict(model="test-model",
                choices=[dict(message={"content":"Final Answer: " + self.answers.get(prompt, "1")}, finish_reason="stop")],
                usage={"prompt_tokens":10,"completion_tokens":5}))

        with patch.dict(os.environ, {}, clear=False), \
                patch("final_eval.inference.client.requests.post", side_effect=reply) as http, \
                patch("final_eval.inference.client.time.sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(), 0)
            attempts = http.call_count
            self.assertEqual(attempts, 240)
            self.assertEqual(pilot.run(), 0)
            self.assertEqual(http.call_count, attempts)
        report = json.loads((self.root / pilot.OUTPUT / "check.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["response_statuses"], {"ok":240})
        self.assertFalse((self.root / pilot.OUTPUT / "access_check.lock").exists())

    def test_modified_pilot_rejected_before_credentials_or_http(self):
        with (self.root / pilot.INPUT).open("a") as f:
            f.write("\n")
        with patch.object(pilot, "dotenv_values", side_effect=AssertionError("key read")):
            with self.assertRaisesRegex(ValueError, "differs"):
                pilot.run()

    def test_truncated_pilot_blocks_unchanged_paid_rerun(self):
        directory = self.root / pilot.OUTPUT
        directory.mkdir(parents=True)
        (directory / "check.json").write_text(json.dumps({"passed":False,"response_statuses":{"truncated":125}}))
        with patch.object(pilot, "dotenv_values", side_effect=AssertionError("key read")):
            with self.assertRaisesRegex(ValueError, "output cap"):
                pilot.run()

    def test_exhausted_attempt_ledger_stops_without_http_even_on_resume(self):
        directory = self.root / pilot.OUTPUT
        directory.mkdir(parents=True)
        cfg = json.loads((self.root / pilot.CONFIG).read_text())
        usage = directory / "responses.csv.usage.json"
        usage.write_text(json.dumps({"config_hash":digest(generation_config(cfg)), "attempts":300}))
        with patch.dict(os.environ, {}, clear=False), \
                patch("final_eval.inference.client.requests.post", side_effect=AssertionError("HTTP despite cap")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(), 1)
            self.assertEqual(pilot.run(), 1)
        self.assertEqual(json.loads(usage.read_text())["attempts"], 300)
        self.assertFalse(json.loads((directory / "check.json").read_text())["passed"])

    def prepare_gptoss(self):
        cfg = json.loads((self.root / pilot.CONFIG).read_text())
        cfg["name"] = "test-gptoss"
        directory = self.root / "outputs/final_eval/jarvis"
        (directory / "target.gptoss.pilot.json").write_text(json.dumps(cfg))
        (directory / "access.gptoss.json.summary.json").write_text('{"passed":true}')
        (directory / "access.gptoss.json.manifest.json").write_text(json.dumps({"config":generation_config(cfg)}))
        fallback = self.root / "outputs/final_eval/target_calibration/gptoss_fallback_v1"
        fallback.mkdir(parents=True)
        (fallback / "summary.json").write_text('{"completed":true,"eligible_for_full_pilot":true}')
        (fallback / "plan.json").write_text(json.dumps({"config":cfg}))
        return fallback

    def test_gptoss_full_pilot_and_resume_preserve_failed_deepseek(self):
        self.prepare_gptoss()
        failed = self.root / pilot.OUTPUT / "check.json"
        failed.parent.mkdir(parents=True)
        failed.write_text('{"passed":false,"response_statuses":{"truncated":125}}')
        original = failed.read_bytes()

        def reply(url, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"]
            self.assertEqual(kwargs["json"]["model"], "test-gptoss")
            return Mock(status_code=200, json=lambda:dict(model="test-gptoss", choices=[dict(
                message={"content":"Final Answer: " + self.answers.get(prompt,"1")}, finish_reason="stop")]))

        with patch.dict(os.environ,{},clear=False), contextlib.redirect_stdout(io.StringIO()), \
                patch("final_eval.inference.client.requests.post",side_effect=reply) as http, \
                patch("final_eval.inference.client.time.sleep"):
            self.assertEqual(pilot.run(model="gptoss"),0)
            self.assertEqual(pilot.run(model="gptoss"),0)
        self.assertEqual(http.call_count,240)
        self.assertEqual(failed.read_bytes(),original)
        report = json.loads((self.root / "outputs/final_eval/target_pilots/gptoss/check.json").read_text())
        self.assertTrue(report["passed"])

    def test_gptoss_needs_successful_same_settings_fallback_before_key_read(self):
        fallback = self.prepare_gptoss()
        (fallback / "summary.json").write_text('{"completed":true,"eligible_for_full_pilot":false}')
        with patch.object(pilot,"dotenv_values",side_effect=AssertionError("key read")):
            with self.assertRaisesRegex(ValueError,"eligible fallback"):
                pilot.run(model="gptoss")

    def prepare_qwen(self):
        cfg = json.loads((self.root / pilot.CONFIG).read_text())
        cfg.update(name="qwen-3.8-27b-fp8", reasoning_effort="low", request_budget=20)
        directory = self.root / "outputs/final_eval/jarvis"
        (directory / "target.qwen.json").write_text(json.dumps(cfg))
        (directory / "access.qwen.json.summary.json").write_text('{"passed":true}')
        (directory / "access.qwen.json.manifest.json").write_text(json.dumps({"config":generation_config(cfg)}))
        return directory

    def test_qwen_preflight_reads_no_keys_and_does_not_touch_prior_readers(self):
        directory = self.prepare_qwen()
        old_config = (self.root / pilot.CONFIG).read_bytes()
        with patch.object(pilot, "dotenv_values", side_effect=AssertionError("key read")), \
                patch.object(pilot.evaluate, "make_client", side_effect=AssertionError("client created")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(True, model="qwen"), 0)
        cfg = json.loads((directory / "target.qwen.pilot.json").read_text())
        self.assertEqual(cfg["request_budget"], 300)
        self.assertEqual(cfg["reasoning_effort"], "low")
        self.assertEqual((self.root / pilot.CONFIG).read_bytes(), old_config)
        self.assertFalse((self.root / "outputs/final_eval/target_pilots/qwen").exists())

    def test_qwen_full_pilot_resume_low_reasoning_usage_and_isolated_outputs(self):
        self.prepare_qwen()
        prior = self.root / "outputs/final_eval/target_pilots/gptoss/responses.csv"
        prior.parent.mkdir(parents=True)
        prior.write_text("preserve prior result\n")
        original = prior.read_bytes()
        def reply(url, **kwargs):
            payload = kwargs["json"]
            self.assertEqual(payload["model"], "qwen-3.8-27b-fp8")
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertEqual(payload["max_tokens"], 4096)
            prompt = payload["messages"][0]["content"]
            return Mock(status_code=200, json=lambda:dict(model="qwen-3.8-27b-fp8", choices=[dict(
                message={"content":"Final Answer: " + self.answers.get(prompt,"1")}, finish_reason="stop")],
                usage={"prompt_tokens":10,"completion_tokens":5}))
        with patch.dict(os.environ,{},clear=False), contextlib.redirect_stdout(io.StringIO()), \
                patch("final_eval.inference.client.requests.post",side_effect=reply) as http, \
                patch("final_eval.inference.client.time.sleep"):
            self.assertEqual(pilot.run(model="qwen"),0)
            self.assertEqual(pilot.run(model="qwen"),0)
        self.assertEqual(http.call_count,240)
        self.assertEqual(prior.read_bytes(), original)
        output = self.root / "outputs/final_eval/target_pilots/qwen"
        self.assertTrue(json.loads((output / "check.json").read_text())["passed"])
        usage = json.loads((output / "usage_summary.json").read_text())
        self.assertEqual(usage["http_attempts"],240)
        self.assertEqual(usage["returned_usage_records"],240)
        self.assertAlmostEqual(usage["returned_usage_cost_estimate_inr"],(2400*28.431+1200*222.709)/1e6)
        self.assertFalse(usage["main_started"])

    def test_qwen_changed_access_or_config_rejected_before_credentials(self):
        directory = self.prepare_qwen()
        path = directory / "target.qwen.json"
        cfg = json.loads(path.read_text())
        cfg["reasoning_effort"] = "high"
        path.write_text(json.dumps(cfg))
        with patch.object(pilot,"dotenv_values",side_effect=AssertionError("key read")):
            with self.assertRaisesRegex(ValueError,"identical settings"):
                pilot.run(model="qwen")
        self.assertFalse((directory / "target.qwen.pilot.json").exists())

    def test_qwen_usage_counts_cached_generation_once(self):
        directory = self.root / "usage_test"
        directory.mkdir()
        rows = [dict(query_hash="same", generation_timestamp_utc="same-time",
                     reused_generation=reused, generation_usage=json.dumps({"prompt_tokens":10,"completion_tokens":20}))
                for reused in (False, True, False)]
        pd.DataFrame(rows).to_csv(directory / "responses.csv",index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            usage = pilot.qwen_usage_summary(directory)
        self.assertEqual(usage["returned_usage_records"],1)
        self.assertEqual(usage["returned_completion_tokens"],20)


if __name__ == "__main__":
    unittest.main()
