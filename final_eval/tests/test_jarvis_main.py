"""Offline main-stage wiring, cohort guard and cumulative request limit."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from final_eval import jarvis_main as main
from final_eval.utils.artifacts import atomic_json, atomic_csv, file_hash, digest
from final_eval.inference.client import generation_config


class MainApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        previous=Path.cwd(); self.addCleanup(os.chdir,previous); os.chdir(self.root)
        root_patch=patch.object(main,"ROOT",self.root); root_patch.start(); self.addCleanup(root_patch.stop)
        self.cfg=dict(backend="hosted",provider="openai-compatible",name="test-gptoss",
            base_url="https://models.jarvislabs.net/v1",api_key_env="JARVISLABS_API_KEY",
            max_tokens=4096,temperature=0,sleep_seconds=2,timeout=180,max_attempts=2,request_budget=3000)
        atomic_json(main.CONFIG,self.cfg)
        atomic_json(main.PILOT,{"passed":True,"target_config_hash":digest(generation_config(self.cfg)),"pilot_base_groups":["pilot"]})
        rows=[dict(uid=f"q{i}",example_id=f"q{i}",base_group_id=f"g{i}",source_dataset="math",
            original_prompt=f"What is {i}+1?",ground_truth_answer=str(i+1),method=m,target_ratio=r,
            actual_ratio=r,compressed_prompt=f"{i}+1 {m} {r}",status="ok",budget_compliant=True)
            for i in range(160) for m in main.METHODS for r in (.5,.75)]
        atomic_csv(pd.DataFrame(rows),main.INPUT)
        atomic_csv(pd.DataFrame({"uid":[f"q{i}" for i in range(160)],"base_group_id":[f"g{i}" for i in range(160)]}),main.BASE/"cohort.csv")
        self.imported=Path("outputs/imports/main_verified_import_20261021.json")
        atomic_json(self.imported,{"passed":True,"compressed_sha256":file_hash(main.INPUT)})
        env=self.root/"final_eval/jarvis.env"; env.parent.mkdir(); env.write_text("JARVISLABS_API_KEY=synthetic-only\n")

    def test_dry_run_never_reads_credentials_or_creates_client(self):
        with patch.object(main,"dotenv_values",side_effect=AssertionError("key read")), \
                patch.object(main.evaluate,"make_client",side_effect=AssertionError("client")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(True),0)
        self.assertFalse(main.OUTPUT.exists())

    def test_full_grid_wiring_and_summary_preserve_pilot(self):
        pilot_before=main.PILOT.read_bytes()
        def fake_evaluation(args):
            self.assertEqual(args.pilot_check,str(main.PILOT))
            self.assertEqual(args.input_csv,str(main.INPUT))
            self.assertIsNone(args.limit)
            self.assertTrue(args.retry_failed)
            comp=pd.read_csv(args.input_csv,keep_default_na=False)
            tasks=main.evaluate.compression_tasks(comp)
            rows=[dict(uid=t["uid"],eval_model="test-gptoss",method=t["method"],target_ratio=t["target_ratio"],
                       response_status="ok",response="Final Answer: 1",correct=True,parse_status="ok",
                       reused_generation=False,generation_usage=json.dumps({"prompt_tokens":10,"completion_tokens":5})) for t in tasks]
            atomic_csv(pd.DataFrame(rows),args.output_csv)
            atomic_json(str(args.output_csv)+".usage.json",{"attempts":2400})
        with patch.dict(os.environ,{},clear=False),patch.object(main.evaluate,"run_eval",side_effect=fake_evaluation), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(),0)
        report=json.loads((main.OUTPUT/"check.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["recorded_conditions"],2400)
        self.assertEqual(report["original_questions"],160)
        self.assertEqual(main.PILOT.read_bytes(),pilot_before)

    def test_existing_cumulative_budget_stops_before_any_http(self):
        atomic_json(main.OUTPUT/"responses.csv.usage.json",{"config_hash":digest(generation_config(self.cfg)),"attempts":3000})
        with patch.dict(os.environ,{},clear=False), \
                patch("final_eval.inference.client.requests.post",side_effect=AssertionError("HTTP after cap")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(),1)
        report=json.loads((main.OUTPUT/"check.json").read_text())
        self.assertEqual(report["http_attempts"],3000)
        self.assertFalse(report["grid_complete"])

    def test_changed_input_and_wrong_generation_settings_rejected_before_key(self):
        changed={**self.cfg,"max_tokens":8192}; atomic_json(main.CONFIG,changed)
        with patch.object(main,"dotenv_values",side_effect=AssertionError("key read")):
            with self.assertRaisesRegex(ValueError,"identical generation"):
                main.run()
        atomic_json(main.CONFIG,self.cfg)
        with main.INPUT.open("a") as stream: stream.write("\n")
        with self.assertRaisesRegex(ValueError,"verified return"):
            main.validate()

    def test_pilot_family_overlap_rejected_even_with_matching_import_hash(self):
        frame=pd.read_csv(main.INPUT,keep_default_na=False); frame.loc[0,"base_group_id"]="pilot"
        atomic_csv(frame,main.INPUT); atomic_json(self.imported,{"passed":True,"compressed_sha256":file_hash(main.INPUT)})
        with self.assertRaisesRegex(ValueError,"families overlap"):
            main.validate()

    def test_archived_failed_attempt_usage_counted_once(self):
        cfg,comp=main.validate()
        atomic_csv(pd.DataFrame([dict(uid="q0",eval_model="test-gptoss",method="none",target_ratio=1.0,
            response_status="ok",correct=True,parse_status="ok",reused_generation=False,
            query_hash="query",generation_timestamp_utc="success",
            generation_usage=json.dumps({"prompt_tokens":10,"completion_tokens":5}))]),main.OUTPUT/"responses.csv")
        atomic_json(main.OUTPUT/"archived_attempts.json",{"attempts":[
            {"query_hash":"query","timestamp_utc":"success","usage":{"prompt_tokens":10,"completion_tokens":5}},
            {"query_hash":"query","timestamp_utc":"truncated","usage":{"prompt_tokens":10,"completion_tokens":4096}}]})
        with contextlib.redirect_stdout(io.StringIO()):
            report=main.summarize(cfg,comp)
        self.assertEqual(report["returned_usage_records"],2)
        self.assertEqual(report["archived_usage_records"],1)
        self.assertEqual(report["returned_completion_tokens"],4101)
