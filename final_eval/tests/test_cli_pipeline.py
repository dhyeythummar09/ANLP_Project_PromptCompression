"""A complete small pipeline with mocked generation and a fixture tokenizer."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

from final_eval.preparation.schema import prepare_problems
from final_eval.compressor.budget import TokenCounter
from final_eval.compressor import evaluate, metrics, compress
from final_eval.inference.client import ModelResponse

class FakeTarget:
    def __init__(self):
        self.calls = 0

    def generate(self, query):
        self.calls += 1
        answer = "0" if "missing facts" in query else "5"
        return ModelResponse(text="fixture reasoning\n" + "x" * 1600 + "\nFinal Answer: " + answer,
                             requested_model="fixture", served_model="fixture", finish_reason="stop", attempts=1)

class PipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.problems = prepare_problems(pd.DataFrame([
            {"example_id": str(i), "source_dataset": "gsm8k",
             "original_prompt": f"Person {i} has 5 apples. How many apples?", "ground_truth_answer": "5"}
            for i in range(2)]))
        self.problem_path = self.root / "problems.csv"
        self.problems.to_csv(self.problem_path, index=False)

    def test_fixture_compression_resume_and_metrics(self):
        out = self.root / "compressed.csv"
        argv = ["compress", "--input-csv", str(self.problem_path), "--output-csv", str(out),
                "--methods", "random", "--ratios", "0.5", "--no-cuda"]
        with patch("sys.argv", argv), patch.object(compress, "TokenCounter", return_value=TokenCounter("regex-test-only")), redirect_stdout(StringIO()):
            compress.main()
            compress.main()
        result = pd.read_csv(out)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.budget_compliant.all())
        self.assertTrue((result.comp_tokens <= result.budget_tokens).all())
        target = self.root / "intrinsic"
        with patch("sys.argv", ["metrics", "--input", str(out), "--output-dir", str(target)]), redirect_stdout(StringIO()):
            metrics.main()
        self.assertTrue((target / "summary.csv").exists())

    def test_evaluation_resume_common_cohort_and_full_response(self):
        rows = []
        for r in self.problems.to_dict("records"):
            for method, compressed in [("grouped_hybrid", "5 apples"), ("random", "missing facts")]:
                rows.append({**r, "method": method, "target_ratio": 0.5, "actual_ratio": 0.5,
                             "compressed_prompt": compressed, "status": "ok", "budget_compliant": True})
        compression_path = self.root / "compression.csv"
        pd.DataFrame(rows).to_csv(compression_path, index=False)
        cfg = self.root / "target.json"
        cfg.write_text(json.dumps({"name": "fixture", "backend": "local"}), encoding="utf-8")
        args = SimpleNamespace(model_config=str(cfg), input_csv=str(compression_path),
            output_csv=str(self.root / "responses.csv"), methods=None, ratios=None, limit=None,
            dry_run=False, retry_failed=False, report_dir=str(self.root / "reports"),
            focus="grouped_hybrid", bootstrap=100)
        fake = FakeTarget()
        with patch.object(evaluate, "make_client", return_value=fake), redirect_stdout(StringIO()):
            evaluate.run_eval(args)
            calls = fake.calls
            evaluate.run_eval(args)
            self.assertEqual(fake.calls, calls)
            evaluate.summarize(args)
        saved = pd.read_csv(args.output_csv)
        self.assertEqual(len(saved), 6)
        self.assertTrue(saved.response.str.len().gt(1500).all())
        paired = pd.read_csv(Path(args.report_dir) / "paired.csv")
        self.assertEqual(paired.n_paired.iloc[0], 2)
        self.assertEqual(paired.difference.iloc[0], 1.0)
        coverage = pd.read_csv(Path(args.report_dir) / "coverage.csv")
        self.assertTrue(coverage.n_common.eq(2).all())
        # Native over-budget outputs remain visible but cannot support a paired win.
        over = pd.read_csv(compression_path)
        over.loc[over.method == "random", "budget_compliant"] = False
        over.to_csv(compression_path, index=False)
        args.report_dir = str(self.root / "over_budget_reports")
        with redirect_stdout(StringIO()):
            evaluate.summarize(args)
        invalid_pair = pd.read_csv(Path(args.report_dir) / "paired.csv")
        self.assertEqual(invalid_pair.n_paired.iloc[0], 0)
        self.assertEqual(invalid_pair.n_excluded_budget_violation.iloc[0], 2)
        self.assertFalse(invalid_pair.ci_excludes_zero.iloc[0])

    def test_dry_run_never_builds_target(self):
        row = {**self.problems.iloc[0].to_dict(), "method": "random", "target_ratio": 0.5,
               "actual_ratio": 0.5, "compressed_prompt": "5", "status": "ok"}
        path = self.root / "compression.csv"; pd.DataFrame([row]).to_csv(path, index=False)
        cfg = self.root / "cfg.json"; cfg.write_text(json.dumps({"name": "SET_UNCONFIRMED", "backend": "hosted"}))
        args = SimpleNamespace(model_config=str(cfg), input_csv=str(path), output_csv=str(self.root / "unused.csv"),
                               methods=None, ratios=None, limit=None, dry_run=True)
        with patch.object(evaluate, "make_client", side_effect=AssertionError("must not initialize")), redirect_stdout(StringIO()):
            evaluate.run_eval(args)
        self.assertFalse(Path(args.output_csv).exists())

if __name__ == "__main__":
    unittest.main()
