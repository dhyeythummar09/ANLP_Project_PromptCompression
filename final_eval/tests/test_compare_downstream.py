import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from final_eval.compare_downstream import build_paired_input, summarize
from final_eval.followup_reader import check_reader_output as _check_reader_output
from final_eval.utils.artifacts import atomic_csv


def compression_rows(texts):
    """One compression table: q1/q2 in different families, method 'ours' at .5."""
    return pd.DataFrame([
        {"uid": uid, "method": "ours", "target_ratio": 0.5, "status": "ok",
         "budget_compliant": True, "compressed_prompt": text,
         "original_prompt": f"full {uid}", "ground_truth_answer": "1",
         "base_group_id": f"family_{uid}", "source_dataset": "gsm8k"}
        for uid, text in zip(("q1", "q2"), texts)
    ])


class CompareDownstreamTests(unittest.TestCase):
    def test_paired_input_renames_methods_and_rejects_mismatched_questions(self):
        historical = compression_rows(["a", "b"])
        v2 = compression_rows(["a", "c"])
        paired = build_paired_input(historical, v2)
        self.assertEqual(set(paired.method), {"ours_historical", "ours_v2"})
        self.assertEqual(set(paired.base_method), {"ours"})
        self.assertEqual(len(paired), 4)

        broken = v2.copy()
        broken.loc[0, "ground_truth_answer"] = "2"
        with self.assertRaises(ValueError):
            build_paired_input(historical, broken)

    def test_gptoss_output_folder_is_protected(self):
        with self.assertRaises(ValueError):
            _check_reader_output("outputs/final_eval/main/target_gptoss/responses.csv")
        _check_reader_output("outputs/followup/old_v2/target_qwen/responses.csv")

    def test_summarize_pairs_one_run_and_counts_identical_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # q1 compressed identically by both checkpoints, q2 differently.
            paired = build_paired_input(compression_rows(["a", "b"]),
                                        compression_rows(["a", "c"]))
            atomic_csv(paired, root / "paired.csv")
            outcomes = {("q1", "none"): True, ("q2", "none"): True,
                        ("q1", "ours_historical"): True, ("q1", "ours_v2"): True,
                        ("q2", "ours_historical"): False, ("q2", "ours_v2"): True}
            responses = pd.DataFrame([
                {"uid": uid, "eval_model": "reader", "method": method,
                 "target_ratio": 1.0 if method == "none" else 0.5,
                 "base_group_id": f"family_{uid}", "response_status": "ok",
                 "correct": correct, "generation_served_model": "reader",
                 "reused_generation": method == "ours_v2" and uid == "q1"}
                for (uid, method), correct in outcomes.items()
            ])
            atomic_csv(responses, root / "responses.csv")
            report = root / "report.json"
            summarize(SimpleNamespace(input=str(root / "paired.csv"),
                                      responses=str(root / "responses.csv"),
                                      report=str(report), bootstrap=100))
            result = json.loads(report.read_text())
            row = result["summary"][0]
            self.assertEqual(row["paired_conditions"], 2)
            self.assertEqual(row["historical_accuracy"], 0.5)
            self.assertEqual(row["v2_accuracy"], 1.0)
            self.assertEqual(row["identical_text_pairs"], 1)
            self.assertEqual(row["delta_on_differing_text"], 1.0)
            self.assertEqual(row["only_v2_correct"], 1)
            self.assertEqual(result["reader_served_counts"], {"reader": 5})

    def test_summarize_refuses_mixed_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_csv(build_paired_input(compression_rows(["a", "b"]),
                                          compression_rows(["a", "b"])), root / "paired.csv")
            atomic_csv(pd.DataFrame([
                {"uid": "q1", "eval_model": reader, "method": "ours_v2", "target_ratio": 0.5,
                 "base_group_id": "family_q1", "response_status": "ok", "correct": True}
                for reader in ("gpt-oss-120b", "qwen-3.8-27b-fp8")
            ]), root / "responses.csv")
            with self.assertRaises(ValueError):
                summarize(SimpleNamespace(input=str(root / "paired.csv"),
                                          responses=str(root / "responses.csv"),
                                          report=str(root / "r.json"), bootstrap=100))


if __name__ == "__main__":
    unittest.main()
