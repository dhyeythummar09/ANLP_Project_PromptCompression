import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from final_eval.compare_checkpoints import audit_common_cohort


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.problems = root / "problems.csv"
        rows = [
            {"example_id": "a1", "source_dataset": "gsm8k",
             "original_prompt": "Alice has 1 apple. How many?", "ground_truth_answer": "1"},
            {"example_id": "a2", "source_dataset": "gsm8k",
             "original_prompt": "Alice has 2 apples. How many?", "ground_truth_answer": "2"},
            {"example_id": "b1", "source_dataset": "bbh",
             "original_prompt": "Bob has 3 apples. How many?", "ground_truth_answer": "3"},
        ]
        pd.DataFrame(rows).to_csv(self.problems, index=False)
        self.old = root / "old.json"
        self.v2 = root / "v2.json"
        # prepare_problems creates a family for each ordinary question.
        self.old.write_text(json.dumps({"folds": {
            "gsm8k|a1": 0, "gsm8k|a2": 1, "bbh|b1": 0}}))
        self.v2.write_text(json.dumps({"folds": {
            "gsm8k|a1": 1, "gsm8k|a2": 0, "bbh|b1": 1},
            "base_groups": {
                "gsm8k|a1": "a", "gsm8k|a2": "b", "bbh|b1": "c"}}))

    def test_intersects_assignments_without_guessing(self):
        cohort, metadata = audit_common_cohort(self.problems, self.old, self.v2)
        self.assertEqual(len(cohort), 3)
        self.assertEqual(metadata["common_families"], 3)

    def test_missing_old_assignment_is_excluded(self):
        split = json.loads(self.old.read_text())
        del split["folds"]["bbh|b1"]
        self.old.write_text(json.dumps(split))
        cohort, metadata = audit_common_cohort(self.problems, self.old, self.v2)
        self.assertEqual(len(cohort), 2)
        self.assertEqual(metadata["excluded_questions_without_both_assignments"], 1)

    def test_invalid_split_is_rejected(self):
        self.old.write_text(json.dumps({"folds": {"gsm8k|a1": "unknown"}}))
        with self.assertRaisesRegex(ValueError, "invalid fold"):
            audit_common_cohort(self.problems, self.old, self.v2)


if __name__ == "__main__":
    unittest.main()
