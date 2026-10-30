import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from final_eval.followup_summary import _digit_retention, mask_deletion, reader_repeatability


class DigitRetentionTest(unittest.TestCase):
    def test_counts_repeated_numbers_once_each(self):
        self.assertEqual(_digit_retention("Tom has 3 apples and 3 pears", "Tom 3 apples pears"), 0.5)

    def test_no_numbers_is_undefined(self):
        self.assertTrue(pd.isna(_digit_retention("no numbers here", "no numbers")))


class RepeatabilityTest(unittest.TestCase):
    def test_only_identical_prompts_with_two_ok_answers_count(self):
        base = {"method": "ours", "target_ratio": 0.5, "response": "x"}
        reference = pd.DataFrame([
            {**base, "uid": "a", "query_hash": "h1", "response_status": "ok", "correct": True},
            {**base, "uid": "b", "query_hash": "h2", "response_status": "ok", "correct": True},
            {**base, "uid": "c", "query_hash": "h3", "response_status": "ok", "correct": True}])
        rerun = pd.DataFrame([
            {**base, "uid": "a", "query_hash": "h1", "response_status": "ok", "correct": "False"},
            {**base, "uid": "b", "query_hash": "changed", "response_status": "ok", "correct": "True"},
            {**base, "uid": "c", "query_hash": "h3", "response_status": "network_error", "correct": ""}])
        rows = reader_repeatability(reference, rerun, "test")
        overall = rows[0]
        self.assertEqual(overall["identical_prompts_both_ok"], 1)
        self.assertEqual(overall["correctness_agreement"], 0.0)


class MaskDeletionTest(unittest.TestCase):
    def test_wrong_originals_and_failed_requests_are_not_eligible(self):
        rows = []
        for i, (orig, mask, delete, status) in enumerate([
                ("True", "True", "False", "ok"), ("True", "True", "True", "ok"),
                ("False", "False", "True", "ok"), ("True", "True", "", "network_error")]):
            rows.append({"uid": f"q{i}", "span_category": "number", "original_status": "ok",
                         "original_correct": orig, "mask_status": "ok", "mask_correct": mask,
                         "delete_status": status, "delete_correct": delete})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spans.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = mask_deletion(SimpleNamespace(mask_deletion=path, bootstrap=50), Path(tmp))
        overall = result["by_category"][0]
        self.assertEqual(overall["eligible_original_correct"], 2)
        self.assertEqual(overall["agreement"], 0.5)
        self.assertEqual(overall["mask_right_delete_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
