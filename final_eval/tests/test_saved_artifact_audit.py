"""Detect corrupt evidence without models, providers or real study artifacts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from final_eval.analysis.audit import audit_responses, returned_usage
from final_eval.compressor.evaluate import compression_tasks, PROMPT_TEMPLATE
from final_eval.utils.artifacts import digest
from final_eval import build_kaggle_main as transfer


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.comp = pd.DataFrame([dict(uid="question", method="ours", target_ratio=.5,
            original_prompt="What is 2 + 2?", compressed_prompt="2+2?",
            ground_truth_answer="4", base_group_id="family", source_dataset="gsm8k")])
        self.manifest = {"config": {"name": "test-reader"}, "wrapper": PROMPT_TEMPLATE}
        self.rows = []
        for task in compression_tasks(self.comp):
            self.rows.append(dict(uid=task["uid"], method=task["method"], target_ratio=task["target_ratio"],
                eval_model="test-reader", ground_truth="4", base_group_id="family", source_dataset="gsm8k",
                generation_requested_model="test-reader", generation_served_model="test-reader",
                query_hash=digest([self.manifest["config"], PROMPT_TEMPLATE.format(question=task["compressed_prompt"])]),
                response="Final Answer: 4", response_status="ok", parse_status="ok", correct=True))

    def test_matching_evidence_passes_and_all_rows_are_checked(self):
        result = audit_responses(self.comp, pd.DataFrame(self.rows), self.manifest)
        self.assertEqual(result["conditions"], 2)
        self.assertEqual(result["query_gold_model_grade_mismatches"], 0)

    def test_changed_query_gold_model_or_grade_is_rejected(self):
        for column, value in [("query_hash", "wrong"), ("ground_truth", "5"),
                              ("generation_served_model", "other"), ("correct", False),
                              ("parse_status", "invalid")]:
            with self.subTest(column=column):
                rows = deepcopy(self.rows); rows[-1][column] = value
                with self.assertRaises(ValueError):
                    audit_responses(self.comp, pd.DataFrame(rows), self.manifest)

    def test_missing_condition_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            audit_responses(self.comp, pd.DataFrame(self.rows[:-1]), self.manifest)

    def test_cached_copies_are_free_but_archived_paid_failures_are_counted_once(self):
        rows = [dict(query_hash="same", generation_timestamp_utc="t1", reused_generation=False,
                     generation_usage=json.dumps({"prompt_tokens": 10, "completion_tokens": 20}))]
        rows += [dict(rows[0]), {**rows[0], "reused_generation": True}]
        history = {"attempts": [dict(query_hash="same", timestamp_utc="t1", usage={"completion_tokens": 20}),
                                dict(query_hash="same", timestamp_utc="t0", usage={"prompt_tokens": 5, "completion_tokens": 30})]}
        usage = returned_usage(pd.DataFrame(rows), history, ("10", "40"))
        self.assertEqual((usage["active_usage_records"], usage["archived_usage_records"]), (1, 1))
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"]), (15, 50))
        self.assertEqual(usage["estimated_inr"], "0.00215")


class FrozenTransferTests(unittest.TestCase):
    def test_existing_transfer_not_rebuilt_from_changed_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive_path = root / transfer.NAME
            source = b"# recorded bootstrap\n"
            manifest = {"bundle": "symbiote-main-stage-c-v1", "files": {
                "final_eval/kaggle_stage_c_bootstrap.py": hashlib.sha256(source).hexdigest()}}
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("MAIN_TRANSFER_MANIFEST.json", json.dumps(manifest))
                archive.writestr("final_eval/kaggle_stage_c_bootstrap.py", source)
            original = archive_path.read_bytes()
            with patch.object(transfer, "ROOT", root), patch.object(transfer, "OUT", root):
                transfer.build()
            self.assertEqual(archive_path.read_bytes(), original)
            notebook = json.loads((root / "notebooks/03_main_compression.ipynb").read_text())
            self.assertEqual(sum(c["cell_type"] == "code" for c in notebook["cells"]), 4)
            self.assertEqual(list(root.rglob("*.md")), [])

    def test_corrupt_transfer_is_rejected_before_writing_notebook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(root / transfer.NAME, "w") as archive:
                archive.writestr("MAIN_TRANSFER_MANIFEST.json", json.dumps({
                    "bundle": "symbiote-main-stage-c-v1", "files": {"bootstrap.py": "wrong"}}))
                archive.writestr("bootstrap.py", "corrupt")
            with patch.object(transfer, "ROOT", root), patch.object(transfer, "OUT", root):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    transfer.build()
            self.assertFalse((root / "notebooks").exists())


if __name__ == "__main__":
    unittest.main()
