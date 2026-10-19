"""Regression checks use fabricated data; never download weights or call APIs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd

from final_eval.utils.answer_matching import answers_match, grade_answer
from final_eval.utils.artifacts import parse_bool, resume_manifest
from final_eval.preparation.schema import normalize_votes, build_consensus, prepare_problems, grouped_folds
from final_eval.scorer.data import build_span_table
from final_eval.compressor.budget import TokenCounter, select_to_budget
from final_eval.compressor.retention import retention, alignment_bounds
from final_eval.inference.client import HostedClient
from final_eval.utils.statistics import cluster_ci
from final_eval.preparation.recover_gsmic import recover

def votes(labels=("critical", "critical", "non_critical")):
    text = "Alice has 5 apples. How many apples?"
    start = text.index("5")
    return pd.DataFrame([{
        "example_id": "example", "source_dataset": "gsm8k", "original_prompt": text,
        "ground_truth_answer": "5", "span_start": start, "span_end": start + 1,
        "span_text": "5", "span_category": "number", "labeling_model": f"model{i}",
        "model_specific_label": label, "model_answer_original": "Final Answer: 5",
        "model_answer_perturbed": "Final Answer: 0" if label == "critical" else "Final Answer: 5",
        "is_correct_original": True,
    } for i, label in enumerate(labels)])

class GraderTests(unittest.TestCase):
    def test_unique_numeric_grammar(self):
        for response, gold in [("42 or 17", "42"), ("1/2", "1"), ("60 and 90", "60"),
                               ("NaN", "1"), ("1/0", "1"), ("-3", "3")]:
            with self.subTest(response=response):
                self.assertFalse(answers_match("Final Answer: " + response, gold))
        for response, gold in [("1/2", "0.5"), ("$1,200", "1200"), ("60 miles", "60"),
                               ("-3.0", "-3"), ("0", "0")]:
            self.assertTrue(answers_match("Final Answer: " + response, gold))

    def test_empty_choice_and_conflicting_text(self):
        q = "Choose.\n(A) Alice\n(B) Bob"
        self.assertFalse(answers_match("", "(A)", q))
        self.assertFalse(answers_match("Final Answer: A or B", "(A)", q))
        self.assertFalse(answers_match("Final Answer: (B) Alice", "(A)", q))
        self.assertTrue(answers_match("Final Answer: (A)", "(A)", q))
        self.assertTrue(answers_match("Final Answer: Alice", "(A)", q))

    def test_reasoning_and_binary(self):
        self.assertTrue(answers_match("We used 6.\nFinal Answer: 5", "5"))
        self.assertFalse(answers_match("<think>Final Answer: 5", "5"))
        self.assertFalse(answers_match("Final Answer: no, actually yes", "yes"))
        self.assertTrue(answers_match("Final Answer: invalid.", "invalid"))
        self.assertNotEqual(grade_answer("Final Answer: cannot tell", "5").status, "ok")

class DataTests(unittest.TestCase):
    def test_deduplicate_regrade_and_soft_labels(self):
        frame = votes()
        normalized = normalize_votes(pd.concat([frame, frame.iloc[[0]]]), regrade=True)
        self.assertEqual(len(normalized), 3)
        consensus = build_consensus(normalized)
        table = build_span_table(consensus)
        self.assertAlmostEqual(table.target.iloc[0], 2 / 3)
        self.assertEqual(table.label.iloc[0], 1)
        held = build_span_table(consensus, "model0")
        self.assertEqual(held.target.iloc[0], 0.5)
        self.assertEqual(held.label.iloc[0], 0)
        self.assertEqual(held.eval_label.iloc[0], 1)

    def test_conflict_not_first_wins(self):
        frame = votes()
        conflict = frame.iloc[[0]].copy()
        conflict.loc[:, "model_specific_label"] = "non_critical"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_votes(pd.concat([frame, conflict]))

    def test_conflicting_consensus_rejected(self):
        consensus = build_consensus(normalize_votes(votes()))
        consensus["agreement_score"] = 0.0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_span_table(consensus)

    def test_family_variants_stay_together(self):
        rows = []
        for i in range(8):
            for j in range(2):
                rows.append({"example_id": f"{i}_{j}", "source_dataset": "gsm_ic",
                    "original_prompt": f"Alice has {i+1} apples. Weather {j}. How many?",
                    "inserted_context": f"Weather {j}.", "ground_truth_answer": str(i+1)})
        problems = prepare_problems(pd.DataFrame(rows))
        self.assertEqual(problems.base_group_id.nunique(), 8)
        for _, roles in grouped_folds(problems, folds=2, val_fraction=0.25):
            group_sets = [set(problems.loc[problems.uid.isin(uids), "base_group_id"]) for uids in roles.values()]
            self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])

    def test_missing_family_data_is_not_guessed(self):
        p = pd.DataFrame([{"example_id": "x", "source_dataset": "gsm_ic",
                           "original_prompt": "Question", "ground_truth_answer": "1"}])
        with self.assertRaisesRegex(ValueError, "Cannot establish"):
            prepare_problems(p)

    def test_verified_source_recovery(self):
        frame = pd.DataFrame([{"example_id": "gsmic_0", "source_dataset": "gsm_ic",
                               "original_prompt": "5 apples. Sunny.", "ground_truth_answer": "5"}])
        source = [{"new_question": "5 apples. Sunny.", "original_question": "5 apples."}]
        self.assertEqual(recover(frame, source).clean_question.iloc[0], "5 apples.")
        source[0]["new_question"] = "A different question"
        with self.assertRaisesRegex(ValueError, "Source mismatch"):
            recover(frame, source)

    def test_missing_original_response_is_not_a_model_error(self):
        frame = votes()
        frame.loc[:, "model_answer_original"] = ""
        normalized = normalize_votes(frame, regrade=True)
        self.assertTrue(normalized.vote_status.eq("missing_original_response").all())
        self.assertTrue(build_consensus(normalized).empty)

class BudgetRetentionTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter("regex-test-only")

    def test_exact_floor_and_source_order(self):
        text = "one\n two three four five"
        result = select_to_budget(text, [5, 4, 3, 2, 1], 0.5, self.counter)
        self.assertEqual(result.budget, 2)
        self.assertLessEqual(self.counter(result.text), 2)
        self.assertEqual(result.text, "one\n two")

    def test_protection_infeasibility_and_whole_groups(self):
        text = "5 red apples left"
        infeasible = select_to_budget(text, [1, 1, 1, 1], 0.25, self.counter, protected_ranges=[(0, 12)])
        self.assertEqual(infeasible.status, "infeasible_protection")
        grouped = select_to_budget(text, [10, 0, 0, 1], 0.5, self.counter, phrase_ranges=[(0, 5)])
        self.assertEqual(grouped.text, "5 red")

    def test_repeated_occurrences_are_separate(self):
        text = "5 apples and 5 pears"
        spans = pd.DataFrame([{"span_start": 0, "span_end": 1, "span_category": "number", "label": 1}])
        second = text.rindex("5")
        result = retention(text, "5 pears", spans, [[second, second + 1], [second + 2, len(text)]])
        self.assertEqual(result["crit_kept_lower"], 0)

    def test_alignment_ambiguity_and_fragments(self):
        low, high, status = alignment_bounds("5 5", "5")
        self.assertEqual(status, "ambiguous_token_alignment")
        self.assertEqual(low, [])
        self.assertEqual(len(high), 2)
        self.assertEqual(alignment_bounds("Lorraine", "Lraine")[2], "unresolved")

    def test_distractor_token_mass_is_not_sentence_survival(self):
        spans = pd.DataFrame([{"span_start": 0, "span_end": 14,
                              "span_category": "distractor_context", "label": 0}])
        result = retention("sunny red days", "sunny days", spans, [[0, 5], [10, 14]])
        category = json.loads(result["cat_retention_json"])["distractor_context"]
        self.assertEqual(category["full_lower"], 0)
        self.assertEqual(category["tokens_lower"], 2)
        self.assertEqual(category["source_tokens"], 3)

class RuntimeTests(unittest.TestCase):
    def test_manifest_blocks_mixed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.csv"
            resume_manifest(output, {"tokenizer": "one"})
            resume_manifest(output, {"tokenizer": "one"})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                resume_manifest(output, {"tokenizer": "two"})

    def test_false_string_is_not_truthy(self):
        self.assertFalse(parse_bool("False"))
        with self.assertRaises(ValueError):
            parse_bool("")

    @patch("final_eval.inference.client.requests.post")
    def test_exact_model_and_full_response(self, post):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"model": "served-model", "choices": [{"message": {"content": "Final Answer: 5"}, "finish_reason": "stop"}], "usage": {"total_tokens": 8}}
        post.return_value = response
        client = HostedClient({"name": "qwen3.7-max", "base_url": "https://example.invalid/v1", "sleep_seconds": 0})
        result = client.generate("prompt")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3.7-max")
        self.assertEqual(result.served_model, "served-model")
        self.assertEqual(result.text, "Final Answer: 5")

    @patch("final_eval.inference.client.requests.post")
    def test_auth_stops_provider_and_attempt_budget(self, post):
        post.return_value = Mock(ok=False, status_code=401)
        client = HostedClient({"name": "exact", "base_url": "https://example.invalid", "sleep_seconds": 0})
        self.assertEqual(client.generate("one").status, "auth_failed")
        self.assertEqual(client.generate("two").status, "auth_failed")
        self.assertEqual(post.call_count, 1)

    @patch("final_eval.inference.client.requests.post")
    def test_attempt_budget_persists_across_resume(self, post):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"model": "exact", "choices": [{"message": {"content": "5"}, "finish_reason": "stop"}]}
        post.return_value = response
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"name": "exact", "base_url": "https://example.invalid/v1", "sleep_seconds": 0, "request_budget": 1}
            path = Path(tmp) / "usage.json"
            self.assertEqual(HostedClient(cfg, usage_path=path).generate("first").status, "ok")
            self.assertEqual(HostedClient(cfg, usage_path=path).generate("second").status, "request_budget_exhausted")
            self.assertEqual(post.call_count, 1)

    def test_clustered_ci_retains_shared_family(self):
        lo, hi = cluster_ci([1, 1, -1, -1], ["a", "a", "b", "b"], bootstrap=100)
        self.assertEqual((lo, hi), (-1.0, 1.0))

if __name__ == "__main__":
    unittest.main()
