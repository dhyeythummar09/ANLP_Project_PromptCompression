from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from final_eval.v3.artifacts import (ROOT, read_json, read_jsonl, write_jsonl, write_csv,
                                    stage_manifest, finish_stage, exclusive_lock)
from final_eval.v3.data import (split_questions, teacher_word_labels, prepare,
                               cached_teacher_labels, local_teacher_labels, load_problems)
from final_eval.v3.model import (encode_question, collate_questions, DenseStudent,
                                question_loss, predict_words, ranking_loss)
from final_eval.v3.compression import reference_fold, atomic_ranges
from final_eval.v3.train import initialize, train, load_model
from final_eval.v3.reader import Reader, evaluate, ReaderStopped, evaluation_tasks
from final_eval.inference.client import ModelResponse
from final_eval.utils.artifacts import atomic_json


class FakeReader:
    """Deterministic simulation only; never a live downstream-quality claim."""
    total_calls = 0

    def __init__(self, cfg=None, usage_path=None):
        self.cfg = cfg or {"name": "fixture-reader"}
        self.attempts = 0

    def generate(self, prompt):
        self.attempts += 1
        FakeReader.total_calls += 1
        nums = [int(x) for x in re.findall(r"\b\d+\b", prompt)]
        return ModelResponse(text=f"Final Answer: {sum(nums)}", requested_model=self.cfg["name"],
            served_model=self.cfg["name"], attempts=1,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 8})


class FakeInfo:
    def __init__(self, *args, **kwargs):
        pass

    def word_info(self, text, units):
        return np.linspace(0, 1, len(units))


def fixture(root, epochs=1):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, DistilBertConfig, DistilBertModel
    from final_eval.scorer.train_scorer import SpanScorer
    base = root / "tiny_base"
    base.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alice", "has", "apples", "and", "buys", "how", "many", "?", "."]
    vocabulary += [str(i) for i in range(60)]
    vocab = {t: i for i, t in enumerate(vocabulary)}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tokenizer.save_pretrained(base)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=12, hidden_dim=24, n_layers=1, n_heads=3,
                          max_position_embeddings=32, dropout=.1, attention_dropout=.1)
    torch.manual_seed(1)
    DistilBertModel(cfg).save_pretrained(base)
    questions, spans, teacher = [], [], []
    for i in range(24):
        text = f"Alice has {i + 3} apples and buys 2 apples. How many apples?"
        q = {"uid": f"math|q{i}", "example_id": f"q{i}", "source_dataset": "math",
             "base_group_id": f"family{i // 2}", "original_prompt": text, "ground_truth_answer": str(i + 5)}
        questions.append(q)
        for token, target in [("buys", 1 / 3), ("2", 1.)]:
            start = text.rindex(token)
            spans.append({**q, "span_start": start, "span_end": start + len(token),
                          "span_text": token, "span_category": "action_verb" if token == "buys" else "number",
                          "agreement_score": target})
        teacher.append({**q, "method": "llmlingua2", "target_ratio": .5,
                        "compressed_prompt": f"Alice {i + 3} apples buys 2 How many?"})
    write_csv(root / "problems.csv", questions)
    write_csv(root / "spans.csv", spans)
    write_csv(root / "teacher.csv", teacher)
    atomic_json(root / "exposure.json", {"exposed_uids": ["math|q0", "math|q1"]})
    v2 = root / "v2"
    (v2 / "tokenizer").mkdir(parents=True)
    tokenizer.save_pretrained(v2 / "tokenizer")
    split = {"protocol": "symbiote-v2", "base_groups": {q["uid"]: q["base_group_id"] for q in questions},
             "folds": {q["uid"]: (i // 2) % 5 for i, q in enumerate(questions)}, "partitions": {}}
    for f in range(5):
        d = v2 / f"fold{f}"
        d.mkdir()
        cfg.save_pretrained(d)
        model = SpanScorer(encoder_config=cfg)
        torch.save(model.state_dict(), d / "scorer.pt")
        atomic_json(d / "category_prior.json", {"number": .5, "action_verb": .1})
        split["partitions"][str(f)] = {"train": [q["uid"] for q in questions if split["folds"][q["uid"]] != f], "val": []}
    atomic_json(v2 / "split.json", split)
    c = read_json(ROOT / "final_eval/v3/configs/student.json")
    c.update(project_root=str(root), problems=str(root / "problems.csv"), span_labels=str(root / "spans.csv"),
             teacher_cache=str(root / "teacher.csv"), v2_model_dir=str(v2), output_dir=str(root / "run"),
             exposure_files=[str(root / "exposure.json")], tokenizer="regex-test-only", test_fixture=True, bootstrap=50)
    c["split"].update(test_families=2, dev_families=2, validation_fraction=.2)
    c["training"].update(encoder=str(base), max_length=32, stride=4, epochs=epochs,
                          batch_size=2, amp=False, device="cpu", require_cuda=False, save_every_steps=1, cpu_threads=2)
    c["references"].update(llmlingua2=False, info_model=str(base))
    c["reader"]["model"].update(name="fixture-reader", api_key_env="", max_tokens=100, sleep_seconds=0)
    c["reader"].update(allowed_served_models=["fixture-reader"], key_file="", pilot_questions=2,
                        pilot_min_accuracy=.5, limit_inr=10)
    c["benchmark"]["enabled"] = False
    return c, tokenizer, cfg


class DataTests(unittest.TestCase):
    def test_repeated_token_alignment_is_masked_not_guessed(self):
        labels, status = teacher_word_labels("red red blue", "red blue")
        self.assertEqual(labels, [-1, -1, 1])
        self.assertEqual(status, "ambiguous_token_alignment")
        self.assertEqual(teacher_word_labels("12.5 apples", "12 apples")[0], [-1, 1])
        self.assertEqual(teacher_word_labels("apples", "invented")[0], [-1])

    def test_family_partition_excludes_siblings_and_prior_exposure(self):
        qs = [{"uid": str(i), "base_group_id": str(i // 2), "source_dataset": str(i % 2)} for i in range(40)]
        settings = {"test_families": 3, "dev_families": 3, "validation_fraction": .2, "seed": 3}
        split = split_questions(qs, {"0"}, settings)
        self.assertNotIn("0", split["families"]["test"])
        for q in qs:
            sibling = next(x for x in qs if x["base_group_id"] == q["base_group_id"] and x["uid"] != q["uid"])
            self.assertEqual(split["roles"][q["uid"]], split["roles"][sibling["uid"]])
        self.assertEqual(split, split_questions(qs, {"0"}, settings))
        with self.assertRaises(ValueError):
            split_questions(qs, {str(i) for i in range(40)}, settings)

    def test_v2_reference_audits_sibling_family_not_only_uid(self):
        split = {"protocol": "symbiote-v2", "folds": {}, "base_groups": {"a": "family"},
                 "partitions": {"0": {"train": ["a"], "val": []}, "1": {"train": [], "val": []}}}
        self.assertEqual(reference_fold({"uid": "unlabelled-sibling", "base_group_id": "family"}, split), 1)
        split["partitions"]["1"]["val"] = ["a"]
        with self.assertRaises(ValueError):
            reference_fold({"uid": "x", "base_group_id": "family"}, split)

    def test_failed_compression_cannot_suppress_original_control(self):
        rows = [{"uid": "q", "original_prompt": "full", "ground_truth_answer": "5", "base_group_id": "f",
                 "budget_compliant": False, "status": "baseline_budget_infeasible", "method": "baseline", "target_ratio": .5}]
        original = evaluation_tasks(rows)[0]
        self.assertEqual(original["method"], "original")
        self.assertTrue(original["budget_compliant"])
        self.assertEqual(original["compressed_prompt"], "full")

    def test_previous_development_pilot_and_pair_queries_are_not_fresh_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            config["output_dir"] = str(Path(directory) / "jobs/new")
            for path, uid in [("older/dev_exposure.json", "math|q2"),
                              ("older/reader/pilot_exposure.json", "math|q4"),
                              ("older/pairs/exposure.json", "math|q6")]:
                atomic_json(Path(directory) / "jobs" / path, {"exposed_uids": [uid]})
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(config)
            split = read_json(Path(config["output_dir"]) / "data/split.json")
            self.assertFalse(set(split["families"]["test"]) & {"family1", "family2", "family3"})


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tokenizer, self.encoder_config = fixture(self.root)

    def test_overflow_owns_every_word_once_and_masks_uncoverable_spans(self):
        text = " ".join(["alice"] * 100)
        q = {"uid": "long", "source_dataset": "math", "original_prompt": text,
             "spans": [{"start": 0, "end": len(text), "target": 1.}]}
        item = encode_question(q, self.tokenizer, [1] * 100, 32, 4)
        self.assertGreater(len(item["windows"]), 1)
        indices = [i for w in item["windows"] for i in w["word_indices"]]
        self.assertEqual(sorted(indices), list(range(100)))
        self.assertEqual(item["uncovered_words"], [])
        self.assertEqual(item["uncovered_spans"], [0])
        from transformers import DistilBertModel
        model = DenseStudent(DistilBertModel(self.encoder_config))
        self.assertEqual(len(predict_words(model, item, "cpu")), 100)

    def test_auxiliary_target_changes_encoder_gradient(self):
        from transformers import DistilBertModel
        q = {"uid": "q", "source_dataset": "math", "original_prompt": "alice buys 2 apples",
             "spans": [{"start": 6, "end": 10, "target": 1.}]}
        item = encode_question(q, self.tokenizer, [1, 1, 1, 0], 32, 4)
        batch = collate_questions([item])
        model = DenseStudent(DistilBertModel(self.encoder_config), dropout=0).eval()
        from final_eval.v3.train import losses
        losses(model, batch, 0, 0)[0].backward()
        first = model.encoder.embeddings.word_embeddings.weight.grad.clone()
        model.zero_grad()
        losses(model, batch, .2, 0)[0].backward()
        second = model.encoder.embeddings.word_embeddings.weight.grad.clone()
        self.assertTrue(torch.isfinite(second).all())
        self.assertGreater(float((first - second).abs().sum()), 0)

    def test_optional_ranking_handles_ties_and_uses_target_order(self):
        valid = torch.tensor([[True, True, True]])
        targets = torch.tensor([[0., .5, 1.]])
        right = torch.tensor([[-1., 0., 1.]], requires_grad=True)
        wrong = torch.tensor([[1., 0., -1.]], requires_grad=True)
        q = torch.tensor([0])
        self.assertLess(float(ranking_loss(right, targets, valid, q).detach()), float(ranking_loss(wrong, targets, valid, q).detach()))
        tied = ranking_loss(right, torch.ones_like(targets), valid, q)
        tied.backward()
        self.assertEqual(float(tied.detach()), 0)

    def test_whitespace_padding_does_not_drop_a_valid_auxiliary_span(self):
        q = {"uid": "space", "source_dataset": "math", "original_prompt": " alice ",
             "spans": [{"start": 0, "end": 7, "target": 1.}]}
        item = encode_question(q, self.tokenizer, [1], 32, 4)
        self.assertEqual(item["uncovered_spans"], [])
        self.assertEqual(sum(len(w["span_targets"]) for w in item["windows"]), 1)

    def test_local_teacher_labels_are_separate_from_test_optimization(self):
        import sys
        from types import SimpleNamespace
        self.config["teacher"].update(mode="local", model=str(self.root / "tiny_base"))
        class Teacher:
            def __init__(self, **kwargs):
                pass
            def compress_prompt_llmlingua2(self, text, **kwargs):
                return {"compressed_prompt": " ".join(text.split()[:4]), "fn_labeled_original_prompt": "fixture"}
            def compress_prompt(self, text, rate, target_token, **kwargs):
                return {"compressed_prompt": " ".join(text.split()[:max(1, target_token // 2)])}
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.config)
            with patch.dict(sys.modules, {"llmlingua": SimpleNamespace(PromptCompressor=Teacher)}):
                local_teacher_labels(self.config)
                from final_eval.v3.compression import generate_teacher
                from final_eval.compressor.budget import TokenCounter
                out = Path(self.config["output_dir"])
                q = next(q for q in read_jsonl(out / "data/questions.jsonl") if q["role"] == "val")
                captures = []
                generate_teacher(self.config, [q], {}, captures.append, TokenCounter("regex-test-only"), "cpu")
                self.assertTrue(all(r["budget_compliant"] for r in captures))
        out = Path(self.config["output_dir"])
        self.assertTrue(all(r["role"] in {"train", "val"} for r in read_jsonl(out / "data/teacher.jsonl")))

    def test_real_tiny_training_resumes_after_interrupted_optimizer_checkpoint(self):
        self.config["training"]["epochs"] = 2
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.config)
            cached_teacher_labels(self.config)
            train(self.config)
        uninterrupted = Path(self.config["output_dir"])
        resumed = copy.deepcopy(self.config)
        resumed["output_dir"] = str(self.root / "resumed")
        from final_eval.v3 import train as training
        original_save = training.torch_save
        interrupted = [False]

        def crash(path, value):
            original_save(path, value)
            if Path(path).name == "last.pt" and Path(path).parent.name == "teacher_only" and not interrupted[0]:
                interrupted[0] = True
                raise RuntimeError("injected interruption")

        with contextlib.redirect_stdout(io.StringIO()):
            prepare(resumed)
            cached_teacher_labels(resumed)
            with patch.object(training, "torch_save", side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    train(resumed)
            train(resumed)
        for arm in ("teacher_only", "teacher_cf"):
            a = torch.load(uninterrupted / "models" / arm / "best.pt", weights_only=True)
            b = torch.load(Path(resumed["output_dir"]) / "models" / arm / "best.pt", weights_only=True)
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)
        a = read_json(uninterrupted / "models/teacher_only/metadata.json")
        b = read_json(uninterrupted / "models/teacher_cf/metadata.json")
        self.assertEqual(a["initial_sha256"], b["initial_sha256"])
        self.assertEqual(a["aux_weight"], 0)
        self.assertEqual(b["aux_weight"], .2)


class ArtifactReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / "final_eval/v3/configs/student.json")
        self.config.update(output_dir=str(self.root / "run"), project_root=str(self.root))
        self.config["reader"]["model"].update(name="fixture-reader", api_key_env="", max_tokens=100)
        self.config["reader"].update(allowed_served_models=["fixture-reader"], limit_inr=1, key_file="")

    def test_success_cache_deduplicates_and_budget_survives_restart(self):
        client = FakeReader()
        reader = Reader(self.config, client)
        a = reader.ask("alice has 2 apples")
        b = reader.ask("alice has 2 apples")
        self.assertEqual(a["query_hash"], b["query_hash"])
        self.assertEqual(client.attempts, 1)
        cost = read_json(self.root / "run/reader/cost.json")["accounted_inr"]
        again = Reader(self.config, FakeReader())
        again.ask("alice has 2 apples")
        self.assertEqual(read_json(self.root / "run/reader/cost.json")["accounted_inr"], cost)
        self.config["output_dir"] = str(self.root / "tiny_allowance")
        self.config["reader"]["limit_inr"] = .00001
        client = FakeReader()
        result = Reader(self.config, client).ask("hello")
        self.assertEqual(result["generation"]["status"], "request_budget_exhausted")
        self.assertEqual(client.attempts, 0)

    def test_interruption_retains_full_unknown_reservation(self):
        class Interrupted(FakeReader):
            def generate(self, prompt):
                self.attempts += 1
                raise RuntimeError("lost response")
        with self.assertRaisesRegex(RuntimeError, "lost response"):
            Reader(self.config, Interrupted()).ask("hello")
        ledger = read_json(self.root / "run/reader/cost.json")
        self.assertGreater(ledger["accounted_inr"], 0)
        self.assertEqual(len(ledger["unsettled_reservations"]), 1)

    def test_malformed_usage_and_wrong_model_fail_closed(self):
        class Malformed(FakeReader):
            def generate(self, prompt):
                r = super().generate(prompt)
                r.usage = {"prompt_tokens": True, "completion_tokens": 0}
                r.served_model = "another-model"
                return r
        r = Reader(self.config, Malformed())
        result = r.ask("hello")
        self.assertEqual(result["generation"]["status"], "served_model_mismatch")
        self.assertEqual(read_json(self.root / "run/reader/cost.json")["unknown_attempts"], 1)
        with self.assertRaises(ReaderStopped):
            r.stop_on(result)

    def test_stage_tampering_and_process_lock(self):
        marker, state, _ = stage_manifest(self.root, "x", {"data": 1})
        output = self.root / "a.txt"
        output.write_text("correct")
        finish_stage(marker, state, [output])
        self.assertTrue(stage_manifest(self.root, "x", {"data": 1})[2])
        output.write_text("tampered")
        with self.assertRaises(ValueError):
            stage_manifest(self.root, "x", {"data": 1})
        with exclusive_lock(self.root / "active.lock"):
            with self.assertRaises(RuntimeError):
                with exclusive_lock(self.root / "active.lock"):
                    pass

    def test_reporting_change_does_not_invalidate_training_scope(self):
        from final_eval.v3.artifacts import stage_sources
        before = stage_sources("train", self.root)
        self.assertNotIn("v3/report.py", before)
        self.assertNotIn("v3/reader.py", before)
        self.assertIn("v3/model.py", before)
        self.assertIn("v3/train.py", before)

    def test_fatal_failure_requires_explicit_retry_and_preserves_ledger(self):
        out = Path(self.config["output_dir"])
        row = {"uid": "q", "base_group_id": "f", "source_dataset": "math", "method": "candidate",
               "target_ratio": .5, "actual_ratio": .5, "original_prompt": "Alice has 2 and buys 3",
               "ground_truth_answer": "5", "compressed_prompt": "2 3", "status": "ok", "budget_compliant": True}
        write_jsonl(out / "dev/compressed.jsonl", [row])
        atomic_json(out / "reader/qualification.json", {"passed": True})
        class FailOnce(FakeReader):
            first = True
            def generate(self, prompt):
                result = super().generate(prompt)
                if FailOnce.first:
                    FailOnce.first = False
                    result.status = "auth_failed"
                return result
        with patch("final_eval.v3.reader.HostedClient", FailOnce):
            with self.assertRaises(ReaderStopped):
                evaluate(self.config, "dev")
            before = read_json(out / "reader/cost.json")["accounted_inr"]
            with self.assertRaises(ReaderStopped):
                evaluate(self.config, "dev")
            self.assertEqual(read_json(out / "reader/cost.json")["accounted_inr"], before)
            evaluate(self.config, "dev", retry_failed=True)
        self.assertTrue(all(r["correct"] for r in read_jsonl(out / "dev/responses.jsonl")))


class EndToEndTests(unittest.TestCase):
    def test_stage_failure_exports_diagnostics_without_entering_test(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            from final_eval.v3.pipeline import run
            with patch("final_eval.v3.data.cached_teacher_labels", side_effect=RuntimeError("injected failure")), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    run(config, through="teacher")
            output = Path(config["output_dir"])
            self.assertEqual(read_json(output / "failure.json")["stage"], "teacher")
            self.assertTrue(Path(str(output) + "_results.zip").exists())
            self.assertFalse((output / "test_exposure.json").exists())

    def test_tiny_real_students_through_freeze_test_report_bundle_and_cached_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = fixture(Path(directory))
            config["benchmark"].update(enabled=True, questions=2, repeats=1)
            from final_eval.v3.pipeline import run
            FakeReader.total_calls = 0
            with patch("final_eval.v3.reader.HostedClient", FakeReader), patch("final_eval.compressor.compress.InfoScorer", FakeInfo), contextlib.redirect_stdout(io.StringIO()):
                run(config)
                first = FakeReader.total_calls
                self.assertGreater(first, 0)
                run(config)
                self.assertEqual(FakeReader.total_calls, first)
            output = Path(config["output_dir"])
            frozen = read_json(output / "freeze.json")
            self.assertIn(frozen["policy"], config["policies"])
            split = read_json(output / "data/split.json")
            self.assertEqual(len(split["downstream"]["test"]), 2)
            report = read_json(output / "test/report/summary.json")
            self.assertTrue(report["complete_reader_stage"])
            self.assertTrue(all(r["budget_violations"] == 0 for r in report["failures"]))
            self.assertTrue((output / "test_exposure.json").exists())
            self.assertTrue(Path(str(output) + "_results.zip").exists())
            self.assertEqual(len(read_json(output / "benchmark.json")["results"]), 4)
            from final_eval.v3.report import regrade_saved
            original = next(r for r in read_jsonl(output / "test/responses.jsonl") if r["method"] == "original")
            original["correct"] = False
            self.assertTrue(regrade_saved(config, [original])[0]["correct"])


if __name__ == "__main__":
    unittest.main()
