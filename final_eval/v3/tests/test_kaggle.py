"""Transfer integrity, recovery and a real tiny LLMLingua-2 adapter smoke."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from final_eval.v3.build_kaggle import notebook
from final_eval.v3.kaggle_bootstrap import (ARCHIVE, BUNDLE, MANIFEST, fetch_weights,
    find_payload, install, relative, restore_saved_output, sha)


class KaggleTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {"final_eval/v3/fixture.py": b"value = 1\n", "data/problems.csv": b"uid\na\n"}
        self.manifest = {"bundle": BUNDLE, "files": {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
                                                   for n, b in self.payload.items()}}

    def archive(self, manifest=None, payload=None):
        manifest = manifest or self.manifest
        raw = json.dumps(manifest).encode()
        target = self.root / ARCHIVE
        with zipfile.ZipFile(target, "w") as archive:
            for name, data in (payload or self.payload).items():
                archive.writestr(name, data)
            archive.writestr(MANIFEST, raw)
        return target, hashlib.sha256(raw).hexdigest()

    def test_zip_and_expanded_upload_roundtrip_and_idempotence(self):
        target, expected = self.archive()
        dest = install(find_payload(self.root, expected), expected, self.root / "installed")
        install(target, expected, dest)
        for name, data in self.payload.items():
            self.assertEqual((dest / name).read_bytes(), data)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(self.root / "expanded")
        target.unlink()
        self.assertEqual(find_payload(self.root / "expanded", expected), self.root / "expanded")
        install(self.root / "expanded", expected, self.root / "installed_expanded")

    def test_tampered_payload_fails_before_any_install(self):
        target, expected = self.archive(payload={**self.payload, "data/problems.csv": b"modified\n"})
        dest = self.root / "installed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            install(target, expected, dest)
        self.assertFalse(dest.exists())

    def test_existing_modified_file_is_preserved(self):
        target, expected = self.archive()
        dest = install(target, expected, self.root / "installed")
        original = dest / "data/problems.csv"
        original.write_text("local work")
        with self.assertRaisesRegex(ValueError, "preserved"):
            install(target, expected, dest)
        self.assertEqual(original.read_text(), "local work")

    def test_unsafe_names_and_extra_zip_members_are_rejected(self):
        for name in ("/absolute", "../escape", "C:/file", "a\\b"):
            with self.assertRaises(ValueError):
                relative(name)
        target, expected = self.archive(payload={**self.payload, "extra.env": b"never install"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            install(target, expected, self.root / "installed")

    def test_manifest_mismatch_rejects_another_version(self):
        target, _ = self.archive()
        with self.assertRaisesRegex(ValueError, "Wrong transfer"):
            install(target, "0" * 64, self.root / "installed")

    def test_weight_download_is_pinned_hash_checked_and_resumable(self):
        data = b"fixture checkpoint, not model weights"
        self.manifest["v2_weights"] = {"repo_id": "fixture/repo", "revision": "fixed-sha",
            "files": {"fold0/scorer.pt": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}
        target, expected = self.archive()
        dest = install(target, expected, self.root / "installed")
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        fetch_weights(dest, download)
        fetch_weights(dest, download)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["revision"], "fixed-sha")
        self.assertIs(calls[0]["token"], False)
        weight = dest / "models/span_scorer_v2_20261021/fold0/scorer.pt"
        weight.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            fetch_weights(dest, download)
        self.assertEqual(weight.read_bytes(), b"wrong")

    def test_full_saved_output_resume_restores_same_ledger_and_optimizer(self):
        target, expected = self.archive()
        old = install(target, expected, self.root / "old")
        run = old / "outputs/v3/student_01"
        for name, data in {"resolved_config.json": b"{}", "reader/cost.json": b'{"accounted_inr": 12}',
                           "models/teacher_only/last.pt": b"optimizer", "pipeline.lock": b"0"}.items():
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        inputs = self.root / "input_previous"
        shutil.copytree(old, inputs / "symbiote_v3")
        new = install(target, expected, self.root / "new")
        restore_saved_output(inputs, expected, new)
        self.assertEqual((new / "outputs/v3/student_01/reader/cost.json").read_bytes(), b'{"accounted_inr": 12}')
        self.assertEqual((new / "outputs/v3/student_01/models/teacher_only/last.pt").read_bytes(), b"optimizer")
        self.assertFalse((new / "outputs/v3/student_01/pipeline.lock").exists())
        (new / "outputs/v3/student_01/reader/cost.json").write_text("different ledger")
        with self.assertRaisesRegex(ValueError, "replace"):
            restore_saved_output(inputs, expected, new)

    def test_missing_full_saved_output_cannot_silently_start_fresh(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            restore_saved_output(self.root, "a" * 64, self.root / "new")

    def test_generated_notebook_compiles_and_uses_secret(self):
        book = notebook("1" * 64, "2" * 64)
        self.assertEqual(sum(c["cell_type"] == "code" for c in book["cells"]), 3)
        text = ""
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                compile(code, "notebook", "exec")
                text += code
        self.assertIn('get_secret("JARVISLABS_API_KEY")', text)
        self.assertIn('env["HF_HUB_DISABLE_XET"] = "1"', text)
        self.assertIn("subprocess.run(command", text)
        self.assertIn("RETRY_FAILED = False", text)

    def test_failed_setup_exports_diagnostics_and_best_weights_without_large_resume_files(self):
        from final_eval.v3 import kaggle_runner as runner
        logs, run = self.root / "logs", self.root / "run"
        logs.mkdir()
        (logs / "setup.log").write_text("a dependency failed")
        for name in ("models/teacher_only/best.pt", "models/teacher_only/last.pt", "initialization/common_initial.pt", "private.env"):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        status = self.root / "status.json"
        status.write_text('{"success":false}')
        result = self.root / "result.zip"
        with patch.multiple(runner, ROOT=self.root, LOGS=logs, RUN=run, STATUS=status,
                            CONFIG=self.root / "config.json", RESULT=result):
            runner.export(None, {})
        with zipfile.ZipFile(result) as archive:
            self.assertIn("run/models/teacher_only/best.pt", archive.namelist())
            self.assertIn("runtime/setup.log", archive.namelist())
            self.assertNotIn("run/models/teacher_only/last.pt", archive.namelist())
            self.assertNotIn("run/initialization/common_initial.pt", archive.namelist())
            self.assertFalse(any(n.endswith(".env") for n in archive.namelist()))

    def test_runner_redacts_keys_and_does_not_pass_python_overlays(self):
        from final_eval.v3.kaggle_runner import environment, redact
        with patch.dict(os.environ, {"JARVISLABS_API_KEY": "fixture-key-to-redact", "PYTHONPATH": "another-experiment"}):
            self.assertEqual(redact("error fixture-key-to-redact"), "error <redacted>")
            env = environment()
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("/kaggle/working", env["HF_HOME"])
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")


@unittest.skipUnless(importlib.util.find_spec("llmlingua") and importlib.util.find_spec("accelerate"),
                     "Optional real LLMLingua smoke needs the Kaggle dependencies")
class RealTeacherAdapterTests(unittest.TestCase):
    def test_real_llmlingua_token_classifier_load_and_compress(self):
        # No pretrained download/paid API. Real package + real tiny token-classifier.
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast, BertConfig, BertForTokenClassification
        from final_eval.compressor.compress import BaselineRunner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-bert-base-multilingual-cased"
            vocab = {t: i for i, t in enumerate(["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]",
                "Alice", "has", "3", "apples", "and", "buys", "2", ".", "How", "many", "?"])}
            tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
            fast.save_pretrained(path)
            cfg = BertConfig(vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
                num_attention_heads=3, intermediate_size=24, max_position_embeddings=512, num_labels=2, pad_token_id=1)
            BertForTokenClassification(cfg).save_pretrained(path)
            # This tokenizer is used only by the library's cost display, not its
            # neural keep decisions. Avoid any tiktoken network/cache dependency.
            with patch("tiktoken.encoding_for_model", return_value=SimpleNamespace(encode=lambda text: list(text))):
                runner = BaselineRunner("cpu", "unused", str(path))
                text, status, details = runner.run_budgeted("llmlingua2_budgeted",
                    "Alice has 3 apples and buys 2 apples . How many apples ?", .5, 8, lambda t: len(t.split()))
            self.assertIsInstance(text, str)
            self.assertIn(status, {"ok", "baseline_budget_infeasible"})
            self.assertEqual(details["budget_controller"], "official-controls-bracket-v1")


if __name__ == "__main__":
    unittest.main()
