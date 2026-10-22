"""Portable release loader matches the trained architecture, using tiny fake weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import DistilBertConfig

from final_eval import model_release
from final_eval.scorer.train_scorer import SpanScorer
from final_eval.utils.artifacts import file_hash


class ModelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "load_scorer.py"
        path.write_text(model_release.loader_source(), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("release_loader_fixture", path)
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)

    def test_tiny_checkpoint_roundtrip_matches_original_logits_without_downloads(self):
        config = DistilBertConfig(vocab_size=16, dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, max_position_embeddings=16)
        original = SpanScorer(encoder_config=config).eval()
        folder = self.root / "fold0"; folder.mkdir()
        config.to_json_file(folder / "config.json")
        torch.save(original.state_dict(), folder / "scorer.pt")
        with patch.object(self.loader.DistilBertModel, "from_pretrained", side_effect=AssertionError("No download")), \
             patch.object(self.loader.DistilBertTokenizerFast, "from_pretrained", return_value="fixture-tokenizer") as tokenizer:
            loaded, saved_tokenizer = self.loader.load_fold(0, directory=self.root)
        tokenizer.assert_called_once_with(str(self.root / "tokenizer"), local_files_only=True)
        self.assertEqual(saved_tokenizer, "fixture-tokenizer")
        ids = torch.tensor([[1, 3, 4, 2]])
        attention = torch.ones_like(ids)
        masks = torch.tensor([[[0., 1., 1., 0.]]])
        with torch.inference_mode():
            torch.testing.assert_close(original(ids, attention, masks), loaded(ids, attention, masks), rtol=0, atol=0)

    def test_checksum_verification_rejects_changed_payload(self):
        payload = self.root / "weights.bin"; payload.write_bytes(b"fake-checkpoint")
        manifest = {"files": {"weights.bin": {"sha256": file_hash(payload)}}}
        (self.root / "release_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(self.loader.verify_bundle(self.root), manifest)
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.loader.verify_bundle(self.root)

    def test_held_out_assignment_is_not_replaced_by_an_arbitrary_fold(self):
        (self.root / "split.json").write_text(json.dumps({"folds": {"known": 3}}))
        self.assertEqual(self.loader.fold_for_uid("known", self.root), 3)
        with self.assertRaisesRegex(ValueError, "Unknown study UID"):
            self.loader.fold_for_uid("new", self.root)

    def test_invalid_fold_rejected_before_loading(self):
        for fold in [True, -1, 5, "0"]:
            with self.subTest(fold=fold), self.assertRaisesRegex(ValueError, "fold"):
                self.loader.load_fold(fold, directory=self.root)


if __name__ == "__main__":
    unittest.main()
