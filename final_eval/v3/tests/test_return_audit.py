"""Reject corrupted artifacts and leaked family metadata before publication."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from final_eval.v3.audit_return import verify_files, verify_families
from final_eval.utils.artifacts import digest


class ReturnAuditTests(unittest.TestCase):
    def archive(self, corrupted=False, wrong_source=False, unsafe=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "fixture.zip"
        payload = {"resolved_config.json": json.dumps({"output_dir": "/run"}).encode(), "sample.txt": b"original"}
        if wrong_source:
            spec = {"stage_sources": {"v3/example.py": hashlib.sha256(b"executed").hexdigest()}}
            payload["example.stage.json"] = json.dumps({"complete": True, "specification": spec,
                "identity": digest(spec), "outputs": {"/run/sample.txt": hashlib.sha256(payload["sample.txt"]).hexdigest()}}).encode()
        manifest = {"files": [{"path": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                              for n, b in payload.items()], "excluded_large_resume_files": []}
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in payload.items():
                archive.writestr("run/" + name, b"modified" if corrupted and name == "sample.txt" else raw)
            archive.writestr("manifest.json", json.dumps(manifest))
            if wrong_source:
                archive.writestr("implementation/final_eval/v3/example.py", b"different implementation")
            if unsafe:
                archive.writestr("../escape", b"never extract")
        return path

    def test_valid_manifest_passes(self):
        with zipfile.ZipFile(self.archive()) as archive:
            manifest, _ = verify_files(archive)
        self.assertEqual(len(manifest["files"]), 2)

    def test_corrupted_bytes_do_not_pass_manifest(self):
        with zipfile.ZipFile(self.archive(corrupted=True)) as archive:
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_files(archive)

    def test_recorded_execution_source_must_match(self):
        with zipfile.ZipFile(self.archive(wrong_source=True)) as archive:
            with self.assertRaisesRegex(ValueError, "Executed source"):
                verify_files(archive)

    def test_unsafe_archive_member_is_rejected(self):
        with zipfile.ZipFile(self.archive(unsafe=True)) as archive:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_files(archive)

    def test_family_or_prior_exposure_overlap_is_rejected(self):
        questions = [{"uid": "a", "role": "train", "base_group_id": "one"},
                     {"uid": "b", "role": "test", "base_group_id": "two"}]
        split = {"roles": {"a": "train", "b": "test"}, "families": {"train": ["one"], "test": ["two"]},
                 "known_exposed_families": [], "downstream": {"test": ["b"]}}
        self.assertEqual(set(verify_families(questions, split)), {"a", "b"})
        split["known_exposed_families"] = ["two"]
        with self.assertRaisesRegex(ValueError, "prior exposure"):
            verify_families(questions, split)
        split["known_exposed_families"] = []
        split["families"]["test"].append("one")
        with self.assertRaisesRegex(ValueError, "Family leakage"):
            verify_families(questions, split)

    def test_imported_published_exposures_cannot_become_a_new_fresh_test(self):
        from final_eval.v3.tests.test_v3 import fixture
        from final_eval.v3.data import prepare
        from final_eval.v3.artifacts import read_json
        from final_eval.utils.artifacts import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = fixture(root)
            ledger = root / "results/v3/exposure.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(json.dumps({"problems_sha256": file_hash(config["problems"]),
                                           "exposed_uids": ["math|q2", "math|q4"]}))
            prepare(config)
            split = read_json(Path(config["output_dir"]) / "data/split.json")
            self.assertFalse({"family1", "family2"} & set(split["families"]["test"]))
            self.assertIn("family1", split["known_exposed_families"])


if __name__ == "__main__":
    unittest.main()
