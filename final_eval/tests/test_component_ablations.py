import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from final_eval.component_ablations import assemble, prepare, summarize
from final_eval.followup_reader import select_pilot_cohort


def _rows(methods):
    rows = []
    for uid in ("q1", "q2"):
        for ratio in (0.5, 0.75):
            for method in methods:
                rows.append({
                    "uid": uid,
                    "method": method,
                    "target_ratio": ratio,
                    "actual_ratio": ratio,
                    "original_prompt": f"{uid} prompt",
                    "ground_truth_answer": "5",
                    "base_group_id": uid,
                    "status": "ok",
                    "budget_compliant": True,
                })
    return pd.DataFrame(rows)


class ComponentAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base_comp = self.root / "base_comp.csv"
        self.abl_comp = self.root / "abl_comp.csv"
        self.base_res = self.root / "base_res.csv"
        self.abl_res = self.root / "abl_res.csv"
        _rows({"ours"}).to_csv(self.base_comp, index=False)
        _rows({"ours_scorer", "ours_signal"}).to_csv(self.abl_comp, index=False)
        base_resp = _rows({"ours"}).assign(eval_model="reader", response_status="ok", correct=True)
        none_rows = pd.DataFrame([{
            "uid": uid,
            "method": "none",
            "target_ratio": 1.0,
            "actual_ratio": 1.0,
            "original_prompt": f"{uid} prompt",
            "ground_truth_answer": "5",
            "base_group_id": uid,
            "status": "ok",
            "budget_compliant": True,
        } for uid in ("q1", "q2")])
        pd.concat([base_resp, none_rows.assign(eval_model="reader", response_status="ok", correct=True)],
                  ignore_index=True).to_csv(self.base_res, index=False)
        _rows({"ours_scorer", "ours_signal"}).assign(
            eval_model="reader", response_status="ok", correct=True).to_csv(self.abl_res, index=False)
        config = {
            "backend": "hosted",
            "name": "reader",
            "base_url": "https://models.jarvislabs.net/v1",
            "api_key_env": "JARVISLABS_API_KEY",
            "temperature": 0,
            "max_tokens": 4096,
        }
        for csv_path in (self.base_res, self.abl_res):
            Path(str(csv_path) + ".manifest.json").write_text(json.dumps({
                "config": config,
                "wrapper": "Solve this problem...",
            }))

    def test_assemble_creates_comparison_ready_outputs(self):
        out_comp = self.root / "out_comp.csv"
        out_res = self.root / "out_res.csv"
        args = SimpleNamespace(
            baseline_compressed=str(self.base_comp),
            baseline_responses=str(self.base_res),
            ablation_compressed=str(self.abl_comp),
            ablation_responses=str(self.abl_res),
            output_compressed=str(out_comp),
            output_responses=str(out_res),
        )
        assemble(args)
        comp = pd.read_csv(out_comp, keep_default_na=False)
        res = pd.read_csv(out_res, keep_default_na=False)
        self.assertEqual(set(comp.method), {"ours", "ours_scorer", "ours_signal"})
        self.assertEqual(set(res.method), {"none", "ours", "ours_scorer", "ours_signal"})
        manifest = json.loads(Path(str(out_res) + ".manifest.json").read_text())
        self.assertEqual(manifest["methods"], ["ours", "ours_scorer", "ours_signal"])

    def test_mismatched_generation_config_is_rejected(self):
        broken = json.loads(Path(str(self.abl_res) + ".manifest.json").read_text())
        broken["config"]["max_tokens"] = 1024
        Path(str(self.abl_res) + ".manifest.json").write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "generation settings differ"):
            assemble(SimpleNamespace(
                baseline_compressed=str(self.base_comp),
                baseline_responses=str(self.base_res),
                ablation_compressed=str(self.abl_comp),
                ablation_responses=str(self.abl_res),
                output_compressed=str(self.root / "x.csv"),
                output_responses=str(self.root / "y.csv"),
            ))


def _arm(methods, texts, root, name, alpha=0.5):
    """A compression table plus the neural.csv manifest that `prepare` audits."""
    rows = []
    for uid in ("q1", "q2", "q3"):
        for ratio in (0.5, 0.75):
            for method in methods:
                rows.append({"uid": uid, "method": method, "target_ratio": ratio,
                             "actual_ratio": ratio, "status": "ok", "budget_compliant": True,
                             "compressed_prompt": texts(uid, method, ratio),
                             "original_prompt": f"{uid} prompt", "ground_truth_answer": "5",
                             "base_group_id": f"family_{uid}", "source_dataset": "gsm8k"})
    folder = root / name
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "compressed.csv", index=False)
    (folder / "neural.csv.manifest.json").write_text(json.dumps({"arguments": {
        "input_csv": "x", "output_csv": "y", "methods": methods, "alpha": alpha,
        "model_dir": "models/v2", "no_bonuses": False}}))
    return str(folder / "compressed.csv")


class MatchedAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # The scorer-only arm copies `ours` on q1, so that pair can only tie.
        text = lambda uid, method, ratio: (
            f"{uid}-ours-{ratio}" if method == "ours" or (method == "ours_scorer" and uid == "q1")
            else f"{uid}-{method}-{ratio}")
        self.reference = _arm(["ours"], text, self.root, "reference")
        self.ablation = _arm(["ours_scorer", "ours_signal"], text, self.root, "ablation")

    def prepare(self, **extra):
        output = self.root / "prepared.csv"
        prepare(SimpleNamespace(reference_input=self.reference, ablation_input=self.ablation,
                                saved_reference=None, ratios=[0.5, 0.75], output=str(output),
                                **extra))
        return output

    def test_prepare_stacks_arms_and_counts_identical_text(self):
        output = self.prepare()
        prepared = pd.read_csv(output)
        self.assertEqual(set(prepared.method), {"ours", "ours_scorer", "ours_signal"})
        self.assertEqual(len(prepared), 18)
        manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
        self.assertEqual(manifest["identical_compressed_text"]["ours_scorer==ours@0.5"], 1)
        self.assertEqual(manifest["logical_conditions_including_originals"], 21)

    def test_prepare_rejects_different_compressor_settings(self):
        self.ablation = _arm(["ours_scorer", "ours_signal"], lambda *a: "t", self.root,
                             "other_alpha", alpha=0.7)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.prepare()

    def test_summarize_reports_paired_contrasts_on_common_questions(self):
        output = self.prepare()
        prepared = pd.read_csv(output, keep_default_na=False)
        # Scorer-only is right on every question, ours only on q1, signal never.
        verdict = {"ours": {"q1"}, "ours_scorer": {"q1", "q2", "q3"}, "ours_signal": set()}
        rows = [{"uid": uid, "eval_model": "reader", "method": "none", "target_ratio": 1.0,
                 "response_status": "ok", "correct": True} for uid in ("q1", "q2", "q3")]
        rows += [{"uid": r.uid, "eval_model": "reader", "method": r.method,
                  "target_ratio": r.target_ratio, "response_status": "ok",
                  "correct": r.uid in verdict[r.method]} for r in prepared.itertuples()]
        pd.DataFrame(rows).to_csv(self.root / "responses.csv", index=False)
        report = self.root / "report.json"
        summarize(SimpleNamespace(input=str(output), responses=str(self.root / "responses.csv"),
                                  report=str(report), bootstrap=100))
        result = json.loads(report.read_text())
        half = [c for c in result["contrasts"] if c["target_ratio"] == 0.5]
        scorer_vs_ours = next(c for c in half if c["focus"] == "ours_scorer" and c["other"] == "ours")
        self.assertAlmostEqual(scorer_vs_ours["difference"], 2 / 3)
        self.assertEqual(scorer_vs_ours["only_focus_correct"], 2)
        self.assertEqual(scorer_vs_ours["identical_text_pairs"], 1)
        self.assertTrue((self.root / "report.per_dataset.csv").exists())


class PilotCohortTests(unittest.TestCase):
    def test_pilot_avoids_main_families_and_takes_one_question_per_family(self):
        source = pd.DataFrame({
            "uid": [f"u{i}" for i in range(8)],
            "base_group_id": ["a", "a", "b", "c", "d", "e", "f", "g"],
            "source_dataset": ["x", "x", "x", "y", "y", "x", "y", "y"]})
        main = pd.DataFrame({"base_group_id": ["b", "g"]})
        pilot = select_pilot_cohort(source, main, n=4, seed=1)
        self.assertEqual(len(pilot), 4)
        self.assertFalse(set(pilot.base_group_id) & {"b", "g"})
        self.assertTrue(pilot.base_group_id.is_unique)
        self.assertEqual(pilot.source_dataset.value_counts().to_dict(), {"x": 2, "y": 2})
        with self.assertRaises(ValueError):
            select_pilot_cohort(source, main, n=6, seed=1)


if __name__ == "__main__":
    unittest.main()
