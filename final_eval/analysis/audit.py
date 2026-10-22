"""Verify the recorded final study and regenerate tables offline, without models/APIs."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from final_eval.analysis import agreement, feature_baseline, strict_results
from final_eval.compressor import evaluate
from final_eval.preparation.schema import POSITION
from final_eval.utils.answer_matching import grade_answer
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, digest, file_hash, parse_bool, require_unique, source_fingerprint

BASE = Path("outputs/final_eval/main_gptoss_v1")
MODEL = Path("models/span_scorer_v2_20261021")
PREPARED = Path("data/processed/team_v2_20261021")
READERS = {
    "gptoss": (BASE / "target_gptoss", BASE / "reports_gptoss_v1", "target.gptoss.main.json", ("10", "40")),
    "qwen": (Path("outputs/final_eval/main_qwen8192_v1/target_qwen"),
             Path("outputs/final_eval/main_qwen8192_v1/reports_qwen_completecase_v1"),
             "target.qwen.main8192.json", ("28.431", "222.709")),
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def compare_tables(actual, saved, keys):
    left = pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    right = pd.read_csv(saved).sort_values(keys).reset_index(drop=True)
    left = left.reindex(sorted(left.columns), axis=1)
    right = right.reindex(sorted(right.columns), axis=1)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def audit_responses(comp, frame, manifest):
    """Reject drift in grids, query hashes, golds, family IDs, model IDs or grades."""
    require_unique(frame, evaluate.KEY, "responses")
    name = manifest["config"]["name"]
    tasks = {(t["uid"], t["method"], t["target_ratio"]): t for t in evaluate.compression_tasks(comp)}
    check(set(frame.eval_model) == {name}, "Unexpected answering-model identity")
    check(set(frame[["uid", "method", "target_ratio"]].itertuples(index=False, name=None)) == set(tasks), "Response grid differs from planned conditions")
    check(manifest["wrapper"] == evaluate.PROMPT_TEMPLATE, "Saved wrapper differs from the recorded evaluator")
    for row in frame.to_dict("records"):
        task = tasks[(row["uid"], row["method"], row["target_ratio"])]
        query = manifest["wrapper"].format(question=task["compressed_prompt"])
        check(row["query_hash"] == digest([manifest["config"], query]), f"Query hash mismatch: {row['uid']}/{row['method']}")
        check(str(row["ground_truth"]) == str(task["ground_truth_answer"]), f"Gold mismatch: {row['uid']}")
        for column in ("base_group_id", "source_dataset"):
            check(row[column] == task[column], f"{column} mismatch: {row['uid']}")
        check(row.get("generation_requested_model") == name and row.get("generation_served_model") == name,
              f"Requested/served model mismatch: {row['uid']}")
        grade = grade_answer(row["response"], task["ground_truth_answer"], task["original_prompt"])
        check(row["parse_status"] == grade.status, f"Parse status mismatch: {row['uid']}/{row['method']}")
        if row["response_status"] == "ok":
            check(parse_bool(row["correct"]) == grade.correct, f"Saved grade mismatch: {row['uid']}/{row['method']}")
        else:
            check(row["correct"] == "" or not parse_bool(row["correct"]), "Provider failure was graded correct")
    originals = frame[frame.method == "none"]
    return {"conditions": len(frame), "original_correct": int(originals.correct.map(parse_bool).sum()),
            "statuses": frame.response_status.value_counts().to_dict(), "parse_statuses": frame.parse_status.value_counts().to_dict(),
            "query_gold_model_grade_mismatches": 0}


def returned_usage(frame, archived, rates):
    seen = set(); tokens_in = tokens_out = active = old = 0
    for row in frame.to_dict("records"):
        if parse_bool(row["reused_generation"]) or not row.get("generation_usage"):
            continue
        identity = (row["query_hash"], row["generation_timestamp_utc"])
        if identity in seen:
            continue
        seen.add(identity)
        usage = json.loads(row["generation_usage"])
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        active += 1
    for row in archived.get("attempts", []):
        identity = (row["query_hash"], row["timestamp_utc"])
        if identity in seen:
            continue
        seen.add(identity); usage = row.get("usage") or {}
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        old += 1
    cost = (tokens_in * Decimal(rates[0]) + tokens_out * Decimal(rates[1])) / 1_000_000
    return {"active_usage_records": active, "archived_usage_records": old,
            "prompt_tokens": tokens_in, "completion_tokens": tokens_out, "estimated_inr": str(cost)}


def run(output_dir, retokenize=False, checkpoints=False):
    target = Path(output_dir); claim_output(target); target.mkdir(parents=True)
    evidence = {}
    def load(path):
        path = Path(path); evidence[path.as_posix()] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        return json.loads(path.read_text(encoding="utf-8"))
    def table(path, **kwargs):
        path = Path(path); evidence[path.as_posix()] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        return pd.read_csv(path, keep_default_na=False, **kwargs)

    comp = table(BASE / "compressed.csv", dtype={"ground_truth_answer": str})
    cohort = table(BASE / "cohort.csv", dtype={"ground_truth_answer": str})
    pilot = table("outputs/final_eval/pilot/cohort.csv")
    require_unique(comp, ["uid", "method", "target_ratio"], "compression")
    check(len(comp) == 2240 and len(cohort) == 160 and cohort.base_group_id.nunique() == 125, "Unexpected main cohort/grid size")
    methods = set(comp.method)
    expected = {(u, m, r) for u in cohort.uid for m in methods for r in (.5, .75)}
    check(len(methods) == 7 and set(comp[["uid", "method", "target_ratio"]].itertuples(index=False, name=None)) == expected, "Incomplete seven-method grid")
    check(not (set(cohort.base_group_id) & set(pilot.base_group_id)), "Pilot/main families overlap")
    check(comp.status.eq("ok").all() and comp.budget_compliant.map(parse_bool).all(), "Failed/over-budget compression")
    check((comp.comp_tokens <= comp.budget_tokens).all(), "Stored output token count exceeds budget")
    check((comp.budget_tokens == np.floor(comp.orig_tokens * comp.target_ratio)).all(), "Incorrect integer budget")
    check(np.allclose(comp.actual_ratio, comp.comp_tokens / comp.orig_tokens, rtol=0, atol=1e-12), "Incorrect actual retention ratios")
    if retokenize:
        import tiktoken
        check(comp.tokenizer.nunique() == 1, "Mixed accounting tokenizers")
        encoder = tiktoken.get_encoding(comp.tokenizer.iloc[0]); cache = {}
        def count(text):
            if text not in cache:
                cache[text] = len(encoder.encode(text))
            return cache[text]
        for row in comp.itertuples():
            check(count(row.original_prompt) == row.orig_tokens and count(row.compressed_prompt) == row.comp_tokens,
                  f"Token recount mismatch: {row.uid}/{row.method}/{row.target_ratio}")

    preparation = load(PREPARED / "manifest.json")
    for name, expected_sha in preparation["inputs"].items():
        name = name.replace("\\", "/")
        check(file_hash(name) == expected_sha, f"Preparation input changed: {name}")
    metadata = load(MODEL / "training_metadata.json"); split = load(MODEL / "split.json")
    check(file_hash(PREPARED / "consensus_dataset.csv") == metadata["consensus_sha256"], "Training input changed")
    problems = table(PREPARED / "problems.csv", usecols=["uid", "base_group_id", "source_dataset", "original_prompt", "ground_truth_answer"], dtype={"ground_truth_answer": str})
    by_uid = problems.set_index("uid")
    for row in cohort.itertuples():
        for name in ("base_group_id", "original_prompt", "ground_truth_answer", "source_dataset"):
            check(str(getattr(row, name)) == str(by_uid.loc[row.uid, name]), f"Cohort metadata drift: {row.uid}/{name}")
    for roles in split["partitions"].values():
        groups = [set(by_uid.loc[roles[role], "base_group_id"]) for role in ("train", "val", "test")]
        check(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "Training family leakage")
    shared = table("outputs/final_eval/feature_baseline/shared_oof.csv")
    feature_report = load("outputs/final_eval/feature_baseline/feature_report.json")
    scorer_metrics = {}
    for name, column in [("neural", "prob"), ("feature_ridge", "feature_prob"), ("category_prior", "prior_prob")]:
        metrics = feature_baseline.metrics(shared.eval_label, shared.target, shared[column])
        for key, value in metrics.items():
            stored = feature_report["shared_metrics"][name][key]
            check(value == stored if value is None or stored is None else abs(value-stored) < 1e-12, f"Scorer metric changed: {name}/{key}")
        scorer_metrics[name] = metrics
    oof = table(MODEL / "scorer_oof_predictions.csv")
    require_unique(oof, POSITION, "scorer OOF")
    check(len(oof) == len(shared) == 13855, "Unexpected shared scored support")
    for row in oof.itertuples():
        check(row.uid in split["partitions"][str(row.fold)]["test"], "OOF prediction is not held out")

    readers = {}; costs = []; response_files = []
    for name, (folder, saved_reports, config_name, rates) in READERS.items():
        response_path = folder / "responses.csv"; response_files.append(str(response_path))
        responses = table(response_path, dtype={"ground_truth": str})
        manifest = load(str(response_path) + ".manifest.json")
        check(manifest["input"] == file_hash(BASE / "compressed.csv"), "Reader compression version mismatch")
        readers[name] = audit_responses(comp, responses, manifest)
        history = load(folder / "archived_attempts.json")
        usage = returned_usage(responses, history, rates)
        saved_check = load(folder / "check.json")
        check(abs(float(usage["estimated_inr"]) - saved_check["returned_usage_cost_estimate_inr"]) < 1e-8, "Main cost estimate mismatch")
        usage["stage"] = name + "_main"; usage["http_attempts"] = load(str(response_path) + ".usage.json")["attempts"]
        costs.append(usage); readers[name]["usage"] = usage
        regenerated = target / (name + "_common")
        args = SimpleNamespace(input_csv=str(BASE / "compressed.csv"), output_csv=str(response_path), report_dir=str(regenerated),
                               methods=None, ratios=None, bootstrap=1000, focus="grouped_hybrid")
        with redirect_stdout(StringIO()):
            evaluate.summarize(args)
        for filename, keys in [("accuracy.csv", ["eval_model", "target_ratio", "method", "source_dataset"]),
                               ("paired.csv", ["eval_model", "target_ratio", "other"]),
                               ("coverage.csv", ["eval_model", "target_ratio", "method"])]:
            compare_tables(regenerated / filename, saved_reports / filename, keys)

    strict = target / "full_cohort"
    with redirect_stdout(StringIO()):
        strict_results.run(BASE / "compressed.csv", response_files, strict, bootstrap=1000)
    for filename, keys in [("accuracy.csv", ["eval_model", "target_ratio", "method", "source_dataset"]),
                           ("paired.csv", ["eval_model", "target_ratio", "other"]),
                           ("coverage.csv", ["eval_model", "target_ratio", "method"]),
                           ("macro_accuracy.csv", ["eval_model", "target_ratio", "method"])]:
        compare_tables(strict / filename, Path("outputs/final_eval/two_reader_strict_v1") / filename, keys)

    agreement_output = target / "agreement"
    agreement.run(PREPARED / "per_model_labels.csv", agreement_output)
    for filename, keys in [("pairwise.csv", ["source_dataset", "category", "model_a", "model_b"]),
                           ("coverage.csv", ["labeling_model", "source_dataset", "vote_status"]),
                           ("prevalence.csv", ["labeling_model", "source_dataset", "span_category"])]:
        compare_tables(agreement_output / filename, Path("outputs/offline_preparation_20261021/agreement") / filename, keys)

    auxiliary = [
        ("deepseek_access", "outputs/final_eval/jarvis/access.deepseek.json.summary.json", "response_usage_cost_estimate_inr"),
        ("gptoss_access", "outputs/final_eval/jarvis/access.gptoss.json.summary.json", "response_usage_cost_estimate_inr"),
        ("qwen_access", "outputs/final_eval/jarvis/access.qwen.json.summary.json", "response_usage_cost_estimate_inr"),
        ("deepseek_pilot_failed", "outputs/final_eval/target_pilots/deepseek/review_summary.json", "usage_cost_estimate_inr"),
        ("deepseek_calibration_failed", "outputs/final_eval/target_calibration/deepseek_v1/summary.json", "returned_usage_cost_estimate_inr"),
        ("gptoss_fallback", "outputs/final_eval/target_calibration/gptoss_fallback_v1/summary.json", "returned_usage_cost_estimate_inr"),
        ("gptoss_pilot", "outputs/final_eval/target_pilots/gptoss/review_summary.json", "returned_usage_cost_estimate_inr"),
        ("qwen_4096_pilot_failed", "outputs/final_eval/target_pilots/qwen/usage_summary.json", "returned_usage_cost_estimate_inr"),
        ("qwen_cap_check", "outputs/final_eval/target_calibration/qwen_cap8192_v1/summary.json", "returned_usage_cost_estimate_inr"),
        ("qwen_8192_pilot", "outputs/final_eval/target_pilots/qwen_8192/usage_summary.json", "returned_usage_cost_estimate_inr"),
    ]
    for name, path, key in auxiliary:
        data = load(path)
        costs.append({"stage": name, "estimated_inr": str(data[key]), "http_attempts": data.get("http_attempts", data.get("total_http_attempts")),
                      "evidence": path})
    total = sum(Decimal(str(row["estimated_inr"])) for row in costs).quantize(Decimal("0.000000001"))
    atomic_csv(pd.DataFrame(costs), target / "compute_estimates.csv")
    weights = {}
    if checkpoints:
        imported = load("outputs/final_eval/main_gptoss_v1/local_model_backup_check.json")
        for path in sorted(MODEL.glob("fold*/scorer.pt")):
            weights[path.as_posix()] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        check(len(weights) == 5, "Missing scorer checkpoint")
        # Original import manifests remain the authoritative hashes; do not deserialize tensors.
        saved = load("outputs/imports/stage_a3_20261021/RESULTS_MANIFEST.json")
        for name, info in weights.items():
            check(info["sha256"] == saved["files"][name], f"Checkpoint changed: {name}")
    report = {"passed": True, "new_inference_calls": 0, "loaded_model_weights": False,
              "core_source_fingerprint": source_fingerprint(), "compression_rows": len(comp), "questions": len(cohort),
              "main_families": cohort.base_group_id.nunique(), "pilot_families": pilot.base_group_id.nunique(),
              "retokenized": retokenize, "scorer_metrics": scorer_metrics, "readers": readers,
              "common_and_full_cohort_tables_reproduced": True, "agreement_tables_reproduced": True,
              "compute_estimate_inr": str(total), "actual_provider_bill_reconciled": False,
              "weights": weights, "evidence": evidence,
              "scope": "Saved-artifact verification, not proof that fresh API generations or retraining reproduce identical outputs."}
    atomic_json(target / "verification.json", report)
    print(f"PASS: {len(comp)} compressed prompts, 4,800 response grades/provenance, scorer metrics and saved tables verified. Estimate INR {total}; actual bill remains pending.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New directory; original results are never overwritten")
    parser.add_argument("--retokenize", action="store_true", help="Recount cl100k tokens; first use may download the small public encoding")
    parser.add_argument("--checkpoints", action="store_true", help="SHA256 the five checkpoints without loading tensors")
    args = parser.parse_args()
    run(args.output_dir, args.retokenize, args.checkpoints)


if __name__ == "__main__":
    main()
