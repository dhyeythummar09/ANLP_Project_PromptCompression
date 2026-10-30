"""Full-cohort estimates, fixed development selection, and explicit uncertainty."""
from __future__ import annotations

from collections import defaultdict
import itertools
from pathlib import Path

import numpy as np

from final_eval.inference.client import generation_config
from final_eval.utils.answer_matching import grade_answer, GRADER_VERSION
from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.utils.statistics import cluster_ci
from final_eval.v3.artifacts import (read_json, read_jsonl, write_csv, stage_manifest,
                                    finish_stage, log)
from final_eval.v3.train import ARMS


def regrade_saved(config, rows):
    """Reporting can correct grading without new generations or overwriting raw evidence."""
    questions = {q["uid"]: q for q in read_jsonl(Path(config["output_dir"]) / "data/questions.jsonl")}
    result = []
    for row in rows:
        q = questions[row["uid"]]
        grade = grade_answer(row.get("response", ""), q["ground_truth_answer"], q["original_prompt"])
        result.append({**row, "correct": bool(row["response_status"] == "ok" and grade.correct),
                       "parse_status": grade.status if row.get("response") else row.get("parse_status", "not_queried"),
                       "report_grader": GRADER_VERSION})
    return result


def freeze(config):
    run = Path(config["output_dir"])
    responses = run / "dev/responses.jsonl"
    if not read_json(run / "dev/reader.stage.json")["complete"]:
        raise ValueError("Finish the development reader grid before freezing")
    rows = regrade_saved(config, read_jsonl(responses))
    initial = file_hash(run / "initialization/common_initial.pt")
    for arm in ARMS:
        if read_json(run / "models" / arm / "metadata.json")["initial_sha256"] != initial:
            raise ValueError("Students did not share the recorded initialization")
    expected = len(read_json(run / "data/split.json")["downstream"]["dev"]) * len(config["ratios"])
    scores = []
    for policy in config["policies"]:
        chosen = [r for r in rows if r["method"] == f"student_teacher_only_{policy}"]
        if len(chosen) != expected:
            raise ValueError("Development policy grid is incomplete")
        scores.append({"policy": policy, "correct": sum(r["correct"] for r in chosen), "conditions": expected})
    policy = max(scores, key=lambda s: (s["correct"], -config["policies"].index(s["policy"])))["policy"]
    value = {"policy": policy, "scores": scores,
        "rule": "Choose shared policy by teacher-only full-cohort correct counts summed across declared ceilings; ties use config policy order. Apply the SAME policy to both arms.",
        "checkpoints": {a: file_hash(run / "models" / a / "best.pt") for a in ARMS},
        "reader": digest(generation_config(config["reader"]["model"])),
        "development_responses": file_hash(responses), "ratios": config["ratios"],
        "decision_grader": GRADER_VERSION,
        "test_uids_hash": digest(read_json(run / "data/split.json")["downstream"]["test"])}
    path = run / "freeze.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Frozen experiment differs; never retune an already frozen test run")
    atomic_json(path, value)
    log("freeze", selected_shared_policy=policy)


def summarize(config, role):
    run = Path(config["output_dir"])
    out = run / role
    compressed = read_jsonl(out / "compressed.jsonl")
    responses = regrade_saved(config, read_jsonl(out / "responses.jsonl"))
    observed = {(r["uid"], r["method"], r["target_ratio"]): r for r in responses}
    if len(observed) != len(responses):
        raise ValueError("Duplicate logical reader rows")
    originals = {r["uid"]: r for r in responses if r["method"] == "original"}
    grouped = defaultdict(list)
    for row in compressed:
        key = (row["uid"], row["method"], row["target_ratio"])
        response = observed.get(key, {"correct": False, "response_status": "missing", "parse_status": "not_queried"})
        grouped[(row["method"], row["target_ratio"])].append({**row, **response})
    accuracy, coverage, per_task, usage = [], [], [], []
    for (method, ratio), rr in sorted(grouped.items()):
        values = [int(r["correct"]) for r in rr]
        groups = [r["base_group_id"] for r in rr]
        ci = cluster_ci(values, groups, config["bootstrap"], config["split"]["seed"])
        eligible = [r for r in rr if originals.get(r["uid"], {}).get("correct", False)]
        accuracy.append({"method": method, "ratio": ratio, "n": len(rr), "families": len(set(groups)),
            "correct": sum(values), "accuracy": float(np.mean(values)), "ci_low": ci[0], "ci_high": ci[1],
            "actual_ratio": float(np.mean([r["actual_ratio"] for r in rr])),
            "conditional_preservation": sum(r["correct"] for r in eligible) / len(eligible) if eligible else None,
            "original_correct_n": len(eligible), "cohort": "all planned questions; technical/missing/parse failures wrong"})
        coverage.append({"method": method, "ratio": ratio, "n_planned": len(rr),
            "response_failures": sum(r["response_status"] != "ok" for r in rr),
            "parse_failures": sum(r.get("parse_status") != "ok" for r in rr),
            "budget_violations": sum(not r["budget_compliant"] for r in rr)})
        for source in sorted({r["source_dataset"] for r in rr}):
            tt = [r for r in rr if r["source_dataset"] == source]
            per_task.append({"method": method, "ratio": ratio, "source_dataset": source,
                             "n": len(tt), "correct": sum(r["correct"] for r in tt),
                             "accuracy": sum(r["correct"] for r in tt) / len(tt)})
        known = [r["usage"] for r in rr if isinstance(r.get("usage"), dict)
                 and type(r["usage"].get("prompt_tokens")) is int and type(r["usage"].get("completion_tokens")) is int]
        if known:
            usage.append({"method": method, "ratio": ratio, "n_with_usage": len(known),
                "mean_prompt_tokens": sum(r["prompt_tokens"] for r in known) / len(known),
                "mean_completion_tokens": sum(r["completion_tokens"] for r in known) / len(known),
                "mean_estimated_cost_inr": sum((r["prompt_tokens"] * config["reader"]["quote_input_inr_per_million"] +
                    r["completion_tokens"] * config["reader"]["quote_output_inr_per_million"]) / 1e6 for r in known) / len(known)})
    paired = []
    for ratio in config["ratios"]:
        methods = sorted(m for m, r in grouped if r == ratio)
        for a, b in itertools.combinations(methods, 2):
            aa = {r["uid"]: r for r in grouped[(a, ratio)]}
            bb = {r["uid"]: r for r in grouped[(b, ratio)]}
            if set(aa) != set(bb):
                raise ValueError("Methods do not share the planned cohort")
            uids = sorted(aa)
            delta = [int(aa[u]["correct"]) - int(bb[u]["correct"]) for u in uids]
            ci = cluster_ci(delta, [aa[u]["base_group_id"] for u in uids], config["bootstrap"], config["split"]["seed"])
            paired.append({"a": a, "b": b, "ratio": ratio, "n": len(uids),
                "delta": float(np.mean(delta)), "ci_low": ci[0], "ci_high": ci[1],
                "a_only_correct": sum(d == 1 for d in delta), "b_only_correct": sum(d == -1 for d in delta),
                "interpretation": "exploratory unadjusted interval; see predeclared primary comparison"})
    report = out / "report"
    report.mkdir(parents=True, exist_ok=True)
    from final_eval.v3.artifacts import write_jsonl
    write_jsonl(report / "regraded_responses.jsonl", responses)
    write_csv(report / "accuracy.csv", accuracy)
    write_csv(report / "paired.csv", paired)
    write_csv(report / "coverage.csv", coverage)
    write_csv(report / "per_task.csv", per_task)
    write_csv(report / "reader_usage.csv", usage)
    original_record = {"n": len(originals), "correct": sum(r["correct"] for r in originals.values()),
                       "technical_failures": sum(r["response_status"] != "ok" for r in originals.values())}
    original_usage = [r["usage"] for r in originals.values() if isinstance(r.get("usage"), dict)]
    if original_usage:
        usage.append({"method": "original", "ratio": 1.0, "n_with_usage": len(original_usage),
            "mean_prompt_tokens": sum(u["prompt_tokens"] for u in original_usage) / len(original_usage),
            "mean_completion_tokens": sum(u["completion_tokens"] for u in original_usage) / len(original_usage),
            "mean_estimated_cost_inr": sum((u["prompt_tokens"] * config["reader"]["quote_input_inr_per_million"] +
                u["completion_tokens"] * config["reader"]["quote_output_inr_per_million"]) / 1e6 for u in original_usage) / len(original_usage)})
        write_csv(report / "reader_usage.csv", usage)
    primary = {"comparison": "student_teacher_cf minus student_teacher_only, same frozen policy",
               "primary_ratio": config["primary_ratio"], "secondary_ratios": [r for r in config["ratios"] if r != config["primary_ratio"]],
               "multiple_testing": "Only the declared primary comparison is confirmatory; other comparisons/ratios exploratory."}
    if role == "test":
        policy = read_json(run / "freeze.json")["policy"]
        contrast = next((p for p in paired if p["a"] == f"student_teacher_cf_{policy}"
                         and p["b"] == f"student_teacher_only_{policy}" and p["ratio"] == config["primary_ratio"]), None)
        if contrast:
            primary["result"] = contrast
            primary["verdict"] = ("positive paired difference in this cohort" if contrast["ci_low"] > 0 else
                                  "negative paired difference in this cohort" if contrast["ci_high"] < 0 else
                                  "no established primary accuracy difference")
            primary["qualification"] = "Inspect coverage; one cohort/reader does not establish universal improvement or equivalence."
    atomic_json(report / "summary.json", {"role": role, "original": original_record, "accuracy": accuracy,
        "report_grader": GRADER_VERSION,
        "primary": primary, "paired": paired, "failures": coverage,
        "complete_reader_stage": read_json(out / "reader.stage.json").get("complete", False),
        "claims": "No guaranteed improvement, necessity, equivalence, speedup or novelty follows from this implementation."})
    log("report", role=role, original=original_record)
