"""Verify a returned Kaggle run and publish small exhibits; no training/API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile

from final_eval.compressor.budget import TokenCounter
from final_eval.compressor.evaluate import PROMPT_TEMPLATE
from final_eval.inference.client import generation_config
from final_eval.utils.answer_matching import grade_answer, extract_final_answer, GRADER_VERSION
from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.utils.statistics import cluster_ci
from final_eval.v3.artifacts import read_csv, write_csv
from final_eval.v3.kaggle_bootstrap import relative
from final_eval.v3.reader import evaluation_tasks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(z, name):
    return json.loads(z.read("run/" + name))


def lines(z, name):
    return [json.loads(row) for row in z.read("run/" + name).decode("utf-8").splitlines() if row]


def verify_files(z):
    names = z.namelist()
    require(len(names) == len(set(names)), "Duplicate archive entries")
    for name in names:
        relative(name)
        require(not name.endswith(".env"), "Unexpected credential file")
    manifest = json.loads(z.read("manifest.json"))
    expected = {}
    for entry in manifest["files"]:
        name = "run/" + entry["path"]
        require(name not in expected, "Duplicate manifest path")
        result = hashlib.sha256()
        with z.open(name) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        require(result.hexdigest() == entry["sha256"], "Archive checksum mismatch: " + name)
        require(z.getinfo(name).file_size == entry["bytes"], "Archive size mismatch: " + name)
        expected[name] = entry["sha256"]
    config = read(z, "resolved_config.json")
    excluded = {e["path"] for e in manifest["excluded_large_resume_files"]}
    stages = []
    for name in names:
        if not name.startswith("run/") or not name.endswith(".stage.json"):
            continue
        state = json.loads(z.read(name))
        require(state["complete"], "Incomplete stage: " + name)
        require(digest(state["specification"]) == state["identity"], "Invalid stage identity: " + name)
        for source, checksum in state["specification"]["stage_sources"].items():
            archived = z.read("implementation/final_eval/" + source)
            require(hashlib.sha256(archived).hexdigest() == checksum, "Executed source mismatch: " + source)
        for path, checksum in state.get("outputs", {}).items():
            prefix = config["output_dir"].rstrip("/") + "/"
            require(path.startswith(prefix), "Stage output is outside the recorded run")
            rel = path[len(prefix):]
            if rel in excluded:
                continue
            require(expected.get("run/" + rel) == checksum, "Stage artifact mismatch: " + path)
        stages.append(name)
    return manifest, stages


def verify_families(questions, split):
    byuid = {q["uid"]: q for q in questions}
    require(len(byuid) == len(questions), "Duplicate question IDs")
    require(set(byuid) == set(split["roles"]), "Question/split IDs differ")
    groups = {r: set(v) for r, v in split["families"].items()}
    for a, b in itertools.combinations(groups, 2):
        require(not groups[a] & groups[b], "Family leakage between " + a + " and " + b)
    require(not groups["test"] & set(split["known_exposed_families"]), "Test overlaps known prior exposure")
    for q in questions:
        role = split["roles"][q["uid"]]
        require(q["role"] == role and q["base_group_id"] in groups[role], "Invalid question role")
    for role, uids in split["downstream"].items():
        require(len(uids) == len(set(uids)), "Duplicate downstream questions")
        require(len({byuid[u]["base_group_id"] for u in uids}) == len(uids), "Downstream family duplicated")
        require(all(byuid[u]["role"] == role for u in uids), "Wrong downstream role")
    return byuid


def verify_run(z):
    manifest, stages = verify_files(z)
    status = read(z, "runtime/v3_status.json")
    require(status["success"] and status["stage"] == "completed", "Kaggle execution did not finish")
    config, split, freeze = (read(z, n) for n in ("resolved_config.json", "data/split.json", "freeze.json"))
    byuid = verify_families(lines(z, "data/questions.jsonl"), split)
    for label in lines(z, "data/teacher.jsonl"):
        require(byuid[label["uid"]]["role"] in {"train", "val"}, "Teacher optimization includes dev/test")
        require(label["text_hash"] == digest(byuid[label["uid"]]["original_prompt"]), "Teacher text mismatch")
    require(digest(split["downstream"]["test"]) == freeze["test_uids_hash"], "Frozen test IDs differ")
    metadata = {arm: read(z, "models/" + arm + "/metadata.json") for arm in ("teacher_only", "teacher_cf")}
    for arm, meta in metadata.items():
        require(meta["initial_sha256"] == read(z, "initialization/source.json")["weights_sha256"], "Different initializations")
        require(meta["best_weights_sha256"] == freeze["checkpoints"][arm], "Frozen best checkpoint differs")
        require(meta["best_weights_sha256"] == next(e["sha256"] for e in manifest["files"]
                if e["path"] == "models/" + arm + "/best.pt"), "Checkpoint bytes differ")
        require(len(meta["history"]) == config["training"]["epochs"], "Training epochs incomplete")
    counter = TokenCounter(config["tokenizer"])
    identity = digest({"generation": generation_config(config["reader"]["model"]), "wrapper": PROMPT_TEMPLATE})
    public_rows, failures = [], []
    for role in ("dev", "test"):
        compressed = lines(z, role + "/compressed.jsonl")
        planned = evaluation_tasks(compressed)
        key = lambda r: (r["uid"], r["method"], r["target_ratio"])
        tasks = {key(r): r for r in planned}
        responses = lines(z, role + "/responses.jsonl")
        require(len(tasks) == len(planned) == len(responses), "Logical grid incomplete/duplicated")
        require(set(tasks) == {key(r) for r in responses}, "Reader/compression grid differs")
        require(set(r["uid"] for r in responses) == set(split["downstream"][role]), "Wrong reader cohort")
        summary = read(z, role + "/report/summary.json")
        require(summary["complete_reader_stage"] and summary["report_grader"] == GRADER_VERSION, "Grader/stage differs")
        grouped = defaultdict(list)
        for row in responses:
            task, q = tasks[key(row)], byuid[row["uid"]]
            require(task["original_prompt"] == q["original_prompt"] and task["ground_truth_answer"] == q["ground_truth_answer"], "Question metadata mismatch")
            require(counter(task["compressed_prompt"]) <= int(row["target_ratio"] * counter(q["original_prompt"])), "Token ceiling violated")
            if row.get("query_hash"):
                expected = digest({"reader": identity, "query": PROMPT_TEMPLATE.format(question=task["compressed_prompt"]), "replicate": None})
                require(row["query_hash"] == expected, "Query hash differs")
                cached = read(z, "reader/cache/" + expected + ".json")["generation"]
                require(row["response"] == cached["text"] and row["response_status"] == cached["status"], "Raw cached answer differs")
            grade = grade_answer(row.get("response", ""), q["ground_truth_answer"], q["original_prompt"])
            correct = row["response_status"] == "ok" and grade.correct
            require(bool(row["correct"]) == bool(correct), "Stored grading mismatch")
            report_row = {"role": role, "uid": q["uid"], "base_group_id": q["base_group_id"],
                "source_dataset": q["source_dataset"], "method": row["method"], "ratio": row["target_ratio"],
                "actual_ratio": row["actual_ratio"], "correct": bool(correct), "response_status": row["response_status"],
                "parse_status": row["parse_status"], "gold": q["ground_truth_answer"],
                "final_answer": extract_final_answer(row.get("response", ""))[:600]}
            public_rows.append(report_row)
            grouped[(row["method"], row["target_ratio"])].append(report_row)
            if row["response_status"] != "ok" or row["parse_status"] != "ok":
                failures.append(report_row)
        for cell in summary["accuracy"]:
            rr = grouped[(cell["method"], cell["ratio"])]
            require(len(rr) == cell["n"] and sum(r["correct"] for r in rr) == cell["correct"], "Reported score differs")
        if role == "dev":
            counts = {p: sum(r["correct"] for r in public_rows if r["role"] == "dev" and
                            r["method"] == "student_teacher_only_" + p) for p in config["policies"]}
            selected = max(counts, key=lambda p: (counts[p], -config["policies"].index(p)))
            require(selected == freeze["policy"], "Shared policy freeze differs")
        else:
            cf = {r["uid"]: r for r in grouped[("student_teacher_cf_" + freeze["policy"], config["primary_ratio"]) ]}
            control = {r["uid"]: r for r in grouped[("student_teacher_only_" + freeze["policy"], config["primary_ratio"]) ]}
            uids = sorted(cf)
            delta = [int(cf[u]["correct"]) - int(control[u]["correct"]) for u in uids]
            ci = cluster_ci(delta, [cf[u]["base_group_id"] for u in uids], config["bootstrap"], config["split"]["seed"])
            require(all(abs(a - b) < 1e-12 for a, b in zip(ci, (summary["primary"]["result"]["ci_low"], summary["primary"]["result"]["ci_high"]))), "Primary interval differs")
    cache_names = [n for n in z.namelist() if n.startswith("run/reader/cache/") and n.endswith(".json")]
    tokens = Counter()
    for name in cache_names:
        usage = json.loads(z.read(name))["generation"].get("usage") or {}
        if all(type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens")):
            tokens.update({k: usage[k] for k in ("prompt_tokens", "completion_tokens")})
    cost = read(z, "reader/cost.json")
    return {"passed": True, "manifest_files": len(manifest["files"]), "completed_stages": len(stages),
        "source_at_execution": manifest["source"], "family_leakage": False, "same_initialization": True,
        "token_ceilings_verified": True, "grades_recomputed": True, "primary_ci_recomputed": True,
        "logical_conditions": len(public_rows), "cache_entries": len(cache_names),
        "tokens_with_reported_usage": dict(tokens), "guard_accounted_inr": cost["accounted_inr"],
        "known_usage_estimate_inr": (tokens["prompt_tokens"] * config["reader"]["quote_input_inr_per_million"] +
                                     tokens["completion_tokens"] * config["reader"]["quote_output_inr_per_million"]) / 1e6,
        "unknown_attempts": cost["unknown_attempts"], "actual_invoice_reconciled": False,
        "new_api_calls": 0, "new_training": False,
        "resume_files_omitted": manifest["excluded_large_resume_files"],
        "caution": "Absolute GPU peaks include any resident process allocations; do not interpret as isolated model footprints."}, public_rows, failures


def extract_verified(z, destination):
    destination = Path(destination).resolve()
    for name in z.namelist():
        target = destination / relative(name)
        require(target.resolve().is_relative_to(destination), "Extraction outside destination")
        if target.exists():
            result = hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    result.update(block)
            require(file_hash(target) == result.hexdigest(), "Existing imported artifact differs: " + name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)


def publish(z, destination, audit, rows, failures):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for role in ("dev", "test"):
        for name in ("accuracy.csv", "paired.csv", "coverage.csv", "per_task.csv", "reader_usage.csv", "summary.json"):
            files[role + "/" + name] = z.read("run/" + role + "/report/" + name)
    for name in ("freeze.json", "benchmark.json", "reference_sources.json", "teacher_reference_source.json", "data/summary.json",
                 "data/split.json", "data/tokenization_audit.json", "initialization/source.json", "runtime/environment.json",
                 "models/teacher_only/metadata.json", "models/teacher_cf/metadata.json", "reader/cost.json"):
        files["metadata/" + name] = z.read("run/" + name)
    for name, raw in files.items():
        path = destination / name
        if path.exists():
            require(path.read_bytes() == raw, "Refusing to replace another published experiment")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    write_csv(destination / "per_question.csv", rows)
    write_csv(destination / "answer_failures.csv", failures)
    atomic_json(destination / "verification.json", audit)
    exposures = [read(z, name) for name in ("test_exposure.json", "dev_exposure.json", "reader/pilot_exposure.json")]
    atomic_json(destination / "exposure.json", {
        "exposed_uids": sorted({u for e in exposures for u in e["exposed_uids"]}),
        "exposed_families": sorted({g for e in exposures for g in e["exposed_families"]}),
        "problems_sha256": read(z, "data/prepare.stage.json")["specification"]["problems"],
        "archive_sha256": audit["archive_sha256"],
        "note": "Completed v3 dev/test/pilot. Future preparation automatically treats these as exposed; do not tune and call them fresh again."})
    figures(destination)
    return destination


def figures(destination):
    """Publication exhibits from verified CSVs, without new inference."""
    from decimal import Decimal, ROUND_HALF_UP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    destination = Path(destination)
    target = destination / "figures"
    target.mkdir(parents=True, exist_ok=True)
    rows = read_csv(destination / "test/accuracy.csv")
    labels = {"llmlingua2_budgeted": "LLMLingua-2",
              "student_teacher_cf_word": "V3 + criticality",
              "student_teacher_only_word": "Teacher-only student",
              "v2_grouped_hybrid": "V2 grouped hybrid", "v2_grouped_signal": "V2 cheap reference"}
    colors = ["#16804a", "#b23a3b", "#2874a6", "#a66716", "#7d5a9b"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for (method, label), color in zip(labels.items(), colors):
        rr = sorted((r for r in rows if r["method"] == method), key=lambda r: float(r["ratio"]))
        y = [float(r["accuracy"]) for r in rr]
        ax.errorbar([float(r["actual_ratio"]) for r in rr], y,
            yerr=[[v - float(r["ci_low"]) for r, v in zip(rr, y)],
                  [float(r["ci_high"]) - v for r, v in zip(rr, y)]],
            label=label, color=color, marker="o", linewidth=1.6, capsize=3, markersize=4)
    summary = json.loads((destination / "test/summary.json").read_text(encoding="utf-8"))
    original = summary["original"]["correct"] / summary["original"]["n"]
    ax.axhline(original, color="#666666", linestyle=":", linewidth=1.2, label="Original prompt")
    ax.set(xlabel="Question tokens retained (observed mean)", ylabel="Downstream accuracy",
           ylim=(0, 1.03), xlim=(.445, .77), title="64-family Qwen test")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower right", fontsize=8, framealpha=.9)
    fig.text(.11, .02, "Two declared ceilings: 50% and 75%. Bars: 95% family-bootstrap intervals.\nTechnical and parsing failures count as incorrect; one reader and one training seed.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(target / ("accuracy_retention." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)
    paired = [r for r in read_csv(destination / "test/paired.csv")
              if r["a"] == "student_teacher_cf_word" and r["b"] == "student_teacher_only_word"]
    paired.sort(key=lambda r: float(r["ratio"]))
    delta = [100 * float(r["delta"]) for r in paired]
    lo, hi = ([100 * float(r[k]) for r in paired] for k in ("ci_low", "ci_high"))
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.bar(range(len(paired)), delta, color=["#b23a3b", "#7d8b99"], width=.5)
    ax.errorbar(range(len(paired)), delta,
                yerr=[[v - l for v, l in zip(delta, lo)], [h - v for v, h in zip(delta, hi)]],
                fmt="none", color="#202020", capsize=6)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(paired)), ["50% retained\nPrimary", "75% retained\nSecondary"])
    ax.set(ylabel="Accuracy difference (percentage points)", title="Criticality supervision minus teacher-only",
           ylim=(min(lo) - 2, max(hi) + 4))
    for i, (value, high) in enumerate(zip(delta, hi)):
        label = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        ax.text(i, high + .8, f"{label:+.2f} pp", ha="center", fontsize=9)
    fig.text(.1, .015, "Same initialization, data and frozen selector. 64 paired families.\nNeither interval excludes zero; the primary gain is not established.", fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(target / ("criticality_ablation." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)


def audit_return(archive, status, destination, public):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        audit, rows, failures = verify_run(z)
        if status:
            outside = json.loads(Path(status).read_text(encoding="utf-8"))
            require(outside == read(z, "runtime/v3_status.json"), "External/internal status differ")
        audit["archive_sha256"] = file_hash(archive)
        extract_verified(z, destination)
        atomic_json(destination / "verification.json", audit)
        publish(z, public, audit, rows, failures)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--status")
    parser.add_argument("--directory", default="outputs/imports/v3_20261030_completed")
    parser.add_argument("--publish", default="results/v3/20261030")
    args = parser.parse_args()
    print(json.dumps(audit_return(args.archive, args.status, args.directory, args.publish), indent=2))


if __name__ == "__main__":
    main()
