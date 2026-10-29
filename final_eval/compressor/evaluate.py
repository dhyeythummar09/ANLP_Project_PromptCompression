"""Resumable target evaluation and paired, family-clustered reporting.

Run uses an explicit backend configuration. Dry-run never initializes a model,
reads a key, or contacts a provider. Legacy response CSVs need explicit regrading.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_eval.utils.answer_matching import grade_answer, GRADER_VERSION
from final_eval.utils.artifacts import source_fingerprint, package_versions, digest, file_hash, resume_manifest, require_unique, parse_bool, atomic_csv, atomic_json, claim_output
from final_eval.utils.statistics import cluster_ci
from final_eval.inference.client import make_client, public_config, generation_config
from final_eval.jarvis_spend import TokenCostGuard

PROMPT_TEMPLATE = "Solve this problem. You may reason briefly, but you MUST end your response with a line in exactly this format:\nFinal Answer: <your answer>\n\nProblem: {question}\n"
KEY = ["uid", "eval_model", "method", "target_ratio"]

def compression_tasks(comp):
    require_unique(comp, ["uid", "method", "target_ratio"], "compressed prompts")
    tasks = []
    for _, g in comp.groupby("uid", sort=False):
        for col in ["original_prompt", "ground_truth_answer", "base_group_id", "source_dataset"]:
            if g[col].nunique(dropna=False) != 1:
                raise ValueError(f"Conflicting {col} for one uid")
        r = g.iloc[0].to_dict()
        tasks.append({**r, "method": "none", "target_ratio": 1.0, "actual_ratio": 1.0,
                      "compressed_prompt": r["original_prompt"], "status": "ok"})
    tasks.extend(comp.to_dict("records"))
    return tasks

def run_eval(args):
    if not args.model_config:
        raise ValueError("--model-config is required; confirm endpoint/model before generation")
    cfg = public_config(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    name = cfg.get("name") or cfg.get("model")
    comp = pd.read_csv(args.input_csv, keep_default_na=False, dtype={"ground_truth_answer": str})
    if "base_group_id" not in comp:
        raise ValueError("Use prepared v2 compression outputs with base_group_id")
    if getattr(args,"pilot_check",None):
        pilot=json.loads(Path(args.pilot_check).read_text())
        if not pilot.get("passed") or pilot.get("target_config_hash")!=digest(generation_config(cfg)):
            raise ValueError("Main inference requires a passed pilot with identical target configuration")
        if set(comp.base_group_id)&set(pilot["pilot_base_groups"]):
            raise ValueError("Pilot families overlap the main cohort")
    if args.methods:
        comp = comp[comp.method.isin(args.methods)]
    if args.ratios:
        comp = comp[comp.target_ratio.isin(args.ratios)]
    if args.limit:
        comp = comp[comp.uid.isin(comp.uid.drop_duplicates().head(args.limit))]
    tasks = compression_tasks(comp)
    print(json.dumps({"backend": cfg.get("backend", "hosted"), "model": name,
                      "questions": comp.uid.nunique(), "planned_conditions": len(tasks),
                      "generation_conditions": sum(t.get("status", "ok") == "ok" for t in tasks),
                      "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return
    manifest_hash = resume_manifest(args.output_csv, {"stage": "evaluation", "input": file_hash(args.input_csv),
        "source": source_fingerprint(), "packages": package_versions(), "config": generation_config(cfg), "grader": GRADER_VERSION, "wrapper": PROMPT_TEMPLATE,
        "methods": args.methods, "ratios": args.ratios, "limit": args.limit,
        "pilot_check":file_hash(args.pilot_check) if getattr(args,"pilot_check",None) else None})
    previous = pd.read_csv(args.output_csv, keep_default_na=False) if Path(args.output_csv).exists() else pd.DataFrame()
    if not previous.empty:
        require_unique(previous, KEY, "evaluation")
    existing = {tuple(r[k] for k in KEY): r for r in previous.to_dict("records")}
    query_cache = {r.get("query_hash"): r for r in existing.values() if r.get("response_status") == "ok"}
    client = None
    for task in tasks:
        key = (task["uid"], name, task["method"], task["target_ratio"])
        old = existing.get(key)
        if old and (not args.retry_failed or old.get("response_status") == "ok"):
            continue
        query = PROMPT_TEMPLATE.format(question=task["compressed_prompt"])
        query_hash = digest([generation_config(cfg), query])
        record = {"uid": task["uid"], "example_id": task.get("example_id", ""),
                  "source_dataset": task["source_dataset"], "base_group_id": task["base_group_id"],
                  "eval_model": name, "method": task["method"], "target_ratio": task["target_ratio"],
                  "actual_ratio": task.get("actual_ratio", ""), "ground_truth": task["ground_truth_answer"],
                  "query_hash": query_hash, "run_hash": manifest_hash, "grader_version": GRADER_VERSION}
        if task.get("status", "ok") != "ok":
            record.update(response_status="compression_" + task["status"], correct="", response="", parse_status="not_queried")
        elif query_hash in query_cache:
            cached = query_cache[query_hash]
            # Copy only the response, never cached correctness tied to another problem.
            grade = grade_answer(cached["response"], task["ground_truth_answer"], task["original_prompt"])
            record.update({k: v for k, v in cached.items() if k.startswith("generation_")})
            record.update(response_status="ok", response=cached["response"], answer=grade.answer,
                          correct=grade.correct, parse_status=grade.status, reused_generation=True)
        else:
            if client is None:
                client = make_client(cfg, usage_path=str(args.output_csv) + ".usage.json")
                # Optional spending cap. The ledger lives next to the responses,
                # so stopping and resuming the run can never reset the allowance.
                if getattr(args, "cost_limit_inr", None) is not None:
                    client = TokenCostGuard(client, cfg, str(args.output_csv) + ".cost.json",
                                            args.cost_limit_inr, args.input_rate, args.output_rate)
            response = client.generate(query)
            record.update({"generation_" + k: json.dumps(v) if isinstance(v, dict) else v
                           for k, v in response.to_dict().items() if k != "text"})
            grade = grade_answer(response.text, task["ground_truth_answer"], task["original_prompt"])
            record.update(response=response.text, answer=grade.answer, parse_status=grade.status,
                          response_status=response.status, correct=grade.correct if response.status == "ok" else "",
                          reused_generation=False)
        existing[key] = record
        if record["response_status"] == "ok":
            query_cache[query_hash] = record
        # Atomic replacement allows retrying a failed key without duplicate rows.
        atomic_csv(pd.DataFrame(existing.values()), args.output_csv)
        if record["response_status"] in {"auth_failed", "quota_exhausted", "request_budget_exhausted", "configuration_error", "model_not_found"}:
            print(f"Stopped: {record['response_status']}. Completed outputs saved; no further requests sent.")
            break

def summarize(args):
    if not args.report_dir:
        raise ValueError("--report-dir required")
    res = pd.read_csv(args.output_csv, keep_default_na=False)
    comp = pd.read_csv(args.input_csv, keep_default_na=False)
    require_unique(res, KEY, "evaluation")
    require_unique(comp, ["uid", "method", "target_ratio"], "compression")
    if "base_group_id" not in res or "response_status" not in res:
        raise ValueError("Legacy outputs require regrading/import before v2 inference")
    if args.methods:
        comp = comp[comp.method.isin(args.methods)]
        res = res[res.method.isin(args.methods + ["none"])]
    if args.ratios:
        comp = comp[comp.target_ratio.isin(args.ratios)]
        res = res[(res.method == "none") | res.target_ratio.isin(args.ratios)]
    successes = res[res.response_status == "ok"].copy()
    successes["correct"] = successes.correct.map(parse_bool).astype(int)
    references = successes[successes.method == "none"].set_index(["eval_model", "uid"])
    summary, paired, coverage, effects = [], [], [], []
    for model in res.eval_model.unique():
        for ratio, planned in comp.groupby("target_ratio"):
            methods = sorted(planned.method.unique())
            expected = planned.groupby("uid").method.nunique()
            full_cohort = set(expected[expected == len(methods)].index)
            observed = successes[(successes.eval_model == model) & (successes.target_ratio == ratio)]
            piv = observed.pivot(index="uid", columns="method", values="correct").reindex(columns=methods)
            original_uids = set(references.loc[model].index) if model in references.index.get_level_values(0) else set()
            common = sorted(full_cohort & set(piv.dropna().index) & original_uids)
            # The primary table uses the same question set for every method at each model/ratio.
            for method in methods:
                plan = planned[planned.method == method]
                successful = observed[observed.method == method]
                coverage.append({"eval_model": model, "target_ratio": ratio, "method": method,
                                 "n_planned": len(plan), "n_success": len(successful), "n_common": len(common),
                                 "n_failed_or_missing": len(plan) - len(successful)})
                subset = successful[successful.uid.isin(common)].merge(
                    plan[["uid", "budget_compliant"]], on="uid", validate="one_to_one")
                for source, g in [("ALL", subset)] + list(subset.groupby("source_dataset")):
                    if g.empty:
                        continue
                    values = g.correct.to_numpy(dtype=float)
                    ci = cluster_ci(values, g.base_group_id.to_numpy(), args.bootstrap)
                    orig = references.loc[[(model, u) for u in g.uid], "correct"].to_numpy(dtype=int)
                    actual = pd.to_numeric(g.actual_ratio)
                    summary.append({"eval_model": model, "target_ratio": ratio, "method": method,
                        "source_dataset": source, "n": len(g), "n_groups": g.base_group_id.nunique(),
                        "accuracy": values.mean(), "ci_lo": ci[0], "ci_hi": ci[1],
                        "full_input_accuracy": orig.mean(), "accuracy_change": values.mean() - orig.mean(),
                        "conditional_preservation": values[orig == 1].mean() if (orig == 1).any() else None,
                        "actual_ratio": actual.mean(), "budget_compliance": g.budget_compliant.map(parse_bool).mean()})
            for other in [m for m in methods if m != args.focus]:
                if args.focus not in piv or not common:
                    continue
                meta = planned[planned.method == args.focus].set_index("uid")
                competitor = planned[planned.method == other].set_index("uid")
                comparable = [u for u in common if parse_bool(meta.loc[u, "budget_compliant"])
                              and parse_bool(competitor.loc[u, "budget_compliant"])]
                diff = piv.loc[comparable, args.focus] - piv.loc[comparable, other]
                groups = meta.loc[comparable, "base_group_id"]
                ci = cluster_ci(diff.to_numpy(), groups.to_numpy(), args.bootstrap)
                actual_focus = pd.to_numeric(meta.loc[comparable, "actual_ratio"], errors="coerce")
                actual_other = pd.to_numeric(competitor.loc[comparable, "actual_ratio"], errors="coerce")
                paired.append({"eval_model": model, "target_ratio": ratio, "focus": args.focus, "other": other,
                               "n_common": len(common), "n_paired": len(comparable),
                               "n_excluded_budget_violation": len(common) - len(comparable),
                               "n_groups": groups.nunique(), "difference": diff.mean(),
                               "actual_ratio_focus": actual_focus.mean(), "actual_ratio_other": actual_other.mean(),
                               "ci_lo": ci[0], "ci_hi": ci[1], "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
    target = Path(args.report_dir); claim_output(target); target.mkdir(parents=True)
    atomic_csv(pd.DataFrame(summary, columns=["eval_model", "target_ratio", "method", "source_dataset", "n", "n_groups", "accuracy", "ci_lo", "ci_hi", "full_input_accuracy", "accuracy_change", "conditional_preservation", "actual_ratio", "budget_compliance"]), target / "accuracy.csv")
    atomic_csv(pd.DataFrame(paired, columns=["eval_model", "target_ratio", "focus", "other", "n_common", "n_paired", "n_excluded_budget_violation", "n_groups", "difference", "actual_ratio_focus", "actual_ratio_other", "ci_lo", "ci_hi", "ci_excludes_zero"]), target / "paired.csv")
    atomic_csv(pd.DataFrame(coverage), target / "coverage.csv")
    atomic_json(target / "manifest.json", {"results_hash": file_hash(args.output_csv),
        "compression_hash": file_hash(args.input_csv), "bootstrap": args.bootstrap,
        "cohort": "common complete conditions including original at each target/ratio",
        "clustering": "base_group_id", "multiple_testing": "unadjusted CIs; exploratory except separately declared primary comparisons"})
    print(f"Reports saved to {target}; inspect coverage.csv before accuracy.csv")

def deletion_check(args):
    """Fresh three-condition assay; never equates new-model outputs to old masks."""
    if not args.model_config:
        raise ValueError("--model-config required")
    cfg = public_config(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    spans = pd.read_csv(args.input_csv, keep_default_na=False, dtype={"ground_truth_answer": str,"span_text":str})
    from final_eval.utils.artifacts import require_columns
    require_columns(spans, ["example_id", "source_dataset", "span_start", "span_end", "span_text", "span_category", "original_prompt", "ground_truth_answer"])
    spans["uid"] = spans.source_dataset.astype(str) + "|" + spans.example_id.astype(str)
    spans = spans.drop_duplicates(["uid", "span_start", "span_end"]).copy()
    if args.n < 1:
        raise ValueError("n must be positive")
    # Shuffle within category, then round-robin categories for a small diagnostic.
    queues = [g.sample(frac=1, random_state=args.seed).to_dict("records") for _, g in spans.groupby("span_category")]
    sample = []
    while len(sample) < args.n and any(queues):
        for queue in queues:
            if queue and len(sample) < args.n:
                sample.append(queue.pop())
    print(json.dumps({"spans": len(sample), "maximum_conditions": 3*len(sample), "dry_run": args.dry_run,
                      "assay": "fresh original/mask/delete on one explicitly configured model"}, indent=2))
    if args.dry_run:
        return
    # Older callers (and tests) may not pass the cost-guard options at all.
    cost_limit = getattr(args, "cost_limit_inr", None)
    resume_manifest(args.output_csv, {"stage": "fresh_mask_deletion", "source": source_fingerprint(), "packages": package_versions(),
        "input": file_hash(args.input_csv), "config": generation_config(cfg), "n": args.n, "seed": args.seed,
        "grader": GRADER_VERSION, "cost_limit_inr": cost_limit,
        "input_rate": getattr(args, "input_rate", None), "output_rate": getattr(args, "output_rate", None)})
    records = pd.read_csv(args.output_csv, keep_default_na=False).to_dict("records") if Path(args.output_csv).exists() else []
    done = {(r["uid"], int(r["span_start"]), int(r["span_end"])) for r in records}
    client = make_client(cfg, usage_path=str(args.output_csv) + ".usage.json")
    # Rebuild the cache from completed records so an interrupted run never
    # repeats an already successful original/mask/delete request.
    cache = {}
    for saved in records:
        for condition in ("original", "mask", "delete"):
            raw = saved.get(condition + "_response_json", "")
            query_hash = saved.get(condition + "_query_hash", "")
            if raw and query_hash and saved.get(condition + "_status") == "ok":
                cache[query_hash] = json.loads(raw)
    if cost_limit is not None:
        client = TokenCostGuard(client, cfg, str(args.output_csv) + ".cost.json",
                                cost_limit, args.input_rate, args.output_rate)
    for r in sample:
        key = (r["uid"], int(r["span_start"]), int(r["span_end"]))
        if key in done:
            continue
        text, start, end = r["original_prompt"], key[1], key[2]
        if text[start:end] != r["span_text"]:
            raise ValueError("Invalid span offsets")
        interventions = {"original": text, "mask": text[:start] + "[MASKED]" + text[end:],
                         "delete": text[:start] + " " + text[end:]}
        record = {"uid": r["uid"], "span_start": start, "span_end": end,
                  "span_category": r["span_category"], "model": cfg.get("name") or cfg.get("model")}
        for condition, question in interventions.items():
            query = PROMPT_TEMPLATE.format(question=question)
            h = digest([generation_config(cfg), query])
            if h not in cache:
                response = client.generate(query)
                cache[h] = response.to_dict()
            else:
                # Cached values are plain dictionaries on disk and live
                # ModelResponse objects during the current process.
                response = cache[h]
            response_text = response["text"] if isinstance(response, dict) else response.text
            response_status = response["status"] if isinstance(response, dict) else response.status
            grade = grade_answer(response_text, r["ground_truth_answer"], text)
            record[condition + "_status"] = response_status
            record[condition + "_correct"] = grade.correct if response_status == "ok" else ""
            record[condition + "_query_hash"] = h
            record[condition + "_response_json"] = json.dumps(
                response if isinstance(response, dict) else response.to_dict())
        records.append(record)
        atomic_csv(pd.DataFrame(records), args.output_csv)
        if any(v in {"auth_failed", "quota_exhausted", "request_budget_exhausted", "configuration_error", "model_not_found"}
               for k, v in record.items() if k.endswith("_status")):
            break
    completed = [r for r in records if all(r[c + "_status"] == "ok" for c in ("original", "mask", "delete"))
                 and parse_bool(r["original_correct"])]
    agree = sum(parse_bool(r["mask_correct"]) == parse_bool(r["delete_correct"]) for r in completed)
    atomic_json(str(args.output_csv) + ".summary.json", {"n_attempted": len(records),
        "n_eligible_complete": len(completed), "agreement": agree / len(completed) if completed else None,
        "sampling": "category round-robin, diagnostic not population representative",
        "scope": "Fresh same-target assay; historical mask outcomes were not reused or certified"})

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["run", "summarize", "deletion-check"], default="run")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--model-config")
    ap.add_argument("--report-dir")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--ratios", nargs="+", type=float)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--focus", default="grouped_hybrid")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--pilot-check",help="Require a passed disjoint pilot with identical target settings before main inference")
    ap.add_argument("--n", type=int, default=30, help="Spans in fresh mask/deletion diagnostic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-limit-inr", type=float,
                    help="Optional conservative INR allowance for run / deletion-check.")
    ap.add_argument("--input-rate", type=float, default=10.0,
                    help="Quoted INR per million input tokens for the cost guard.")
    ap.add_argument("--output-rate", type=float, default=40.0,
                    help="Quoted INR per million output tokens for the cost guard.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error("At least 100 bootstrap resamples required")
    {"run": run_eval, "summarize": summarize, "deletion-check": deletion_check}[args.mode](args)

if __name__ == "__main__":
    from final_eval.utils.profiling import profile_cli
    profile_cli(main, "--output-csv")
