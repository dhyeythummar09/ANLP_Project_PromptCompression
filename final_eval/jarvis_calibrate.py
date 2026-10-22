"""Bounded, user-run DeepSeek settings calibration using saved pilot cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from final_eval import jarvis_pilot
from final_eval.compressor.evaluate import PROMPT_TEMPLATE
from final_eval.inference.client import HostedClient, generation_config
from final_eval.jarvis_api import ROOT, exclusive_check
from final_eval.utils.answer_matching import grade_answer
from final_eval.utils.artifacts import atomic_json, digest, file_hash, resume_manifest, source_fingerprint

OUTPUT = Path("outputs/final_eval/target_calibration/deepseek_v1")
VARIANTS = {
    "low_8192": {"thinking": {"type": "enabled"}, "reasoning_effort": "low", "max_tokens": 8192},
    "no_thinking_4096": {"thinking": {"type": "disabled"}, "max_tokens": 4096},
}


def reasoning_tokens(record):
    details = (record.get("usage") or {}).get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) and value >= 0 else None


def prepare():
    cfg = jarvis_pilot.validate()
    prior = jarvis_pilot.OUTPUT / "responses.csv"
    gate = json.loads((jarvis_pilot.OUTPUT / "check.json").read_text())
    if gate.get("passed") or not gate.get("response_statuses", {}).get("truncated"):
        raise ValueError("Expected the saved output-truncated pilot; no new test needed otherwise")
    if gate["responses_sha256"] != file_hash(prior):
        raise ValueError("Saved responses differ from their target check")
    comp = pd.read_csv(jarvis_pilot.INPUT, keep_default_na=False, dtype={"ground_truth_answer": str})
    res = pd.read_csv(prior, keep_default_na=False)
    failed = res[res.response_status.eq("truncated")]
    selected, used = [], set()
    # Fixed method coverage; choose by failure/identity only, never answer correctness.
    for method in ("ours", "llmlingua_budgeted", "random", "grouped_signal"):
        options = failed[failed.method.eq(method)].to_dict("records")
        if not options:
            raise ValueError(f"No saved truncated case for {method}")
        options.sort(key=lambda r: (r["uid"] in used,
            digest([r["uid"], r["method"], r["target_ratio"]])))
        item = options[0]
        row = comp[comp.uid.eq(item["uid"]) & comp.method.eq(method)
                   & comp.target_ratio.eq(item["target_ratio"])].iloc[0].to_dict()
        selected.append(row)
        used.add(row["uid"])
    if len(used) < 2:
        raise ValueError("Need two different original-question controls")
    first = selected[0]
    second = next((r for r in selected[1:] if r["uid"] != first["uid"]
                   and r["source_dataset"] != first["source_dataset"]),
                  next(r for r in selected[1:] if r["uid"] != first["uid"]))
    controls = [{**row, "method": "none", "target_ratio": 1.0,
                 "compressed_prompt": row["original_prompt"]} for row in (first, second)]
    cases = []
    for row in controls + selected:
        cases.append({k: row[k] for k in ("uid", "method", "target_ratio", "original_prompt", "ground_truth_answer")})
        cases[-1].update(query=PROMPT_TEMPLATE.format(question=row["compressed_prompt"]),
                         case_id=digest([row["uid"], row["method"], row["target_ratio"]]))
    configs = {name: {**cfg, **settings, "request_budget": 8, "max_attempts": 2}
               for name, settings in VARIANTS.items()}
    return {"stage": "deepseek_settings_calibration_v1", "source": source_fingerprint(),
            "runner_sha256": file_hash(__file__), "prior_responses_sha256": file_hash(prior),
            "input_sha256": file_hash(jarvis_pilot.INPUT), "cases": cases, "configs": configs,
            "total_http_attempt_limit": 16,
            "selection_rule": "Prefer low effort if all six finish with a final-answer marker, both originals correct and thinking observed; otherwise consider verified thinking-disabled variant. Compressed-answer correctness/parse failures are diagnostic, not criteria for choosing settings.",
            "scope": "Pilot families only; provisional settings test, not full target gate or main accuracy. No automatic full run.",
            "support_note": "Upstream controls documented; Jarvislabs acceptance/enforcement unknown until tested. Low effort enforcement cannot be proven from token counts alone.",
            "docs": ["https://api-docs.deepseek.com/guides/thinking_mode/", "https://jarvislabs.ai/products/models"]}


def summarize(plan):
    variants = {}
    estimated_cost = 0.0
    total_attempts = 0
    for name in VARIANTS:
        path = OUTPUT / f"{name}.json"
        records = json.loads(path.read_text()).get("responses", {}) if path.exists() else {}
        items = list(records.values())
        usage_path = Path(str(path) + ".usage.json")
        attempts = json.loads(usage_path.read_text())["attempts"] if usage_path.exists() else 0
        total_attempts += attempts
        complete = len(items) == len(plan["cases"])
        operational = complete and all(r["status"] == "ok" and r["has_final_answer_marker"] for r in items)
        originals = [r for r in items if r["method"] == "none"]
        originals_correct = len(originals) == 2 and all(r["grade"]["correct"] for r in originals)
        thinking_observed = any(r.get("reasoning", "").strip() or (reasoning_tokens(r) or 0) > 0 for r in items)
        disabled_observed = complete and all(not r.get("reasoning", "").strip() and
            reasoning_tokens(r) == 0 for r in items)
        for r in items:
            u = r.get("usage") or {}
            estimated_cost += ((u.get("prompt_tokens") or 0) * 13 + (u.get("completion_tokens") or 0) * 25) / 1e6
        variants[name] = {"records":len(items), "http_attempts":attempts,
            "statuses":{s:sum(r["status"] == s for r in items) for s in sorted({r["status"] for r in items})},
            "all_complete_with_final_answer":operational, "both_original_controls_correct":originals_correct,
            "parse_statuses":{s:sum(r["grade"]["status"] == s for r in items) for s in sorted({r["grade"]["status"] for r in items})},
            "thinking_observed":bool(thinking_observed), "thinking_disabled_observed":disabled_observed,
            "eligible_for_full_pilot":bool(operational and originals_correct and
                (thinking_observed if name == "low_8192" else disabled_observed))}
    selected = next((name for name in VARIANTS if variants[name]["eligible_for_full_pilot"]), None)
    report = {"completed":all(v["records"] == 6 for v in variants.values()), "variants":variants,
        "provisional_recommendation":selected, "total_http_attempts":total_attempts,
        "total_http_attempt_limit":16, "returned_usage_cost_estimate_inr":estimated_cost,
        "billing_note":"Estimate at user-quoted rates, not actual bill; missing usage and failed attempts may add costs. Check wallet.",
        "full_target_pilot_passed":False, "main_started":False,
        "next_step":"Review this summary, then a separate full pilot with one uniform configuration. No automatic rerun; do not mix with old settings/answers."}
    atomic_json(OUTPUT / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run(dry_run=False):
    os.chdir(ROOT)
    plan = prepare()
    if dry_run:
        print(json.dumps({"questions":len({c["uid"] for c in plan["cases"]}),
            "cases_per_setting":6, "settings":list(VARIANTS), "planned_conditions":12,
            "http_attempt_limit":16, "estimated_max_returned_output_cost_inr":2.4576,
            "case_methods":[c["method"] for c in plan["cases"]],
            "dry_run":True, "api_calls":0, "keys_read":False}, indent=2))
        return 0
    with exclusive_check(OUTPUT):
        resume_manifest(OUTPUT / "plan.json", plan)
        atomic_json(OUTPUT / "plan.json", plan)
        values = dotenv_values(ROOT / "final_eval/jarvis.env", interpolate=False)
        key = (values.get("JARVISLABS_API_KEY") or "").strip()
        if not key:
            raise ValueError("Fill JARVISLABS_API_KEY in final_eval/jarvis.env")
        os.environ["JARVISLABS_API_KEY"] = key
        stop_all = False
        for name, cfg in plan["configs"].items():
            path = OUTPUT / f"{name}.json"
            resume_manifest(path, {"plan":digest(plan), "generation_config":generation_config(cfg)})
            atomic_json(OUTPUT / f"candidate.{name}.json", cfg)
            saved = json.loads(path.read_text()) if path.exists() else {"responses":{}}
            client = HostedClient(cfg, usage_path=str(path) + ".usage.json")
            # HTTP/schema rejection is terminal for this setting, also across resumptions.
            if any(r["status"] in {"auth_failed", "quota_exhausted", "configuration_error", "model_not_found"}
                   for r in saved["responses"].values()):
                break
            if any(r["status"] not in {"ok", "truncated", "empty_response"} for r in saved["responses"].values()):
                continue
            for case in plan["cases"]:
                if case["case_id"] in saved["responses"]:
                    continue
                print(f"{name}: {case['method']} / {case['target_ratio']}; attempts {client.attempts}/8", flush=True)
                response = client.generate(case["query"])
                grade = grade_answer(response.text, case["ground_truth_answer"], case["original_prompt"])
                saved["responses"][case["case_id"]] = {**response.to_dict(),
                    "uid":case["uid"], "method":case["method"], "target_ratio":case["target_ratio"],
                    "grade":grade.to_dict(), "has_final_answer_marker":"final answer:" in response.text.lower()}
                atomic_json(path, saved)
                print(f"{name}: status={response.status}, parse={grade.status}, attempts={client.attempts}/8", flush=True)
                if response.status in {"auth_failed", "quota_exhausted", "configuration_error", "model_not_found"}:
                    stop_all = True
                    break
                if response.status not in {"ok", "truncated", "empty_response"}:
                    break
            if stop_all:
                break
        report = summarize(plan)
        print(f"Calibration saved: {OUTPUT / 'summary.json'}. No full pilot or main run started.", flush=True)
        return 0 if report["provisional_recommendation"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.dry_run)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        print(f"Calibration stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
