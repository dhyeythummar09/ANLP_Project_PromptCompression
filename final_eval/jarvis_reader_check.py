"""User-run, six-case GPT-OSS fallback check; no full pilot or GPU work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from final_eval.inference.client import HostedClient, generation_config, public_config
from final_eval.jarvis_api import ROOT, exclusive_check
from final_eval.jarvis_pilot import INPUT
from final_eval.utils.answer_matching import grade_answer
from final_eval.utils.artifacts import atomic_json, file_hash, resume_manifest, source_fingerprint

DIRECTORY = Path("outputs/final_eval/target_calibration/gptoss_fallback_v1")
DEEPSEEK = Path("outputs/final_eval/target_calibration/deepseek_v1")


def prepare():
    calibration = json.loads((DEEPSEEK / "summary.json").read_text())
    if not calibration.get("completed") or calibration.get("provisional_recommendation"):
        raise ValueError("Review/finish the existing calibration before considering another reader")
    prior = json.loads((DEEPSEEK / "plan.json").read_text())
    if prior["input_sha256"] != file_hash(INPUT) or len(prior["cases"]) != 6:
        raise ValueError("Calibration cases/input have changed")
    access = Path("outputs/final_eval/jarvis/access.gptoss.json")
    cfg = public_config(json.loads(Path("outputs/final_eval/jarvis/target.gptoss.json").read_text()))
    access_cfg = json.loads(Path(str(access) + ".manifest.json").read_text())["config"]
    if not json.loads(Path(str(access) + ".summary.json").read_text())["passed"] or generation_config(cfg) != access_cfg:
        raise ValueError("GPT-OSS configuration must match its passed access diagnostic")
    if cfg.get("backend") != "hosted" or cfg.get("base_url") != "https://models.jarvislabs.net/v1" or cfg.get("api_key_env") != "JARVISLABS_API_KEY":
        raise ValueError("Expected the reviewed Jarvislabs hosted configuration")
    cfg.update(request_budget=8, max_attempts=2)
    return {"stage":"gptoss_fallback_check_v1", "cases":prior["cases"], "config":cfg,
        "source":source_fingerprint(), "runner_sha256":file_hash(__file__),
        "calibration_plan_sha256":file_hash(DEEPSEEK / "plan.json"),
        "calibration_summary_sha256":file_hash(DEEPSEEK / "summary.json"),
        "selection":"Operational completion/final-answer format and original controls only; no compressed-correctness selection.",
        "scope":"Fallback candidate only. This is not a full pilot, main experiment or proof of model performance."}


def run(dry_run=False):
    os.chdir(ROOT)
    plan = prepare()
    if dry_run:
        print("GPT-OSS fallback: same 6 cases; maximum 8 HTTP attempts across restarts; no keys read/API calls.")
        return 0
    with exclusive_check(DIRECTORY):
        resume_manifest(DIRECTORY / "plan.json", plan)
        atomic_json(DIRECTORY / "plan.json", plan)
        values = dotenv_values(ROOT / "final_eval/jarvis.env", interpolate=False)
        key = (values.get("JARVISLABS_API_KEY") or "").strip()
        if not key:
            raise ValueError("Fill JARVISLABS_API_KEY in final_eval/jarvis.env")
        os.environ["JARVISLABS_API_KEY"] = key
        path = DIRECTORY / "responses.json"
        saved = json.loads(path.read_text()) if path.exists() else {"responses":{}}
        client = HostedClient(plan["config"], usage_path=str(path) + ".usage.json")
        terminal = any(r["status"] not in {"ok", "truncated", "empty_response"} for r in saved["responses"].values())
        for case in plan["cases"]:
            if terminal:
                break
            if case["case_id"] in saved["responses"]:
                continue
            print(f"GPT-OSS: {case['method']} / {case['target_ratio']}; attempts {client.attempts}/8", flush=True)
            response = client.generate(case["query"])
            grade = grade_answer(response.text, case["ground_truth_answer"], case["original_prompt"])
            saved["responses"][case["case_id"]] = {**response.to_dict(), "method":case["method"],
                "uid":case["uid"], "target_ratio":case["target_ratio"], "grade":grade.to_dict(),
                "has_final_answer_marker":"final answer:" in response.text.lower()}
            atomic_json(path, saved)
            print(f"GPT-OSS: status={response.status}, parse={grade.status}, attempts={client.attempts}/8", flush=True)
            terminal = response.status not in {"ok", "truncated", "empty_response"}
        rows = list(saved["responses"].values())
        originals = [r for r in rows if r["method"] == "none"]
        eligible = (len(rows) == 6 and all(r["status"] == "ok" and r["has_final_answer_marker"] for r in rows)
                    and len(originals) == 2 and all(r["grade"]["correct"] for r in originals))
        estimate = sum((((r.get("usage") or {}).get("prompt_tokens") or 0) * 10
                       + ((r.get("usage") or {}).get("completion_tokens") or 0) * 40) / 1e6 for r in rows)
        report = {"completed":len(rows) == 6, "eligible_for_full_pilot":eligible,
            "statuses":{s:sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
            "original_controls_correct":sum(r["grade"]["correct"] for r in originals),
            "http_attempts":client.attempts, "http_attempt_limit":8,
            "returned_usage_cost_estimate_inr":estimate, "billing_note":"Quoted rates; reconcile actual provider bill. No financial cutoff.",
            "full_pilot_passed":False, "main_started":False,
            "next_step":"If eligible, run a separately recorded full GPT-OSS pilot before main. Otherwise review; no automatic additional calls.",
            "reader_caveat":"Historical labeling used GPT-OSS; do not describe this fallback as proven unseen-model transfer."}
        atomic_json(DIRECTORY / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
        print("Small fallback check finished. No full pilot or main run started.", flush=True)
        return 0 if eligible else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.dry_run)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Fallback check stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
