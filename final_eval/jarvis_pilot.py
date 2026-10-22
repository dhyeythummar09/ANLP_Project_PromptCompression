"""User-run target pilot: saved prompts and hosted API only, no GPU imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import dotenv_values

from final_eval.compressor import evaluate
from final_eval.handoff import check
from final_eval.inference.client import generation_config, public_config
from final_eval.jarvis_api import ROOT, RATES, exclusive_check
from final_eval.utils.artifacts import atomic_json, file_hash, parse_bool, require_unique
from final_eval.utils.answer_matching import grade_answer

INPUT = Path("outputs/final_eval/pilot_llmlingua_window_v2/compressed.csv")
CONFIG = Path("outputs/final_eval/jarvis/target.deepseek.pilot.json")
OUTPUT = Path("outputs/final_eval/target_pilots/deepseek")
METHODS = {"ours", "grouped_hybrid", "grouped_signal", "random",
           "llmlingua_budgeted", "llmlingua2_budgeted", "selective_context_budgeted"}


def approved_qwen_8192_config():
    """Verify the saved cap diagnostic before admitting changed generation settings."""
    from final_eval import jarvis_qwen_cap as cap
    plan = cap.prepare()
    saved_plan = json.loads((cap.DIRECTORY / 'plan.json').read_text())
    summary = json.loads((cap.DIRECTORY / 'summary.json').read_text())
    saved = json.loads((cap.DIRECTORY / 'responses.json').read_text())['responses']
    if (saved_plan != plan or not summary.get('completed')
            or not summary.get('eligible_for_uniform_8192_pilot')
            or set(saved) != {case['case_id'] for case in plan['cases']}):
        raise ValueError('Need the completed eligible Qwen8192 cap diagnostic with unchanged provenance')
    for case in plan['cases']:
        record = saved[case['case_id']]
        grade = grade_answer(record['text'], case['ground_truth_answer'], case['original_prompt'])
        if (record['status'] != 'ok' or 'final answer:' not in record['text'].lower()
                or record.get('requested_model') != plan['config']['name']
                or (case['method'] == 'none' and not grade.correct)):
            raise ValueError('Qwen8192 cap diagnostic does not meet the recorded completion/control gate')
    return plan['config']


def prepare_qwen_config(output_cap=4096):
    """Derive the pilot from the passed public access config; never read keys."""
    directory = Path("outputs/final_eval/jarvis")
    cfg = public_config(json.loads((directory / "target.qwen.json").read_text(encoding="utf-8")))
    summary = json.loads((directory / "access.qwen.json.summary.json").read_text())
    checked = json.loads((directory / "access.qwen.json.manifest.json").read_text())["config"]
    if not summary.get("passed") or generation_config(cfg) != checked:
        raise ValueError("Need the passed Qwen access check with identical settings")
    if (cfg.get("name") != "qwen-3.8-27b-fp8" or cfg.get("reasoning_effort") != "low"
            or cfg.get("max_tokens") != 4096 or cfg.get("temperature") != 0):
        raise ValueError("Expected confirmed Qwen ID and reviewed Low/4096/temperature0 settings")
    cfg.update(request_budget=300, max_attempts=2)
    if output_cap == 8192:
        approved = approved_qwen_8192_config()
        cfg['max_tokens'] = 8192
        if generation_config(cfg) != generation_config(approved):
            raise ValueError('Changed Qwen pilot must match the passed8192 cap diagnostic')
    elif output_cap != 4096:
        raise ValueError('Reviewed Qwen output caps are4096 and8192 only')
    destination = directory / ("target.qwen.pilot8192.json" if output_cap == 8192 else "target.qwen.pilot.json")
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != cfg:
            raise ValueError("Existing Qwen pilot config differs; preserve it rather than overwriting")
    else:
        atomic_json(destination, cfg)


def qwen_usage_summary(output):
    """Count returned tokens once per generation, excluding cached copies."""
    frame = pd.read_csv(output / "responses.csv", keep_default_na=False)
    seen = set()
    tokens_in = tokens_out = records = 0
    for row in frame.to_dict("records"):
        if parse_bool(row.get("reused_generation", False)) or not row.get("generation_usage"):
            continue
        identity = (row.get("query_hash"), row.get("generation_timestamp_utc"))
        if all(identity) and identity in seen:
            continue
        if all(identity):
            seen.add(identity)
        usage = json.loads(row["generation_usage"])
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        records += 1
    ledger = Path(str(output / "responses.csv") + ".usage.json")
    rate_in, rate_out = RATES["qwen"]
    summary = {"model": "qwen-3.8-27b-fp8", "returned_usage_records": records,
        "returned_prompt_tokens": tokens_in, "returned_completion_tokens": tokens_out,
        "http_attempts": json.loads(ledger.read_text())["attempts"] if ledger.exists() else 0,
        "input_inr_per_million": rate_in, "output_inr_per_million": rate_out,
        "returned_usage_cost_estimate_inr": (tokens_in * rate_in + tokens_out * rate_out) / 1e6,
        "billing_note": "Retained returned usage only; provider bill is authoritative. Retries/timeouts or replaced failures may add cost. Full input rate used, no assumed cache discount.",
        "main_started": False}
    atomic_json(output / "usage_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def validate(input_csv=INPUT, config_path=CONFIG, model="deepseek"):
    if model not in {"deepseek", "gptoss", "qwen"}:
        raise ValueError("Expected deepseek, gptoss or qwen")
    audit = json.loads(Path("outputs/imports/llmlingua_verified_import_20261021.json").read_text())
    if file_hash(input_csv) != audit["compressed_sha256"]:
        raise ValueError("Pilot input differs from the verified passing import")
    comp = pd.read_csv(input_csv, keep_default_na=False)
    require_unique(comp, ["uid", "method", "target_ratio"])
    if (len(comp) != 224 or comp.uid.nunique() != 16
            or set(comp.method) != METHODS or set(comp.target_ratio) != {0.5, 0.75}
            or not comp.groupby(["uid", "target_ratio"]).size().eq(7).all()
            or not comp.status.eq("ok").all()
            or not comp.budget_compliant.map(parse_bool).all()):
        raise ValueError("Expected the complete passing 16-question, 224-condition grid")
    cfg = public_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    access_path = Path(f"outputs/final_eval/jarvis/access.{model}.json")
    access = json.loads(Path(str(access_path) + ".summary.json").read_text())
    access_config = json.loads(Path(str(access_path) + ".manifest.json").read_text())["config"]
    if model == 'qwen' and cfg.get('max_tokens') == 8192:
        access_config = generation_config(approved_qwen_8192_config())
    if not access.get("passed") or generation_config(cfg) != access_config:
        raise ValueError(f"Pilot configuration must match the passed {model} access check")
    if (cfg.get("backend") != "hosted" or cfg.get("base_url") != "https://models.jarvislabs.net/v1"
            or cfg.get("api_key_env") != "JARVISLABS_API_KEY"
            or cfg.get("request_budget") != 300 or cfg.get("max_attempts") != 2):
        raise ValueError("Expected reviewed Jarvislabs hosted configuration and 300-attempt limit")
    if model == "gptoss":
        folder = Path("outputs/final_eval/target_calibration/gptoss_fallback_v1")
        fallback = json.loads((folder / "summary.json").read_text())
        fallback_config = json.loads((folder / "plan.json").read_text())["config"]
        if (not fallback.get("completed") or not fallback.get("eligible_for_full_pilot")
                or generation_config(cfg) != generation_config(fallback_config)):
            raise ValueError("Full GPT-OSS pilot needs the completed eligible fallback with identical settings")
    return cfg


def run(dry_run=False, model="deepseek", qwen_output_cap=4096):
    os.chdir(ROOT)
    if model != 'qwen' and qwen_output_cap != 4096:
        raise ValueError('The Qwen output-cap option applies only to Qwen')
    if model == "qwen":
        prepare_qwen_config(qwen_output_cap)
    config = CONFIG if model == "deepseek" else Path(f"outputs/final_eval/jarvis/target.{model}.pilot.json")
    output = OUTPUT if model == "deepseek" else Path(f"outputs/final_eval/target_pilots/{model}")
    if model == 'qwen' and qwen_output_cap == 8192:
        config = Path('outputs/final_eval/jarvis/target.qwen.pilot8192.json')
        output = Path('outputs/final_eval/target_pilots/qwen_8192')
    cfg = validate(config_path=config, model=model)
    args = SimpleNamespace(model_config=str(config), input_csv=str(INPUT),
        output_csv=str(output / "responses.csv"), pilot_check=None,
        methods=None, ratios=None, limit=None, dry_run=dry_run, retry_failed=True)
    if dry_run:
        # No env file is read, client created, key loaded, or HTTP request sent.
        evaluate.run_eval(args)
        print("Verified pilot ready. No API calls, keys read, GPU or weights loaded.")
        return 0
    previous_check = output / "check.json"
    if previous_check.exists():
        previous = json.loads(previous_check.read_text(encoding="utf-8"))
        if previous.get("response_statuses", {}).get("truncated", 0):
            raise ValueError("Saved pilot hit its output cap. Do not rerun unchanged settings; use a separate bounded settings calibration first. Existing answers and usage ledger are preserved.")
    with exclusive_check(output):
        values = dotenv_values(ROOT / "final_eval/jarvis.env", interpolate=False)
        key = (values.get("JARVISLABS_API_KEY") or "").strip()
        if not key:
            raise ValueError("Fill JARVISLABS_API_KEY in final_eval/jarvis.env")
        os.environ["JARVISLABS_API_KEY"] = key
        factory = evaluate.make_client

        def progress_client(cfg, usage_path):
            client = factory(cfg, usage_path=usage_path)
            generate = client.generate

            def logged(prompt):
                print(f"API request starting; cumulative HTTP attempts so far: {client.attempts}/300", flush=True)
                result = generate(prompt)
                print(f"API response: {result.status}; HTTP attempts: {client.attempts}/300", flush=True)
                return result

            client.generate = logged
            return client

        print(f"{model} target pilot only: 16 originals + 224 compressed conditions.", flush=True)
        print(f"Output cap:{cfg['max_tokens']}; requested reasoning effort:{cfg.get('reasoning_effort','provider default')}", flush=True)
        print("Resumes saved successes; maximum 300 cumulative HTTP attempts. No automatic INR cap.", flush=True)
        evaluate.make_client = progress_client
        try:
            evaluate.run_eval(args)
        finally:
            evaluate.make_client = factory
        report = check(INPUT, output / "responses.csv", output / "check.json")
        if model == "qwen":
            qwen_usage_summary(output)
        print("TARGET PILOT PASS" if report["passed"] else "TARGET PILOT NEEDS REVIEW", flush=True)
        print("Check Jarvislabs wallet/usage and share check.json before starting main evaluation.", flush=True)
        return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=["deepseek", "gptoss", "qwen"], default="deepseek")
    parser.add_argument('--qwen-output-cap', type=int, choices=[4096, 8192], default=4096)
    args = parser.parse_args()
    try:
        return run(args.dry_run, args.model, args.qwen_output_cap)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Pilot stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
