"""User-run hosted main evaluation of the verified main compression return."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import dotenv_values

from final_eval.compressor import evaluate
from final_eval.inference.client import generation_config, public_config
from final_eval.jarvis_api import ROOT, RATES, exclusive_check
from final_eval.jarvis_pilot import METHODS, INPUT as PILOT_INPUT
from final_eval.jarvis_spend import TokenCostGuard
from final_eval.utils.artifacts import atomic_json, digest, file_hash, parse_bool, require_unique

BASE = Path("outputs/final_eval/main_gptoss_v1")
INPUT = BASE / "compressed.csv"
OUTPUT = BASE / "target_gptoss"
PILOT = Path("outputs/final_eval/target_pilots/gptoss/check.json")
CONFIG = Path("outputs/final_eval/jarvis/target.gptoss.main.json")


def stage_paths(model):
    if model == 'gptoss':
        return CONFIG, PILOT, OUTPUT
    if model == 'qwen':
        return (Path('outputs/final_eval/jarvis/target.qwen.main8192.json'),
                Path('outputs/final_eval/target_pilots/qwen_8192/check.json'),
                Path('outputs/final_eval/main_qwen8192_v1/target_qwen'))
    raise ValueError('Expected gptoss or qwen')


def prepare_qwen_main():
    """Derive the main config from the completed uniform8192 pilot, without keys."""
    config_path, pilot_path, _ = stage_paths('qwen')
    cfg = public_config(json.loads(Path('outputs/final_eval/jarvis/target.qwen.pilot8192.json').read_text()))
    gate = json.loads(pilot_path.read_text())
    responses = pilot_path.parent / 'responses.csv'
    manifest = json.loads(Path(str(responses) + '.manifest.json').read_text())
    if (not gate.get('passed') or gate.get('target_config_hash') != digest(generation_config(cfg))
            or gate.get('responses_sha256') != file_hash(responses)
            or gate.get('compression_sha256') != file_hash(PILOT_INPUT)
            or manifest.get('config') != generation_config(cfg)):
        raise ValueError('Need the unchanged passed uniform8192 Qwen pilot and response provenance')
    frame = pd.read_csv(responses, keep_default_na=False)
    require_unique(frame, evaluate.KEY, 'Qwen pilot responses')
    originals = frame[frame.method.eq('none')]
    if (len(frame) != 240 or not frame.response_status.eq('ok').all()
            or len(originals) != 16 or not originals.correct.map(parse_bool).all()
            or not originals.parse_status.eq('ok').all()
            or set(frame.eval_model) != {'qwen-3.8-27b-fp8'}
            or cfg.get('name') != 'qwen-3.8-27b-fp8' or cfg.get('max_tokens') != 8192
            or cfg.get('reasoning_effort') != 'low' or cfg.get('temperature') != 0):
        raise ValueError('Expected full240-condition Qwen8192 pilot with16 correct original controls')
    cfg.update(request_budget=3000, max_attempts=2)
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('Existing Qwen main config differs; preserve it rather than overwriting')
    else:
        atomic_json(config_path, cfg)


def validate(model='gptoss'):
    config_path, pilot_path, _ = stage_paths(model)
    imported = json.loads(Path("outputs/imports/main_verified_import_20261021.json").read_text())
    if not imported.get("passed") or imported["compressed_sha256"] != file_hash(INPUT):
        raise ValueError("Main compression differs from the independently verified return")
    cfg = public_config(json.loads(config_path.read_text(encoding="utf-8")))
    pilot = json.loads(pilot_path.read_text())
    if not pilot.get("passed") or pilot["target_config_hash"] != digest(generation_config(cfg)):
        raise ValueError(f"Need passed {model} pilot with identical generation settings")
    if (cfg.get("backend") != "hosted" or cfg.get("base_url") != "https://models.jarvislabs.net/v1"
            or cfg.get("api_key_env") != "JARVISLABS_API_KEY" or cfg.get("request_budget") != 3000
            or cfg.get("max_attempts") != 2):
        raise ValueError("Expected reviewed Jarvislabs hosted config and 3000-attempt ceiling")
    comp = pd.read_csv(INPUT, keep_default_na=False, dtype={"ground_truth_answer":str})
    cohort = pd.read_csv(BASE / "cohort.csv", keep_default_na=False)
    require_unique(comp,["uid","method","target_ratio"],"main compression")
    expected = {(uid,method,ratio) for uid in cohort.uid for method in METHODS for ratio in (.5,.75)}
    actual = set(comp[["uid","method","target_ratio"]].itertuples(index=False,name=None))
    if (len(cohort) != 160 or cohort.uid.nunique() != 160 or len(comp) != 2240 or actual != expected
            or not comp.status.eq("ok").all() or not comp.budget_compliant.map(parse_bool).all()):
        raise ValueError("Need the frozen160-question,2240-condition passing compression grid")
    if set(comp.base_group_id) & set(pilot["pilot_base_groups"]):
        raise ValueError("Main families overlap the target pilot")
    return cfg,comp


def summarize(cfg,comp,model='gptoss'):
    _, pilot_path, output = stage_paths(model)
    path = output / "responses.csv"
    frame = pd.read_csv(path,keep_default_na=False)
    require_unique(frame,evaluate.KEY,"main responses")
    expected={(t["uid"],cfg.get("name") or cfg["model"],t["method"],t["target_ratio"])
              for t in evaluate.compression_tasks(comp)}
    actual=set(frame[evaluate.KEY].itertuples(index=False,name=None))
    originals=frame[frame.method.eq("none")]
    successful=frame.response_status.eq("ok")
    tokens_in=tokens_out=usage_records=0
    seen_usage=set()
    for row in frame.to_dict("records"):
        if parse_bool(row.get("reused_generation",False)) or not row.get("generation_usage"):
            continue
        identity=(row.get("query_hash"),row.get("generation_timestamp_utc"))
        # Legacy/synthetic tables may omit identities; do not collapse those rows.
        if all(identity) and identity in seen_usage:
            continue
        if all(identity):
            seen_usage.add(identity)
        usage=json.loads(row["generation_usage"])
        tokens_in+=usage.get("prompt_tokens") or 0
        tokens_out+=usage.get("completion_tokens") or 0
        usage_records+=1
    active_usage_records=usage_records
    archived_usage_records=0
    archive=output/"archived_attempts.json"
    if archive.exists():
        for attempt in json.loads(archive.read_text())["attempts"]:
            identity=(attempt["query_hash"],attempt["timestamp_utc"])
            if identity in seen_usage:
                continue
            usage=attempt.get("usage") or {}
            if not usage:
                continue
            seen_usage.add(identity)
            tokens_in+=usage.get("prompt_tokens") or 0
            tokens_out+=usage.get("completion_tokens") or 0
            usage_records+=1
            archived_usage_records+=1
    usage_path=Path(str(path)+".usage.json")
    attempts=json.loads(usage_path.read_text())["attempts"] if usage_path.exists() else 0
    rate_in, rate_out = RATES[model]
    report={"passed":actual==expected and bool(successful.all()),
        "expected_conditions":2400,"recorded_conditions":len(frame),"grid_complete":actual==expected,
        "response_statuses":frame.response_status.value_counts().to_dict(),
        "successful_cached_reuses":int(frame.reused_generation.map(parse_bool).sum()),
        "http_attempts":attempts,"http_attempt_limit":3000,
        "original_questions":len(originals),
        "original_accuracy":float(originals.correct.map(lambda v: parse_bool(v) if str(v) else False).mean()) if len(originals) else None,
        "original_parse_statuses":originals.parse_status.value_counts().to_dict(),
        "returned_usage_records":usage_records,"active_csv_usage_records":active_usage_records,
        "archived_usage_records":archived_usage_records,
        "returned_prompt_tokens":tokens_in,"returned_completion_tokens":tokens_out,
        "input_inr_per_million":rate_in,"output_inr_per_million":rate_out,
        "returned_usage_cost_estimate_inr":(tokens_in*rate_in+tokens_out*rate_out)/1e6,
        "billing_note":"Quoted rates, estimate only. Reconcile provider usage/wallet; missing usage/failed attempts may add costs. Qwen has a separate INR500 conservative token-cost allowance; GPT-OSS has no currency cutoff.",
        "responses_sha256":file_hash(path),"compression_sha256":file_hash(INPUT),"pilot_check_sha256":file_hash(pilot_path),
        "scope":f"Main {model} answers only; no automatic superiority claim or additional paid calls.",
        "next_step":"Inspect coverage, grading and billing, then run offline analysis on these saved answers. Preserve failures; do not tune settings against main outcomes."}
    cost_path = output / 'token_cost_allowance.json'
    if cost_path.exists():
        report['token_cost_allowance'] = json.loads(cost_path.read_text())
    atomic_json(output / "check.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def run(dry_run=False,model='gptoss'):
    os.chdir(ROOT)
    if model == 'qwen':
        prepare_qwen_main()
    config_path, pilot_path, output = stage_paths(model)
    cfg,comp=validate(model)
    args=SimpleNamespace(model_config=str(config_path),input_csv=str(INPUT),output_csv=str(output/"responses.csv"),
        pilot_check=str(pilot_path),methods=None,ratios=None,limit=None,dry_run=dry_run,retry_failed=True)
    if dry_run:
        evaluate.run_eval(args)
        tasks=evaluate.compression_tasks(comp)
        unique=len({t["compressed_prompt"] for t in tasks})
        print(f"Up to{unique} distinct prompts before response caching/retries;3000 is a ceiling, not expected count.")
        print("Verified main ready. No keys read, GPU/model loading or API calls.")
        return 0
    with exclusive_check(output):
        values=dotenv_values(ROOT/"final_eval/jarvis.env",interpolate=False)
        key=(values.get("JARVISLABS_API_KEY") or "").strip()
        if not key:
            raise ValueError("Fill JARVISLABS_API_KEY in final_eval/jarvis.env")
        os.environ["JARVISLABS_API_KEY"]=key
        factory=evaluate.make_client
        def progress_client(config,usage_path):
            client=factory(config,usage_path=usage_path)
            if model == 'qwen':
                client=TokenCostGuard(client,config,output/'token_cost_allowance.json',500,*RATES['qwen'])
            generate=client.generate
            def logged(prompt):
                print(f"Main API request starting; cumulative attempts:{client.attempts}/3000",flush=True)
                response=generate(prompt)
                print(f"Main API response:{response.status}; cumulative attempts:{client.attempts}/3000",flush=True)
                return response
            client.generate=logged
            return client
        print(f"MAIN {model}:160 originals+2240 compressed conditions; no GPU. Saved successes resume.",flush=True)
        print(f"Output cap:{cfg['max_tokens']}; requested reasoning effort:{cfg.get('reasoning_effort','provider default')}",flush=True)
        print("Maximum3000 cumulative HTTP attempts including retries.",flush=True)
        if model == 'qwen':
            print('INR500 token-cost allowance: reserves worst-case attempts before calls, accounts returned usage, persists across restarts. Provider billing remains authoritative.',flush=True)
        else:
            print('No automatic INR spending cutoff.',flush=True)
        evaluate.make_client=progress_client
        try:
            evaluate.run_eval(args)
        finally:
            evaluate.make_client=factory
        report=summarize(cfg,comp,model)
        print("MAIN ANSWERS COMPLETE" if report["passed"] else "MAIN ANSWERS NEED REVIEW",flush=True)
        print(f"Saved:{output/'check.json'}. Check provider wallet/usage; no extra calls started.",flush=True)
        return 0 if report["passed"] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument('--model',choices=['gptoss','qwen'],default='gptoss')
    args=parser.parse_args()
    try:
        return run(args.dry_run,args.model)
    except (ValueError,OSError,KeyError) as exc:
        print(f"Main evaluation stopped:{exc}",flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
