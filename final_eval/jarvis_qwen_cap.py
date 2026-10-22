"""User-run three-case Qwen output-cap check; no automatic pilot/main launch."""
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
from final_eval.jarvis_api import ROOT, RATES, exclusive_check
from final_eval.utils.answer_matching import grade_answer
from final_eval.utils.artifacts import atomic_json, digest, file_hash, parse_bool, resume_manifest, source_fingerprint

PRIOR = Path('outputs/final_eval/target_pilots/qwen')
DIRECTORY = Path('outputs/final_eval/target_calibration/qwen_cap8192_v1')
CONFIG = Path('outputs/final_eval/jarvis/target.qwen.pilot.json')


def prepare():
    cfg = jarvis_pilot.validate(config_path=CONFIG, model='qwen')
    gate = json.loads((PRIOR / 'check.json').read_text())
    previous = PRIOR / 'responses.csv'
    if (gate.get('passed') or gate.get('responses_sha256') != file_hash(previous)
            or gate.get('target_config_hash') != digest(generation_config(cfg))):
        raise ValueError('Expected the unchanged failed Qwen pilot/configuration')
    if (cfg.get('name') != 'qwen-3.8-27b-fp8' or cfg.get('max_tokens') != 4096
            or cfg.get('reasoning_effort') != 'low' or cfg.get('temperature') != 0):
        raise ValueError('Expected the reviewed Low/4096/temperature0 Qwen pilot')
    res = pd.read_csv(previous, keep_default_na=False)
    original = res[res.method.eq('none')]
    failed = res[~res.response_status.eq('ok')]
    if (len(res) != 240 or len(original) != 16 or not original.response_status.eq('ok').all()
            or not original.correct.map(parse_bool).all() or len(failed) != 3
            or not failed.response_status.eq('truncated').all() or failed.query_hash.nunique() != 1):
        raise ValueError('Expected 16 correct controls and three truncations of one identical query')
    comp = pd.read_csv(jarvis_pilot.INPUT, keep_default_na=False, dtype={'ground_truth_answer': str})
    failure = failed.sort_values(['uid', 'method', 'target_ratio']).iloc[0]
    row = comp[comp.uid.eq(failure.uid) & comp.method.eq(failure.method)
               & comp.target_ratio.eq(failure.target_ratio)].iloc[0].to_dict()
    if digest([generation_config(cfg), PROMPT_TEMPLATE.format(question=row['compressed_prompt'])]) != failure.query_hash:
        raise ValueError('Failed query does not match its saved compression')
    other = comp[~comp.source_dataset.eq(row['source_dataset'])].sort_values(['uid', 'method', 'target_ratio']).iloc[0].to_dict()
    controls = [{**r, 'method': 'none', 'target_ratio': 1.0,
                 'compressed_prompt': r['original_prompt']} for r in (row, other)]
    cases = []
    for r in controls + [row]:
        case = {k:r[k] for k in ('uid', 'method', 'target_ratio', 'original_prompt', 'ground_truth_answer')}
        case.update(query=PROMPT_TEMPLATE.format(question=r['compressed_prompt']),
                    case_id=digest([r['uid'], r['method'], r['target_ratio']]))
        cases.append(case)
    return {'stage':'qwen_cap8192_v1', 'source':source_fingerprint(),
        'runner_sha256':file_hash(__file__), 'prior_responses_sha256':file_hash(previous),
        'prior_check_sha256':file_hash(PRIOR / 'check.json'), 'input_sha256':file_hash(jarvis_pilot.INPUT),
        'config':{**cfg, 'max_tokens':8192, 'request_budget':4}, 'cases':cases,
        'selection':'Two deterministic original controls and the sole failed query; completion/original correctness only. Compressed correctness is not a selection criterion.',
        'scope':'Cap diagnostic only. A changed cap requires a separate uniform full pilot; never merge generations with the old pilot.'}


def run(dry_run=False):
    os.chdir(ROOT)
    plan = prepare()
    if dry_run:
        print('Qwen cap check: 2 originals + 1 failed compressed query; cap8192/Low; maximum4 HTTP attempts across restarts. No keys read/API calls.')
        return 0
    with exclusive_check(DIRECTORY):
        resume_manifest(DIRECTORY / 'plan.json', plan)
        atomic_json(DIRECTORY / 'plan.json', plan)
        values = dotenv_values(ROOT / 'final_eval/jarvis.env', interpolate=False)
        key = (values.get('JARVISLABS_API_KEY') or '').strip()
        if not key:
            raise ValueError('Fill the existing Jarvislabs API key privately')
        os.environ['JARVISLABS_API_KEY'] = key
        path = DIRECTORY / 'responses.json'
        saved = json.loads(path.read_text()) if path.exists() else {'responses':{}}
        client = HostedClient(plan['config'], usage_path=str(path) + '.usage.json')
        stopped = any(r['status'] != 'ok' for r in saved['responses'].values())
        for case in plan['cases']:
            if stopped:
                break
            if case['case_id'] in saved['responses']:
                continue
            print(f"Qwen8192: {case['uid']} {case['method']}/{case['target_ratio']}; attempts {client.attempts}/4", flush=True)
            response = client.generate(case['query'])
            grade = grade_answer(response.text, case['ground_truth_answer'], case['original_prompt'])
            saved['responses'][case['case_id']] = {**response.to_dict(),
                'uid':case['uid'], 'method':case['method'], 'target_ratio':case['target_ratio'],
                'grade':grade.to_dict(), 'has_final_answer_marker':'final answer:' in response.text.lower()}
            atomic_json(path, saved)
            print(f'Qwen8192: status={response.status}, parse={grade.status}; attempts {client.attempts}/4', flush=True)
            stopped = response.status != 'ok'
        rows = list(saved['responses'].values())
        controls = [r for r in rows if r['method'] == 'none']
        eligible = (len(rows) == 3 and all(r['status'] == 'ok' and r['has_final_answer_marker'] for r in rows)
                    and len(controls) == 2 and all(r['grade']['correct'] for r in controls))
        ti = sum((r.get('usage') or {}).get('prompt_tokens') or 0 for r in rows)
        to = sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in rows)
        ri, ro = RATES['qwen']
        summary = {'completed':len(rows) == 3, 'eligible_for_uniform_8192_pilot':eligible,
            'response_statuses':{s:sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
            'original_controls_correct':sum(r['grade']['correct'] for r in controls),
            'http_attempts':client.attempts, 'http_attempt_limit':4,
            'returned_prompt_tokens':ti, 'returned_completion_tokens':to,
            'returned_usage_cost_estimate_inr':(ti*ri + to*ro)/1e6,
            'billing_note':'Returned-token estimate, full input rate; missing usage/retries may add provider charges. No currency cutoff.',
            'full_pilot_passed':False, 'main_started':False,
            'next_step':'If eligible, review before a separate all-condition8192 pilot. If not, stop cap increases. No automatic next run.'}
        atomic_json(DIRECTORY / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if eligible else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        return run(args.dry_run)
    except (ValueError, OSError, KeyError) as exc:
        print(f'Qwen cap check stopped: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
