"""Small hosted-API access checks. No torch, model weights or GPU required.

This is an access/format diagnostic, not the frozen downstream pilot. Run it
yourself after filling final_eval/jarvis.env. Repeating a completed check reuses
its saved answers. Two prompts and at most four HTTP attempts per model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from final_eval.inference.client import HostedClient, generation_config
from final_eval.utils.answer_matching import grade_answer
from final_eval.utils.artifacts import atomic_json, resume_manifest

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"deepseek": "DEEPSEEK", "gptoss": "GPTOSS", "glm": "GLM", "qwen": "QWEN"}
# Qwen rates confirmed from the user's Token Factory catalog on 22 October.
RATES = {"deepseek": (13, 25), "gptoss": (10, 40), "glm": (14, 42.67), "qwen": (28.431, 222.709)}
PROMPTS = [
    ["Mira has 3 apples and gets 2 more. How many apples does she have?", "5"],
    ["There are 4 chairs and 3 tables. How many objects are there in total?", "7"],
]
WRAPPER = ("Solve this problem. You may reason briefly, but you MUST end your response "
           "with a line in exactly this format:\nFinal Answer: <your answer>\n\nProblem: {question}\n")


def configurations(env_file, selected):
    path = Path(env_file)
    if not path.is_file():
        raise ValueError("Copy final_eval/jarvis.env.example to final_eval/jarvis.env and fill it first")
    values = dotenv_values(path, interpolate=False)
    names = list(MODELS) if selected == ["all"] else selected
    if "all" in names:
        raise ValueError("Use --models all on its own")
    configs = {}
    for name in dict.fromkeys(names):
        model = (values.get(f"JARVISLABS_{MODELS[name]}_MODEL") or "").strip()
        if not model and selected == ["all"]:
            continue
        if not model or model.startswith("SET_"):
            raise ValueError(f"Fill the exact Token Factory API ID for {name} in the env file")
        cfg = dict(backend="hosted", provider="openai-compatible", name=model,
                   base_url=(values.get("JARVISLABS_BASE_URL") or "").strip(),
                   api_key_env="JARVISLABS_API_KEY", max_tokens=4096, temperature=0,
                   sleep_seconds=2, max_attempts=2, request_budget=4, timeout=180)
        if name == "qwen":
            # Catalog supports Low; avoid leaving reasoning at default Extra high.
            # This is a requested control, not proof the serving proxy honors it.
            cfg["reasoning_effort"] = "low"
        # Validate endpoint syntax without authenticating or loading any .env.
        from urllib.parse import urlsplit
        parsed = urlsplit(cfg["base_url"])
        if (parsed.scheme != "https" or not parsed.netloc or parsed.query or
                parsed.fragment or parsed.username or parsed.password):
            raise ValueError("JARVISLABS_BASE_URL must be an HTTPS URL without credentials/query/fragment")
        if parsed.hostname != "models.jarvislabs.net":
            raise ValueError("This helper sends Jarvislabs keys only to models.jarvislabs.net")
        configs[name] = cfg
    if not configs:
        raise ValueError("Fill at least one exact model ID in the env file")
    return configs, (values.get("JARVISLABS_API_KEY") or "").strip()


@contextmanager
def exclusive_check(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "access_check.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Check lock exists: another check may be running. If it was killed, confirm it stopped before removing access_check.lock") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def access_check(name, cfg, directory, retry_failed=False):
    output = directory / f"access.{name}.json"
    resume_manifest(output, {"stage": "jarvis_access_only", "config": generation_config(cfg),
                             "prompts": PROMPTS, "wrapper": WRAPPER})
    saved = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    records = saved.get("responses", {})
    client = HostedClient(cfg, usage_path=str(output) + ".usage.json")
    for index, (question, gold) in enumerate(PROMPTS):
        old = records.get(str(index))
        if old and (old["status"] == "ok" or not retry_failed):
            if old["status"] != "ok":
                break
            continue
        response = client.generate(WRAPPER.format(question=question))
        grade = grade_answer(response.text, gold, question)
        records[str(index)] = {**response.to_dict(), "correct": grade.correct,
                                "parse_status": grade.status}
        atomic_json(output, {"diagnostic_only": True, "model": cfg["name"],
                             "checked_utc": datetime.now(timezone.utc).isoformat(),
                             "responses": records})
        print(f"{name}: check {index + 1}/2, status={response.status}, final_answer_correct={grade.correct}", flush=True)
        if response.status != "ok":
            break
    # Usage saved with answers is counted once, never multiplied by a resume.
    usage = [r.get("usage") or {} for r in records.values()]
    complete_usage = len(usage) == 2 and all("prompt_tokens" in u and "completion_tokens" in u for u in usage)
    token_cost = None
    if complete_usage:
        token_cost = sum(u["prompt_tokens"] * RATES[name][0] + u["completion_tokens"] * RATES[name][1] for u in usage) / 1e6
    passed = len(records) == 2 and all(r["status"] == "ok" and r["correct"] for r in records.values())
    summary = dict(passed=passed, diagnostic_only=True, http_attempts=client.attempts,
                   response_count=len(records), response_usage_cost_estimate_inr=token_cost,
                   billing_note="User-quoted rates; provider balance/export determines actual charges. Failed/timed-out attempts may add costs not returned here.")
    atomic_json(str(output) + ".summary.json", summary)
    print(f"{name}: {'PASS' if passed else 'NEEDS REVIEW'}; cumulative HTTP attempts={client.attempts}/4; saved {output}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check"])
    parser.add_argument("--env-file", default=str(ROOT / "final_eval/jarvis.env"))
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=["deepseek"])
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/final_eval/jarvis"))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    try:
        configs, key = configurations(args.env_file, args.models)
        directory = Path(args.output_dir)
        if args.action == "check" and not key:
            raise ValueError("Fill JARVISLABS_API_KEY privately in the env file")
        with exclusive_check(directory):
            for name, cfg in configs.items():
                # Ready for subsequent pilot commands; no inline credential.
                target = {**cfg, "request_budget": 20}
                destination = directory / f"target.{name}.json"
                if destination.exists():
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                    if generation_config(previous) != generation_config(target):
                        raise ValueError(f"Existing {name} target settings differ; preserve that experiment and use a new --output-dir")
                else:
                    atomic_json(destination, target)
            print(f"Prepared credential-free configs: {', '.join(configs)}. No GPU/model loading.", flush=True)
            if args.action == "prepare":
                print("No API calls made. Output cap/decoding remain draft settings until target-pilot validation.", flush=True)
                return 0
            os.environ["JARVISLABS_API_KEY"] = key
            all_passed = True
            for name, cfg in configs.items():
                if not access_check(name, cfg, directory, args.retry_failed):
                    all_passed = False
                    break
            print("Check provider usage/wallet before further calls. This is not the downstream pilot.", flush=True)
            return 0 if all_passed else 1
    except ValueError as exc:
        print(f"Setup/check stopped: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
