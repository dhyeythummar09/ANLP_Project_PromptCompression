"""
test_models.py -- smoke test for model API keys and reasoning correctness.

Purpose: before committing to a multi-day labeling run, verify:
  1. Does the API key work? (auth / 401)
  2. Is the model ID recognized by the provider? (404 / "model not found")
  3. Does the model output match our expected answer format (Final Answer: ...)?
  4. Accuracy on a tiny sample (default 1 question per family = 3 questions total).

Usage:
  # Test all configured active models:
  python src/dataset_creation/test_models.py

  # Test only motapis:
  python src/dataset_creation/test_models.py --provider motapis

  # Test a specific model:
  python src/dataset_creation/test_models.py --model deepseek-v4-flash --provider motapis

  # Test with custom number of questions per dataset family (default is 1 to save tokens):
  python src/dataset_creation/test_models.py --questions 1
"""

import os
import sys
import time
import argparse
import textwrap

import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.utils.answer_matching import answers_match, extract_final_answer

load_dotenv()

# ---------------------------------------------------------------------------
# Models under test.
# Adjust or filter using CLI flags (--provider, --model).
# ---------------------------------------------------------------------------
TEST_MODELS = [
    {
        "name": "qwen3.7-max",
        "provider": "motapis_qwen",
        "sleep_seconds": 0.5,
        "daily_cap": 2000,
    },
    {
        "name": "glm-5.3-flash",
        "provider": "motapis_glm",
        "sleep_seconds": 0.5,
        "daily_cap": 2000,
    },
    {
        "name": "openai/gpt-oss-120b",
        "provider": "groq",
        "sleep_seconds": 2.2,
        "daily_cap": 1000,
    },
    {
        "name": "gemini-3.6-flash",
        "provider": "google",
        "sleep_seconds": 4.2,
        "daily_cap": 1000,
    },
]

DATASET_FAMILIES = {
    "GSM8K": lambda s: s == "gsm8k",
    "GSM-IC": lambda s: s == "gsm_ic",
    "BBH": lambda s: s.startswith("bbh_"),
}

PROMPT_TEMPLATE = """Solve this problem. You may reason briefly, but you MUST end your response with a line in exactly this format:
Final Answer: <your answer>

Problem: {question}
"""

def _resolve_motapis_url() -> str:
    base = (os.environ.get("MOTAPIS_BASE_URL") or config.MOTAPIS_BASE_URL or "https://api.motapis.com/v1").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"

OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "ollama": "http://localhost:11434/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "huggingface": "https://router.huggingface.co/hf-inference/v1/chat/completions",
}

PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "motapis_qwen": "MOTAPIS_API_KEY_QWEN",
    "motapis_glm": "MOTAPIS_API_KEY_GLM",
    "huggingface": "HF_TOKEN",
}


class ApiError(Exception):
    def __init__(self, kind: str, detail: str):
        self.kind = kind  # "auth" | "rate_limit" | "not_found" | "quota" | "network" | "parse" | "other"
        self.detail = detail
        super().__init__(f"[{kind}] {detail}")


def _diagnose_http_error(resp: requests.Response) -> ApiError:
    status = resp.status_code
    body = resp.text[:400]

    if status in (401, 403):
        return ApiError("auth", f"HTTP {status} -- API key rejected or unauthorized. Check .env. Body: {body}")
    if status == 402 or "insufficient" in body.lower() or "balance" in body.lower():
        return ApiError("quota", f"HTTP {status} -- Quota/balance exhausted on this provider. Body: {body}")
    if status == 404:
        return ApiError("not_found", f"HTTP 404 -- Model ID not found. Verify model name with provider docs. Body: {body}")
    if status == 429:
        return ApiError("rate_limit", f"HTTP 429 -- Rate limit or request ceiling exceeded. Body: {body}")
    if status >= 500:
        return ApiError("other", f"HTTP {status} -- Provider server error/overloaded. Body: {body}")
    return ApiError("other", f"HTTP {status}. Body: {body}")


def _call_openai_compatible(model_cfg: dict, prompt: str, api_key: str) -> str:
    provider = model_cfg["provider"]
    if provider.startswith("motapis"):
        url = _resolve_motapis_url()
    else:
        url = OPENAI_COMPATIBLE_BASE_URLS[provider]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model_cfg["name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=45)
    except requests.exceptions.RequestException as e:
        raise ApiError("network", f"Could not reach {url}: {e}")

    if not resp.ok:
        raise _diagnose_http_error(resp)

    try:
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip() and ("reasoning" in msg or "reasoning_content" in msg):
            content = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return content.strip()
    except (KeyError, IndexError, ValueError) as e:
        raise ApiError("parse", f"Unexpected response shape: {e}. Body: {resp.text[:300]}")


def _call_google(model_cfg: dict, prompt: str, api_key: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_cfg['name']}:generateContent?key={api_key}"
    )
    body = {
        "systemInstruction": {
            "parts": [{"text": "You are a mathematical and logical reasoning assistant. You may reason briefly, but you MUST end your response with a line in exactly this format:\nFinal Answer: <your answer>"}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, timeout=45)
            if resp.status_code == 503:
                print(f"      [Google 503] Server busy, retrying in 10s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
                continue
            if not resp.ok:
                raise _diagnose_http_error(resp)
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            raise ApiError("network", f"Could not reach Gemini endpoint: {e}")

    raise ApiError("other", f"HTTP 503 -- Google servers remained busy after {max_retries} retries.")


def call_model(model_cfg: dict, prompt: str) -> str:
    provider = model_cfg["provider"]
    env_var = PROVIDER_ENV_VARS.get(provider)
    if not env_var:
        raise ApiError("other", f"No env var mapped for provider: {provider}")

    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise ApiError("auth", f"{env_var} is not set in your .env file")

    if provider in OPENAI_COMPATIBLE_BASE_URLS or provider.startswith("motapis"):
        return _call_openai_compatible(model_cfg, prompt, api_key)
    elif provider == "google":
        return _call_google(model_cfg, prompt, api_key)
    raise ApiError("other", f"Unknown provider: {provider}")


def load_test_questions(questions_per_dataset: int = 1) -> list[dict]:
    if not os.path.exists(config.RAW_PROBLEMS_FILE):
        print(f"ERROR: {config.RAW_PROBLEMS_FILE} not found.")
        print("Run `python src/dataset_creation/01_load_data.py` first.")
        sys.exit(1)

    df = pd.read_csv(config.RAW_PROBLEMS_FILE).fillna("")
    questions = []

    for family_name, matches in DATASET_FAMILIES.items():
        subset = df[df["source_dataset"].apply(matches)]
        if subset.empty:
            print(f"  [warning] no problems found for family '{family_name}' -- skipping")
            continue
        sampled = subset.sample(n=min(questions_per_dataset, len(subset)), random_state=42)
        for _, row in sampled.iterrows():
            questions.append({
                "family": family_name,
                "example_id": row["example_id"],
                "source_dataset": row["source_dataset"],
                "question": row["original_prompt"],
                "ground_truth": str(row["ground_truth_answer"]),
            })

    return questions


def test_one_model(model_cfg: dict, questions: list[dict]) -> dict:
    name = model_cfg["name"]
    provider = model_cfg["provider"]
    print("\n" + "=" * 78)
    print(f"TESTING: {name}  (provider: {provider}, sleep: {model_cfg.get('sleep_seconds', 1.0)}s)")
    print("=" * 78)

    n_calls_made = 0
    results = []

    for i, q in enumerate(questions, 1):
        if n_calls_made >= model_cfg.get("daily_cap", 10000):
            print(f"\n  ! Reached model daily cap -- stopping early.")
            break

        print(f"\n  [{i}/{len(questions)}] {q['family']} / {q['example_id']}")
        print(f"      Q: {textwrap.shorten(q['question'], width=100)}")
        print(f"      Expected: {q['ground_truth']}")

        try:
            raw = call_model(model_cfg, PROMPT_TEMPLATE.format(question=q["question"]))
            n_calls_made += 1
        except ApiError as e:
            print(f"      FAILED -> {e}")
            results.append({**q, "model": name, "status": f"error:{e.kind}", "correct": False,
                            "parsed_answer": "", "raw_response": ""})
            if e.kind in ("auth", "rate_limit", "not_found", "quota"):
                print(f"      -> Skipping remaining test questions for {name}.")
                break
            time.sleep(model_cfg.get("sleep_seconds", 1.0))
            continue

        parsed = extract_final_answer(raw)
        correct = answers_match(raw, q["ground_truth"], q["question"])
        symbol = "PASS" if correct else "FAIL"
        print(f"      Got: {textwrap.shorten(parsed, width=80)}")
        print(f"      Result: {symbol}")

        results.append({**q, "model": name, "status": "ok", "correct": correct,
                        "parsed_answer": parsed, "raw_response": raw})

        time.sleep(model_cfg.get("sleep_seconds", 1.0))

    return {"model": name, "provider": provider, "results": results, "calls_made": n_calls_made}


def print_summary(all_results: list[dict]):
    print("\n\n" + "=" * 78)
    print("SMOKE TEST SUMMARY")
    print("=" * 78)

    for entry in all_results:
        name = entry["model"]
        provider = entry.get("provider", "")
        rows = entry["results"]
        if not rows:
            print(f"\n{name} ({provider}): no results")
            continue

        ok_rows = [r for r in rows if r["status"] == "ok"]
        errors = [r for r in rows if r["status"] != "ok"]
        n_correct = sum(1 for r in ok_rows if r["correct"])

        print(f"\n{name} ({provider})")
        print(f"  Calls succeeded: {len(ok_rows)}/{len(rows)}")
        if ok_rows:
            print(f"  Answers correct: {n_correct}/{len(ok_rows)}")
        if errors:
            kinds = {}
            for r in errors:
                kinds[r["status"]] = kinds.get(r["status"], 0) + 1
            print(f"  Errors encountered: {kinds}")

        by_family = {}
        for r in ok_rows:
            by_family.setdefault(r["family"], []).append(r["correct"])
        for family, flags in sorted(by_family.items()):
            print(f"    {family:8s}: {sum(flags)}/{len(flags)} correct")

    print("\n" + "-" * 78)


def main():
    parser = argparse.ArgumentParser(description="Smoke test models on GSM8K / GSM-IC / BBH")
    parser.add_argument("--provider", type=str, help="Only test models for this provider (e.g. motapis, groq, google, huggingface)")
    parser.add_argument("--model", type=str, help="Only test this specific model name")
    parser.add_argument("--questions", type=int, default=1, help="Number of questions per dataset family (default 1 to save tokens)")
    args = parser.parse_args()

    models_to_test = TEST_MODELS
    if args.provider:
        models_to_test = [m for m in models_to_test if m["provider"].lower() == args.provider.lower()]
    if args.model:
        models_to_test = [m for m in models_to_test if args.model.lower() in m["name"].lower()]

    if not models_to_test:
        print(f"No models matched provider={args.provider}, model={args.model}.")
        print("Available models:")
        for m in TEST_MODELS:
            print(f"  - {m['name']} ({m['provider']})")
        sys.exit(1)

    questions = load_test_questions(questions_per_dataset=args.questions)
    total_calls = len(questions) * len(models_to_test)
    print(f"Loaded {len(questions)} test questions across {len(DATASET_FAMILIES)} dataset families.")
    print(f"Testing {len(models_to_test)} model(s). Max calls planned: {total_calls} ({args.questions} per family/model).")

    all_results = []
    for model_cfg in models_to_test:
        all_results.append(test_one_model(model_cfg, questions))

    print_summary(all_results)

    flat = [r for entry in all_results for r in entry["results"]]
    if flat:
        out_path = os.path.join(config.DATA_DIR, "00_model_smoke_test_results.csv")
        pd.DataFrame(flat).to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)