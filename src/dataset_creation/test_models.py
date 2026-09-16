"""
test_models.py -- a one-off smoke test, NOT part of the labeling pipeline.

Purpose: before committing to a multi-day labeling run, verify three things
for each of our 3 models:
  1. Does the API key work at all? (auth / 401)
  2. Is the model ID correct? (404 / "model not found")
  3. Do we actually get correct answers back on real problems from each of
     our 3 dataset families?

It also reports HTTP 429s distinctly, so you can tell "my quota is gone"
apart from "my key is wrong" -- those look similar in a generic error
message but need completely different fixes.

IMPORTANT: this script is deliberately self-contained for API calls (it does
not import src/utils/api_clients.py). That's so you can run it BEFORE
refactoring api_clients.py to add Cerebras -- it's an independent check, not
a test of our own wrapper code. It does reuse answer_matching.py, since
verifying that our correctness-checking logic works on real model output is
part of what we want to test here.

Cost warning: with the defaults below (5 questions x 3 models = 15 calls),
this burns 5 of Gemini's ~20 daily requests. Run it once; don't re-run
casually.

Place in: src/dataset_creation/test_models.py
Run:      python src/dataset_creation/test_models.py
"""

import os
import sys
import time
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
#
# NOTE on sleep values: these come straight from the measured rate-limit
# table, NOT from config.MODELS. Gemini in particular needs 12.5s (5 RPM ->
# 60/5 = 12s + buffer); a 4.2s value would be ~14 RPM and would 429 within
# seconds. Fix config.MODELS to match before running the real pipeline.
# ---------------------------------------------------------------------------
TEST_MODELS = [
    # {
    #     "name": "openai/gpt-oss-120b",
    #     "provider": "groq",
    #     "sleep_seconds": 2.2,      # 30 RPM
    #     "daily_cap": 1000,
    # },
    # {
    #     "name": "gemini-3.6-flash",
    #     "provider": "google",
    #     "sleep_seconds": 12.5,     # 5 RPM -- NOT 4.2
    #     "daily_cap": 20,           # brutally low; see note in the chat
    # },
    {
        "name": "qwen2.5",
        "provider": "ollama",
        "sleep_seconds": 0.0,      # No rate limits on your own laptop!
        "daily_cap": 999999,       # Unlimited calls
    },
    # {
    #     "name": "qwen-3.8-27b",
    #     "provider": "cerebras",
    #     "sleep_seconds": 0.2,      # 450 RPM
    #     "daily_cap": 648_000,
    # },
]

# How many questions to pull from each dataset family. 2 per family = 6
# questions = 18 calls total, which stays under Gemini's 20/day ceiling.
# Raising this to 5 (15 questions, 45 calls) will exhaust Gemini partway
# through -- the script handles that gracefully, but you'll get an
# incomplete picture for that one model.
QUESTIONS_PER_DATASET = 2

# Which dataset families to sample from. BBH is matched by prefix since it's
# stored as "bbh_<subtask>" in the raw problems file.
DATASET_FAMILIES = {
    "GSM8K": lambda s: s == "gsm8k",
    "GSM-IC": lambda s: s == "gsm_ic",
    "BBH": lambda s: s.startswith("bbh_"),
}

PROMPT_TEMPLATE = """Solve this problem. You may reason briefly, but you MUST end your \
response with a line in exactly this format:
Final Answer: <your answer>

Problem: {question}
"""

# OpenAI-compatible providers: same request/response shape, different host.
OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "ollama": "http://localhost:11434/v1/chat/completions",
}

PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


class ApiError(Exception):
    """Wraps an API failure with a human-readable diagnosis of what went wrong."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind          # "auth" | "rate_limit" | "not_found" | "network" | "parse" | "other"
        self.detail = detail
        super().__init__(f"[{kind}] {detail}")


def _diagnose_http_error(resp: requests.Response) -> ApiError:
    """
    Turn an HTTP error response into a specific, actionable diagnosis.
    The whole point of this script is telling these cases apart -- a generic
    "request failed" tells you nothing about whether to fix your key, wait
    a day, or correct a model ID.
    """
    status = resp.status_code
    body = resp.text[:300]

    if status in (401, 403):
        return ApiError("auth", f"HTTP {status} -- API key rejected or lacks access. Check your .env. Body: {body}")
    if status == 404:
        return ApiError("not_found", f"HTTP 404 -- model ID not recognized by this provider. Body: {body}")
    if status == 429:
        return ApiError("rate_limit", f"HTTP 429 -- rate limit or daily quota exhausted. Body: {body}")
    if status >= 500:
        return ApiError("other", f"HTTP {status} -- provider-side error, usually transient. Body: {body}")
    return ApiError("other", f"HTTP {status}. Body: {body}")


def _call_openai_compatible(model_cfg: dict, prompt: str, api_key: str) -> str:
    """Call Groq or Cerebras (both expose an OpenAI-compatible chat endpoint)."""
    url = OPENAI_COMPATIBLE_BASE_URLS[model_cfg["provider"]]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model_cfg["name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=45)
    except requests.exceptions.RequestException as e:
        raise ApiError("network", f"Could not reach {url}: {e}")

    if not resp.ok:
        raise _diagnose_http_error(resp)

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as e:
        raise ApiError("parse", f"Unexpected response shape: {e}. Body: {resp.text[:300]}")


# def _call_google(model_cfg: dict, prompt: str, api_key: str) -> str:
#     """Call a Gemini model via the generateContent endpoint."""
#     url = (
#         f"https://generativelanguage.googleapis.com/v1beta/models/"
#         f"{model_cfg['name']}:generateContent?key={api_key}"
#     )
#     body = {
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
#     }
#     try:
#         resp = requests.post(url, json=body, timeout=45)
#     except requests.exceptions.RequestException as e:
#         raise ApiError("network", f"Could not reach Gemini endpoint: {e}")

#     if not resp.ok:
#         raise _diagnose_http_error(resp)

#     try:
#         return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
#     except (KeyError, IndexError, ValueError) as e:
#         # A common cause here is a safety block or an empty candidates list,
#         # which returns HTTP 200 but no usable text -- so this is not
#         # necessarily a bug in our parsing.
#         raise ApiError("parse", f"No usable text in response ({e}). Body: {resp.text[:300]}")


def _call_google(model_cfg: dict, prompt: str, api_key: str) -> str:
    """Call a Gemini model with retries and strict system instructions."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_cfg['name']}:generateContent?key={api_key}"
    )
    
    # Force Gemini to obey formatting by using a strict System Instruction
    body = {
        "systemInstruction": {
            "parts": [{"text": "You are a mathematical and logical reasoning assistant. You may reason briefly, but you MUST end your response with a line in exactly this format:\nFinal Answer: <your answer>"}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
    }
    
    # Add a retry loop to wait out the free-tier 503 High Demand spikes
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, timeout=45)
            
            # If we hit a 503, wait 10 seconds and try again
            if resp.status_code == 503:
                print(f"      [Google 503] Server busy, retrying in 10s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
                continue
                
            if not resp.ok:
                raise _diagnose_http_error(resp)

            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            raise ApiError("network", f"Could not reach Gemini endpoint: {e}")
            
    # If it fails 3 times in a row, then raise the error
    raise ApiError("other", f"HTTP 503 -- Google servers remained busy after {max_retries} retries.")


def call_model(model_cfg: dict, prompt: str) -> str:
    """Dispatch to the right provider. Raises ApiError with a diagnosis on failure."""
    provider = model_cfg["provider"]
    api_key = os.environ.get(PROVIDER_ENV_VARS[provider], "")
    if not api_key:
        raise ApiError("auth", f"{PROVIDER_ENV_VARS[provider]} is not set in your environment/.env")

    if provider in OPENAI_COMPATIBLE_BASE_URLS:
        return _call_openai_compatible(model_cfg, prompt, api_key)
    elif provider == "google":
        return _call_google(model_cfg, prompt, api_key)
    raise ApiError("other", f"Unknown provider: {provider}")


def load_test_questions() -> list[dict]:
    """
    Pull a few problems from each dataset family out of the raw problems CSV.
    Requires 01_load_data.py to have been run first.
    """
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
        # Fixed seed so repeated runs test the same questions and results
        # stay comparable across runs.
        sampled = subset.sample(n=min(QUESTIONS_PER_DATASET, len(subset)), random_state=42)
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
    """Run every test question against one model and collect results."""
    name = model_cfg["name"]
    print("\n" + "=" * 78)
    print(f"TESTING: {name}  (provider: {model_cfg['provider']}, sleep: {model_cfg['sleep_seconds']}s)")
    print("=" * 78)

    n_calls_made = 0
    results = []

    for i, q in enumerate(questions, 1):
        # Respect the daily cap -- stop cleanly rather than hammering 429s.
        if n_calls_made >= model_cfg["daily_cap"]:
            print(f"\n  ! Reached this model's daily cap ({model_cfg['daily_cap']}) -- stopping early.")
            print(f"    {len(questions) - i + 1} question(s) not tested for this model.")
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
            # A rate-limit or auth error will hit every subsequent question
            # too, so there's no point continuing with this model.
            if e.kind in ("auth", "rate_limit", "not_found"):
                print(f"      -> This error affects all remaining questions; skipping rest of {name}.")
                break
            time.sleep(model_cfg["sleep_seconds"])
            continue

        parsed = extract_final_answer(raw)
        correct = answers_match(raw, q["ground_truth"])
        symbol = "PASS" if correct else "FAIL"
        print(f"      Got: {textwrap.shorten(parsed, width=80)}")
        print(f"      {symbol}")

        results.append({**q, "model": name, "status": "ok", "correct": correct,
                        "parsed_answer": parsed, "raw_response": raw})

        time.sleep(model_cfg["sleep_seconds"])

    return {"model": name, "results": results, "calls_made": n_calls_made}


def print_summary(all_results: list[dict]):
    """Print a compact per-model and per-dataset breakdown."""
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for entry in all_results:
        name = entry["model"]
        rows = entry["results"]
        if not rows:
            print(f"\n{name}: no results (model never responded)")
            continue

        ok_rows = [r for r in rows if r["status"] == "ok"]
        errors = [r for r in rows if r["status"] != "ok"]
        n_correct = sum(1 for r in ok_rows if r["correct"])

        print(f"\n{name}")
        print(f"  API calls succeeded: {len(ok_rows)}/{len(rows)}")
        if ok_rows:
            print(f"  Answers correct:     {n_correct}/{len(ok_rows)}")
        if errors:
            kinds = {}
            for r in errors:
                kinds[r["status"]] = kinds.get(r["status"], 0) + 1
            print(f"  Errors: {kinds}")

        # Per-dataset-family accuracy -- useful for spotting a family that
        # systematically fails (e.g. answer-format mismatches on BBH, which
        # would point at answer_matching.py rather than the model).
        by_family = {}
        for r in ok_rows:
            by_family.setdefault(r["family"], []).append(r["correct"])
        for family, flags in sorted(by_family.items()):
            print(f"    {family:8s}: {sum(flags)}/{len(flags)} correct")

    print("\n" + "-" * 78)
    print("How to read this:")
    print("  error:auth       -> wrong/missing key in .env for that provider")
    print("  error:not_found  -> model ID string is wrong; check the provider's docs")
    print("  error:rate_limit -> quota gone; wait, or lower QUESTIONS_PER_DATASET")
    print("  error:parse      -> got HTTP 200 but no usable text (often a safety block)")
    print("  Low accuracy but no errors -> the API works; check whether it's the")
    print("    model genuinely failing, or answer_matching.py mis-parsing the format.")
    print("-" * 78)


def main():
    questions = load_test_questions()
    if not questions:
        print("No test questions could be loaded. Aborting.")
        sys.exit(1)

    total_calls = len(questions) * len(TEST_MODELS)
    print(f"Loaded {len(questions)} test questions across {len(DATASET_FAMILIES)} dataset families.")
    print(f"Will make up to {total_calls} API calls ({len(questions)} per model x {len(TEST_MODELS)} models).")

    gemini = next((m for m in TEST_MODELS if m["provider"] == "google"), None)
    if gemini and len(questions) > gemini["daily_cap"]:
        print(f"\n  ! WARNING: {len(questions)} questions exceeds {gemini['name']}'s "
              f"daily cap of {gemini['daily_cap']}. It will stop partway through.")

    all_results = []
    for model_cfg in TEST_MODELS:
        all_results.append(test_one_model(model_cfg, questions))

    print_summary(all_results)

    # Save raw responses so you can eyeball exactly what each model returned
    # -- essential for debugging answer_matching.py against real output.
    flat = [r for entry in all_results for r in entry["results"]]
    out_path = os.path.join(config.DATA_DIR, "00_model_smoke_test_results.csv")
    pd.DataFrame(flat).to_csv(out_path, index=False)
    print(f"\nFull results (including raw responses) saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)