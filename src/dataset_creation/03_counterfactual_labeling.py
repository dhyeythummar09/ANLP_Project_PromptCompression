"""
STEP 3: The core loop. For every candidate span, evaluate the ORIGINAL
question once per problem per model (cached to disk), then evaluate the
PERTURBED question. Record whether each model's answer flipped from
correct to incorrect.

Includes:
  - provider-level rate-limiting (requests per MINUTE)
  - daily-quota tracking (requests per DAY) -- this is the cap that
    actually matters. RPM just controls how fast you queue up against the
    RPD ceiling; it doesn't prevent hitting it.
  - disk-cached baseline ("original question") answers, so they're never
    re-asked even across separate days
  - problem-level filtering (skip a model on a problem it already got
    wrong before touching any spans)
  - automatic checkpointing, so a crash or a deliberate stop never loses
    completed work
  - round-robin workload distribution across teammates
  - graceful handling when a model's daily cap is hit: that ONE model is
    skipped for the rest of today, other models keep going, and the whole
    script only exits once every model is exhausted for the day -- never
    just spins retrying against a wall it can't get past.

Run: python src/dataset_creation/03_counterfactual_labeling.py
Output: data/03_per_model_labels_part{WORKER_ID}.csv
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.utils.api_clients import call_model
from src.utils.answer_matching import answers_match

# --- TEAM DISTRIBUTION SETTINGS ---
WORKER_ID = int(os.environ.get("WORKER_ID", getattr(config, "WORKER_ID", 1)))
TOTAL_WORKERS = 4

# --- CLI: optional single-model mode for parallel execution ---
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", type=str, default=None,
                     help="Run only this model (by name). Launches an isolated process "
                          "with its own output/cache/log files — safe to run in parallel.")
_args, _ = _parser.parse_known_args()
MODEL_FILTER = _args.model  # None = run all models sequentially (default)

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)

# Slug used in filenames: empty when running all models, model-name-based when single-model
_model_slug = ("_" + MODEL_FILTER.replace("/", "-")) if MODEL_FILTER else ""

# Each parallel process gets fully isolated files — no write conflicts
WORKER_LABELS_FILE = config.LABELS_FILE.replace(".csv", f"_part{WORKER_ID}{_model_slug}.csv")
WORKER_CACHE_FILE = os.path.join(config.DATA_DIR, f"03_original_answers_cache_part{WORKER_ID}{_model_slug}.json")
WORKER_LOG_FILE = config.LABELING_LOG_FILE.replace(".log", f"_part{WORKER_ID}{_model_slug}.log")
WORKER_USAGE_FILE = os.path.join(config.DATA_DIR, f"03_daily_usage_part{WORKER_ID}{_model_slug}.json")

logging.basicConfig(
    filename=WORKER_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("counterfactual_labeling")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

PROMPT_TEMPLATE = """Solve this problem. You may reason briefly, but you MUST end your \
response with a line in exactly this format:
Final Answer: <your answer>

Problem: {question}
"""

CHECKPOINT_EVERY = 25

# Minimum seconds between calls to the same provider, to stay under rate limits
PROVIDER_MIN_SLEEP = {
    "motapis_qwen": 0.5,     # Account 1
    "motapis_glm": 0.5,      # Account 2
    "groq": 2.2,        # ~30 RPM free tier
    "google": 4.2,      # Google AI Studio
}

# Daily request caps per model
DAILY_CALL_CAP = {
    "qwen3.7-max": 5000,
    "glm-5.3-flash": 5000,
    "openai/gpt-oss-120b": 1000,
    "gemini-3.6-flash": 1500,
}


# ---------------------------------------------------------------------------
# Daily usage tracking
# ---------------------------------------------------------------------------
def _today_str() -> str:
    return datetime.date.today().isoformat()


def _load_daily_usage() -> dict:
    """
    Load today's per-model call counts. If the stored date isn't today
    (i.e. we're running on a new day, or for the first time), start fresh
    -- this is what makes "try again tomorrow" actually work: the cap
    resets automatically just by the date changing, no manual reset needed.
    """
    if os.path.exists(WORKER_USAGE_FILE):
        try:
            with open(WORKER_USAGE_FILE) as f:
                saved = json.load(f)
            if saved.get("date") == _today_str():
                return saved.get("counts", {})
        except Exception as e:
            logger.warning("Could not read usage file, starting fresh: %s", e)
    return {}


def _save_daily_usage(counts: dict):
    try:
        with open(WORKER_USAGE_FILE, "w") as f:
            json.dump({"date": _today_str(), "counts": counts}, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save usage file: %s", e)


def _is_exhausted(model_name: str, counts: dict) -> bool:
    cap = DAILY_CALL_CAP.get(model_name)
    if cap is None:
        return False  # no configured cap -- verify this is intentional
    return counts.get(model_name, 0) >= cap


def _record_call(model_name: str, counts: dict):
    counts[model_name] = counts.get(model_name, 0) + 1
    _save_daily_usage(counts)


# ---------------------------------------------------------------------------
# Existing-work loading (resumability)
# ---------------------------------------------------------------------------
def _load_cache() -> dict:
    if os.path.exists(WORKER_CACHE_FILE):
        try:
            with open(WORKER_CACHE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read existing cache file, starting fresh: %s", e)
    return {}


def _save_cache(cache: dict):
    try:
        with open(WORKER_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save cache file: %s", e)


def _load_existing_labels():
    if os.path.exists(WORKER_LABELS_FILE):
        try:
            done = pd.read_csv(WORKER_LABELS_FILE)
            done_keys = set(
                zip(
                    done["example_id"].astype(str),
                    done["span_text"].astype(str),
                    done["span_start"].astype(int),
                    done["labeling_model"].astype(str),
                )
            )
            return done, done_keys
        except Exception as e:
            logger.warning("Could not parse existing labels file: %s", e)
    return pd.DataFrame(), set()


def _wait_for_rate_limit(provider: str, model_cfg: dict, last_call_timestamps: dict):
    configured_sleep = model_cfg.get("sleep_seconds", 0.0)
    min_required_sleep = PROVIDER_MIN_SLEEP.get(provider.lower(), 2.0)
    delay = max(configured_sleep, min_required_sleep)

    last_call = last_call_timestamps.get(provider, 0.0)
    elapsed = time.time() - last_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    last_call_timestamps[provider] = time.time()


def _get_original_answer(
    cache: dict,
    example_id: str,
    model_cfg: dict,
    original_prompt: str,
    last_call_timestamps: dict,
    daily_counts: dict,
) -> str | None:
    """
    Fetch (or return the cached) baseline answer for an unperturbed
    question. Cached PERMANENTLY across days -- once a model has answered
    a given problem's original question, we never ask again, regardless
    of how many days this takes overall.
    """
    cache_key = f"{example_id}::{model_cfg['name']}"
    if cache_key in cache:
        return cache[cache_key]

    provider = model_cfg.get("provider", "groq")
    _wait_for_rate_limit(provider, model_cfg, last_call_timestamps)

    prompt = PROMPT_TEMPLATE.format(question=original_prompt)
    response = call_model(model_cfg, prompt)
    _record_call(model_cfg["name"], daily_counts)

    if response is not None:
        cache[cache_key] = response
        _save_cache(cache)
    else:
        logger.error("Original call failed after retries: example=%s model=%s", example_id, model_cfg["name"])
        time.sleep(5.0)

    return response


def main():

    if not os.path.exists(config.SPANS_FILE):
        logger.error("Spans file not found: %s", config.SPANS_FILE)
        sys.exit(1)

    # Filter to a single model when --model is specified (parallel mode)
    active_models = config.MODELS
    if MODEL_FILTER:
        active_models = [m for m in config.MODELS if MODEL_FILTER.lower() in m["name"].lower()]
        if not active_models:
            logger.error(
                "--model '%s' didn't match any model in config.MODELS. Available: %s",
                MODEL_FILTER, [m['name'] for m in config.MODELS]
            )
            sys.exit(1)
        logger.info("Running in single-model mode: %s", active_models[0]['name'])

    spans_df = pd.read_csv(config.SPANS_FILE).fillna("")
    done_df, done_keys = _load_existing_labels()
    original_cache = _load_cache()
    daily_counts = _load_daily_usage()

    last_call_timestamps = {}
    new_results = []

    logger.info("Loaded %d candidate spans from %s", len(spans_df), config.SPANS_FILE)
    logger.info("Loaded %d previously completed labels for Worker %d", len(done_keys), WORKER_ID)
    logger.info("Today's usage so far: %s", daily_counts)

    exhausted_models = set()

    for model_cfg in active_models:
        model_name = model_cfg["name"]
        provider = model_cfg.get("provider", "groq")

        if _is_exhausted(model_name, daily_counts):
            logger.info("Model %s already hit today's cap before this run started -- skipping.", model_name)
            exhausted_models.add(model_name)
            continue

        logger.info("--- Starting Model: %s (Provider: %s) ---", model_name, provider)
        problem_groups = spans_df.groupby("example_id", sort=False)

        for problem_index, (example_id, group) in enumerate(problem_groups):
            if (problem_index % TOTAL_WORKERS) + 1 != WORKER_ID:
                continue

            # Check BEFORE every call, not just once at the top of the model
            # loop -- we can cross the cap mid-problem, and need to notice
            # immediately rather than after wastefully finishing the batch.
            if _is_exhausted(model_name, daily_counts):
                logger.info(
                    "Model %s's daily limit is reached for today -- try again tomorrow. "
                    "Moving on to remaining models.",
                    model_name,
                )
                exhausted_models.add(model_name)
                break  # stop THIS model's problem loop; other models keep going

            first_row = group.iloc[0]
            orig_prompt = first_row["original_prompt"]
            ground_truth = str(first_row["ground_truth_answer"])

            orig_ans = _get_original_answer(
                original_cache, example_id, model_cfg, orig_prompt, last_call_timestamps, daily_counts
            )
            if orig_ans is None:
                continue

            is_correct_orig = answers_match(orig_ans, ground_truth, orig_prompt)
            if not is_correct_orig:
                continue  # no clean criticality signal possible for this model+problem

            for _, row in group.iterrows():
                span_key = (str(row["example_id"]), str(row["span_text"]), int(row["span_start"]), str(model_name))
                if span_key in done_keys:
                    continue

                if _is_exhausted(model_name, daily_counts):
                    logger.info("Model %s's daily limit is reached for today -- try again tomorrow.", model_name)
                    exhausted_models.add(model_name)
                    break  # stop spans for this problem; outer break below stops the model

                _wait_for_rate_limit(provider, model_cfg, last_call_timestamps)
                pert_prompt = PROMPT_TEMPLATE.format(question=row["perturbed_prompt"])
                pert_ans = call_model(model_cfg, pert_prompt)
                _record_call(model_name, daily_counts)

                if pert_ans is None:
                    logger.error(
                        "Perturbed call failed: example=%s span=%r model=%s",
                        row["example_id"], row["span_text"], model_name,
                    )
                    time.sleep(5.0)
                    continue

                is_correct_pert = answers_match(pert_ans, ground_truth, row["perturbed_prompt"])
                label = "critical" if not is_correct_pert else "non_critical"

                new_results.append({
                    **row.to_dict(),
                    "labeling_model": model_name,
                    "model_answer_original": orig_ans,
                    "is_correct_original": is_correct_orig,
                    "model_answer_perturbed": pert_ans,
                    "is_correct_perturbed": is_correct_pert,
                    "model_specific_label": label,
                    "decoding_settings": "temperature=0",
                })
                done_keys.add(span_key)

                if len(new_results) % CHECKPOINT_EVERY == 0:
                    combined = (
                        pd.concat([done_df, pd.DataFrame(new_results)], ignore_index=True)
                        if not done_df.empty else pd.DataFrame(new_results)
                    )
                    combined.to_csv(WORKER_LABELS_FILE, index=False)
                    logger.info("Checkpoint: %d total labels saved (%d from this session)", len(combined), len(new_results))

            if model_name in exhausted_models:
                break  # exit the problem loop for this model; move to the next model

    # Final save, regardless of how we got here
    if new_results:
        final_df = (
            pd.concat([done_df, pd.DataFrame(new_results)], ignore_index=True)
            if not done_df.empty else pd.DataFrame(new_results)
        )
        final_df.to_csv(WORKER_LABELS_FILE, index=False)
        logger.info("Session complete. Total labels: %d saved to %s", len(final_df), WORKER_LABELS_FILE)
    else:
        logger.info("No new labels produced this session.")

    remaining_models = [m["name"] for m in active_models if m["name"] not in exhausted_models]
    if not remaining_models:
        logger.info(
            "All models' daily limits are reached for today -- try again tomorrow. "
            "Progress is saved; re-running this script tomorrow will pick up exactly where it left off."
        )
    else:
        logger.info("All available spans processed for today's remaining-quota models: %s", remaining_models)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user. Existing checkpoints remain intact.")
        sys.exit(0)