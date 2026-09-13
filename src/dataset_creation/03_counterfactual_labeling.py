"""
STEP 3: The core loop. For every candidate span, ask each of the 4 models
the ORIGINAL question once (cached per example+model), then ask the
PERTURBED question, and record whether each model's answer changed from
correct to incorrect.

This is the expensive step (many API calls) -- run it in the background,
it can take a while depending on rate limits. It's resumable: already-
labeled rows are skipped if you re-run it after a crash or a stop.

Run: python src/dataset_creation/03_counterfactual_labeling.py
Output: data/03_per_model_labels.csv
"""

import os
import sys
import time
import logging

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.utils.api_clients import call_model
from src.utils.answer_matching import answers_match

os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=config.LABELING_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("counterfactual_labeling")

# Allowing brief reasoning (unlike the original bare-answer prompt) and asking for an explicit marker before the final answer, so
# extract_final_answer() in answer_matching.py has something reliable to find regardless of how much the model reasons first.
PROMPT_TEMPLATE = """Solve this problem. You may reason briefly, but you MUST end your \
response with a line in exactly this format:
Final Answer: <your answer>

Problem: {question}
"""

# How often (in number of new labels produced) to write a checkpoint to disk, so a crash partway through doesn't lose everything.
CHECKPOINT_EVERY = 50


def _load_existing_labels():
    """Load already-labeled rows (if any) so a re-run skips them, and build
    a lookup set of (example_id, span_text, span_start, model) keys already done."""
    if os.path.exists(config.LABELS_FILE):
        done = pd.read_csv(config.LABELS_FILE)
        done_keys = set(zip(done["example_id"], done["span_text"], done["span_start"], done["labeling_model"]))
    else:
        done = pd.DataFrame()
        done_keys = set()
    return done, done_keys


def _get_original_answer(cache: dict, example_id, model_cfg, original_prompt):
    """
    Get (and cache) a model's answer to the UNPERTURBED question. Cached per
    (example_id, model) so we ask the original question once per problem per
    model, no matter how many spans that problem has.
    """
    cache_key = (example_id, model_cfg["name"])
    if cache_key in cache:
        return cache[cache_key]

    response = call_model(model_cfg, PROMPT_TEMPLATE.format(question=original_prompt))
    if response is None:
        logger.error("Original-question call failed after retries: example=%s model=%s", example_id, model_cfg["name"])
    cache[cache_key] = response
    time.sleep(model_cfg["sleep_seconds"])
    return response


def main():
    spans = pd.read_csv(config.SPANS_FILE)
    done, done_keys = _load_existing_labels()

    original_answer_cache: dict = {}
    results = []
    total_calls_planned = len(spans) * len(config.MODELS)
    n_attempted = 0

    for _, row in spans.iterrows():
        for model_cfg in config.MODELS:
            n_attempted += 1
            key = (row["example_id"], row["span_text"], row["span_start"], model_cfg["name"])
            if key in done_keys:
                continue

            orig_ans = _get_original_answer(original_answer_cache, row["example_id"], model_cfg, row["original_prompt"])
            if orig_ans is None:
                continue  # API call failed after retries; skip, already logged

            is_correct_original = answers_match(orig_ans, row["ground_truth_answer"])
            if not is_correct_original:
                # This model was already wrong before we touched anything ==> removing a span can't give us a clean "did this cause the
                # failure" signal here, so skip this model for this problem.
                continue

            pert_ans = call_model(model_cfg, PROMPT_TEMPLATE.format(question=row["perturbed_prompt"]))
            time.sleep(model_cfg["sleep_seconds"])
            if pert_ans is None:
                logger.error(
                    "Perturbed-question call failed after retries: example=%s span=%r model=%s",
                    row["example_id"], row["span_text"], model_cfg["name"],
                )
                continue

            is_correct_perturbed = answers_match(pert_ans, row["ground_truth_answer"])
            label = "critical" if (is_correct_original and not is_correct_perturbed) else "non_critical"

            results.append({
                **row.to_dict(),
                "labeling_model": model_cfg["name"],
                "model_answer_original": orig_ans,
                "is_correct_original": is_correct_original,
                "model_answer_perturbed": pert_ans,
                "is_correct_perturbed": is_correct_perturbed,
                "model_specific_label": label,
                "decoding_settings": "temperature=0",
            })

            if len(results) % CHECKPOINT_EVERY == 0:
                print(f"  progress: {n_attempted}/{total_calls_planned} calls attempted, {len(results)} new labels so far")
                pd.concat([done, pd.DataFrame(results)], ignore_index=True).to_csv(config.LABELS_FILE, index=False)

    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True) if results else done
    final.to_csv(config.LABELS_FILE, index=False)
    print(f"Done. {len(final)} total per-model labels saved to {config.LABELS_FILE}")
    print(f"See {config.LABELING_LOG_FILE} for any calls that failed after retries.")


if __name__ == "__main__":
    main()
