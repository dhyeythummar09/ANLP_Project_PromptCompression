"""
STEP 1: Load GSM8K, GSM-IC, and the 4 BBH subtasks, sample down to the
agreed sizes, and save one flat CSV of raw problems.

GSM-IC (Shi et al., 2023) is GSM8K problems with one irrelevant sentence
deliberately inserted. We include it because pure GSM8K risks a trivial
dataset: in a math word problem almost every number and operation is
load-bearing, so "critical vs non-critical" can collapse into "numbers are
critical, filler words aren't" ==> a distinction that doesn't need
counterfactual testing to discover. GSM-IC gives us a genuine, non-trivial
non-critical class (the inserted sentence) inside a math-reasoning setting.

Run: python src/dataset_creation/01_load_data.py
Output: data/01_raw_problems.csv
"""

import os
import sys
import json

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Let this script be run directly (e.g. `python src/dataset_creation/01_load_data.py`) regardless of the current working directory, by putting the project root on sys.path so `import config` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

os.makedirs(config.DATA_DIR, exist_ok=True)

# Load GSM8K's test split and return a list of raw-problem dicts
def load_gsm8k() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=config.GSM8K_SPLIT)
    ds = ds.shuffle(seed=42)
    n = min(config.GSM8K_N, len(ds))
    if n < config.GSM8K_N:
        print(f"  [warning] requested {config.GSM8K_N} GSM8K problems but only {len(ds)} available")
    ds = ds.select(range(n))

    rows = []
    for i, ex in enumerate(ds):
        # GSM8K answers look like "... #### 42" ==> the line after #### is the ground truth; everything before it is the worked solution, which we don't need here since we're only compressing the question.
        answer = ex["answer"].split("####")[-1].strip()
        rows.append({
            "example_id": f"gsm8k_{i}",
            "source_dataset": "gsm8k",
            "original_prompt": ex["question"],
            "ground_truth_answer": answer,
            "inserted_context": "",  # not applicable outside GSM-IC
        })
    return rows

# Load the 4 configured BBH subtasks and return a list of raw-problem dicts
def load_bbh() -> list[dict]:
    rows = []
    for subtask, n in config.BBH_SUBTASKS.items():
        try:
            bbh = load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception as e:
            # This mirror occasionally goes stale/unavailable. If this fails,
            # try "maveriq/bigbenchhard" as an alternative mirror, or load
            # directly from the official BIG-Bench-Hard GitHub CSVs.
            print(f"  [error] could not load BBH subtask '{subtask}' from lukaemon/bbh: {e}")
            print("  -> try 'maveriq/bigbenchhard' as a fallback mirror, or the official BBH GitHub repo")
            continue

        bbh = bbh.shuffle(seed=42)
        n_available = min(n, len(bbh))
        if n_available < n:
            print(f"  [warning] requested {n} '{subtask}' problems but only {len(bbh)} available")
        bbh = bbh.select(range(n_available))

        # The BBH mirror's schema is inconsistent across subtasks, so we can't trust a single field name for the prompt or answer. Instead, we look for the first field that looks like a question and the first that looks like an answer.
        for i, ex in enumerate(bbh):
            rows.append({
                "example_id": f"bbh_{subtask}_{i}",
                "source_dataset": f"bbh_{subtask}",
                "original_prompt": ex["input"],
                "ground_truth_answer": ex["target"],
                "inserted_context": "",
            })
    return rows


def _find_inserted_sentence(original_question: str, new_question: str) -> str:
    """
    Given a GSM-IC pair (the clean original question and the version with an
    irrelevant sentence inserted), find the sentence that's new.

    We do this by sentence-splitting both on periods and diffing the sets,
    rather than trusting a specific field name -- GSM-IC's exact JSON schema
    isn't guaranteed to match across mirrors/versions, so this is a more
    robust way to recover the inserted span regardless of schema details.
    """
    def sentences(text):
        return [s.strip() for s in text.split(".") if s.strip()]

    orig_sents = set(sentences(original_question))
    new_sents = sentences(new_question)
    inserted = [s for s in new_sents if s not in orig_sents]
    return ". ".join(inserted).strip()


def load_gsm_ic() -> list[dict]:
    """
    Load GSM-IC (Shi et al., 2023) via its HuggingFace mirror and return a
    list of raw-problem dicts, with the inserted irrelevant sentence
    identified for each one (used later in extract_spans.py to tag it as
    the "distractor_context" category directly, rather than re-detecting it
    with a parser).
    """
    try:
        path = hf_hub_download(
            repo_id="voidful/GSM-IC", filename="GSM-IC_2step.json", repo_type="dataset"
        )
    except Exception as e:
        print(f"  [error] could not download GSM-IC: {e}")
        print("  -> check https://huggingface.co/datasets/voidful/GSM-IC for the current filename")
        return []

    with open(path) as f:
        raw = json.load(f)

    # Print one record so you can eyeball the actual field names before trusting the guesses below -- GSM-IC's schema has varied across releases/mirrors.
    if raw:
        print("  [debug] sample GSM-IC record keys:", list(raw[0].keys()))

    # Helper to find the first present field in a dict from a list of possible keys, or None if none are present
    # GSM-IC's schema has varied across releases/mirrors, so we can't trust a single field name for the new question, original question, or answer.
    def first_present(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    rows = []
    n = min(config.GSM_IC_N, len(raw))
    for i, ex in enumerate(raw[:n]):
        new_q = first_present(ex, "new_question", "question_with_context", "question")
        orig_q = first_present(ex, "original_question", "orig_question")
        answer = first_present(ex, "answer", "ground_truth", "target")

        if new_q is None or orig_q is None or answer is None:
            print(f"  [warning] skipping GSM-IC record {i}, missing expected fields")
            continue

        # Find the inserted sentence by diffing the original and new questions, rather than trusting a specific field name ==> GSM-IC's schema isn't guaranteed to be consistent across mirrors/versions.
        inserted = _find_inserted_sentence(orig_q, new_q)
        rows.append({
            "example_id": f"gsmic_{i}",
            "source_dataset": "gsm_ic",
            "original_prompt": new_q,
            "ground_truth_answer": answer,
            "inserted_context": inserted,
        })
    return rows


def main():
    all_rows = []
    print("Loading GSM8K...")
    all_rows.extend(load_gsm8k())
    print("Loading GSM-IC...")
    all_rows.extend(load_gsm_ic())
    print("Loading BBH subtasks...")
    all_rows.extend(load_bbh())

    df = pd.DataFrame(all_rows)
    df.to_csv(config.RAW_PROBLEMS_FILE, index=False)        # Save the raw problems to a CSV file
    print(f"\nSaved {len(df)} problems to {config.RAW_PROBLEMS_FILE}")
    print(df["source_dataset"].value_counts())


if __name__ == "__main__":
    main()
