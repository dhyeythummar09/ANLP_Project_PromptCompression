"""Recover clean GSM-IC questions from the exact source JSON, without API calls.

Uses the original loader's gsmic_<source-index> convention and verifies the full
distracted question before attaching any clean text. Never guesses a pairing.
"""
import argparse
import json
import re
import pandas as pd
from final_eval.utils.artifacts import atomic_csv, atomic_json, file_hash, claim_output

def recover(problems, source):
    out = problems.copy().fillna("")
    if "clean_question" not in out:
        out["clean_question"] = ""
    def present(row, names):
        return next((row[n] for n in names if n in row), None)
    for index, row in out[out.source_dataset == "gsm_ic"].iterrows():
        match = re.fullmatch(r"gsmic_(\d+)", str(row.example_id))
        if not match:
            raise ValueError("Unrecognized source-index convention; supply explicit provenance")
        position = int(match.group(1))
        if position >= len(source):
            raise ValueError("Source JSON does not contain the saved source index")
        item = source[position]
        new = present(item, ["new_question", "question_with_context", "question"])
        clean = present(item, ["original_question", "orig_question"])
        if new != row.original_prompt or not isinstance(clean, str) or not clean.strip():
            raise ValueError(f"Source mismatch or missing clean text: {row.example_id}")
        out.loc[index, "clean_question"] = clean
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--source-json", required=True, help="Exact GSM-IC_2step.json used for the original collection")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    claim_output(args.output)
    frame = pd.read_csv(args.problems, keep_default_na=False, dtype={"ground_truth_answer": str})
    source = json.loads(open(args.source_json, encoding="utf-8").read())
    if not isinstance(source, list):
        raise ValueError("Expected the original list-of-records source JSON")
    atomic_csv(recover(frame, source), args.output)
    atomic_json(str(args.output) + ".source.json", {"problem_input": file_hash(args.problems),
        "gsmic_source": file_hash(args.source_json), "pairing": "source index plus exact distracted-question equality"})

if __name__ == "__main__":
    main()
