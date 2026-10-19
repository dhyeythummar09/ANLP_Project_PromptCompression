"""Freeze a dataset-balanced scorer-compatible cohort, keeping families intact."""
import argparse
import json
from pathlib import Path
import random
import pandas as pd
from final_eval.preparation.schema import prepare_problems
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-questions", type=int, default=None, help="Upper cap; families are never split to fill it")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    claim_output(args.output)
    problems = prepare_problems(pd.read_csv(args.problems, keep_default_na=False, dtype={"ground_truth_answer": str}))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if split.get("protocol") != "symbiote-v2":
        raise ValueError("This cohort builder requires a v2 family-safe split")
    eligible = problems[problems.uid.isin(split["folds"])].copy()
    # A family may contain label-ineligible variants: do not add those variants
    # without their own held-out scorer assignment.
    strata = {}
    for group, rows in eligible.groupby("base_group_id"):
        source = "+".join(sorted(rows.source_dataset.unique()))
        strata.setdefault(source, []).append((group, len(rows)))
    rng = random.Random(args.seed)
    for queue in strata.values():
        rng.shuffle(queue)
    selected, size = [], 0
    cap = args.max_questions if args.max_questions is not None else len(eligible)
    if cap < 1:
        raise ValueError("Question cap must be positive")
    while any(strata.values()):
        for source in sorted(strata):
            if strata[source]:
                group, count = strata[source].pop()
                if size + count <= cap:
                    selected.append(group); size += count
    result = eligible[eligible.base_group_id.isin(selected)].copy()
    if result.empty:
        raise ValueError("No complete eligible families fit the cap")
    result = result.sort_values(["source_dataset", "base_group_id", "uid"])
    atomic_csv(result, args.output)
    atomic_json(str(args.output) + ".cohort.json", {"problems": file_hash(args.problems), "split": file_hash(args.split),
        "seed": args.seed, "requested_question_cap": cap, "n_questions": len(result),
        "n_base_groups": result.base_group_id.nunique(), "datasets": result.source_dataset.value_counts().to_dict(),
        "excluded_without_fold": len(problems) - len(eligible),
        "selection": "round-robin dataset strata, complete label-eligible families, no outcome-based selection"})
    print(f"Frozen {len(result)} questions / {result.base_group_id.nunique()} base groups at {args.output}")

if __name__ == "__main__":
    main()
