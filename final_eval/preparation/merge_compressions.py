"""Merge explicit method-stage outputs after running GPU-heavy baselines separately."""
import argparse
import pandas as pd
from final_eval.utils.artifacts import atomic_csv, atomic_json, file_hash, claim_output, require_unique, require_columns

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-methods", nargs="+", required=True)
    args = ap.parse_args()
    claim_output(args.output)
    frames = [pd.read_csv(p, keep_default_na=False) for p in args.inputs]
    combined = pd.concat(frames, ignore_index=True)
    require_columns(combined, ["uid", "method", "target_ratio", "original_prompt", "ground_truth_answer", "base_group_id", "tokenizer", "status"])
    require_unique(combined, ["uid", "method", "target_ratio"], "merged compression")
    if set(combined.method) != set(args.expected_methods):
        raise ValueError("Actual and expected method sets differ")
    if combined.tokenizer.nunique() != 1:
        raise ValueError("Cannot combine different token-accounting tokenizers")
    for col in ["original_prompt", "ground_truth_answer", "base_group_id"]:
        if combined.groupby("uid")[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Input versions disagree in {col}")
    for _, g in combined.groupby(["uid", "target_ratio"]):
        if set(g.method) != set(args.expected_methods):
            raise ValueError("Incomplete method grid; run the same cohort/ratios for every stage")
    # Also require the same question/ratio grid in all method stages.
    grids = [set(g[["uid", "target_ratio"]].itertuples(index=False, name=None)) for _, g in combined.groupby("method")]
    if any(grid != grids[0] for grid in grids):
        raise ValueError("Different method cohorts/budgets")
    atomic_csv(combined, args.output)
    atomic_json(str(args.output) + ".merge.json", {"inputs": {p: file_hash(p) for p in args.inputs},
        "methods": args.expected_methods, "rows": len(combined), "questions": combined.uid.nunique(),
        "status_counts": combined.status.value_counts().to_dict()})
    print(f"Merged {len(combined)} conditions; inspect the status counts before evaluation")

if __name__ == "__main__":
    main()
