"""Explicit, non-destructive team-shard import and response regrading."""
import argparse
import json
from pathlib import Path
import pandas as pd
from final_eval.preparation.schema import prepare_problems, normalize_votes, build_consensus, POSITION
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, PROTOCOL_VERSION
from final_eval.utils.answer_matching import GRADER_VERSION

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True, help="Explicit complete shard list or merged raw-label CSV")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--output-dir", required=True, help="New directory, never an existing dataset")
    ap.add_argument("--minimum-models", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="Audit and regrade in memory; write nothing")
    args = ap.parse_args()
    if len({str(Path(p).resolve()) for p in args.labels}) != len(args.labels):
        ap.error("Duplicate input paths")
    inputs = {str(Path(p)): file_hash(p) for p in [args.problems] + args.labels}
    problems = prepare_problems(pd.read_csv(args.problems, keep_default_na=False, dtype={"ground_truth_answer": str}))
    raw = pd.concat([pd.read_csv(p, keep_default_na=False, dtype={"ground_truth_answer": str}).assign(input_file=str(p))
                     for p in args.labels], ignore_index=True)
    votes = normalize_votes(raw, problems, regrade=True)
    consensus = build_consensus(votes, args.minimum_models)
    report = {"protocol": PROTOCOL_VERSION, "grader": GRADER_VERSION, "inputs": inputs,
              "raw_vote_rows": len(raw), "unique_model_votes": len(votes),
              "vote_status_counts": votes.vote_status.value_counts().to_dict(),
              "original_parse_status_counts": votes.parse_status_original.value_counts().to_dict(),
              "perturbed_parse_status_counts": votes.parse_status_perturbed.value_counts().to_dict(),
              "problem_count": len(problems), "base_groups": problems.base_group_id.nunique(),
              "labeled_questions": votes.uid.nunique(), "reliable_questions": consensus.uid.nunique(),
              "reliable_spans": len(consensus.drop_duplicates(POSITION)),
              "minimum_models": args.minimum_models,
              "scope": "Only the explicitly supplied inputs; completeness requires team confirmation",
              "perturbed_parse_policy": "Nonempty returned but unparseable final answers count as operational failures; parse status is retained"}
    print(json.dumps(report, indent=2, default=int))
    if not args.dry_run:
        target = Path(args.output_dir)
        claim_output(target)
        target.mkdir(parents=True)
        atomic_csv(problems, target / "problems.csv")
        atomic_csv(votes, target / "per_model_labels.csv")
        atomic_csv(consensus, target / "consensus_dataset.csv")
        atomic_json(target / "manifest.json", report)

if __name__ == "__main__":
    main()
