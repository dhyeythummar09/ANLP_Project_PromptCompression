"""Regrade legacy raw evaluation responses without new model calls."""
import argparse
import json
import pandas as pd
from final_eval.preparation.schema import prepare_problems
from final_eval.utils.answer_matching import grade_answer, GRADER_VERSION
from final_eval.utils.artifacts import atomic_csv, atomic_json, file_hash, claim_output, require_columns, require_unique

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = pd.read_csv(args.results, keep_default_na=False)
    require_columns(result, ["uid", "eval_model", "method", "target_ratio", "response"])
    require_unique(result, ["uid", "eval_model", "method", "target_ratio"], "legacy evaluation")
    problems = prepare_problems(pd.read_csv(args.problems, keep_default_na=False, dtype={"ground_truth_answer": str})).set_index("uid")
    records = []
    for r in result.to_dict("records"):
        if r["uid"] not in problems.index:
            raise ValueError(f"Unknown question: {r['uid']}")
        problem = problems.loc[r["uid"]]
        if "ground_truth" in r and str(r["ground_truth"]) != str(problem.ground_truth_answer):
            raise ValueError("Gold-answer conflict with provided problem version")
        grade = grade_answer(r["response"], problem.ground_truth_answer, problem.original_prompt)
        # Historical evaluator sliced raw text at 1,500 chars. Do not claim a
        # regrade is verifiable if the final answer may have been cut off.
        cut = len(r["response"]) == 1500 and grade.status != "ok"
        r.update(legacy_correct=r.get("correct", ""), correct="" if cut else grade.correct,
                 parse_status=grade.status, answer=grade.answer, grader_version=GRADER_VERSION,
                 response_status="legacy_truncated_unverifiable" if cut else "ok" if r["response"].strip() else "empty_response",
                 base_group_id=problem.base_group_id, source_dataset=problem.source_dataset,
                 legacy_model_identity="unverified_historical_alias")
        records.append(r)
    graded = pd.DataFrame(records)
    print(json.dumps({"rows": len(graded), "status": graded.response_status.value_counts().to_dict()}, indent=2))
    if not args.dry_run:
        claim_output(args.output)
        atomic_csv(graded, args.output)
        atomic_json(str(args.output) + ".regrade.json", {"input_hash": file_hash(args.results),
                    "problems_hash": file_hash(args.problems), "grader": GRADER_VERSION,
                    "limitations": "Original responses may be truncated; original source-model identity requires separate provenance."})

if __name__ == "__main__":
    main()
