"""Paired historical-checkpoint versus v2 downstream comparison (NEXT_STEPS step 2).

Both checkpoints' compressed prompts go to the *same* reader in *one* run:

1. ``prepare`` merges the two compression tables into one input. Methods are
   renamed to distinct IDs (``ours_historical`` / ``ours_v2``) so the two
   checkpoints never collide, and the provenance is kept in extra columns.
2. ``run`` sends that table through ``final_eval.compressor.evaluate``. Because
   it is a single output file, each original question is asked once and any
   compressed prompt that both checkpoints produced identically is asked once
   (the evaluator's query cache), which is most of them.
3. ``summarize`` pairs historical and v2 answers per question, budget and
   method, and reports the v2-minus-historical accuracy delta with
   family-clustered bootstrap intervals.

Reader answers from different readers or runs are never mixed: the summary
refuses a responses file that contains more than one ``eval_model``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_eval.compressor.evaluate import KEY, PROMPT_TEMPLATE
from final_eval.followup_reader import add_run_arguments, launch
from final_eval.utils.artifacts import atomic_csv, atomic_json, file_hash, parse_bool, require_unique
from final_eval.utils.statistics import cluster_ci

CHECKPOINTS = ("historical", "v2")
# Columns that describe the question itself; both checkpoints must agree on them.
QUESTION_COLUMNS = ["original_prompt", "ground_truth_answer", "base_group_id", "source_dataset"]


def paired_method(method: str, checkpoint: str) -> str:
    return f"{method}_{checkpoint}"


def _load_compression(path: str, methods: list[str], ratios: list[float]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype={"ground_truth_answer": str})
    require_unique(frame, ["uid", "method", "target_ratio"], f"compression: {path}")
    required = {"uid", "method", "target_ratio", "status", "budget_compliant",
                "compressed_prompt", *QUESTION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = frame[frame.method.isin(methods) & frame.target_ratio.isin(ratios)].copy()
    found = set(zip(selected.method, selected.target_ratio))
    absent = {(m, r) for m in methods for r in ratios} - found
    if absent:
        raise ValueError(f"{path} lacks method/ratio combinations: {sorted(absent)}")
    if not selected.status.eq("ok").all() or not selected.budget_compliant.map(parse_bool).all():
        raise ValueError(f"{path} contains failed or over-budget compression rows")
    return selected


def build_paired_input(historical: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """Stack both checkpoints with distinct method IDs after checking they match."""
    if set(historical.uid) != set(v2.uid):
        raise ValueError("Historical and v2 compression inputs have different question IDs")
    grid = ["uid", "method", "target_ratio"]
    if set(map(tuple, historical[grid].values)) != set(map(tuple, v2[grid].values)):
        raise ValueError("Historical and v2 compression grids differ")
    # The evaluator insists on one original prompt / answer / family per uid,
    # so check that up front with a readable message.
    questions = pd.concat([historical, v2])[["uid", *QUESTION_COLUMNS]].drop_duplicates()
    if questions.uid.duplicated().any():
        bad = questions[questions.uid.duplicated(keep=False)].uid.unique()[:3]
        raise ValueError(f"Checkpoints disagree on question details, e.g. {list(bad)}")
    stacked = []
    for checkpoint, frame in zip(CHECKPOINTS, (historical, v2)):
        frame = frame.copy()
        frame["base_method"] = frame.method
        frame["checkpoint"] = checkpoint
        frame["method"] = frame.method.map(lambda m: paired_method(m, checkpoint))
        stacked.append(frame)
    return pd.concat(stacked, ignore_index=True).sort_values(
        ["uid", "base_method", "target_ratio", "checkpoint"]).reset_index(drop=True)


def prepare(args: argparse.Namespace) -> None:
    historical = _load_compression(args.historical_input, args.methods, args.ratios)
    v2 = _load_compression(args.v2_input, args.methods, args.ratios)
    paired = build_paired_input(historical, v2)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"{output} already exists; keep it or choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(paired, output)

    # How often the two checkpoints wrote exactly the same prompt. Those
    # conditions are answered once and can only tie, which is worth reporting.
    merged = historical.merge(v2, on=["uid", "method", "target_ratio"], suffixes=("_h", "_v"))
    identical = {f"{m}@{r}": int((g.compressed_prompt_h == g.compressed_prompt_v).sum())
                 for (m, r), g in merged.groupby(["method", "target_ratio"])}
    unique_queries = len(set(paired.original_prompt) | set(paired.compressed_prompt))
    manifest = {
        "protocol": "old-v2-paired-v1",
        "historical_input": args.historical_input,
        "historical_sha256": file_hash(args.historical_input),
        "v2_input": args.v2_input,
        "v2_sha256": file_hash(args.v2_input),
        "paired_sha256": file_hash(output),
        "methods": args.methods,
        "ratios": args.ratios,
        "questions": int(paired.uid.nunique()),
        "families": int(paired.base_group_id.nunique()),
        "logical_conditions_including_originals":
            int(paired.uid.nunique() + len(paired)),
        "unique_reader_queries_before_retries": int(unique_queries),
        "identical_compressed_text": identical,
    }
    atomic_json(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def run(args: argparse.Namespace) -> None:
    paired = pd.read_csv(args.input, keep_default_na=False)
    if not {"checkpoint", "base_method"} <= set(paired.columns):
        raise ValueError(f"{args.input} is not a paired table; run 'prepare' first")
    launch(args, paired)


def _discordance(group: pd.DataFrame) -> dict:
    """Counts behind the paired delta: who got which question right."""
    h, v = group.historical_correct, group.v2_correct
    return {"both_correct": int(((h == 1) & (v == 1)).sum()),
            "only_v2_correct": int(((h == 0) & (v == 1)).sum()),
            "only_historical_correct": int(((h == 1) & (v == 0)).sum()),
            "both_wrong": int(((h == 0) & (v == 0)).sum())}


def summarize(args: argparse.Namespace) -> None:
    responses = pd.read_csv(args.responses, keep_default_na=False)
    require_unique(responses, KEY, "responses")
    if not {"base_group_id", "response_status", "correct"} <= set(responses.columns):
        raise ValueError("Responses are missing required grading columns")
    readers = sorted(responses.eval_model.unique())
    if len(readers) != 1:
        raise ValueError(f"Expected answers from exactly one reader, found {readers}")
    paired_input = pd.read_csv(args.input, keep_default_na=False)
    lookup = paired_input[["uid", "method", "target_ratio", "base_method", "checkpoint",
                           "compressed_prompt"]]

    graded = responses.merge(lookup, on=["uid", "method", "target_ratio"], how="left",
                             validate="one_to_one")
    originals = graded[graded.method.eq("none")].set_index("uid")
    compressed = graded[graded.method.ne("none")]
    if compressed.checkpoint.eq("").any() or compressed.checkpoint.isna().any():
        raise ValueError("Responses contain methods that are not in the paired input")

    columns = ["uid", "base_group_id", "base_method", "target_ratio", "response_status",
               "correct", "compressed_prompt"]
    left = compressed[compressed.checkpoint.eq("historical")][columns]
    right = compressed[compressed.checkpoint.eq("v2")][columns]
    merged = left.merge(right, on=["uid", "base_method", "target_ratio"], how="outer",
                        suffixes=("_historical", "_v2"), validate="one_to_one")
    if merged.isna().any(axis=None):
        raise ValueError("Historical and v2 response grids do not match; finish the run first")

    # Only questions where both arms (and the original) answered cleanly are
    # compared, and the excluded count is reported rather than hidden.
    ok = (merged.response_status_historical.eq("ok") & merged.response_status_v2.eq("ok")
          & merged.uid.map(originals.response_status).eq("ok"))
    successful = merged[ok].copy()
    successful["base_group_id"] = successful.base_group_id_historical
    successful["historical_correct"] = successful.correct_historical.map(parse_bool).astype(int)
    successful["v2_correct"] = successful.correct_v2.map(parse_bool).astype(int)
    successful["original_correct"] = successful.uid.map(originals.correct).map(parse_bool).astype(int)
    successful["identical_text"] = (successful.compressed_prompt_historical
                                    == successful.compressed_prompt_v2)
    successful["delta_v2_minus_historical"] = successful.v2_correct - successful.historical_correct

    summaries = []
    for (method, ratio), group in successful.groupby(["base_method", "target_ratio"], sort=True):
        planned = merged[merged.base_method.eq(method) & merged.target_ratio.eq(ratio)]
        delta = group.delta_v2_minus_historical.to_numpy()
        ci_low, ci_high = cluster_ci(delta, group.base_group_id.to_numpy(), args.bootstrap)
        differing = group[~group.identical_text]
        if len(differing) and differing.base_group_id.nunique() > 1:
            diff_low, diff_high = cluster_ci(differing.delta_v2_minus_historical.to_numpy(),
                                             differing.base_group_id.to_numpy(), args.bootstrap)
        else:
            diff_low = diff_high = float("nan")
        summaries.append({
            "method": method,
            "target_ratio": float(ratio),
            "planned_pairs": len(planned),
            "paired_conditions": len(group),
            "excluded_failed_or_missing": len(planned) - len(group),
            "families": int(group.base_group_id.nunique()),
            "original_accuracy": float(group.original_correct.mean()),
            "historical_accuracy": float(group.historical_correct.mean()),
            "v2_accuracy": float(group.v2_correct.mean()),
            "delta_v2_minus_historical": float(delta.mean()),
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "identical_text_pairs": int(group.identical_text.sum()),
            # The informative subset: identical prompts get one shared answer
            # and so always tie, which dilutes the overall delta toward zero.
            "differing_text_pairs": len(differing),
            "delta_on_differing_text": float(differing.delta_v2_minus_historical.mean())
                if len(differing) else float("nan"),
            "delta_on_differing_ci95_low": diff_low,
            "delta_on_differing_ci95_high": diff_high,
            **_discordance(group),
        })

    # What the provider says it actually served, counted over fresh HTTP answers
    # only (cache reuses copy the same metadata and would double count).
    served = {}
    if {"generation_served_model", "reused_generation"} <= set(responses.columns):
        fresh = responses[responses.response_status.eq("ok")
                          & ~responses.reused_generation.map(parse_bool)]
        served = fresh.generation_served_model.replace("", "(not reported)").value_counts().to_dict()

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "comparison": "historical checkpoint versus v2, same reader, single paired run",
        "reader_requested": readers[0],
        "reader_served_counts": served,
        "wrapper": PROMPT_TEMPLATE,
        "responses_sha256": file_hash(args.responses),
        "paired_input_sha256": file_hash(args.input),
        "response_statuses": responses.response_status.value_counts().to_dict(),
        "original_questions_ok": int(originals.response_status.eq("ok").sum()),
        "rows_with_all_successful": len(successful),
        "bootstrap": args.bootstrap,
        "clustering": "base_group_id",
        "summary": summaries,
        "note": "Unadjusted 95% family-clustered bootstrap CIs. Identical prompts share one answer.",
    }
    atomic_json(output, report)
    successful.drop(columns=["compressed_prompt_historical", "compressed_prompt_v2"]).to_csv(
        output.with_suffix(".paired.csv"), index=False)
    pd.DataFrame(summaries).to_csv(output.with_suffix(".summary.csv"), index=False)
    print(json.dumps({"report": str(output), "paired_rows": len(successful),
                      "summary": summaries}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Merge both checkpoints into one paired input")
    prep.add_argument("--historical-input", required=True)
    prep.add_argument("--v2-input", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--methods", nargs="+", default=["ours", "grouped_hybrid"])
    prep.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.75])
    prep.set_defaults(function=prepare)

    run_parser = subparsers.add_parser("run", help="Ask the reader every paired condition")
    add_run_arguments(run_parser)
    run_parser.set_defaults(function=run)

    summary_parser = subparsers.add_parser("summarize", help="Paired old-vs-v2 report")
    summary_parser.add_argument("--input", required=True, help="Paired table from 'prepare'")
    summary_parser.add_argument("--responses", required=True)
    summary_parser.add_argument("--report", required=True)
    summary_parser.add_argument("--bootstrap", type=int, default=1000)
    summary_parser.set_defaults(function=summarize)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
