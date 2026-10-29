"""Prepare a controlled historical-checkpoint versus v2 comparison.

This module deliberately stops before model loading or hosted inference. The
historical release has a different split, so the first task is to prove which
questions can be evaluated by both checkpoints without inventing a fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_eval.preparation.schema import prepare_problems
from final_eval.utils.artifacts import atomic_json, file_hash


OLD_REPO = "flashstep/reasoning-critical-span-scorer"
OLD_REVISION = "c24d8ce9ef2cfbdc7b11185958dee5810327e432"


def load_split(path: str | Path, label: str) -> dict:
    """Load a split and require the explicit historical/v2 fold mapping."""
    split_path = Path(path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    folds = split.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError(f"{label} split must contain a non-empty object named 'folds'")
    invalid = [uid for uid, fold in folds.items() if not isinstance(fold, int) or fold < 0]
    if invalid:
        raise ValueError(f"{label} split has invalid fold assignments, including {invalid[:3]}")
    return split


def audit_common_cohort(problems_path: str | Path, old_split_path: str | Path,
                        v2_split_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return only questions assigned to both checkpoints with matching families."""
    raw = pd.read_csv(problems_path, keep_default_na=False,
                      dtype={"ground_truth_answer": str})
    try:
        problems = prepare_problems(raw)
    except ValueError as error:
        # The checked-in raw export predates the recovered clean GSM-IC
        # questions. We can still proceed only when the v2 release supplies
        # explicit family IDs for every question used in the comparison.
        if "Cannot establish clean GSM-IC family" not in str(error):
            raise
        required = {"example_id", "source_dataset", "original_prompt",
                    "ground_truth_answer"}
        if not required.issubset(raw.columns):
            raise
        problems = raw.copy()
        problems["uid"] = problems.source_dataset.astype(str) + "|" + problems.example_id.astype(str)
        problems["base_group_id"] = ""
    old = load_split(old_split_path, "Historical")
    v2 = load_split(v2_split_path, "v2")

    by_uid = problems.set_index("uid")
    old_ids, v2_ids = set(old["folds"]), set(v2["folds"])
    common_ids = old_ids & v2_ids & set(by_uid.index)
    if not common_ids:
        raise ValueError("No common prepared questions have assignments in both splits")

    common = problems[problems.uid.isin(common_ids)].copy()
    v2_groups = v2.get("base_groups", {})
    missing_v2_groups = sorted(set(common.uid) - set(v2_groups))
    if missing_v2_groups:
        raise ValueError(
            f"v2 split lacks base-family metadata for {len(missing_v2_groups)} common questions")
    common["base_group_id"] = common.uid.map(v2_groups)
    old_groups = old.get("base_groups", {})
    historical_family_safety = "verified_from_historical_metadata" if old_groups else (
        "conservative_v2_family_proxy; historical split has no base_groups")
    if old_groups:
        mismatched = common[common.uid.map(old_groups) != common.base_group_id]
        if not mismatched.empty:
            raise ValueError("Historical split base families disagree with prepared problem families")

    # A common family is retained only when every question in that family is
    # represented in both split maps; partial families would make exposure
    # auditing ambiguous.
    common_families = set(common.base_group_id)
    complete_families = {
        family for family in common_families
        if set(problems.loc[problems.base_group_id == family, "uid"]).issubset(old_ids)
        and set(problems.loc[problems.base_group_id == family, "uid"]).issubset(v2_ids)
    }
    if not old_groups:
        # Without historical family IDs, reject any v2 family split across
        # historical folds. This is conservative and makes exposure leakage
        # less likely, but it does not recreate the missing legacy metadata.
        old_fold_counts = common.groupby("base_group_id")["uid"].transform(
            lambda ids: len({old["folds"][uid] for uid in ids}))
        complete_families = {
            family for family in complete_families
            if old_fold_counts[common.base_group_id == family].iloc[0] == 1
        }
    result = common[common.base_group_id.isin(complete_families)].copy()
    if result.empty:
        raise ValueError("No complete common families remain after exposure filtering")
    result = result.sort_values(["source_dataset", "base_group_id", "uid"])
    metadata = {
        "protocol": "checkpoint-comparison-v1",
        "historical_repo": OLD_REPO,
        "historical_revision": OLD_REVISION,
        "problems_sha256": file_hash(problems_path),
        "historical_split_sha256": file_hash(old_split_path),
        "v2_split_sha256": file_hash(v2_split_path),
        "common_questions": int(len(result)),
        "common_families": int(result.base_group_id.nunique()),
        "excluded_questions_without_both_assignments": int(len(problems) - len(common)),
        "excluded_partial_family_questions": int(len(common) - len(result)),
        "historical_family_safety": historical_family_safety,
        "datasets": result.source_dataset.value_counts().to_dict(),
        "selection": "intersection of both split maps, complete prepared families only; no outcome-based selection",
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--old-split", required=True)
    parser.add_argument("--v2-split", required=True)
    parser.add_argument("--output-cohort", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    cohort, metadata = audit_common_cohort(
        args.problems, args.old_split, args.v2_split)
    output = Path(args.output_cohort)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    metadata["cohort_sha256"] = file_hash(output)
    atomic_json(args.output_manifest, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
