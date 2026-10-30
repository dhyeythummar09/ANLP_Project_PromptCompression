"""Step 6 of NEXT_STEPS: paired summary of the Qwen follow-up runs (steps 2-5).

Reads only finished artifacts, so it costs nothing and can be rerun freely:

* step 2  results/followup/old_vs_v2_qwen/summary.csv
* step 3  results/followup/components_qwen/{contrasts,per_question}.csv
* step 4  bonuses off: Abhinav's no-bonuses responses + compressions;
          bonuses on: our step-3 run, whose prompts his with-bonuses rerun
          reproduced exactly (same query hashes, so same text and settings)
* step 5  results/followup_v2/step5/deletion_check_results.csv

Bonuses on and off come from two separate runs of the same reader. Before
pairing them we measure how often that reader gives the same correctness to
identical prompts across runs; that repeat rate is reported next to the
contrast so the cross-run noise is visible.

Writes everything to one folder (default results/followup/summary/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_eval.utils.artifacts import atomic_json, file_hash, parse_bool, require_unique
from final_eval.utils.statistics import cluster_ci

KEY = ["uid", "method", "target_ratio"]
BONUS_ARMS = ["ours_scorer", "ours_signal"]
CONDITIONS = ["original", "mask", "delete"]


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype={"ground_truth": str, "ground_truth_answer": str})


def _correct(series: pd.Series) -> pd.Series:
    return series.map(parse_bool).astype(int)


def _digit_retention(original: str, compressed: str) -> float:
    """Share of number-bearing words that survive compression (bonuses target these)."""
    wanted = [w for w in original.split() if any(c.isdigit() for c in w)]
    if not wanted:
        return np.nan
    kept = compressed.split()
    hits = 0
    for word in wanted:
        if word in kept:
            kept.remove(word)
            hits += 1
    return hits / len(wanted)


def _paired_row(focus: np.ndarray, other: np.ndarray, groups: np.ndarray, bootstrap: int) -> dict:
    low, high = cluster_ci(focus - other, groups, bootstrap)
    return {"n_paired": int(len(focus)), "families": int(len(set(groups))),
            "focus_accuracy": float(focus.mean()), "other_accuracy": float(other.mean()),
            "difference": float((focus - other).mean()), "ci95_low": low, "ci95_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "only_focus_correct": int(((focus == 1) & (other == 0)).sum()),
            "only_other_correct": int(((focus == 0) & (other == 1)).sum())}


# ---------------------------------------------------------------- step 4 ----

def reader_repeatability(reference: pd.DataFrame, rerun: pd.DataFrame, label: str) -> list[dict]:
    """Same prompt, same settings, two runs: how often does correctness agree?"""
    merged = reference.merge(rerun, on=KEY, suffixes=("_ref", "_rerun"))
    same_query = merged.query_hash_ref.eq(merged.query_hash_rerun)
    both_ok = merged.response_status_ref.eq("ok") & merged.response_status_rerun.eq("ok")
    usable = merged[same_query & both_ok].copy()
    usable["agree"] = _correct(usable.correct_ref).eq(_correct(usable.correct_rerun))
    usable["same_text"] = usable.response_ref.eq(usable.response_rerun)
    rows = []
    for method, part in [("all", usable), *usable.groupby("method")]:
        rows.append({"comparison": label, "method": method,
                     "rows_matched": int(len(merged) if method == "all" else merged.method.eq(method).sum()),
                     "identical_prompts_both_ok": int(len(part)),
                     "correctness_agreement": float(part.agree.mean()) if len(part) else np.nan,
                     "identical_response_text": float(part.same_text.mean()) if len(part) else np.nan})
    return rows


def bonuses(args, out: Path) -> dict:
    on = _read(args.bonus_on_responses)
    off = _read(args.bonus_off_responses)
    rerun = _read(args.bonus_on_rerun_responses)
    for name, frame in [("bonus-on", on), ("bonus-off", off), ("bonus-on rerun", rerun)]:
        require_unique(frame, KEY, name)
        readers = set(frame.eval_model)
        if readers != {"qwen-3.8-27b-fp8"}:
            raise ValueError(f"{name} responses come from {readers}, expected Qwen only")

    # Provenance: the with-bonuses rerun must have asked exactly our step-3 prompts,
    # otherwise "bonus on" would not be the same condition in both runs.
    check = on.merge(rerun, on=KEY, suffixes=("_on", "_rerun"))
    if not check.query_hash_on.eq(check.query_hash_rerun).all():
        raise ValueError("With-bonuses rerun does not reproduce the step-3 prompts")

    on_text = _read(args.bonus_on_compressed)
    off_text = _read(args.bonus_off_compressed)
    texts = on_text[on_text.method.isin(BONUS_ARMS)][KEY + ["original_prompt", "compressed_prompt", "actual_ratio"]].merge(
        off_text[KEY + ["compressed_prompt", "actual_ratio", "status"]], on=KEY, suffixes=("_on", "_off"))
    if len(texts) != 640 or not texts.status.eq("ok").all():
        raise ValueError(f"Expected 640 compressed bonus pairs with status ok, got {len(texts)}")
    texts["identical_text"] = texts.compressed_prompt_on.eq(texts.compressed_prompt_off)
    texts["digits_kept_on"] = [_digit_retention(o, c) for o, c in zip(texts.original_prompt, texts.compressed_prompt_on)]
    texts["digits_kept_off"] = [_digit_retention(o, c) for o, c in zip(texts.original_prompt, texts.compressed_prompt_off)]

    families = on.drop_duplicates("uid").set_index("uid").base_group_id
    datasets = on.drop_duplicates("uid").set_index("uid").source_dataset
    originals = on[on.method.eq("none") & on.response_status.eq("ok")].set_index("uid").correct.pipe(_correct)

    contrasts, per_question, compression = [], [], []
    for (method, ratio), part in texts.groupby(["method", "target_ratio"], sort=True):
        compression.append({
            "method": method, "target_ratio": float(ratio), "pairs": int(len(part)),
            "identical_text": int(part.identical_text.sum()),
            "actual_ratio_on": float(pd.to_numeric(part.actual_ratio_on).mean()),
            "actual_ratio_off": float(pd.to_numeric(part.actual_ratio_off).mean()),
            "number_words_kept_on": float(part.digits_kept_on.mean()),
            "number_words_kept_off": float(part.digits_kept_off.mean()),
            "questions_with_numbers": int(part.digits_kept_on.notna().sum())})
        a = on[on.method.eq(method) & on.target_ratio.eq(ratio) & on.response_status.eq("ok")].set_index("uid")
        b = off[off.method.eq(method) & off.target_ratio.eq(ratio) & off.response_status.eq("ok")].set_index("uid")
        common = sorted(set(a.index) & set(b.index) & set(originals.index))
        focus, other = _correct(b.loc[common].correct).to_numpy(), _correct(a.loc[common].correct).to_numpy()
        same = part.set_index("uid").identical_text.loc[common].to_numpy()
        groups = families.loc[common].to_numpy()
        row = {"step": 4, "comparison": "bonuses off - bonuses on", "method": method,
               "target_ratio": float(ratio), "focus": "bonuses_off", "other": "bonuses_on",
               "planned": int(len(part)), "excluded_failed_or_missing": int(len(part) - len(common)),
               **_paired_row(focus, other, groups, args.bootstrap),
               "identical_text_pairs": int(same.sum())}
        if (~same).sum() >= 2:
            low, high = cluster_ci((focus - other)[~same], groups[~same], args.bootstrap)
            row.update(difference_on_differing_text=float((focus - other)[~same].mean()),
                       differing_ci95_low=low, differing_ci95_high=high)
        contrasts.append(row)
        per_question.append(pd.DataFrame({
            "uid": common, "source_dataset": datasets.loc[common].values, "base_group_id": groups,
            "method": method, "target_ratio": float(ratio), "original_correct": originals.loc[common].values,
            "bonus_on_correct": other, "bonus_off_correct": focus, "identical_text": same}))

    repeat = reader_repeatability(on, rerun, "step-3 run vs Abhinav with-bonuses rerun (identical prompts)")
    repeat += reader_repeatability(on, off, "step-3 run vs Abhinav no-bonuses run (prompts unchanged by bonuses)")
    coverage = []
    for name, frame in [("bonus_on (step 3)", on), ("bonus_off", off), ("bonus_on rerun", rerun)]:
        for (method, ratio), part in frame.groupby(["method", "target_ratio"]):
            coverage.append({"run": name, "method": method, "target_ratio": float(ratio), "rows": int(len(part)),
                             **{f"status_{k}": int(v) for k, v in part.response_status.value_counts().items()}})

    pd.DataFrame(contrasts).to_csv(out / "bonuses_contrasts.csv", index=False)
    pd.concat(per_question).to_csv(out / "bonuses_per_question.csv", index=False)
    pd.DataFrame(compression).to_csv(out / "bonuses_compression.csv", index=False)
    pd.DataFrame(repeat).to_csv(out / "reader_repeatability.csv", index=False)
    pd.DataFrame(coverage).fillna(0).to_csv(out / "bonuses_coverage.csv", index=False)
    return {"contrasts": contrasts, "compression": compression, "repeatability": repeat}


# ---------------------------------------------------------------- step 5 ----

def mask_deletion(args, out: Path) -> dict:
    spans = _read(args.mask_deletion)
    complete = spans[spans[[c + "_status" for c in CONDITIONS]].eq("ok").all(axis=1)].copy()
    for c in CONDITIONS:
        complete[c + "_correct"] = _correct(complete[c + "_correct"])
    # Only spans whose original question was answered correctly can show a deletion effect.
    eligible = complete[complete.original_correct.eq(1)].copy()
    eligible["agree"] = eligible.mask_correct.eq(eligible.delete_correct)
    rows = []
    for category, part in [("all", eligible), *eligible.groupby("span_category")]:
        groups = part.uid.to_numpy()
        diff = (part.delete_correct - part.mask_correct).to_numpy()
        low, high = cluster_ci(diff, groups, args.bootstrap) if len(part) > 1 else (np.nan, np.nan)
        agree_low, agree_high = cluster_ci(part.agree.astype(float).to_numpy(), groups, args.bootstrap) \
            if len(part) > 1 else (np.nan, np.nan)
        rows.append({
            "span_category": category,
            "spans_attempted": int(len(spans) if category == "all" else spans.span_category.eq(category).sum()),
            "eligible_original_correct": int(len(part)),
            "agreement": float(part.agree.mean()), "agreement_ci95_low": agree_low, "agreement_ci95_high": agree_high,
            "mask_accuracy": float(part.mask_correct.mean()), "delete_accuracy": float(part.delete_correct.mean()),
            "delete_minus_mask": float(diff.mean()), "ci95_low": low, "ci95_high": high,
            "mask_wrong_delete_right": int(((part.mask_correct == 0) & (part.delete_correct == 1)).sum()),
            "mask_right_delete_wrong": int(((part.mask_correct == 1) & (part.delete_correct == 0)).sum()),
            "both_wrong": int(((part.mask_correct == 0) & (part.delete_correct == 0)).sum())})
    pd.DataFrame(rows).to_csv(out / "mask_deletion_by_category.csv", index=False)
    return {"by_category": rows, "status_counts": {
        c: spans[c + "_status"].value_counts().to_dict() for c in CONDITIONS}}


# ------------------------------------------------------------- failures ----

def failure_tables(args, out: Path) -> dict:
    """Two kinds of failure: requests that never produced a usable answer,
    and questions the original answered correctly but compression broke."""
    runs = {
        "step2 old vs v2": _read(args.step2_per_question),
        "step3 components": _read(args.step3_per_question),
    }
    status = []
    old_v2 = runs["step2 old vs v2"]
    for checkpoint in ("historical", "v2"):
        for (method, ratio), part in old_v2.groupby(["base_method", "target_ratio"]):
            counts = part[f"response_status_{checkpoint}"].value_counts().to_dict()
            status.append({"run": "step2 old vs v2", "method": f"{method} ({checkpoint})",
                           "target_ratio": float(ratio), "rows": int(len(part)), **counts})
    for name, path in [("step3 components", args.step3_per_question),
                       ("step4 bonuses off", args.bonus_off_responses),
                       ("step4 bonuses on rerun", args.bonus_on_rerun_responses)]:
        frame = _read(path)
        for (method, ratio), part in frame.groupby(["method", "target_ratio"]):
            status.append({"run": name, "method": method, "target_ratio": float(ratio),
                           "rows": int(len(part)), **part.response_status.value_counts().to_dict()})
    status = pd.DataFrame(status).fillna(0)
    status.to_csv(out / "failures_request_status.csv", index=False)

    # Compression-induced errors per dataset, from the complete step-3 run
    # plus the bonus-off arms paired against the same originals.
    components = runs["step3 components"]
    components = components[components.response_status.eq("ok")].copy()
    components["correct"] = _correct(components.correct)
    originals = components[components.method.eq("none")].set_index("uid").correct
    compressed = components[~components.method.eq("none") & components.uid.isin(originals.index)].copy()
    compressed["original_correct"] = compressed.uid.map(originals)
    bonus = _read(out / "bonuses_per_question.csv")
    bonus_off = bonus.assign(method=bonus.method + "_nobonus", correct=bonus.bonus_off_correct)[
        ["uid", "source_dataset", "method", "target_ratio", "original_correct", "correct"]]
    rows = pd.concat([compressed[["uid", "source_dataset", "method", "target_ratio", "original_correct", "correct"]],
                      bonus_off])
    broken = rows[rows.original_correct.eq(1)]
    by_dataset = (broken.assign(broken=broken.correct.eq(0))
                  .groupby(["target_ratio", "method", "source_dataset"])
                  .agg(original_correct=("uid", "size"), broken_by_compression=("broken", "sum"))
                  .reset_index())
    by_dataset["broken_rate"] = by_dataset.broken_by_compression / by_dataset.original_correct
    by_dataset.to_csv(out / "failures_compression_by_dataset.csv", index=False)
    rescued = rows[rows.original_correct.eq(0) & rows.correct.eq(1)]
    return {"request_status_rows": int(len(status)),
            "compression_breaks": int(broken.correct.eq(0).sum()),
            "compressed_correct_when_original_wrong": int(len(rescued))}


# --------------------------------------------------------------- combine ----

def combined_table(args, out: Path, bonus: dict, step5: dict) -> pd.DataFrame:
    rows = []
    step2 = _read(args.step2_summary)
    for r in step2.itertuples():
        rows.append({"step": 2, "comparison": "v2 - historical checkpoint", "method": r.method,
                     "target_ratio": float(r.target_ratio), "n_paired": int(r.paired_conditions),
                     "families": int(r.families), "focus_accuracy": float(r.v2_accuracy),
                     "other_accuracy": float(r.historical_accuracy),
                     "difference": float(r.delta_v2_minus_historical),
                     "ci95_low": float(r.delta_ci95_low), "ci95_high": float(r.delta_ci95_high),
                     "ci_excludes_zero": parse_bool(r.ci_excludes_zero)})
    step3 = _read(args.step3_contrasts)
    for r in step3.itertuples():
        rows.append({"step": 3, "comparison": f"{r.focus} - {r.other}", "method": r.focus,
                     "target_ratio": float(r.target_ratio), "n_paired": int(r.n_paired),
                     "families": int(r.families), "focus_accuracy": float(r.focus_accuracy),
                     "other_accuracy": float(r.other_accuracy), "difference": float(r.difference),
                     "ci95_low": float(r.ci95_low), "ci95_high": float(r.ci95_high),
                     "ci_excludes_zero": parse_bool(r.ci_excludes_zero)})
    for r in bonus["contrasts"]:
        rows.append({k: r[k] for k in ["step", "comparison", "method", "target_ratio", "n_paired", "families",
                                       "focus_accuracy", "other_accuracy", "difference", "ci95_low",
                                       "ci95_high", "ci_excludes_zero"]})
    for r in step5["by_category"]:
        rows.append({"step": 5, "comparison": "delete - mask (spans, original correct)",
                     "method": f"span category: {r['span_category']}", "target_ratio": np.nan,
                     "n_paired": r["eligible_original_correct"], "families": np.nan,
                     "focus_accuracy": r["delete_accuracy"], "other_accuracy": r["mask_accuracy"],
                     "difference": r["delete_minus_mask"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                     "ci_excludes_zero": bool(r["ci95_low"] > 0 or r["ci95_high"] < 0)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "paired_results.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/followup/summary")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--step2-summary", default="results/followup/old_vs_v2_qwen/summary.csv")
    parser.add_argument("--step2-per-question", default="results/followup/old_vs_v2_qwen/per_question.csv")
    parser.add_argument("--step3-contrasts", default="results/followup/components_qwen/contrasts.csv")
    parser.add_argument("--step3-per-question", default="results/followup/components_qwen/per_question.csv")
    parser.add_argument("--bonus-on-responses", default="outputs/followup/components_v2/target_qwen/responses.csv",
                        help="Our complete step-3 responses (with query hashes)")
    parser.add_argument("--bonus-on-compressed", default="outputs/followup/components_v2/matched/prepared.csv")
    parser.add_argument("--bonus-on-rerun-responses", default="results/followup_v2/step4/with_bonuses_responses.csv")
    parser.add_argument("--bonus-off-responses", default="results/followup_v2/step4/no_bonuses_responses.csv")
    parser.add_argument("--bonus-off-compressed", default="results/followup_v2/step4/no_bonuses_compressed.csv")
    parser.add_argument("--mask-deletion", default="results/followup_v2/step5/deletion_check_results.csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bonus = bonuses(args, out)
    step5 = mask_deletion(args, out)
    failures = failure_tables(args, out)
    table = combined_table(args, out, bonus, step5)

    inputs = {name: getattr(args, name) for name in vars(args) if name not in {"output_dir", "bootstrap"}}
    atomic_json(out / "manifest.json", {
        "reader": "qwen-3.8-27b-fp8 (temperature 0, max_tokens 8192, reasoning low)",
        "inputs": {name: {"path": path, "sha256": file_hash(path)} for name, path in inputs.items()},
        "bootstrap": args.bootstrap, "clustering": "base_group_id (step 5: uid)",
        "multiple_testing": "unadjusted 95% CIs; exploratory",
        "step4_design": ("bonus on = our step-3 run; bonus off = Abhinav's no-bonuses run; paired by "
                         "question across two runs of the same reader; see reader_repeatability.csv"),
        "step4_caveats": ["no pilot gate for the step-4 runs",
                          "frozen 160-question cohort instead of separate development families",
                          "bonuses off was run for ours_scorer and ours_signal only, not the alpha=.5 mixture"],
        "failures": failures, "paired_rows": int(len(table))})
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
