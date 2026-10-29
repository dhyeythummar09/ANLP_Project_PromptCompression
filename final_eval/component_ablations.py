"""Scorer-only and signal-only component ablations (NEXT_STEPS step 3).

``ours`` mixes the neural span scorer with DistilGPT2 surprisal. This module
asks whether either half alone does as well, on the frozen 160-question cohort:

* ``pilot-cohort`` picks a small pilot whose families avoid the main cohort.
* ``prepare`` stacks the ``ours`` reference arm and the ``ours_scorer`` /
  ``ours_signal`` arms into one reader input, after checking they describe the
  same questions with the same settings.
* ``run`` sends that table to the reader in one matched run (originals included).
* ``summarize`` reports accuracy and paired, family-clustered differences.

``assemble`` is the older helper that combines two separate response runs; it
is kept for the historical GPT-OSS path but is not used for the Qwen follow-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_eval.compressor.evaluate import KEY as EVAL_KEY, PROMPT_TEMPLATE
from final_eval.followup_reader import (add_pilot_cohort_arguments, add_run_arguments,
                                        launch, pilot_cohort)
from final_eval.inference.client import generation_config
from final_eval.utils.artifacts import (
    atomic_csv,
    atomic_json,
    claim_output,
    file_hash,
    parse_bool,
    require_unique,
)
from final_eval.utils.statistics import cluster_ci

KEY = ["uid", "method", "target_ratio"]
RESPONSE_KEY = KEY + ["eval_model"]
ABLATION_METHODS = {"ours_scorer", "ours_signal"}
REFERENCE_METHOD = "ours"
# Paired contrasts reported by `summarize`: (focus, other), delta = focus - other.
CONTRASTS = [("ours_scorer", "ours"), ("ours_signal", "ours"), ("ours_scorer", "ours_signal")]
QUESTION_COLUMNS = ["original_prompt", "ground_truth_answer", "base_group_id", "source_dataset"]


def _read_manifest(csv_path: str | Path) -> dict:
    manifest_path = Path(str(csv_path) + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for {csv_path}: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _assert_expected_methods(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    actual = set(frame.method.unique())
    if actual != expected:
        raise ValueError(
            f"{name} methods mismatch. Expected {sorted(expected)}, got {sorted(actual)}")


def _assert_grid_complete(frame: pd.DataFrame, methods: set[str], name: str) -> None:
    expected = frame.groupby("uid").method.nunique()
    full = expected[expected == len(methods)]
    if len(full) != frame.uid.nunique():
        raise ValueError(f"{name} has incomplete uid/method coverage for {sorted(methods)}")
    per_ratio = frame.groupby("target_ratio").uid.nunique()
    if len(set(per_ratio.values)) != 1:
        raise ValueError(f"{name} ratio coverage is inconsistent across methods")


def _assert_overlap_consistency(baseline: pd.DataFrame, ablation: pd.DataFrame) -> None:
    common = set(baseline.uid) & set(ablation.uid)
    if not common:
        raise ValueError("No common questions between baseline and ablation compressions")
    for frame_name, frame in (("baseline", baseline), ("ablation", ablation)):
        for column in ("original_prompt", "ground_truth_answer", "base_group_id"):
            if frame.groupby("uid")[column].nunique(dropna=False).gt(1).any():
                raise ValueError(f"{frame_name} compression has conflicting {column} for a uid")
    left = baseline[baseline.uid.isin(common)][
        ["uid", "original_prompt", "ground_truth_answer", "base_group_id"]
    ].drop_duplicates("uid").set_index("uid")
    right = ablation[ablation.uid.isin(common)][
        ["uid", "original_prompt", "ground_truth_answer", "base_group_id"]
    ].drop_duplicates("uid").set_index("uid")
    for column in ("original_prompt", "ground_truth_answer", "base_group_id"):
        mismatch = (left[column].sort_index() != right[column].sort_index()).any()
        if mismatch:
            raise ValueError(f"Baseline and ablation artifacts disagree on {column}")


def assemble(args: argparse.Namespace) -> None:
    baseline_comp = pd.read_csv(args.baseline_compressed, keep_default_na=False)
    ablation_comp = pd.read_csv(args.ablation_compressed, keep_default_na=False)
    baseline_res = pd.read_csv(args.baseline_responses, keep_default_na=False)
    ablation_res = pd.read_csv(args.ablation_responses, keep_default_na=False)
    require_unique(baseline_comp, KEY, "baseline compression")
    require_unique(ablation_comp, KEY, "ablation compression")
    require_unique(baseline_res, RESPONSE_KEY, "baseline responses")
    require_unique(ablation_res, RESPONSE_KEY, "ablation responses")

    # Baseline must provide the historical "ours" rows to preserve continuity.
    _assert_expected_methods(
        baseline_comp[baseline_comp.method.eq("ours")], {"ours"}, "Baseline compression")
    _assert_expected_methods(
        ablation_comp[ablation_comp.method.isin(ABLATION_METHODS)], ABLATION_METHODS,
        "Ablation compression")
    _assert_grid_complete(
        baseline_comp[baseline_comp.method.eq("ours")], {"ours"},
        "Baseline compression subset")
    _assert_grid_complete(
        ablation_comp[ablation_comp.method.isin(ABLATION_METHODS)], ABLATION_METHODS,
        "Ablation compression subset")
    _assert_overlap_consistency(
        baseline_comp[baseline_comp.method.eq("ours")],
        ablation_comp[ablation_comp.method.isin(ABLATION_METHODS)])

    baseline_manifest = _read_manifest(args.baseline_responses)
    ablation_manifest = _read_manifest(args.ablation_responses)
    base_cfg = generation_config(baseline_manifest["config"])
    ablation_cfg = generation_config(ablation_manifest["config"])
    if base_cfg != ablation_cfg:
        raise ValueError("Response generation settings differ; rerun with identical model configuration")
    if baseline_manifest.get("wrapper") != ablation_manifest.get("wrapper"):
        raise ValueError("Prompt wrapper differs between runs; comparisons would be confounded")

    chosen_comp = pd.concat([
        baseline_comp[baseline_comp.method.eq("ours")],
        ablation_comp[ablation_comp.method.isin(ABLATION_METHODS)],
    ], ignore_index=True)
    chosen_res = pd.concat([
        baseline_res[baseline_res.method.isin({"ours", "none"})],
        ablation_res[ablation_res.method.isin(ABLATION_METHODS)],
    ], ignore_index=True)
    require_unique(chosen_comp, KEY, "assembled compression")
    require_unique(chosen_res, RESPONSE_KEY, "assembled responses")

    claim_output(args.output_compressed)
    claim_output(args.output_responses)
    atomic_csv(chosen_comp, args.output_compressed)
    atomic_csv(chosen_res, args.output_responses)
    atomic_json(str(args.output_responses) + ".manifest.json", {
        "stage": "component_ablation_assembly",
        "baseline_compression_sha256": file_hash(args.baseline_compressed),
        "ablation_compression_sha256": file_hash(args.ablation_compressed),
        "baseline_responses_sha256": file_hash(args.baseline_responses),
        "ablation_responses_sha256": file_hash(args.ablation_responses),
        "baseline_generation_config": base_cfg,
        "ablation_generation_config": ablation_cfg,
        "wrapper": baseline_manifest.get("wrapper"),
        "methods": ["ours", "ours_scorer", "ours_signal"],
        "ratios": sorted(map(float, chosen_comp.target_ratio.unique())),
        "n_questions": int(chosen_comp.uid.nunique()),
        "selection": "baseline ours + ablation scorer/signal, shared configuration required",
    })
    print(
        f"Assembled {len(chosen_comp)} compression rows and {len(chosen_res)} responses "
        f"for ours/ours_scorer/ours_signal.")


# ---------------------------------------------------------------------------
# Matched single-run workflow (Qwen follow-up)
# ---------------------------------------------------------------------------

def _load_arm(path: str, methods: list[str], ratios: list[float]) -> pd.DataFrame:
    """Read one compression table and keep only fully successful requested rows."""
    frame = pd.read_csv(path, keep_default_na=False, dtype={"ground_truth_answer": str})
    require_unique(frame, KEY, f"compression: {path}")
    selected = frame[frame.method.isin(methods) & frame.target_ratio.isin(ratios)].copy()
    missing = {(m, r) for m in methods for r in ratios} - set(zip(selected.method, selected.target_ratio))
    if missing:
        raise ValueError(f"{path} lacks method/ratio combinations: {sorted(missing)}")
    if not selected.status.eq("ok").all() or not selected.budget_compliant.map(parse_bool).all():
        raise ValueError(f"{path} contains failed or over-budget rows for {methods}")
    # Every question must have every requested method/ratio; no silent gaps.
    if selected.groupby("uid").size().ne(len(methods) * len(ratios)).any():
        raise ValueError(f"{path} has questions missing some {methods} conditions")
    return selected


def _arguments_without_paths(path: str) -> dict:
    """Compression settings from a run manifest, minus file names and method lists."""
    manifest = _read_manifest(path.replace("compressed.csv", "neural.csv"))
    ignore = {"input_csv", "output_csv", "methods", "limit"}
    return {k: v for k, v in manifest["arguments"].items() if k not in ignore}


def prepare(args: argparse.Namespace) -> None:
    reference = _load_arm(args.reference_input, [REFERENCE_METHOD], args.ratios)
    ablation = _load_arm(args.ablation_input, sorted(ABLATION_METHODS), args.ratios)
    if set(reference.uid) != set(ablation.uid):
        raise ValueError("Reference and ablation compressions cover different questions")
    questions = pd.concat([reference, ablation])[["uid", *QUESTION_COLUMNS]].drop_duplicates()
    if questions.uid.duplicated().any():
        raise ValueError("Reference and ablation compressions disagree on question details")
    # All three arms must come from the same compressor settings (checkpoint,
    # alpha, bonuses, tokenizer, ...); otherwise a difference is not a component effect.
    settings = {path: _arguments_without_paths(path)
                for path in dict.fromkeys([args.reference_input, args.ablation_input])}
    if len({json.dumps(v, sort_keys=True) for v in settings.values()}) != 1:
        raise ValueError(f"Compression settings differ between arms: {settings}")

    prepared = pd.concat([reference, ablation], ignore_index=True).sort_values(
        ["uid", "target_ratio", "method"]).reset_index(drop=True)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"{output} already exists; keep it or choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(prepared, output)

    # Identical compressed strings get one shared answer, so they can only tie.
    by_condition = prepared.pivot_table(index=["uid", "target_ratio"], columns="method",
                                        values="compressed_prompt", aggfunc="first")
    identical = {f"{a}=={b}@{ratio}": int((group[a] == group[b]).sum())
                 for a, b in CONTRASTS
                 for ratio, group in by_condition.groupby(level="target_ratio")}
    manifest = {
        "protocol": "component-ablation-matched-v1",
        "reference_input": args.reference_input,
        "reference_sha256": file_hash(args.reference_input),
        "ablation_input": args.ablation_input,
        "ablation_sha256": file_hash(args.ablation_input),
        "prepared_sha256": file_hash(output),
        "compression_settings": next(iter(settings.values())),
        "methods": [REFERENCE_METHOD, *sorted(ABLATION_METHODS)],
        "ratios": args.ratios,
        "questions": int(prepared.uid.nunique()),
        "families": int(prepared.base_group_id.nunique()),
        "logical_conditions_including_originals": int(prepared.uid.nunique() + len(prepared)),
        "unique_reader_queries_before_retries":
            int(len(set(prepared.original_prompt) | set(prepared.compressed_prompt))),
        "identical_compressed_text": identical,
    }
    # Optional provenance check: does the freshly generated `ours` reproduce the
    # text that the completed main study evaluated?
    if args.saved_reference:
        saved = pd.read_csv(args.saved_reference, keep_default_na=False)
        saved = saved[saved.method.eq(REFERENCE_METHOD) & saved.target_ratio.isin(args.ratios)]
        joined = reference.merge(saved, on=KEY, suffixes=("", "_saved"))
        manifest["saved_reference"] = {
            "path": args.saved_reference, "sha256": file_hash(args.saved_reference),
            "matched_conditions": int(len(joined)),
            "identical_text": int((joined.compressed_prompt == joined.compressed_prompt_saved).sum()),
        }
    atomic_json(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _paired_contrast(piv: pd.DataFrame, texts: pd.DataFrame, families: pd.Series,
                     focus: str, other: str, bootstrap: int) -> dict:
    diff = piv[focus] - piv[other]
    low, high = cluster_ci(diff.to_numpy(), families.loc[diff.index].to_numpy(), bootstrap)
    same_text = texts[focus] == texts[other]
    return {
        "focus": focus, "other": other, "n_paired": int(len(diff)),
        "families": int(families.loc[diff.index].nunique()),
        "focus_accuracy": float(piv[focus].mean()), "other_accuracy": float(piv[other].mean()),
        "difference": float(diff.mean()), "ci95_low": low, "ci95_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "only_focus_correct": int(((piv[focus] == 1) & (piv[other] == 0)).sum()),
        "only_other_correct": int(((piv[focus] == 0) & (piv[other] == 1)).sum()),
        "identical_text_pairs": int(same_text.sum()),
    }


def summarize(args: argparse.Namespace) -> None:
    responses = pd.read_csv(args.responses, keep_default_na=False)
    require_unique(responses, EVAL_KEY, "responses")
    readers = sorted(responses.eval_model.unique())
    if len(readers) != 1:
        raise ValueError(f"Expected answers from exactly one reader, found {readers}")
    prepared = pd.read_csv(args.input, keep_default_na=False)
    methods = [REFERENCE_METHOD, *sorted(ABLATION_METHODS)]

    ok = responses[responses.response_status.eq("ok")].copy()
    ok["correct"] = ok.correct.map(parse_bool).astype(int)
    originals = ok[ok.method.eq("none")].set_index("uid").correct
    families = prepared.drop_duplicates("uid").set_index("uid").base_group_id
    datasets = prepared.drop_duplicates("uid").set_index("uid").source_dataset

    accuracy, contrasts, per_dataset, coverage = [], [], [], []
    for ratio, planned in prepared.groupby("target_ratio", sort=True):
        observed = ok[ok.target_ratio.eq(ratio) & ok.method.isin(methods)]
        piv = observed.pivot(index="uid", columns="method", values="correct").reindex(columns=methods)
        # Primary table: questions where the original and every arm answered cleanly.
        common = sorted(set(piv.dropna().index) & set(originals.index))
        piv = piv.loc[common].astype(int)
        texts = planned.pivot(index="uid", columns="method", values="compressed_prompt").loc[common]
        actual = planned.pivot(index="uid", columns="method", values="actual_ratio").loc[common]
        for method in methods:
            values = piv[method].to_numpy()
            low, high = cluster_ci(values, families.loc[common].to_numpy(), args.bootstrap)
            accuracy.append({
                "target_ratio": float(ratio), "method": method, "n": len(common),
                "families": int(families.loc[common].nunique()),
                "accuracy": float(values.mean()), "ci95_low": low, "ci95_high": high,
                "original_accuracy": float(originals.loc[common].mean()),
                "actual_ratio": float(pd.to_numeric(actual[method]).mean())})
            coverage.append({"target_ratio": float(ratio), "method": method,
                             "planned": int(planned.method.eq(method).sum()),
                             "answered_ok": int(observed.method.eq(method).sum()),
                             "in_common_table": len(common)})
            for dataset, uids in pd.Series(common, index=datasets.loc[common].values).groupby(level=0):
                per_dataset.append({"target_ratio": float(ratio), "method": method,
                                    "source_dataset": dataset, "n": len(uids),
                                    "accuracy": float(piv.loc[uids.values, method].mean()),
                                    "original_accuracy": float(originals.loc[uids.values].mean())})
        for focus, other in CONTRASTS:
            contrasts.append({"target_ratio": float(ratio),
                              **_paired_contrast(piv, texts, families, focus, other, args.bootstrap)})

    served = {}
    if {"generation_served_model", "reused_generation"} <= set(responses.columns):
        fresh = responses[responses.response_status.eq("ok") & ~responses.reused_generation.map(parse_bool)]
        served = fresh.generation_served_model.replace("", "(not reported)").value_counts().to_dict()

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "comparison": "ours versus scorer-only and signal-only, same reader, single matched run",
        "reader_requested": readers[0], "reader_served_counts": served,
        "wrapper": PROMPT_TEMPLATE,
        "responses_sha256": file_hash(args.responses), "prepared_sha256": file_hash(args.input),
        "response_statuses": responses.response_status.value_counts().to_dict(),
        "bootstrap": args.bootstrap, "clustering": "base_group_id",
        "accuracy": accuracy, "contrasts": contrasts, "coverage": coverage,
        "note": "Unadjusted 95% family-clustered bootstrap CIs; exploratory component analysis.",
    }
    atomic_json(output, report)
    stem = output.with_suffix("")
    pd.DataFrame(accuracy).to_csv(f"{stem}.accuracy.csv", index=False)
    pd.DataFrame(contrasts).to_csv(f"{stem}.contrasts.csv", index=False)
    pd.DataFrame(per_dataset).to_csv(f"{stem}.per_dataset.csv", index=False)
    print(json.dumps({"report": str(output), "accuracy": accuracy, "contrasts": contrasts},
                     indent=2, default=str))


def run(args: argparse.Namespace) -> None:
    prepared = pd.read_csv(args.input, keep_default_na=False)
    if set(prepared.method) - ({REFERENCE_METHOD} | ABLATION_METHODS):
        raise ValueError(f"{args.input} has unexpected methods; run 'prepare' first")
    launch(args, prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ap = sub.add_parser("assemble")
    ap.add_argument("--baseline-compressed", required=True)
    ap.add_argument("--baseline-responses", required=True)
    ap.add_argument("--ablation-compressed", required=True)
    ap.add_argument("--ablation-responses", required=True)
    ap.add_argument("--output-compressed", required=True)
    ap.add_argument("--output-responses", required=True)

    pilot = sub.add_parser("pilot-cohort", help="Small pilot avoiding the main families")
    add_pilot_cohort_arguments(pilot)

    prep = sub.add_parser("prepare", help="Stack ours + scorer/signal arms into one reader input")
    prep.add_argument("--reference-input", required=True, help="Compression containing 'ours'")
    prep.add_argument("--ablation-input", required=True,
                      help="Compression containing 'ours_scorer' and 'ours_signal'")
    prep.add_argument("--saved-reference",
                      help="Optional earlier compression whose 'ours' text is checked for identity")
    prep.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.75])
    prep.add_argument("--output", required=True)

    run_parser = sub.add_parser("run", help="Ask the reader every prepared condition")
    add_run_arguments(run_parser)

    summary = sub.add_parser("summarize", help="Accuracy and paired component contrasts")
    summary.add_argument("--input", required=True, help="Prepared table from 'prepare'")
    summary.add_argument("--responses", required=True)
    summary.add_argument("--report", required=True)
    summary.add_argument("--bootstrap", type=int, default=1000)

    args = parser.parse_args()
    {"assemble": assemble, "pilot-cohort": pilot_cohort, "prepare": prepare,
     "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
