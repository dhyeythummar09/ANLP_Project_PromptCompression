"""Normalize legacy tables without guessing missing model or family identities."""
from __future__ import annotations
import json
import re
import pandas as pd
from final_eval.utils.artifacts import digest, require_columns, require_unique, parse_bool
from final_eval.utils.answer_matching import grade_answer

POSITION = ["uid", "span_start", "span_end"]

def canonical_text(text):
    return re.sub(r"\W+", " ", str(text).casefold()).strip()

def prepare_problems(frame):
    frame = frame.copy().fillna("")
    require_columns(frame, ["example_id", "source_dataset", "original_prompt", "ground_truth_answer"], "problems")
    frame["uid"] = frame.source_dataset.astype(str) + "|" + frame.example_id.astype(str)
    require_unique(frame, ["uid"], "problems")
    groups, methods = [], []
    for r in frame.to_dict("records"):
        text = str(r["original_prompt"])
        if not text.strip() or not str(r["ground_truth_answer"]).strip():
            raise ValueError(f"Empty prompt/gold for {r['uid']}")
        if r.get("base_group_id"):
            groups.append(str(r["base_group_id"])); methods.append("supplied")
            continue
        clean = r.get("clean_question") or r.get("original_question")
        if not clean and r["source_dataset"] == "gsm_ic":
            inserted = r.get("inserted_context", "")
            if not inserted or str(inserted) not in text:
                raise ValueError(f"Cannot establish clean GSM-IC family for {r['uid']}; supply clean_question/base_group_id")
            clean = text.replace(str(inserted), "", 1)
        groups.append(digest({"canonical_question": canonical_text(clean or text)})[:24])
        methods.append("derived_clean_text" if clean else "derived_question_text")
    frame["base_group_id"] = groups
    frame["grouping_method"] = methods
    frame["prompt_hash"] = frame.original_prompt.map(digest)
    return frame

def normalize_votes(frame, problems=None, regrade=False):
    frame = frame.copy().fillna("")
    require_columns(frame, ["example_id", "source_dataset", "original_prompt", "ground_truth_answer",
                            "span_start", "span_end", "span_text", "span_category", "labeling_model"], "votes")
    frame["uid"] = frame.source_dataset.astype(str) + "|" + frame.example_id.astype(str)
    for col in ("span_start", "span_end"):
        nums = pd.to_numeric(frame[col], errors="raise")
        if (nums % 1 != 0).any():
            raise ValueError(f"Nonintegral {col}")
        frame[col] = nums.astype(int)
    if frame.labeling_model.astype(str).str.strip().eq("").any():
        raise ValueError("Every vote needs a nonempty labeling_model identity")
    for r in frame.itertuples():
        if not (0 <= r.span_start < r.span_end <= len(r.original_prompt)) or r.original_prompt[r.span_start:r.span_end] != r.span_text:
            raise ValueError(f"Invalid span offsets/text: {r.uid}:{r.span_start}:{r.span_end}")
    frame["prompt_hash"] = frame.original_prompt.map(digest)
    if problems is not None:
        expected = problems.set_index("uid")
        for col in ("original_prompt", "ground_truth_answer"):
            matched = frame.uid.map(expected[col])
            if matched.isna().any() or (matched.astype(str) != frame[col].astype(str)).any():
                raise ValueError(f"Vote/problem conflict in {col}; verify source dataset versions")
        frame["base_group_id"] = frame.uid.map(expected.base_group_id)
    # A reused source ID with different text or gold is always a conflict.
    for col in ("prompt_hash", "ground_truth_answer"):
        if frame.groupby("uid")[col].nunique().gt(1).any():
            raise ValueError(f"Conflicting {col} for the same uid")
    if regrade:
        require_columns(frame, ["model_answer_original", "model_answer_perturbed"], "responses for regrading")
        orig, pert, status = [], [], []
        for r in frame.itertuples():
            a = grade_answer(r.model_answer_original, r.ground_truth_answer, r.original_prompt)
            b = grade_answer(r.model_answer_perturbed, r.ground_truth_answer, r.original_prompt)
            orig.append(a); pert.append(b)
            original_status = getattr(r, "original_response_status", "ok")
            perturbed_status = getattr(r, "perturbed_response_status", "ok")
            status.append("missing_original_response" if not str(r.model_answer_original).strip() else
                          "original_unavailable" if original_status != "ok" else
                          "original_wrong" if not a.correct else
                          "perturbed_unavailable" if perturbed_status != "ok" else
                          "missing_perturbed_response" if not str(r.model_answer_perturbed).strip() else "valid")
        for prefix, grades in (("original", orig), ("perturbed", pert)):
            frame[f"is_correct_{prefix}"] = [g.correct for g in grades]
            frame[f"parse_status_{prefix}"] = [g.status for g in grades]
        frame["vote_status"] = status
        frame["model_specific_label"] = ["critical" if not g.correct else "non_critical" for g in pert]
        frame["grader_version"] = orig[0].grader_version if orig else "unique-final-v2"
    else:
        if "model_specific_label" not in frame and "label" in frame:
            frame["model_specific_label"] = frame["label"]
        require_columns(frame, ["model_specific_label"], "votes")
        if not frame.model_specific_label.isin(["critical", "non_critical"]).all():
            raise ValueError("Unknown model-specific labels")
        frame["vote_status"] = frame.get("vote_status", pd.Series("valid", index=frame.index))
        if "is_correct_original" in frame:
            frame.loc[~frame.is_correct_original.map(parse_bool), "vote_status"] = "original_wrong"
    frame["span_id"] = [digest([u, h, s, e]) for u, h, s, e in
                        zip(frame.uid, frame.prompt_hash, frame.span_start, frame.span_end)]
    key = POSITION + ["labeling_model"]
    conflict_cols = ["model_specific_label", "vote_status", "span_text"]
    conflict_cols += [c for c in ["model_answer_original", "model_answer_perturbed", "perturbed_prompt", "decoding_settings"] if c in frame]
    for col in conflict_cols:
        if frame.groupby(key)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting duplicate model/span outcomes: {col}. Resolve explicitly; no first/last wins.")
    categories = frame.groupby(POSITION).span_category.agg(lambda s: json.dumps(sorted(set(s))))
    frame["span_categories"] = [categories.loc[(u, s, e)] for u, s, e in frame[POSITION].itertuples(index=False, name=None)]
    frame = frame.drop_duplicates(key).copy()
    # Deterministic primary category; all memberships remain in span_categories.
    frame["span_category"] = frame.span_categories.map(lambda value: json.loads(value)[0])
    return frame.reset_index(drop=True)

def build_consensus(votes, minimum_models=2):
    if minimum_models < 1:
        raise ValueError("minimum_models must be positive")
    valid = votes[votes.vote_status == "valid"].copy()
    require_unique(valid, POSITION + ["labeling_model"], "valid votes")
    valid["_critical"] = valid.model_specific_label.eq("critical").astype(int)
    agg = valid.groupby(POSITION).agg(n_models_labeled=("labeling_model", "nunique"),
                                     n_models_critical=("_critical", "sum")).reset_index()
    agg = agg[agg.n_models_labeled >= minimum_models].copy()
    agg["agreement_score"] = agg.n_models_critical / agg.n_models_labeled
    agg["consensus_label"] = agg.agreement_score.map(lambda n: "critical" if n > 0.5 else "non_critical")
    old = [c for c in agg if c not in POSITION and c in valid]
    return valid.drop(columns=["_critical"] + old).merge(agg, on=POSITION, validate="many_to_one")

def attach_groups(spans, problems=None):
    if "base_group_id" in spans and spans.base_group_id.astype(str).str.len().gt(0).all():
        if spans.groupby("uid").base_group_id.nunique().gt(1).any():
            raise ValueError("A question cannot belong to multiple base families")
        return spans
    if problems is None:
        raise ValueError("Base-family metadata required. Run final_eval.preparation.prepare or supply --problems-file.")
    table = prepare_problems(problems).set_index("uid")
    spans = spans.copy()
    spans["base_group_id"] = spans.uid.map(table.base_group_id)
    if spans.base_group_id.isna().any():
        raise ValueError("Missing base family for a labeled question")
    return spans

def grouped_folds(span_table, folds=5, val_fraction=0.15, seed=42):
    import numpy as np
    from sklearn.model_selection import KFold, train_test_split
    if not 0 < val_fraction < 1 or folds < 2:
        raise ValueError("Need folds >= 2 and 0 < validation fraction < 1")
    groups = np.array(sorted(span_table.base_group_id.unique()))
    if len(groups) < folds:
        raise ValueError("Fewer independent groups than folds")
    for fold, (rest, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(groups)):
        if len(rest) < 2:
            raise ValueError("Need at least two remaining groups for train/validation")
        tr, va = train_test_split(groups[rest], test_size=val_fraction, random_state=seed)
        roles = {role: set(span_table.loc[span_table.base_group_id.isin(values), "uid"])
                 for role, values in (("train", tr), ("val", va), ("test", groups[test]))}
        yield fold, roles
