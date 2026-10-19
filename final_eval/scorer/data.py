"""Scorer target construction without importing PyTorch or loading weights."""
import json
import pandas as pd
from final_eval.preparation.schema import POSITION, normalize_votes
from final_eval.utils.artifacts import require_columns

def build_span_table(df, holdout_model=None):
    df = df.copy()
    require_columns(df, ["example_id", "source_dataset", "original_prompt", "span_text", "span_start", "span_end", "span_category"])
    df["uid"] = df.source_dataset.astype(str) + "|" + df.example_id.astype(str)
    key = POSITION
    for c in ["span_start", "span_end"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    base = df.drop_duplicates(key).set_index(key)
    cols = ["example_id", "source_dataset", "original_prompt", "span_text", "span_category"]
    cols += [c for c in ["base_group_id", "span_categories", "span_id"] if c in base]
    out = base[cols].copy()
    label_col = next((c for c in ["model_specific_label", "label"] if c in df), None)
    panel = None
    if label_col and "labeling_model" in df:
        valid = df.copy()
        if "vote_status" in valid:
            valid = valid[valid.vote_status == "valid"].copy()
        if not valid[label_col].isin(["critical", "non_critical"]).all():
            raise ValueError("Unknown per-model label")
        valid["_y"] = valid[label_col].eq("critical").astype(int)
        if valid.groupby(key + ["labeling_model"])._y.nunique().gt(1).any():
            raise ValueError("Conflicting model/span labels")
        panel = valid.drop_duplicates(key + ["labeling_model"]).pivot(index=key, columns="labeling_model", values="_y")
    if holdout_model:
        if panel is None or holdout_model not in panel:
            raise ValueError("Held-out model requires its per-model labels")
        others = panel.drop(columns=holdout_model)
        out["target"] = others.mean(axis=1)
        out["eval_label"] = panel[holdout_model]
        out["n_target_models"] = others.count(axis=1)
        out = out.dropna(subset=["target", "eval_label"])
    else:
        score_col = next((c for c in ["agreement_score", "consensus_score", "consensus_criticality", "criticality_score"] if c in base), None)
        if score_col:
            for _, group in df.groupby(key):
                if pd.to_numeric(group[score_col], errors="raise").nunique() != 1:
                    raise ValueError("Inconsistent consensus score for a span")
            out["target"] = pd.to_numeric(base[score_col], errors="raise")
            if panel is not None:
                expected = panel.mean(axis=1).reindex(out.index)
                if ((out.target - expected).abs() > 1e-8).any():
                    raise ValueError("Consensus score disagrees with per-model votes; rebuild consensus")
        elif panel is not None:
            out["target"] = panel.mean(axis=1)
        else:
            require_columns(base, ["consensus_label"])
            out["target"] = base.consensus_label.eq("critical").astype(float)
        out["eval_label"] = (out.target > 0.5).astype(int)
        out["n_target_models"] = panel.count(axis=1) if panel is not None else 0
    if out.target.isna().any() or not out.target.between(0, 1).all():
        raise ValueError("Missing or out-of-range targets")
    out["label"] = (out.target > 0.5).astype(int)
    out["eval_label"] = out.eval_label.astype(int)
    return out.reset_index()
