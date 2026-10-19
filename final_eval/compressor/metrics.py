"""No-new-answer-call diagnostics with occurrence-aware retention bounds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_eval.compressor.retention import retention
from final_eval.scorer.data import build_span_table
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, parse_bool, require_unique

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels-file")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--recompute-retention", action="store_true", help="Required for legacy substring annotations")
    args = ap.parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    require_unique(df, ["uid", "method", "target_ratio"], "compression")
    if "alignment_status" not in df and not args.recompute_retention:
        raise ValueError("Legacy substring retention must be recomputed with --labels-file --recompute-retention")
    labels = {}
    if args.labels_file:
        tbl = build_span_table(pd.read_csv(args.labels_file, keep_default_na=False))
        labels = {u: g for u, g in tbl.groupby("uid")}
    if args.recompute_retention and not labels:
        raise ValueError("Recomputing retention requires --labels-file with usable annotations")
    details, cats = [], []
    for r in df.to_dict("records"):
        if args.recompute_retention and r["uid"] in labels and r.get("status", "ok") == "ok":
            encoded = r.get("kept_ranges_json", "")
            ranges = json.loads(encoded) if encoded else None
            r.update(retention(r["original_prompt"], r["compressed_prompt"], labels[r["uid"]], ranges))
        total = pd.to_numeric(r.get("crit_total"), errors="coerce")
        for kind in ("lower", "upper"):
            count = pd.to_numeric(r.get("crit_kept_" + kind), errors="coerce")
            r["critical_recall_" + kind] = count / total if pd.notna(total) and total > 0 else np.nan
        encoded = r.get("cat_retention_json") or "{}"
        for category, counts in json.loads(encoded).items():
            if not isinstance(counts, dict):
                raise ValueError("Legacy category annotations must be recomputed; full-sentence survival is not token retention")
            cats.append({"uid": r["uid"], "method": r["method"], "target_ratio": r["target_ratio"],
                         "source_dataset": r["source_dataset"], "category": category, **counts})
        details.append(r)
    detail = pd.DataFrame(details)
    rows = []
    for columns in [["method", "target_ratio"], ["method", "target_ratio", "source_dataset"]]:
        for keys, group in detail.groupby(columns):
            valid = group[group.status == "ok"] if "status" in group else group
            row = dict(zip(columns, keys if isinstance(keys, tuple) else (keys,)))
            actual = pd.to_numeric(valid.actual_ratio, errors="coerce")
            row.update(n_planned=len(group), n_valid=len(valid), n_labeled=pd.to_numeric(valid.get("crit_total"), errors="coerce").notna().sum(),
                       mean_actual_ratio=actual.mean(), mean_absolute_budget_error=(actual - row["target_ratio"]).abs().mean(),
                       budget_overflow_rate=(actual > row["target_ratio"]).mean(),
                       macro_critical_recall_lower=valid.critical_recall_lower.mean(),
                       macro_critical_recall_upper=valid.critical_recall_upper.mean())
            for kind in ["lower", "upper"]:
                denom = pd.to_numeric(valid.get("crit_total"), errors="coerce").sum()
                count = pd.to_numeric(valid.get("crit_kept_" + kind), errors="coerce").sum()
                row["micro_critical_recall_" + kind] = count / denom if denom else np.nan
            row["alignment_unresolved"] = (valid.alignment_status == "unresolved").sum() if "alignment_status" in valid else len(valid)
            rows.append(row)
    target = Path(args.output_dir); claim_output(target); target.mkdir(parents=True)
    atomic_csv(detail, target / "per_question.csv")
    atomic_csv(pd.DataFrame(rows), target / "summary.csv")
    if cats:
        category = pd.DataFrame(cats).groupby(["method", "target_ratio", "source_dataset", "category"], as_index=False)[
            ["spans", "full_lower", "full_upper", "source_tokens", "tokens_lower", "tokens_upper"]].sum()
        category["token_retention_lower"] = category.tokens_lower / category.source_tokens
        category["token_retention_upper"] = category.tokens_upper / category.source_tokens
        category["full_span_retention_lower"] = category.full_lower / category.spans
        atomic_csv(category, target / "categories.csv")
    atomic_json(target / "manifest.json", {"input_hash": file_hash(args.input),
        "labels_hash": file_hash(args.labels_file) if args.labels_file else None,
        "retention_protocol": "source_ranges_or_monotone_token_bounds-v2",
        "note": "Intrinsic diagnostics do not establish downstream correctness; unresolved alignments have missing retention."})
    print(f"Saved diagnostics to {target}")

if __name__ == "__main__":
    main()
