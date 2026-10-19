"""Pairwise agreement on common valid model/span occurrences, without new calls."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import pandas as pd
from final_eval.preparation.schema import POSITION
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, require_columns, require_unique


def pair_statistics(a, b):
    a, b = a.astype(bool), b.astype(bool)
    both = int((a & b).sum()); neither = int((~a & ~b).sum())
    left = int((a & ~b).sum()); right = int((~a & b).sum()); n = len(a)
    observed = (both + neither) / n if n else None
    pa, pb = (a.mean(), b.mean()) if n else (0, 0)
    expected = pa * pb + (1-pa) * (1-pb)
    return {"n_common": n, "both_critical": both, "only_a_critical": left,
            "only_b_critical": right, "neither_critical": neither,
            "prevalence_a": float(pa) if n else None, "prevalence_b": float(pb) if n else None,
            "agreement": observed,
            "cohen_kappa": (observed-expected)/(1-expected) if n and expected < 1 else None,
            "positive_agreement": 2*both/(2*both+left+right) if 2*both+left+right else None,
            "negative_agreement": 2*neither/(2*neither+left+right) if 2*neither+left+right else None}


def run(input_csv, output_dir):
    frame = pd.read_csv(input_csv, keep_default_na=False)
    require_columns(frame, POSITION + ["labeling_model", "model_specific_label", "vote_status", "base_group_id", "source_dataset", "span_category"])
    require_unique(frame, POSITION + ["labeling_model"], "prepared votes")
    valid = frame[frame.vote_status == "valid"].copy()
    if not valid.model_specific_label.isin(["critical", "non_critical"]).all():
        raise ValueError("Unknown valid vote label")
    valid["critical"] = valid.model_specific_label.eq("critical").astype(int)
    rows = []
    strata = [("ALL", "ALL", valid)]
    strata += [(str(source), "ALL", group) for source, group in valid.groupby("source_dataset")]
    memberships = valid.copy()
    memberships["category_membership"] = memberships.apply(
        lambda r: json.loads(r.span_categories) if "span_categories" in r and r.span_categories else [r.span_category], axis=1)
    memberships = memberships.explode("category_membership")
    strata += [(str(source), str(category), group) for (source, category), group in
               memberships.groupby(["source_dataset", "category_membership"])]
    for source, category, group in strata:
        panel = group.pivot(index=POSITION, columns="labeling_model", values="critical")
        families = group.drop_duplicates(POSITION).set_index(POSITION).base_group_id
        for a, b in combinations(sorted(panel.columns), 2):
            common = panel[[a,b]].dropna()
            rows.append({"source_dataset": source, "category": category, "model_a": a, "model_b": b,
                         "n_base_groups": families.reindex(common.index).nunique(),
                         **pair_statistics(common[a], common[b])})
    target = Path(output_dir); claim_output(target); target.mkdir(parents=True)
    atomic_csv(pd.DataFrame(rows), target / "pairwise.csv")
    atomic_csv(frame.groupby(["labeling_model", "source_dataset", "vote_status"]).size().rename("n_votes").reset_index(), target / "coverage.csv")
    prevalence = valid.groupby(["labeling_model", "source_dataset", "span_category"]).critical.agg(["mean", "count"]).reset_index()
    atomic_csv(prevalence.rename(columns={"mean":"critical_prevalence", "count":"n_valid"}), target / "prevalence.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overall=pd.DataFrame(rows)
    if not overall.empty:
        overall=overall[(overall.source_dataset=="ALL") & (overall.category=="ALL")].copy()
        overall.index=overall.model_a+" / "+overall.model_b
        axis=overall[["agreement","positive_agreement","negative_agreement"]].plot.bar(figsize=(8,4),ylim=(0,1))
        axis.set_ylabel("Agreement on common valid occurrences")
        axis.figure.tight_layout(); axis.figure.savefig(target/"agreement.png",dpi=180); plt.close(axis.figure)
    atomic_json(target / "manifest.json", {"input_sha256": file_hash(input_csv), "n_valid": len(valid),
        "note": "Pair-specific common eligible occurrences; category memberships can overlap. Agreement is conditional on original-answer eligibility and is not universal model equivalence."})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes", required=True); parser.add_argument("--output-dir", required=True)
    args=parser.parse_args(); run(args.votes, args.output_dir)


if __name__ == "__main__":
    main()
