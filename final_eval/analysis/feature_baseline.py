"""CPU-only soft-target ridge baseline on the scorer's exact saved partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from final_eval.scorer.data import build_span_table
from final_eval.preparation.schema import POSITION, attach_groups
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, require_unique, package_versions

NUMERIC = ["relative_start", "relative_length", "log_prompt_length", "log_span_length", "contains_digit"]
CATEGORICAL = ["source_dataset", "span_category"]


def features(table):
    result = table[CATEGORICAL].astype(str).copy()
    length = table.original_prompt.astype(str).str.len().clip(lower=1)
    result["relative_start"] = table.span_start / length
    result["relative_length"] = (table.span_end-table.span_start) / length
    result["log_prompt_length"] = np.log1p(length)
    result["log_span_length"] = np.log1p(table.span_text.astype(str).str.len())
    result["contains_digit"] = table.span_text.astype(str).str.contains(r"\d", regex=True).astype(float)
    return result


def metrics(y, target, prediction):
    y=np.asarray(y); prediction=np.asarray(prediction)
    return {"n":len(y), "roc_auc":float(roc_auc_score(y,prediction)) if len(set(y))>1 else None,
            "pr_auc":float(average_precision_score(y,prediction)) if len(set(y))>1 else None,
            "f1_critical":float(f1_score(y,prediction>=0.5,zero_division=0)),
            "brier_soft":float(np.mean((prediction-np.asarray(target))**2))}


def run(consensus_file, model_dir, output_dir):
    model_dir=Path(model_dir)
    split=json.loads((model_dir/"split.json").read_text())
    metadata=json.loads((model_dir/"training_metadata.json").read_text())
    if split.get("protocol")!="symbiote-v2" or not split.get("partitions"):
        raise ValueError("Need saved v2 training partitions")
    if file_hash(consensus_file)!=metadata["consensus_sha256"]:
        raise ValueError("Consensus differs from the trained scorer's input")
    table=attach_groups(build_span_table(pd.read_csv(consensus_file,keep_default_na=False),split.get("holdout_model")))
    X=features(table); parts=[]
    for fold, roles in sorted(split["partitions"].items(),key=lambda x:int(x[0])):
        group_sets=[]
        for role in ("train","val","test"):
            group_sets.append(set(table.loc[table.uid.isin(roles[role]),"base_group_id"]))
        if group_sets[0]&group_sets[1] or group_sets[0]&group_sets[2] or group_sets[1]&group_sets[2]:
            raise ValueError("Family leakage in supplied partitions")
        train=table.uid.isin(roles["train"]); test=table.uid.isin(roles["test"])
        if not train.any() or not test.any():
            raise ValueError("Empty training or test partition")
        transform=ColumnTransformer([("category",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL),
                                     ("numeric",StandardScaler(),NUMERIC)])
        estimator=make_pipeline(transform,Ridge(alpha=1.0,solver="lsqr"))
        # Fit preprocessing and weights exclusively on this fold's train questions.
        estimator.fit(X.loc[train],table.loc[train,"target"])
        part=table.loc[test,POSITION+["base_group_id","target","eval_label"]].copy()
        part["feature_prob"]=np.clip(estimator.predict(X.loc[test]),0,1)
        part["fold"]=int(fold); parts.append(part)
    oof=pd.concat(parts,ignore_index=True); require_unique(oof,POSITION,"feature OOF")
    neural=pd.read_csv(model_dir/"scorer_oof_predictions.csv",keep_default_na=False)
    require_unique(neural,POSITION,"neural OOF")
    shared=oof.merge(neural[POSITION+["prob","prior_prob"]],on=POSITION,validate="one_to_one")
    if shared.empty:
        raise ValueError("No shared evaluated occurrences")
    report={"n_feature_oof":len(oof),"n_shared_with_neural":len(shared),
            "n_feature_only":len(oof)-len(shared),"features":CATEGORICAL+NUMERIC,
            "definition":"Fixed alpha=1 ridge on soft training targets; outputs clipped to [0,1]. Scorer comparison only, not an extra compressor or a calibrated probability claim.",
            "shared_metrics":{name:metrics(shared.eval_label,shared.target,shared[column]) for name,column in
                              [("feature_ridge","feature_prob"),("neural","prob"),("category_prior","prior_prob")]},
            "consensus_sha256":file_hash(consensus_file),"split_sha256":file_hash(model_dir/"split.json"),"packages":package_versions()}
    target=Path(output_dir); claim_output(target); target.mkdir(parents=True)
    atomic_csv(oof,target/"feature_oof.csv"); atomic_csv(shared,target/"shared_oof.csv")
    atomic_json(target/"feature_report.json",report)
    if shared.eval_label.nunique()>1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig,axis=plt.subplots(figsize=(5,4))
        for label,column in [("Feature ridge","feature_prob"),("Neural scorer","prob"),("Category prior","prior_prob")]:
            fpr,tpr,_=roc_curve(shared.eval_label,shared[column]); axis.plot(fpr,tpr,label=label)
        axis.plot([0,1],[0,1],linestyle="--",color="grey")
        axis.set(xlabel="False positive rate",ylabel="True positive rate",title="Shared out-of-fold occurrences")
        axis.legend(fontsize=8); fig.tight_layout(); fig.savefig(target/"scorer_roc.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-file",required=True); parser.add_argument("--model-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args(); run(args.consensus_file,args.model_dir,args.output_dir)


if __name__ == "__main__":
    main()
