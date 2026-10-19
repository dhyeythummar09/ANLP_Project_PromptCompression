"""Generate offline figures, retention prediction checks, and a claims ledger."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, parse_bool, require_columns, require_unique

KEY=["uid","method","target_ratio"]


def predictive_check(group):
    """Fixed regularized model, OOF by family; no causal or p-value interpretation."""
    report={"n":len(group),"n_families":group.base_group_id.nunique()}
    if report["n_families"]<5 or group.correct.nunique()<2:
        return {**report,"status":"insufficient_families_or_outcome_variation"},pd.DataFrame()
    numeric=["actual_ratio","log_original_tokens","original_correct"]
    y=group.correct.to_numpy(); predictions={name:np.full(len(group),np.nan) for name in ("controls","controls_and_retention")}
    assignment=np.full(len(group),-1)
    for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(group,y,group.base_group_id)):
        if len(np.unique(y[train]))<2:
            return {**report,"status":"single_class_training_fold"},pd.DataFrame()
        assignment[test]=fold
        for name in predictions:
            columns=numeric+(["critical_recall_lower"] if name!="controls" else [])
            transform=ColumnTransformer([("numeric",StandardScaler(),columns),
                                         ("method",OneHotEncoder(handle_unknown="ignore"),["method"])])
            estimator=make_pipeline(transform,LogisticRegression(C=1.0,max_iter=2000,random_state=42))
            estimator.fit(group.iloc[train],y[train])
            predictions[name][test]=estimator.predict_proba(group.iloc[test])[:,1]
    report["status"]="ok"
    for name,prediction in predictions.items():
        report[name+"_roc_auc"]=float(roc_auc_score(y,prediction))
        report[name+"_brier"]=float(np.mean((prediction-y)**2))
    report["brier_improvement_with_retention"]=report["controls_brier"]-report["controls_and_retention_brier"]
    detail=group[KEY+["base_group_id","correct"]].copy()
    detail["analysis_fold"]=assignment
    for name,prediction in predictions.items():
        detail[name+"_prediction"]=prediction
    return report,detail


def run(compressed,responses,reports,output_dir):
    provenance=json.loads((Path(reports)/"manifest.json").read_text())
    if provenance["results_hash"]!=file_hash(responses) or provenance["compression_hash"]!=file_hash(compressed):
        raise ValueError("Summary reports have stale inputs; rebuild summaries in a new report directory. Do not rerun model inference.")
    comp=pd.read_csv(compressed,keep_default_na=False); res=pd.read_csv(responses,keep_default_na=False)
    require_unique(comp,KEY,"compression"); require_unique(res,KEY+["eval_model"],"responses")
    require_columns(comp,["crit_total","crit_kept_lower","crit_kept_upper","orig_tokens","budget_compliant","status"])
    require_columns(res,["response_status","correct","base_group_id","source_dataset","eval_model"])
    successful=res[res.response_status=="ok"].copy()
    successful["correct"]=successful.correct.map(parse_bool).astype(int)
    original=successful[successful.method=="none"][["eval_model","uid","correct"]].rename(columns={"correct":"original_correct"})
    require_unique(original,["eval_model","uid"],"full-input responses")
    detail=successful[successful.method!="none"].merge(
        comp[KEY+["crit_total","crit_kept_lower","crit_kept_upper","orig_tokens","budget_compliant","status"]],on=KEY,validate="many_to_one")
    detail=detail.merge(original,on=["eval_model","uid"],validate="many_to_one")
    # Match the main report's common complete cohort, then retain compliant conditions.
    cohorts=[]
    for (model,ratio),group in detail.groupby(["eval_model","target_ratio"]):
        n_methods=comp.loc[comp.target_ratio==ratio,"method"].nunique()
        eligible=group.groupby("uid").method.nunique()
        cohorts.append(group[group.uid.isin(eligible[eligible==n_methods].index)])
    detail=pd.concat(cohorts,ignore_index=True) if cohorts else detail.iloc[:0].copy()
    detail=detail[(detail.status=="ok") & detail.budget_compliant.map(parse_bool)].copy()
    for column in ("crit_total","crit_kept_lower","crit_kept_upper","orig_tokens","actual_ratio"):
        detail[column]=pd.to_numeric(detail[column],errors="coerce")
    detail["critical_recall_lower"]=detail.crit_kept_lower/detail.crit_total.where(detail.crit_total>0)
    detail["critical_recall_upper"]=detail.crit_kept_upper/detail.crit_total.where(detail.crit_total>0)
    detail["log_original_tokens"]=np.log1p(detail.orig_tokens)
    finite=detail.dropna(subset=["critical_recall_lower","critical_recall_upper","actual_ratio","log_original_tokens"])
    strata=[]; predictions=[]
    for keys,group in finite.groupby(["eval_model","target_ratio","source_dataset"]):
        report,part=predictive_check(group.reset_index(drop=True))
        identity=dict(zip(["eval_model","target_ratio","source_dataset"],keys))
        strata.append({**identity,**report})
        if not part.empty:
            predictions.append(part.assign(**identity))
    target=Path(output_dir); claim_output(target); target.mkdir(parents=True)
    atomic_csv(detail,target/"retention_conditions.csv")
    atomic_csv(pd.DataFrame(strata),target/"retention_prediction.csv")
    if predictions:
        atomic_csv(pd.concat(predictions,ignore_index=True),target/"retention_prediction_oof.csv")
    accuracy=pd.read_csv(Path(reports)/"accuracy.csv")
    atomic_csv(accuracy[accuracy.source_dataset!="ALL"].groupby(["eval_model","target_ratio","method"],as_index=False).agg(
        dataset_macro_accuracy=("accuracy","mean"),n_datasets=("source_dataset","nunique"),n_questions=("n","sum")),target/"macro_accuracy.csv")
    timings=[]
    if "latency_ms" in comp:
        for keys,group in comp[comp.status=="ok"].groupby(["method","target_ratio"]):
            values=pd.to_numeric(group.latency_ms,errors="coerce").dropna()
            timings.append({"method":keys[0],"target_ratio":keys[1],"n":len(values),
                            "median_ms":values.median(),"p95_ms":values.quantile(.95),
                            "scope":"Condition timings may include cold baseline initialization; process profiles report total wall time and peak PyTorch memory."})
    atomic_csv(pd.DataFrame(timings),target/"compression_timing.csv")
    usage_rows=[]
    if "generation_usage" in res:
        for model,group in res.groupby("eval_model"):
            fresh=group[group.get("reused_generation",pd.Series("",index=group.index)).map(
                lambda v:not parse_bool(v) if str(v).strip() else False)].copy()
            fresh=fresh.drop_duplicates([c for c in ("query_hash","generation_timestamp_utc") if c in fresh])
            records=[]
            for encoded in fresh.generation_usage:
                value=json.loads(encoded) if isinstance(encoded,str) and encoded.strip().startswith("{") else {}
                if value:
                    records.append({"prompt_tokens":value.get("prompt_tokens",value.get("promptTokenCount")),
                                    "completion_tokens":value.get("completion_tokens",value.get("candidatesTokenCount")),
                                    "total_tokens":value.get("total_tokens",value.get("totalTokenCount"))})
            row={"eval_model":model,"n_fresh_generation_records":len(fresh),"n_records_with_usage":len(records)}
            for column in ("prompt_tokens","completion_tokens","total_tokens"):
                values=pd.to_numeric(pd.Series([r[column] for r in records],dtype=object),errors="coerce")
                row[column]=values.sum() if values.notna().any() else None
            usage_rows.append(row)
    atomic_csv(pd.DataFrame(usage_rows,columns=["eval_model","n_fresh_generation_records","n_records_with_usage","prompt_tokens","completion_tokens","total_tokens"]),target/"target_usage.csv")
    paired=pd.read_csv(Path(reports)/"paired.csv")
    ledger=[]
    for row in paired.to_dict("records"):
        interval=(pd.notna(row["ci_lo"]) and pd.notna(row["ci_hi"]))
        status=("no_compliant_pairs" if row["n_paired"]==0 else "insufficient_independent_groups" if not interval else
                "positive_paired_interval" if row["ci_lo"]>0 else "negative_paired_interval" if row["ci_hi"]<0 else "inconclusive_interval")
        ledger.append({**row,"evidence_status":status,
                       "scope":"This target/cohort/budget only; unadjusted intervals. Model equivalence, universal transfer, causality and overall SOTA are not established."})
    atomic_csv(pd.DataFrame(ledger),target/"claims_ledger.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i,(model,group) in enumerate(accuracy[accuracy.source_dataset=="ALL"].groupby("eval_model")):
        fig,axis=plt.subplots(figsize=(7,4))
        for method,points in group.groupby("method"):
            points=points.sort_values("actual_ratio")
            axis.plot(points.actual_ratio,points.accuracy,marker="o",label=method)
        axis.set(xlabel="Mean actual retained-token fraction",ylabel="Accuracy",title=str(model),ylim=(0,1))
        axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(target/f"accuracy_tokens_{i}.png",dpi=180); plt.close(fig)
    if not finite.empty:
        binned=finite.copy()
        binned["recall_bin"]=pd.cut(binned.critical_recall_lower,[0,.25,.5,.75,1],include_lowest=True).astype(str)
        bins=binned.groupby(["eval_model","target_ratio","source_dataset","recall_bin"],as_index=False).agg(
            n_conditions=("correct","size"),accuracy=("correct","mean"),mean_critical_recall=("critical_recall_lower","mean"))
        atomic_csv(bins,target/"retention_bins.csv")
        for i,(identity,group) in enumerate(bins.groupby(["eval_model","target_ratio"])):
            fig,axis=plt.subplots(figsize=(7,4))
            for source,points in group.groupby("source_dataset"):
                points=points.sort_values("mean_critical_recall")
                axis.plot(points.mean_critical_recall,points.accuracy,marker="o",label=source)
            axis.set(xlabel="Critical occurrence recall (lower bound)",ylabel="Accuracy",ylim=(0,1),
                     title=f"{identity[0]} / retained fraction {identity[1]} (descriptive)")
            axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(target/f"retention_accuracy_{i}.png",dpi=180); plt.close(fig)
    atomic_json(target/"manifest.json",{"compressed_sha256":file_hash(compressed),"responses_sha256":file_hash(responses),
        "accuracy_sha256":file_hash(Path(reports)/"accuracy.csv"),"paired_sha256":file_hash(Path(reports)/"paired.csv"),
        "n_compliant_common_conditions":len(detail),"n_with_resolved_positive_critical_support":len(finite),
        "retention_analysis":"Within target, dataset and ratio; 5-fold family-disjoint predictive analysis controlling for method, actual ratio, original correctness and input length. Annotation-based diagnostic, not a deployment model or causal test. Small strata explicitly skipped.",
        "plot_scope":"Native-output descriptive accuracy plot; inspect budget compliance and paired tables before claiming fixed-budget gains."})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("compressed","responses","reports","output-dir"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args(); run(args.compressed,args.responses,args.reports,args.output_dir)


if __name__=="__main__":
    main()
