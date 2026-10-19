"""Environment inventory, pilot checks, and a local results ZIP (never uploads)."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
import pandas as pd
from final_eval.inference.client import public_config, generation_config
from final_eval.utils.artifacts import atomic_json, claim_output, file_hash, package_versions, parse_bool, require_unique, digest


def environment(output):
    report={"python":sys.version,"platform":platform.platform(),"packages":package_versions()}
    import torch
    report["torch_cuda"]=torch.version.cuda
    report["gpus"]=[{"name":torch.cuda.get_device_name(i),"total_bytes":torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    try:
        report["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        report["git_commit"]=None
    # Name/version inventory only: direct-install URLs in pip freeze can contain credentials.
    report["installed_packages"]=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()})
    atomic_json(output,report)
    print(f"Saved environment inventory to {output}")


def check(compressed,responses,output,min_questions=12,min_original_accuracy=0.5):
    comp=pd.read_csv(compressed,keep_default_na=False); res=pd.read_csv(responses,keep_default_na=False)
    require_unique(comp,["uid","method","target_ratio"],"pilot compression")
    require_unique(res,["uid","eval_model","method","target_ratio"],"pilot responses")
    problems=[]
    if comp.empty:
        problems.append("Empty compression table")
    if not comp.empty and not comp.status.eq("ok").all():
        problems.append("One or more compression conditions failed")
    compliant=comp.budget_compliant.map(parse_bool)
    if not compliant.all():
        problems.append("A compressor exceeds its requested budget; inspect native output/settings before the full run")
    success=res[res.response_status=="ok"].copy()
    full=success[success.method=="none"].copy()
    if not res.response_status.eq("ok").all():
        problems.append("A provider/target condition failed, was empty, or was truncated")
    model_results=[]
    models=sorted(res.eval_model.unique())
    if not models:
        problems.append("No target model responses")
    for model in models:
        originals=full[full.eval_model==model]
        accuracy=originals.correct.map(parse_bool).mean() if len(originals) else None
        expected=set(comp[["uid","method","target_ratio"]].itertuples(index=False,name=None))
        observed=set(success[(success.eval_model==model)&(success.method!="none")][["uid","method","target_ratio"]].itertuples(index=False,name=None))
        if observed!=expected or set(originals.uid)!=set(comp.uid):
            problems.append(f"Incomplete target condition grid for {model}")
        if len(originals)<min_questions or accuracy is None or accuracy<min_original_accuracy:
            problems.append(f"Insufficient pilot coverage/full-input accuracy for {model}")
        model_results.append({"model":model,"n_originals":len(originals),"full_input_accuracy":accuracy,
            "per_dataset":{str(source):{"n":len(g),"accuracy":g.correct.map(parse_bool).mean()} for source,g in originals.groupby("source_dataset")}})
    if "parse_status" in res:
        poor=full.parse_status.ne("ok").sum()
        if poor:
            problems.append("Some successful original responses have unparseable final answers")
    manifest=Path(str(responses)+".manifest.json")
    cfg=json.loads(manifest.read_text()).get("config") if manifest.exists() else None
    if cfg is None:
        problems.append("Missing target configuration provenance")
    report={"passed":not problems,"problems":problems,"models":model_results,
        "target_config_hash":digest(generation_config(cfg)) if cfg is not None else None,
        "pilot_base_groups":sorted(comp.base_group_id.unique()),
        "compression_sha256":file_hash(compressed),"responses_sha256":file_hash(responses),
        "n_questions":comp.uid.nunique(),"compression_statuses":comp.status.value_counts().to_dict(),
        "budget_violations":int((~compliant).sum()),"response_statuses":res.response_status.value_counts().to_dict(),
        "minimum_original_questions":min_questions,"minimum_original_accuracy":min_original_accuracy,
        "note":"Operational gate, not a statistical adequacy guarantee. Inspect each dataset's small pilot and failures. Do not change thresholds after seeing main results."}
    atomic_json(output,report)
    print(json.dumps(report,indent=2,default=str))
    return report


def bundle(run_dir,prepared_dir,model_dir,output):
    root=Path.cwd().resolve(); claim_output(output)
    selections=[]; missing=[]
    for directory in [run_dir,prepared_dir]:
        folder=Path(directory).resolve()
        folder.relative_to(root)
        if not folder.is_dir():
            missing.append(str(directory))
            continue
        selections += [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".csv",".json",".png",".svg",".md",".txt"}]
    folder=Path(model_dir).resolve(); folder.relative_to(root)
    selections += [folder/name for name in ["training_metadata.json","scorer_report.json","scorer_oof_predictions.csv","split.json"] if (folder/name).is_file()]
    selections += [p for p in folder.rglob("*.json") if p.is_file()]
    package=root/"final_eval"
    selections += [p for p in package.rglob("*") if p.is_file() and p.suffix in {".py",".md",".txt"}]
    selections += [p for p in (root/"PROJECT_COMPLETION_PLAN.md",) if p.exists()]
    selected=sorted(set(selections))
    for path in selected:
        path.resolve().relative_to(root)
        if path.suffix==".json":
            try:
                value=json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeError,json.JSONDecodeError):
                raise ValueError(f"Invalid JSON artifact: {path.name}")
            if isinstance(value,dict) and ("backend" in value or "base_url" in value):
                public_config(value)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    manifest={str(p.relative_to(root)).replace("\\","/"):file_hash(p) for p in selected}
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path,str(path.relative_to(root)))
        archive.writestr("HANDOFF_MANIFEST.json",json.dumps({"files":manifest,"missing_directories":missing},indent=2))
    print(f"Saved {len(selected)} artifacts to {output}; no checkpoint weights, .env, or uploads included.")


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    env=sub.add_parser("environment"); env.add_argument("--output",required=True)
    pilot=sub.add_parser("check")
    for name in ("compressed","responses","output"):
        pilot.add_argument("--"+name,required=True)
    pilot.add_argument("--min-questions",type=int,default=12)
    pilot.add_argument("--min-original-accuracy",type=float,default=.5)
    pack=sub.add_parser("bundle")
    for name in ("run-dir","prepared-dir","model-dir","output"):
        pack.add_argument("--"+name,required=True)
    args=parser.parse_args()
    if args.command=="environment":
        environment(args.output)
    elif args.command=="check":
        report=check(args.compressed,args.responses,args.output,args.min_questions,args.min_original_accuracy)
        if not report["passed"]:
            raise SystemExit(2)
    else:
        bundle(args.run_dir,args.prepared_dir,args.model_dir,args.output)


if __name__=="__main__":
    main()
