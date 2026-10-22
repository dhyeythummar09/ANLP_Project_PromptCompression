"""Run compression methods in separate processes, then merge and audit outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from final_eval.utils.artifacts import file_hash, parse_bool

METHODS=["ours","grouped_hybrid","grouped_signal","random","llmlingua_budgeted","llmlingua2_budgeted","selective_context_budgeted"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cohort","work-dir","model-dir","labels-file"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--pilot-check")
    parser.add_argument("--plan",action="store_true",help="Print commands, write nothing and execute nothing")
    args=parser.parse_args(); folder=Path(args.work_dir)
    cohort=pd.read_csv(args.cohort,keep_default_na=False)
    if args.pilot:
        if len(cohort)>16:
            raise ValueError("Pilot cap is 16 questions")
    else:
        if not args.pilot_check:
            raise ValueError("Main batch requires --pilot-check")
        check=json.loads(Path(args.pilot_check).read_text())
        if not check.get("passed") or set(cohort.base_group_id)&set(check["pilot_base_groups"]):
            raise ValueError("Need a passed pilot and disjoint main families")
    commands=[]; stages=[]
    for name,methods in [("ours",METHODS[:4])]+[(m,[m]) for m in METHODS[4:]]:
        output=folder/f"{name}.csv"; stages.append(output)
        commands.append([sys.executable,"-m","final_eval.compressor.compress","--input-csv",args.cohort,
            "--output-csv",str(output),"--model-dir",args.model_dir,"--labels-file",args.labels_file,
            "--methods",*methods,"--ratios","0.75","0.5"])
    merged=folder/"compressed.csv"
    for command in commands:
        print(json.dumps(command),flush=True)
        if not args.plan:
            subprocess.run(command,check=True)
    if args.plan:
        print("Then merge, intrinsic diagnostics, and compression budget/failure checks. No inference is performed by batch.")
        return
    expected_hashes={str(path):file_hash(path) for path in stages}
    if merged.exists():
        manifest=json.loads(Path(str(merged)+".merge.json").read_text())
        if manifest["inputs"]!=expected_hashes:
            raise ValueError("Existing merge belongs to different stage files; use a new merge/report path")
    else:
        subprocess.run([sys.executable,"-m","final_eval.preparation.merge_compressions","--inputs",*map(str,stages),
            "--expected-methods",*METHODS,"--output",str(merged)],check=True)
    intrinsic=folder/"intrinsic"
    if intrinsic.exists():
        manifest=json.loads((intrinsic/"manifest.json").read_text())
        if manifest["input_hash"]!=file_hash(merged):
            raise ValueError("Existing intrinsic report has different inputs; use a new report directory")
    else:
        subprocess.run([sys.executable,"-m","final_eval.compressor.metrics","--input",str(merged),"--output-dir",str(intrinsic)],check=True)
    frame=pd.read_csv(merged,keep_default_na=False)
    failed=frame.status.ne("ok"); over=~frame.budget_compliant.map(parse_bool)
    if failed.any() or over.any():
        print(f"STOP before target calls: {int(failed.sum())} failed and {int(over.sum())} over-budget compression conditions. Return the artifacts; do not truncate a baseline manually.")
        raise SystemExit(2)
    print("Compression grid complete and budget compliant. See the recorded target workflow in README.md.")


if __name__=="__main__":
    main()
