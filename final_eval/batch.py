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
    parser.add_argument("--methods", nargs="+", choices=[
        "ours", "ours_scorer", "ours_signal", "grouped_hybrid",
        "grouped_signal", "random", "surprisal_word",
        "llmlingua_budgeted", "llmlingua2_budgeted",
        "selective_context_budgeted"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.75, 0.5])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-bonuses", action="store_true")
    parser.add_argument("--protect-interface", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true",
                        help="Permit a historical checkpoint for descriptive comparison.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force CPU compression; useful for local dry runs and scorer-only tests.")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--pilot-check")
    parser.add_argument("--plan",action="store_true",help="Print commands, write nothing and execute nothing")
    args=parser.parse_args(); folder=Path(args.work_dir)
    methods = args.methods or METHODS
    if len(set(methods)) != len(methods):
        parser.error("Duplicate methods are not allowed")
    if any(not 0 < ratio <= 1 for ratio in args.ratios):
        parser.error("Ratios must be in (0, 1]")
    cohort=pd.read_csv(args.cohort,keep_default_na=False)
    if args.pilot:
        if len(cohort)>16:
            raise ValueError("Pilot cap is 16 questions")
    else:
        if args.pilot_check:
            check=json.loads(Path(args.pilot_check).read_text())
            if not check.get("passed") or set(cohort.base_group_id)&set(check["pilot_base_groups"]):
                raise ValueError("Need a passed pilot and disjoint main families")
        else:
            print("No target pilot supplied: compression-only run; reader inference remains gated separately.",
                  flush=True)
    commands=[]; stages=[]
    own_methods = [method for method in methods if method in {
        "ours", "ours_scorer", "ours_signal", "grouped_hybrid", "grouped_signal",
        "random", "surprisal_word"}]
    stage_specs = ([("neural", own_methods)] if own_methods else [])
    stage_specs.extend((method, [method]) for method in methods if method not in own_methods)
    for name, stage_methods in stage_specs:
        output=folder/f"{name}.csv"; stages.append(output)
        commands.append([sys.executable,"-m","final_eval.compressor.compress","--input-csv",args.cohort,
            "--output-csv",str(output),"--model-dir",args.model_dir,"--labels-file",args.labels_file,
            "--methods",*stage_methods,"--ratios",*map(str,args.ratios),
            "--alpha",str(args.alpha)] +
            (["--allow-legacy-checkpoint"] if args.allow_legacy_checkpoint else []) +
            (["--no-bonuses"] if args.no_bonuses else []) +
            (["--protect-interface"] if args.protect_interface else []) +
            (["--no-cuda"] if args.no_cuda else []))
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
            "--expected-methods",*methods,"--output",str(merged)],check=True)
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
