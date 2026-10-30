"""python -m final_eval.v3 --help"""
from __future__ import annotations

import argparse
from pathlib import Path

from final_eval.v3.artifacts import ROOT, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled v3 students; original data collection and v2 remain unchanged.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = ["preflight", "prepare", "teacher", "model-smoke", "train", "qualify", "compress", "evaluate",
                "freeze", "report", "benchmark", "bundle", "run", "snapshot-exposure"]
    parsers = {}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(ROOT / "final_eval/v3/configs/student.json"))
        parsers[name] = p
    infer = sub.add_parser("infer", help="Compress a new question with the saved v3 model, without API calls")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--text", required=True)
    infer.add_argument("--ratio", type=float, default=.75, help="Fraction of question tokens to retain")
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parsers["preflight"].add_argument("--offline", action="store_true")
    for name in ("compress", "evaluate", "report"):
        parsers[name].add_argument("--role", choices=["dev", "test"], required=True)
    from final_eval.v3.pipeline import STAGES
    parsers["run"].add_argument("--through", choices=STAGES, default="bundle")
    for name in ("run", "qualify", "evaluate"):
        parsers[name].add_argument("--retry-failed", action="store_true")
    parsers["snapshot-exposure"].add_argument("--inputs", nargs="+", required=True)
    parsers["snapshot-exposure"].add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "infer":
        import json
        from final_eval.v3.compression import infer_text
        print(json.dumps(infer_text(args.run_dir, args.text, args.ratio, args.device), indent=2))
        return
    config = load_config(args.config)
    if args.command not in {"run", "preflight", "snapshot-exposure"}:
        from final_eval.v3.artifacts import exclusive_lock
        with exclusive_lock(Path(config["output_dir"]) / "pipeline.lock"):
            dispatch(args, config)
    else:
        dispatch(args, config)


def dispatch(args, config):
    name = args.command
    if name == "preflight":
        from final_eval.v3.pipeline import preflight
        preflight(config, offline=args.offline)
    elif name == "prepare":
        from final_eval.v3.data import prepare
        prepare(config)
    elif name == "teacher":
        from final_eval.v3.data import cached_teacher_labels, local_teacher_labels
        (cached_teacher_labels if config["teacher"]["mode"] == "cached" else local_teacher_labels)(config)
    elif name == "train":
        from final_eval.v3.train import train
        train(config)
    elif name == "model-smoke":
        from final_eval.v3.compression import smoke_dependencies
        smoke_dependencies(config)
    elif name == "qualify":
        from final_eval.v3.reader import qualify
        qualify(config, args.retry_failed)
    elif name == "compress":
        from final_eval.v3.compression import generate
        generate(config, args.role)
    elif name == "evaluate":
        from final_eval.v3.reader import evaluate
        evaluate(config, args.role, args.retry_failed)
    elif name == "freeze":
        from final_eval.v3.report import freeze
        freeze(config)
    elif name == "report":
        from final_eval.v3.report import summarize
        summarize(config, args.role)
    elif name == "benchmark":
        from final_eval.v3.benchmark import benchmark
        benchmark(config)
    elif name in {"bundle", "run"}:
        from final_eval.v3.pipeline import bundle, run
        run(config, args.through, args.retry_failed) if name == "run" else bundle(config)
    elif name == "snapshot-exposure":
        from final_eval.v3.data import snapshot_exposures
        snapshot_exposures(config["problems"], args.inputs, args.output)


if __name__ == "__main__":
    main()
