"""Shared plumbing for the follow-up hosted-reader runs (NEXT_STEPS steps 2-9).

Every follow-up comparison sends one prepared table through
``final_eval.compressor.evaluate`` with the same guards:

* the reader config must name the expected model (Qwen after GPT-OSS was retired),
* a passed pilot with identical generation settings and disjoint families is required,
* outputs of the retired GPT-OSS runs are never written to or resumed,
* a persistent INR cost ledger caps spending across restarts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from dotenv import dotenv_values

from final_eval.inference.client import generation_config, public_config
from final_eval.utils.artifacts import atomic_csv, atomic_json, digest, file_hash

# Outputs from the retired GPT-OSS attempt must never be resumed under another reader.
PROTECTED_PATH_PARTS = ("target_gptoss",)
DEFAULT_READER = "qwen-3.8-27b-fp8"
# Token Factory catalog: $0.30 / $2.35 per million tokens (rechecked 28 Oct 2026).
QWEN_INPUT_INR = 28.431
QWEN_OUTPUT_INR = 222.709


def read_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing reader config: {config_path}")
    return public_config(json.loads(config_path.read_text(encoding="utf-8")))


def read_pilot(path: str, config: dict) -> dict:
    pilot_path = Path(path)
    if not pilot_path.is_file():
        raise FileNotFoundError(
            f"Missing passed pilot: {pilot_path}. Run a disjoint pilot with this exact "
            "reader config first (see README).")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("passed"):
        raise ValueError(f"Pilot did not pass: {pilot_path}: {pilot.get('problems')}")
    if pilot.get("target_config_hash") != digest(generation_config(config)):
        raise ValueError("Reader config does not match the passed pilot settings")
    return pilot


def load_api_environment(config: dict) -> None:
    """Pull the API key from final_eval/jarvis.env without printing it."""
    env_path = Path("final_eval/jarvis.env")
    values = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
    variable = config.get("api_key_env")
    if variable and not os.environ.get(variable):
        value = (values.get(variable) or "").strip()
        if value:
            os.environ[variable] = value
    if variable and not os.environ.get(variable):
        raise ValueError(f"Missing {variable}; place it in final_eval/jarvis.env or the environment")


def check_reader_output(path: str) -> None:
    """Refuse to write into (or resume) another reader's response folder."""
    if any(part in PROTECTED_PATH_PARTS for part in Path(path).parts):
        raise ValueError(f"{path} belongs to a historical reader run; use a new target_* directory")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Prepared table from 'prepare'")
    parser.add_argument("--output", required=True, help="New responses CSV, e.g. target_qwen/")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--pilot-check", help="Passed pilot report (required unless --pilot-only)")
    parser.add_argument("--pilot-only", action="store_true",
                        help="Run the small pilot itself; skips the pilot-check gate")
    parser.add_argument("--expect-model", default=DEFAULT_READER,
                        help="Fail fast if the config names a different reader")
    parser.add_argument("--cost-limit-inr", type=float, required=True)
    parser.add_argument("--input-rate", type=float, default=QWEN_INPUT_INR,
                        help="INR per million input tokens")
    parser.add_argument("--output-rate", type=float, default=QWEN_OUTPUT_INR,
                        help="INR per million output tokens")
    parser.add_argument("--dry-run", action="store_true")


def launch(args: argparse.Namespace, prepared: pd.DataFrame) -> None:
    """Gate the run, then hand the prepared table to the resumable evaluator."""
    if not args.pilot_only and not args.pilot_check:
        raise ValueError("--pilot-check is required for the main run")
    config = read_config(args.model_config)
    if args.expect_model and config.get("name") != args.expect_model:
        raise ValueError(f"Config names {config.get('name')!r}, expected {args.expect_model!r}")
    check_reader_output(args.output)
    if not args.pilot_only:
        pilot = read_pilot(args.pilot_check, config)
        overlap = set(pilot.get("pilot_base_groups", [])) & set(prepared.base_group_id)
        if overlap:
            raise ValueError(f"Pilot overlaps the main cohort in {len(overlap)} families")
    command = [
        sys.executable, "-m", "final_eval.compressor.evaluate",
        "--mode", "run",
        "--input-csv", args.input,
        "--output-csv", args.output,
        "--model-config", args.model_config,
        "--retry-failed",
        "--cost-limit-inr", str(args.cost_limit_inr),
        "--input-rate", str(args.input_rate),
        "--output-rate", str(args.output_rate),
    ]
    if not args.pilot_only:
        command += ["--pilot-check", args.pilot_check]
    if args.dry_run:
        command.append("--dry-run")
    else:
        load_api_environment(config)
    print(json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def select_pilot_cohort(source: pd.DataFrame, exclude: pd.DataFrame, n: int,
                        seed: int) -> pd.DataFrame:
    """Pick one question per family, round-robin over datasets, avoiding excluded families.

    Selection uses only IDs and dataset names, never outcomes, so the pilot
    cannot be tuned toward any result.
    """
    eligible = source[~source.base_group_id.isin(set(exclude.base_group_id))]
    # One deterministic question per family keeps every pilot family independent.
    eligible = eligible.sort_values("uid").drop_duplicates("base_group_id")
    queues = {name: group.sample(frac=1, random_state=seed).uid.tolist()
              for name, group in eligible.groupby("source_dataset", sort=True)}
    chosen = []
    while len(chosen) < n and any(queues.values()):
        for name in sorted(queues):
            if queues[name] and len(chosen) < n:
                chosen.append(queues[name].pop(0))
    if len(chosen) < n:
        raise ValueError(f"Only {len(chosen)} eligible pilot families; need {n}")
    return eligible[eligible.uid.isin(chosen)].sort_values(["source_dataset", "uid"])


def pilot_cohort(args: argparse.Namespace) -> None:
    source = pd.read_csv(args.source_cohort, keep_default_na=False, dtype={"ground_truth_answer": str})
    exclude = pd.concat([pd.read_csv(path, keep_default_na=False) for path in args.exclude_cohort])
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"{output} already exists; keep it or choose a new path")
    pilot = select_pilot_cohort(source, exclude, args.n, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(pilot, output)
    atomic_json(str(output) + ".cohort.json", {
        "source_cohort": args.source_cohort, "source_sha256": file_hash(args.source_cohort),
        "excluded_cohorts": {path: file_hash(path) for path in args.exclude_cohort},
        "seed": args.seed, "n_questions": int(len(pilot)),
        "n_base_groups": int(pilot.base_group_id.nunique()),
        "datasets": pilot.source_dataset.value_counts().to_dict(),
        "selection": "one question per family, dataset round-robin, excluded families removed; no outcomes used"})
    print(f"Pilot cohort: {len(pilot)} questions / {pilot.base_group_id.nunique()} families -> {output}")


def add_pilot_cohort_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-cohort", required=True)
    parser.add_argument("--exclude-cohort", nargs="+", required=True,
                        help="Cohorts whose families the pilot must avoid (the main cohort at least)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261028)
