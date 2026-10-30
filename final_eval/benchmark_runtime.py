"""Controlled runtime/memory benchmark for our compressor (NEXT_STEPS step 6).

The main-study timings in RESULTS.md are cold Kaggle processes with different
batching, so they cannot be compared per stage. Here everything runs in one
process, on one device, with a fixed thread count, after a warm-up, and each
stage is timed on its own:

* parser    spaCy span extraction (``extract_spans``) and the second spaCy
            pass that ``prepare_ours`` uses for bonuses/groups
* models    DistilBERT span scorer and DistilGPT2 surprisal
* prepare   the full ``prepare_ours`` call as the pipeline runs it
* select    priority mixing + ``select_to_budget`` (exact final-string budget)

Stage timings are measured in isolation, so parser + scorer + surprisal can
add up to slightly more or less than ``prepare``; ``prepare + select`` is the
real per-condition cost. Model load time and resident memory are recorded
separately. The produced compressions are checked against the saved step-3
texts so we know we timed the same pipeline that was evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd

from final_eval.compressor.budget import TokenCounter, select_to_budget, words
from final_eval.compressor.compress import (MODEL_NAME, OURS_ALPHA, InfoScorer, extract_spans,
                                            load_scorer, prepare_ours, score_spans)
from final_eval.utils.artifacts import atomic_json, file_hash, package_versions

METHODS = ["ours", "ours_scorer", "ours_signal"]


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, (time.perf_counter() - start) * 1000


def _select(prep, method, ratio, text, counter):
    alpha = 0.5 if method == "ours" else OURS_ALPHA[method]
    priority = alpha * prep["crit"] + (1 - alpha) * prep["info"] + prep["bonus"]
    return select_to_budget(text, priority, ratio, counter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default="outputs/followup/components_v2/cohort.csv",
                        help="Frozen 160-question cohort with scorer folds")
    parser.add_argument("--model-dir", default="models/span_scorer_v2_20261021")
    parser.add_argument("--reference", default="outputs/followup/components_v2/matched/prepared.csv",
                        help="Saved step-3 compressions to check output identity against")
    parser.add_argument("--output-dir", default="results/followup/runtime")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    import spacy
    from transformers import DistilBertTokenizerFast

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device("cpu")
    cohort = pd.read_csv(args.cohort, keep_default_na=False, dtype={"ground_truth_answer": str})
    cohort = cohort.sort_values(["fold", "uid"])
    if args.limit:
        cohort = cohort.head(args.limit)

    # Load costs, each with the resident-memory step it adds.
    loads, rss = [], _rss_mb()
    counter, ms = _timed(TokenCounter, "cl100k_base")
    loads.append({"component": "tiktoken cl100k_base", "load_ms": ms, "rss_after_mb": _rss_mb(), "rss_added_mb": _rss_mb() - rss})
    rss = _rss_mb()
    nlp, ms = _timed(spacy.load, "en_core_web_sm")
    loads.append({"component": "spaCy en_core_web_sm (parser)", "load_ms": ms, "rss_after_mb": _rss_mb(), "rss_added_mb": _rss_mb() - rss})
    rss = _rss_mb()
    info, ms = _timed(InfoScorer, "distilgpt2", device)
    loads.append({"component": "DistilGPT2 surprisal", "load_ms": ms, "rss_after_mb": _rss_mb(), "rss_added_mb": _rss_mb() - rss})
    rss = _rss_mb()
    location = Path(args.model_dir) / "tokenizer"
    tokenizer, ms = _timed(DistilBertTokenizerFast.from_pretrained, str(location) if location.exists() else MODEL_NAME)
    loads.append({"component": "DistilBERT tokenizer", "load_ms": ms, "rss_after_mb": _rss_mb(), "rss_added_mb": _rss_mb() - rss})

    models = {}
    for fold in sorted(cohort.fold.unique()):
        rss = _rss_mb()
        models[fold], ms = _timed(load_scorer, Path(args.model_dir) / f"fold{fold}", device)
        loads.append({"component": f"DistilBERT span scorer fold{fold}", "load_ms": ms,
                      "rss_after_mb": _rss_mb(), "rss_added_mb": _rss_mb() - rss})

    def one_question(row) -> tuple[dict, dict]:
        text, model = row.original_prompt, models[row.fold]
        timing = {"uid": row.uid, "source_dataset": row.source_dataset, "fold": int(row.fold),
                  "orig_tokens": counter(text), "words": len(words(text))}
        spans, timing["parser_spans_ms"] = _timed(extract_spans, text, nlp)
        timing["spans"] = len(spans)
        _, timing["parser_doc_ms"] = _timed(nlp, text)
        _, timing["scorer_ms"] = _timed(score_spans, text, spans, model, tokenizer, device)
        _, timing["surprisal_ms"] = _timed(info.word_info, text, words(text))
        _, timing["token_count_ms"] = _timed(counter, text)
        neural, timing["prepare_neural_ms"] = _timed(prepare_ours, text, model, tokenizer, nlp, info, device)
        signal, timing["prepare_signal_ms"] = _timed(prepare_ours, text, None, tokenizer, nlp, info, device, None)
        outputs = {}
        for method in METHODS:
            prep = signal if method == "ours_signal" else neural
            for ratio in args.ratios:
                selection, ms = _timed(_select, prep, method, ratio, text, counter)
                timing[f"select_{method}_{ratio}_ms"] = ms
                outputs[(row.uid, method, ratio)] = selection.text
        return timing, outputs

    for row in cohort.head(args.warmup).itertuples():
        one_question(row)

    timings, produced = [], {}
    started = time.perf_counter()
    for repeat in range(args.repeats):
        for row in cohort.itertuples():
            timing, outputs = one_question(row)
            timings.append({"repeat": repeat, **timing})
            produced.update(outputs)
    wall = time.perf_counter() - started
    timings = pd.DataFrame(timings)

    # Per-condition cost as the pipeline pays it: one preparation + one selection.
    for method in METHODS:
        prep = "prepare_signal_ms" if method == "ours_signal" else "prepare_neural_ms"
        for ratio in args.ratios:
            timings[f"end_to_end_{method}_{ratio}_ms"] = timings[prep] + timings[f"select_{method}_{ratio}_ms"]

    stage_columns = [c for c in timings.columns if c.endswith("_ms")]
    summary = pd.DataFrame([{
        "stage": c.removesuffix("_ms"), "n": int(timings[c].size),
        "mean_ms": timings[c].mean(), "median_ms": timings[c].median(),
        "p90_ms": timings[c].quantile(.9), "max_ms": timings[c].max()} for c in stage_columns])

    identity = None
    if args.reference and Path(args.reference).is_file():
        reference = pd.read_csv(args.reference, keep_default_na=False)
        reference = reference[reference.method.isin(METHODS)].set_index(["uid", "method", "target_ratio"]).compressed_prompt
        keys = [k for k in produced if k in reference.index]
        identity = {"compared": len(keys), "identical": int(sum(produced[k] == reference.loc[k] for k in keys))}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings.to_csv(out / "per_question.csv", index=False)
    summary.to_csv(out / "stages.csv", index=False)
    pd.DataFrame(loads).to_csv(out / "model_loads.csv", index=False)
    atomic_json(out / "manifest.json", {
        "purpose": "controlled single-process CPU benchmark of ours / ours_scorer / ours_signal",
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count(),
                    "python": platform.python_version()},
        "device": "cpu", "torch_threads": args.threads, "packages": package_versions(),
        "cohort": {"path": args.cohort, "sha256": file_hash(args.cohort), "questions": int(len(cohort))},
        "model_dir": args.model_dir, "ratios": args.ratios, "repeats": args.repeats, "warmup_questions": args.warmup,
        "timed_wall_seconds": wall, "peak_rss_mb": _peak_rss_mb(), "final_rss_mb": _rss_mb(),
        "output_identity_vs_step3": identity,
        "notes": ["stage timings are isolated calls; prepare_* is the real preparation cost",
                  "end_to_end = preparation + selection for one condition; variants sharing a "
                  "question share one preparation in the batch pipeline",
                  "no GPU; LLMLingua baselines are not installed locally and are not timed here"]})
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(json.dumps({"peak_rss_mb": _peak_rss_mb(), "identity": identity, "wall_s": wall}, indent=2))


if __name__ == "__main__":
    main()
