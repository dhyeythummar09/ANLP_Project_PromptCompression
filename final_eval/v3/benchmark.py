"""Controlled warm-compressor timing, separate from API/network latency."""
from __future__ import annotations

import gc
from pathlib import Path
import statistics
import time

from final_eval.utils.artifacts import atomic_json, file_hash
from final_eval.v3.artifacts import read_json, read_jsonl, stage_manifest, finish_stage


def benchmark(config):
    import spacy
    import torch
    from transformers import AutoTokenizer
    from final_eval.compressor.budget import TokenCounter, select_to_budget
    from final_eval.compressor.compress import load_scorer, InfoScorer, prepare_ours, BaselineRunner
    from final_eval.v3.compression import reference_fold, select
    from final_eval.v3.model import encode_question, predict_words
    from final_eval.v3.train import ARMS, load_model, choose_device
    from final_eval.v3.data import select_families
    run = Path(config["output_dir"])
    cfg = config["benchmark"]
    freeze = read_json(run / "freeze.json")
    marker, state, done = stage_manifest(run, "benchmark", {"settings": cfg,
        "freeze": file_hash(run / "freeze.json"), "primary_ratio": config["primary_ratio"]})
    if done:
        return
    v2 = Path(config["v2_model_dir"])
    oldsplit = read_json(v2 / "split.json")
    val = [q for q in read_jsonl(run / "data/questions.jsonl") if q["role"] == "val"]
    counts = {f: sum(reference_fold(q, oldsplit) == f for q in val) for f in range(5)}
    fold = max(counts, key=counts.get)
    pool = [q for q in val if reference_fold(q, oldsplit) == fold]
    n = min(cfg["questions"], len({q["base_group_id"] for q in pool}))
    if n < 1:
        raise ValueError("No validation families for controlled timing")
    questions = select_families(pool, n, config["split"]["seed"] + 50)
    device = choose_device(config)
    nlp = spacy.load(config["spacy_model"])
    counter = TokenCounter(config["tokenizer"])
    ratio = config["primary_ratio"]
    results = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(name, function, load_seconds):
        function(questions[0])  # explicit discarded warmup
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for repeat in range(cfg["repeats"]):
            for q in questions:
                synchronize()
                started = time.perf_counter()
                output = function(q)
                synchronize()
                samples.append({"uid": q["uid"], "repeat": repeat,
                                "milliseconds": (time.perf_counter() - started) * 1000,
                                "output_tokens": counter(output)})
        results.append({"method": name, "model_loading_seconds": load_seconds,
            "median_milliseconds": statistics.median(s["milliseconds"] for s in samples),
            "mean_milliseconds": statistics.mean(s["milliseconds"] for s in samples),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "samples": samples})

    tokenizer = AutoTokenizer.from_pretrained(run / "initialization/tokenizer", local_files_only=True)
    for arm in ARMS:
        started = time.perf_counter()
        model = load_model(config, run / "models" / arm / "best.pt", device).eval()
        loaded = time.perf_counter() - started

        def student(q):
            item = encode_question(q, tokenizer, max_length=config["training"]["max_length"], stride=config["training"]["stride"])
            prob = predict_words(model, item, device)
            return select(q["original_prompt"], prob, ratio, counter, freeze["policy"], nlp).text

        measure("student_" + arm, student, loaded)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    oldtok = AutoTokenizer.from_pretrained(v2 / "tokenizer", local_files_only=True)
    prior = read_json(v2 / f"fold{fold}/category_prior.json")
    from final_eval.v3.compression import info_location
    info_name = info_location(config)
    for neural in (True, False):
        started = time.perf_counter()
        info = InfoScorer(info_name, device)
        model = load_scorer(v2 / f"fold{fold}", device) if neural else None
        loaded = time.perf_counter() - started
        def original(q):
            pp = prepare_ours(q["original_prompt"], model if neural else None, oldtok, nlp, info, device, prior)
            return select_to_budget(q["original_prompt"], .5 * pp["crit"] + .5 * pp["info"] + pp["bonus"], ratio, counter, pp["groups"]).text
        measure("v2_grouped_hybrid" if neural else "v2_grouped_signal", original, loaded)
        del model, info
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if config["references"].get("llmlingua2", True):
        from final_eval.v3.train import download_model
        source = read_json(run / "teacher_reference_source.json")
        location = download_model(source)
        runner = BaselineRunner(str(device), "unused", location)
        # Instantiate separately so model loading is not mislabeled compression.
        from llmlingua import PromptCompressor
        started = time.perf_counter()
        runner.objects["llmlingua2"] = PromptCompressor(model_name=location, use_llmlingua2=True, device_map=str(device))
        loaded = time.perf_counter() - started
        def teacher(q):
            output, status, _ = runner.run_budgeted("llmlingua2_budgeted", q["original_prompt"], ratio,
                                                   int(ratio * counter(q["original_prompt"])), counter)
            if status != "ok":
                raise ValueError("Teacher could not meet the benchmark ceiling")
            return output
        measure("llmlingua2_budgeted", teacher, loaded)
    path = run / "benchmark.json"
    atomic_json(path, {"results": results, "device": str(device), "question_uids": [q["uid"] for q in questions],
        "ratio": ratio, "scope": "Warm full compressor: tokenization, parsing where used, all model forwards and exact selection. Excludes loading (reported separately), reader/API time and network downloads. No automatic speedup claim.",
        "memory_scope": "Peak CUDA allocated memory, not system RAM or total device memory",
        "baseline_fold": fold})
    finish_stage(marker, state, [path])
