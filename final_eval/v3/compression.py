"""Dense selection and matched historical/external references, with exact budgets."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import time

import numpy as np

from final_eval.compressor.budget import TokenCounter, select_to_budget
from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.v3.artifacts import read_json, read_jsonl, write_jsonl, stage_manifest, finish_stage, versions, log

BASELINES = ("v2_grouped_hybrid", "v2_grouped_signal", "llmlingua2_budgeted")


def infer_text(run_dir, text, ratio=.75, device="cpu"):
    """Compress a new question with the frozen combined student; no reader calls."""
    import torch
    from transformers import AutoTokenizer
    from final_eval.v3.train import load_model
    from final_eval.v3.model import encode_question, predict_words
    if not text.strip() or not 0 < ratio <= 1:
        raise ValueError("A nonempty question and retention ratio in (0,1] are required")
    run = Path(run_dir).resolve()
    config, frozen = read_json(run / "resolved_config.json"), read_json(run / "freeze.json")
    config["output_dir"] = str(run)
    weights = run / "models/teacher_cf/best.pt"
    if file_hash(weights) != frozen["checkpoints"]["teacher_cf"]:
        raise ValueError("Model weights differ from the verified frozen combined student")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(config["training"].get("cpu_threads", 2))
    tokenizer = AutoTokenizer.from_pretrained(run / "initialization/tokenizer", local_files_only=True)
    model = load_model(config, weights, target).eval()
    q = {"uid": "user-question", "source_dataset": "custom", "original_prompt": text, "spans": []}
    item = encode_question(q, tokenizer, max_length=config["training"]["max_length"], stride=config["training"]["stride"])
    probability = predict_words(model, item, target)
    nlp = None
    if frozen["policy"] == "atomic":
        import spacy
        nlp = spacy.load(config["spacy_model"])
    counter = TokenCounter(config["tokenizer"])
    result = select(text, probability, ratio, counter, frozen["policy"], nlp)
    return {"compressed_prompt": result.text, "original_tokens": counter(text), "compressed_tokens": counter(result.text),
            "actual_retention": counter(result.text) / counter(text), "target_retention": ratio,
            "policy": frozen["policy"], "checkpoint_sha256": frozen["checkpoints"]["teacher_cf"],
            "reader_calls": 0, "status": result.status}


def prepare_reference_sources(config):
    from final_eval.v3.train import resolve_model
    run = Path(config["output_dir"])
    v2 = Path(config["v2_model_dir"])
    paths = [v2 / "split.json"]
    paths += [p for p in (v2 / "tokenizer").iterdir() if p.is_file()]
    for fold in range(5):
        paths += [v2 / f"fold{fold}" / name for name in ("scorer.pt", "config.json", "category_prior.json")]
    hashes = {str(p.relative_to(v2)): file_hash(p) for p in paths}
    path = run / "reference_sources.json"
    if path.exists():
        source = read_json(path)
        if source["v2_files"] != hashes or source["requested_info"] != config["references"]:
            raise ValueError("Historical reference weights/tokenizer/settings changed")
    else:
        name, identity = resolve_model(config["references"]["info_model"], config["references"]["info_revision"])
        source = {"info": {"name": name, **identity}, "v2_files": hashes,
                  "requested_info": config["references"]}
        atomic_json(path, source)
    return source


def info_location(config):
    from final_eval.v3.train import download_model
    source = read_json(Path(config["output_dir"]) / "reference_sources.json")["info"]
    return download_model(source)


def smoke_dependencies(config):
    """Exercise real loading/forwards before the long training job, without paid calls."""
    import torch
    import spacy
    from transformers import AutoTokenizer
    from final_eval.v3.train import initialize, choose_device
    from final_eval.compressor.compress import InfoScorer, prepare_ours, load_scorer
    run = Path(config["output_dir"])
    device = choose_device(config)
    initialize(config)
    prepare_reference_sources(config)
    marker, state, done = stage_manifest(run, "model_smoke", {
        "initialization": file_hash(run / "initialization/source.json"),
        "references": file_hash(run / "reference_sources.json"), "teacher": config["teacher"],
        "packages": versions()})
    if done:
        return
    q = next(q for q in read_jsonl(run / "data/questions.jsonl") if q["role"] == "val")
    v2 = Path(config["v2_model_dir"])
    fold = reference_fold(q, read_json(v2 / "split.json"))
    nlp = spacy.load(config["spacy_model"])
    tok = AutoTokenizer.from_pretrained(v2 / "tokenizer", local_files_only=True)
    model = load_scorer(v2 / f"fold{fold}", device)
    info = InfoScorer(info_location(config), device)
    pp = prepare_ours(q["original_prompt"], model, tok, nlp, info, device)
    if not np.isfinite(pp["crit"]).all() or not np.isfinite(pp["info"]).all():
        raise ValueError("Nonfinite historical reference scores")
    del pp, model, info
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    status = {"base_encoder_loaded": True, "v2_and_signal_forward": True,
              "scope": "Model/library compatibility only; not an accuracy experiment"}
    if config["references"].get("llmlingua2", True):
        captures = []
        generate_teacher(config, [q], {}, captures.append, TokenCounter(config["tokenizer"]), device)
        failures = [r for r in captures if r["status"] == "compression_error"]
        if failures:
            raise RuntimeError(f"LLMLingua-2 compatibility smoke failed: {[r.get('error') for r in failures]}")
        status["teacher_forward"] = True
        status["teacher_budget_statuses"] = [r["status"] for r in captures]
    path = run / "model_smoke.json"
    atomic_json(path, status)
    finish_stage(marker, state, [path])


def atomic_ranges(text, nlp):
    """Same declared number/comparison/negation grouping as v2, no extra bonuses."""
    from final_eval.compressor.compress import extract_spans
    spans = extract_spans(text, nlp)
    result = [(s["span_start"], s["span_end"]) for s in spans
              if s["span_category"] in {"number", "comparison_conditional", "negation"}]
    for token in nlp(text):
        if token.dep_ == "neg":
            result.append((min(token.idx, token.head.idx),
                           max(token.idx + len(token.text), token.head.idx + len(token.head.text))))
    return result


def reference_fold(question, split):
    """A sibling outside the labelled UID set can still have family exposure."""
    if split.get("protocol") != "symbiote-v2" or not split.get("base_groups"):
        raise ValueError("A family-audited v2 split is required")
    preferred = split.get("folds", {}).get(question["uid"])
    choices = sorted(split["partitions"], key=int)
    if preferred is not None:
        choices = [str(preferred)] + [f for f in choices if f != str(preferred)]
    for fold in choices:
        membership = split["partitions"][fold]
        used = membership["train"] + membership["val"]
        groups = {split["base_groups"][u] for u in used}
        if question["base_group_id"] not in groups:
            return int(fold)
    raise ValueError(f"No v2 checkpoint is held out from family: {question['uid']}")


def select(text, priority, ratio, counter, policy, nlp):
    ranges = atomic_ranges(text, nlp) if policy == "atomic" else []
    result = select_to_budget(text, priority, ratio, counter, ranges)
    if result.tokens > math.floor(ratio * counter(text)):
        raise AssertionError("Final rendered text exceeds the common ceiling")
    return result


def record(question, method, ratio, text, status, counter, elapsed_ms, **extra):
    original = question["original_prompt"]
    orig, comp = counter(original), counter(text)
    budget = math.floor(ratio * orig)
    return {"uid": question["uid"], "base_group_id": question["base_group_id"],
        "source_dataset": question["source_dataset"], "original_prompt": original,
        "ground_truth_answer": question["ground_truth_answer"], "method": method,
        "target_ratio": ratio, "compressed_prompt": text, "status": status,
        "orig_tokens": orig, "comp_tokens": comp, "budget_tokens": budget,
        "actual_ratio": comp / orig, "budget_compliant": comp <= budget,
        "latency_ms": elapsed_ms, **extra}


def generate(config, role):
    import spacy
    import torch
    from transformers import AutoTokenizer
    from final_eval.v3.train import load_model, ARMS, choose_device
    from final_eval.v3.model import encode_question, predict_words
    run = Path(config["output_dir"])
    split = read_json(run / "data/split.json")
    chosen = read_json(run / "freeze.json") if role == "test" else None
    policies = [chosen["policy"]] if chosen else config["policies"]
    if chosen:
        from final_eval.inference.client import generation_config
        if digest(generation_config(config["reader"]["model"])) != chosen["reader"] or config["ratios"] != chosen["ratios"]:
            raise ValueError("Reader/ratios differ from the frozen development decision")
        if digest(split["downstream"]["test"]) != chosen["test_uids_hash"]:
            raise ValueError("Frozen test cohort changed")
        for arm in ARMS:
            if file_hash(run / "models" / arm / "best.pt") != chosen["checkpoints"][arm]:
                raise ValueError("Frozen checkpoint changed")
    all_questions = read_jsonl(run / "data/questions.jsonl")
    byuid = {q["uid"]: q for q in all_questions}
    questions = [byuid[u] for u in split["downstream"][role]]
    output = run / role
    v2 = Path(config["v2_model_dir"])
    prepare_reference_sources(config)
    marker, state, done = stage_manifest(output, "compress", {
        "questions": digest(questions), "ratios": config["ratios"], "policies": policies,
        "students": {arm: file_hash(run / "models" / arm / "best.pt") for arm in ARMS},
        "reference_split": file_hash(v2 / "split.json"), "references": config["references"],
        "reference_sources": file_hash(run / "reference_sources.json"),
        "packages": versions(),
        "teacher_settings": config["teacher"],
        "freeze": file_hash(run / "freeze.json") if chosen else None})
    if done:
        return
    torch.set_num_threads(config["training"].get("cpu_threads", 2))
    device = choose_device(config)
    counter = TokenCounter(config["tokenizer"])
    nlp = spacy.load(config["spacy_model"])
    tokenizer = AutoTokenizer.from_pretrained(run / "initialization/tokenizer", local_files_only=True)
    partial = output / "compressed.jsonl"
    existing = read_jsonl(partial) if partial.exists() else []
    records = {(r["uid"], r["method"], r["target_ratio"]): r for r in existing}

    def save(row):
        records[(row["uid"], row["method"], row["target_ratio"])] = row
        write_jsonl(partial, records.values())

    for arm in ARMS:
        model = load_model(config, run / "models" / arm / "best.pt", device).eval()
        for q in questions:
            if all((q["uid"], f"student_{arm}_{p}", r) in records for p in policies for r in config["ratios"]):
                continue
            started = time.perf_counter()
            item = encode_question(q, tokenizer, max_length=config["training"]["max_length"], stride=config["training"]["stride"])
            try:
                probability = predict_words(model, item, device)
                inference_ms = (time.perf_counter() - started) * 1000
                for policy in policies:
                    for ratio in config["ratios"]:
                        method = f"student_{arm}_{policy}"
                        if (q["uid"], method, ratio) in records:
                            continue
                        t = time.perf_counter()
                        selected = select(q["original_prompt"], probability, ratio, counter, policy, nlp)
                        save(record(q, method, ratio, selected.text, selected.status, counter,
                            inference_ms + (time.perf_counter() - t) * 1000,
                            kept_ranges=selected.kept_ranges,
                            timing_scope="tokenization+student windows+policy parser/selection; warm models; no loading",
                            scorer_forward_passes=len(item["windows"])))
            except ValueError as exc:
                for policy in policies:
                    for ratio in config["ratios"]:
                        save(record(q, f"student_{arm}_{policy}", ratio, "", "compression_error", counter,
                                    (time.perf_counter() - started) * 1000, error=str(exc)))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log("compress", role=role, completed_method=arm)
    generate_v2(config, questions, records, save, counter, nlp, device)
    if config["references"].get("llmlingua2", True):
        generate_teacher(config, questions, records, save, counter, device)
    expected = len(questions) * len(config["ratios"]) * (len(ARMS) * len(policies) + len(BASELINES))
    if not config["references"].get("llmlingua2", True):
        expected -= len(questions) * len(config["ratios"])
    if len(records) != expected:
        raise ValueError(f"Incomplete compression grid: {len(records)}/{expected}")
    finish_stage(marker, state, [partial], expected_conditions=expected,
        failures=sum(r["status"] != "ok" for r in records.values()))


def generate_v2(config, questions, records, save, counter, nlp, device):
    import torch
    from transformers import AutoTokenizer
    from final_eval.compressor.compress import load_scorer, InfoScorer, prepare_ours
    v2 = Path(config["v2_model_dir"])
    split = read_json(v2 / "split.json")
    todo = [q for q in questions if any((q["uid"], m, r) not in records for m in BASELINES[:2] for r in config["ratios"])]
    if not todo:
        return
    tokenizer = AutoTokenizer.from_pretrained(v2 / "tokenizer", local_files_only=True)
    info = InfoScorer(info_location(config), device)
    model = None
    for fold in sorted({reference_fold(q, split) for q in todo}):
        selected = [q for q in todo if reference_fold(q, split) == fold]
        model = load_scorer(v2 / f"fold{fold}", device)
        prior = read_json(v2 / f"fold{fold}/category_prior.json")
        for q in selected:
            for method in BASELINES[:2]:
                if all((q["uid"], method, r) in records for r in config["ratios"]):
                    continue
                started = time.perf_counter()
                pp = prepare_ours(q["original_prompt"], model if method == BASELINES[0] else None,
                                  tokenizer, nlp, info, device, prior=prior)
                prep_ms = (time.perf_counter() - started) * 1000
                priority = .5 * pp["crit"] + .5 * pp["info"] + pp["bonus"]
                for ratio in config["ratios"]:
                    if (q["uid"], method, ratio) in records:
                        continue
                    t = time.perf_counter()
                    result = select_to_budget(q["original_prompt"], priority, ratio, counter, pp["groups"])
                    save(record(q, method, ratio, result.text, result.status, counter,
                                prep_ms + (time.perf_counter() - t) * 1000, fold=fold,
                                kept_ranges=result.kept_ranges,
                                timing_scope="full warm compressor; parser+scorer/signal+selection, no loading"))
        del model
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del info
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def generate_teacher(config, questions, records, save, counter, device):
    from final_eval.compressor.compress import BaselineRunner
    from final_eval.v3.train import resolve_model, download_model
    run = Path(config["output_dir"])
    path = run / "teacher_reference_source.json"
    todo = [q for q in questions if any((q["uid"], BASELINES[2], r) not in records for r in config["ratios"])]
    if not todo:
        return
    if path.exists():
        source = read_json(path)
        if source["requested_teacher"] != {k: config["teacher"][k] for k in ("model", "revision")}:
            raise ValueError("Pinned teacher reference differs from requested settings")
    else:
        if config["teacher"]["mode"] == "local":
            source = read_json(run / "data/teacher_source.json")
        else:
            name, identity = resolve_model(config["teacher"]["model"], config["teacher"]["revision"])
            source = {"name": name, **identity}
        source["requested_teacher"] = {k: config["teacher"][k] for k in ("model", "revision")}
        atomic_json(path, source)
    location = download_model(source)
    runner = BaselineRunner(str(device), "unused", location)
    for q in todo:
        for ratio in config["ratios"]:
            if (q["uid"], BASELINES[2], ratio) in records:
                continue
            started = time.perf_counter()
            try:
                budget = math.floor(ratio * counter(q["original_prompt"]))
                text, status, details = runner.run_budgeted(BASELINES[2], q["original_prompt"], ratio, budget, counter)
                save(record(q, BASELINES[2], ratio, text, status, counter,
                    (time.perf_counter() - started) * 1000, **details,
                    timing_scope="full official-controls budget search; warm teacher except first query"))
            except Exception as exc:
                # Never replace a failing baseline with a truncated or handcrafted output.
                save(record(q, BASELINES[2], ratio, "", "compression_error", counter,
                    (time.perf_counter() - started) * 1000, error=type(exc).__name__))
    del runner
    gc.collect()
