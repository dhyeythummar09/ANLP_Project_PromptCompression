"""Matched initialization, two controlled objectives, and restartable optimizer steps."""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.v3.artifacts import (read_json, read_jsonl, stage_manifest, finish_stage, versions, log)
from final_eval.v3.model import DenseStudent, encode_question, collate_questions, question_loss, ranking_loss

ARMS = ("teacher_only", "teacher_cf")


def training_versions():
    values = versions()
    return {k: values[k] for k in ("python", "torch", "transformers", "tokenizers", "numpy", "huggingface-hub")}


def choose_device(config):
    value = config["training"]["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value)
    if config["training"].get("require_cuda", True) and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("This run requires CUDA; enable the GPU or explicitly configure a CPU smoke run")
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device is unavailable")
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resolve_model(name, revision):
    if Path(name).is_dir():
        files = {p.name: file_hash(p) for p in Path(name).iterdir() if p.is_file()}
        return name, {"local_files": files}
    from huggingface_hub import HfApi
    pin = HfApi().model_info(name, revision=revision).sha
    return name, {"revision": pin}


def download_model(source):
    """Download the pinned native checkpoint/tokenizer, not TF/ONNX duplicate weights."""
    if "revision" not in source:
        return source["name"]
    from huggingface_hub import HfApi, snapshot_download
    files = HfApi().list_repo_files(source["name"], revision=source["revision"])
    native = [p for p in files if "/" not in p]
    safe = [p for p in native if p.endswith(".safetensors")]
    weights = safe or [p for p in native if p.startswith("pytorch_model") and p.endswith(".bin")]
    if not weights:
        raise ValueError("Pinned model has no supported native weights")
    tokenizer = [p for p in native if p.endswith((".json", ".txt", ".model", ".bpe"))]
    return snapshot_download(source["name"], revision=source["revision"], allow_patterns=weights + tokenizer)


def initialize(config):
    cfg = config["training"]
    out = Path(config["output_dir"]) / "initialization"
    spec = {"encoder": cfg["encoder"], "revision": cfg["revision"], "seed": cfg["seed"],
            "dropout": cfg["dropout"], "packages": training_versions()}
    marker, state, done = stage_manifest(out, "initialize", spec)
    if done:
        return
    seed_all(cfg["seed"])
    name, source = resolve_model(cfg["encoder"], cfg["revision"])
    kwargs = {"revision": source["revision"]} if "revision" in source else {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)
    if not tokenizer.is_fast:
        raise ValueError("Offset-safe training requires a fast tokenizer")
    encoder = AutoModel.from_pretrained(name, **kwargs)
    if encoder.config.model_type != "distilbert" or tokenizer.pad_token_id is None:
        raise ValueError("The controlled v3 architecture is DistilBERT with an explicit PAD token")
    if cfg["max_length"] > encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds the selected encoder context")
    model = DenseStudent(encoder, cfg["dropout"])
    tokenizer.save_pretrained(out / "tokenizer")
    model.encoder.config.save_pretrained(out / "encoder_config")
    torch_save(out / "common_initial.pt", model.state_dict())
    atomic_json(out / "source.json", {"name": name, **source, "seed": cfg["seed"],
        "weights_sha256": file_hash(out / "common_initial.pt"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": "Both arms start from these identical encoder AND keep-head weights; no v2 warm start."})
    files = [out / "common_initial.pt", out / "source.json"]
    files += [p for directory in ("tokenizer", "encoder_config") for p in (out / directory).iterdir() if p.is_file()]
    finish_stage(marker, state, files)
    log("initialize", source=source, parameters=sum(p.numel() for p in model.parameters()))
    del model, encoder
    gc.collect()


def load_model(config, checkpoint, device):
    init = Path(config["output_dir"]) / "initialization"
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(init / "encoder_config", local_files_only=True))
    model = DenseStudent(encoder, config["training"]["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device)


def training_items(config, tokenizer):
    data = Path(config["output_dir"]) / "data"
    labels = {r["uid"]: r for r in read_jsonl(data / "teacher.jsonl")}
    roles = {"train": [], "val": []}
    audit = {"ignored_no_teacher_words": [], "uncovered_words": 0, "uncovered_spans": 0}
    for q in read_jsonl(data / "questions.jsonl"):
        if q["role"] not in roles:
            continue
        row = labels[q["uid"]]
        if row["text_hash"] != digest(q["original_prompt"]):
            raise ValueError("Teacher/question text changed")
        item = encode_question(q, tokenizer, row["labels"], config["training"]["max_length"], config["training"]["stride"])
        if not any(y >= 0 for w in item["windows"] for y in w["teacher"]):
            audit["ignored_no_teacher_words"].append(q["uid"])
            continue
        audit["uncovered_words"] += len(item["uncovered_words"])
        audit["uncovered_spans"] += len(item["uncovered_spans"])
        roles[q["role"]].append(item)
    if not roles["train"] or not roles["val"]:
        raise ValueError("No usable teacher train/validation questions")
    if not any(w["span_targets"] for x in roles["train"] for w in x["windows"]):
        raise ValueError("Combined arm has no eligible span labels")
    return roles, audit


def epoch_order(items, seed, balanced=False):
    rng = random.Random(seed)
    if balanced:
        counts = {s: sum(x["source"] == s for x in items) for s in {x["source"] for x in items}}
        return rng.choices(range(len(items)), weights=[1 / counts[x["source"]] for x in items], k=len(items))
    indices = list(range(len(items)))
    rng.shuffle(indices)
    return indices


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def losses(model, batch, aux_weight, rank_weight):
    keep, span = model(*(batch[k] for k in ("input_ids", "attention_mask", "word_masks", "span_masks")))
    teacher = question_loss(keep, batch["teacher"], batch["teacher_valid"], batch["question_index"], batch["n_questions"])
    criticality = question_loss(span, batch["span_targets"], batch["span_valid"], batch["question_index"], batch["n_questions"])
    rank = ranking_loss(span, batch["span_targets"], batch["span_valid"], batch["question_index"]) if rank_weight else span.sum() * 0
    return teacher + aux_weight * criticality + rank_weight * rank, teacher, criticality


@torch.inference_mode()
def validate(model, items, batch_size, pad_id, device):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(items), batch_size):
        part = items[i:i + batch_size]
        batch = to_device(collate_questions(part, pad_id), device)
        _, teacher, _ = losses(model, batch, 0, 0)
        total += teacher.item() * len(part)
        n += len(part)
    return total / n


def train_arm(config, arm, items, tokenizer, device):
    cfg = config["training"]
    run = Path(config["output_dir"])
    out = run / "models" / arm
    initial = run / "initialization/common_initial.pt"
    aux = cfg["aux_weight"] if arm == "teacher_cf" else 0.0
    rank = cfg.get("ranking_weight", 0.0) if arm == "teacher_cf" else 0.0
    marker, state, done = stage_manifest(out, "train", {"settings": cfg, "arm": arm,
        "initial_weights": file_hash(initial), "teacher": file_hash(run / "data/teacher.jsonl"),
        "questions": file_hash(run / "data/questions.jsonl"), "packages": training_versions()})
    if done:
        return
    seed_all(cfg["seed"])
    model = load_model(config, initial, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    batches = math.ceil(len(items["train"]) / cfg["batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(batches * cfg["epochs"] * cfg["warmup_fraction"]), batches * cfg["epochs"])
    amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    progress = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_val": float("inf"), "best_epoch": -1, "history": []}
    checkpoint = out / "last.pt"
    if checkpoint.exists():
        # Only our own locally-created optimizer/RNG checkpoint is loaded here.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != state["identity"]:
            raise ValueError("Optimizer checkpoint is from another training run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        progress = saved["progress"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def save():
        torch_save(checkpoint, {"identity": state["identity"], "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "progress": progress, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(progress["epoch"], cfg["epochs"]):
        order = epoch_order(items["train"], cfg["seed"] + epoch, cfg.get("task_balanced", False))
        start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
        model.train()
        for b in range(start_batch, batches):
            part = [items["train"][j] for j in order[b * cfg["batch_size"]:(b + 1) * cfg["batch_size"]]]
            batch = to_device(collate_questions(part, tokenizer.pad_token_id), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, teacher, span = losses(model, batch, aux, rank)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {arm}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.update(epoch=epoch, next_batch=b + 1, global_step=progress["global_step"] + 1)
            if progress["global_step"] % cfg["save_every_steps"] == 0:
                save()
            if b == 0 or (b + 1) % 20 == 0:
                log("train", arm=arm, epoch=epoch + 1, batch=b + 1, batches=batches,
                    loss=float(loss.detach()), teacher_loss=float(teacher.detach()), span_loss=float(span.detach()))
        val = validate(model, items["val"], cfg["batch_size"], tokenizer.pad_token_id, device)
        if val < progress["best_val"]:
            progress.update(best_val=val, best_epoch=epoch + 1)
            torch_save(out / "best.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
        progress["history"].append({"epoch": epoch + 1, "teacher_validation_bce": val})
        progress.update(epoch=epoch + 1, next_batch=0)
        save()
        log("validation", arm=arm, epoch=epoch + 1, teacher_validation_bce=val)
    metadata = {"arm": arm, "aux_weight": aux, "ranking_weight": rank,
        "initial_sha256": file_hash(initial), "best_weights_sha256": file_hash(out / "best.pt"),
        "checkpoint_selection": "minimum teacher BCE on validation families, same criterion for both arms",
        "best_epoch": progress["best_epoch"], "validation_bce": progress["best_val"],
        "history": progress["history"], "seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "training_questions": len(items["train"]), "validation_questions": len(items["val"]),
        "optimization_scope": "training families only; checkpoint selection uses teacher validation loss, not test answers"}
    atomic_json(out / "metadata.json", metadata)
    finish_stage(marker, state, [out / "best.pt", out / "metadata.json"], best_epoch=progress["best_epoch"])
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train(config):
    device = choose_device(config)
    initialize(config)
    cfg = config["training"]
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    tokenizer = AutoTokenizer.from_pretrained(Path(config["output_dir"]) / "initialization/tokenizer", local_files_only=True)
    items, audit = training_items(config, tokenizer)
    atomic_json(Path(config["output_dir"]) / "data/tokenization_audit.json", audit)
    for arm in ARMS:
        train_arm(config, arm, items, tokenizer, device)
