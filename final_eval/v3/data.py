"""Family partitions, strict source validation, and auditable single-budget labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import zipfile

from final_eval.compressor.budget import words
from final_eval.compressor.retention import alignment_bounds
from final_eval.preparation.schema import prepare_problems
from final_eval.scorer.data import build_span_table
from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.v3.artifacts import (read_csv, read_json, read_jsonl, write_jsonl,
                                    stage_manifest, finish_stage, log)


def load_problems(path):
    import pandas as pd
    frame = prepare_problems(pd.read_csv(path, keep_default_na=False, dtype=str))
    return frame.to_dict("records")


def select_families(questions, n, seed):
    """Source round-robin, one representative per family, without answer outcomes."""
    rng = random.Random(seed)
    queues = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["uid"]):
        queues[q["source_dataset"]].append(q)
    for queue in queues.values():
        rng.shuffle(queue)
    selected, families = [], set()
    while len(selected) < n and any(queues.values()):
        for name in sorted(queues):
            while queues[name] and queues[name][-1]["base_group_id"] in families:
                queues[name].pop()
            if queues[name] and len(selected) < n:
                q = queues[name].pop()
                selected.append(q)
                families.add(q["base_group_id"])
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} independent eligible families; requested {n}")
    return selected


def exposure_uids(path):
    """Only recorded outcomes/cohorts, not every proposed span input, are exposures."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("experiment/split_manifest.json"))
        return {q["uid"] for role in ("dev", "sealed_test") for q in manifest[role]}
    if path.suffix == ".csv":
        rows = read_csv(path)
        if rows and not {"uid", "example_id"}.intersection(rows[0]):
            raise ValueError(f"Exposure file has no question identity: {path}")
        return {r.get("uid") or r["source_dataset"] + "|" + r["example_id"] for r in rows}
    obj = read_json(path)
    if "exposed_uids" in obj:
        return set(obj["exposed_uids"])
    if "pilot_uids" in obj:
        return set(obj["pilot_uids"])
    raise ValueError(f"Unsupported exposure JSON schema: {path}")


def snapshot_exposures(problems_path, paths, output):
    questions = {q["uid"]: q for q in load_problems(problems_path)}
    exposed = set()
    for path in paths:
        exposed.update(exposure_uids(path))
    unknown = exposed - questions.keys()
    if unknown:
        raise ValueError(f"Unknown exposure UIDs: {sorted(unknown)[:5]}")
    atomic_json(output, {"exposed_uids": sorted(exposed),
        "exposed_families": sorted({questions[u]["base_group_id"] for u in exposed}),
        "problems_sha256": file_hash(problems_path),
        "sources": {str(p): file_hash(p) for p in paths},
        "note": "Append future experiment cohorts before preparing a new run. No outcomes used in selection."})


def split_questions(questions, exposed_uids, settings):
    byuid = {q["uid"]: q for q in questions}
    if not exposed_uids.issubset(byuid):
        raise ValueError("Exposure IDs are not all in this dataset version")
    exposed_groups = {byuid[u]["base_group_id"] for u in exposed_uids}
    fresh = [q for q in questions if q["base_group_id"] not in exposed_groups]
    test_representatives = select_families(fresh, settings["test_families"], settings["seed"])
    test_groups = {q["base_group_id"] for q in test_representatives}
    rest = [q for q in questions if q["base_group_id"] not in test_groups]
    # Development can reuse previously exposed families; it is never reported as fresh.
    dev_representatives = select_families(rest, settings["dev_families"], settings["seed"] + 1)
    dev_groups = {q["base_group_id"] for q in dev_representatives}
    rest = [q for q in rest if q["base_group_id"] not in dev_groups]
    n_val = max(2, round(len({q["base_group_id"] for q in rest}) * settings["validation_fraction"]))
    val_representatives = select_families(rest, n_val, settings["seed"] + 2)
    val_groups = {q["base_group_id"] for q in val_representatives}
    train = [q for q in rest if q["base_group_id"] not in val_groups]
    if not train:
        raise ValueError("No training questions remain")
    role_groups = {"test": test_groups, "dev": dev_groups, "val": val_groups,
                   "train": {q["base_group_id"] for q in train}}
    roles = {q["uid"]: next(role for role, groups in role_groups.items() if q["base_group_id"] in groups)
             for q in questions}
    assert len(roles) == len(questions)
    assert not test_groups & exposed_groups
    return {"roles": roles, "families": {k: sorted(v) for k, v in role_groups.items()},
            "downstream": {"dev": [q["uid"] for q in dev_representatives],
                           "test": [q["uid"] for q in test_representatives]},
            "known_exposed_families": sorted(exposed_groups), "settings": settings,
            "fresh_pool_families_before_split": len({q["base_group_id"] for q in fresh})}


def teacher_word_labels(original, compressed):
    """Known keep/drop labels only; ambiguous occurrences and word fragments are masked."""
    lower, upper, status = alignment_bounds(original, compressed)
    units = words(original)
    if status == "unresolved":
        return [-1] * len(units), status
    lo, hi = set(map(tuple, lower)), set(map(tuple, upper))
    targets = []
    for s, e, text in units:
        pieces = [(s + m.start(), s + m.end(), m.group())
                  for m in re.finditer(r"\w+|[^\w\s]", text)]
        meaningful = [(a, b) for a, b, t in pieces if any(c.isalnum() for c in t)]
        meaningful = meaningful or [(a, b) for a, b, _ in pieces]
        if meaningful and all(p in lo for p in meaningful):
            targets.append(1)
        elif meaningful and all(p not in hi for p in meaningful):
            targets.append(0)
        else:
            targets.append(-1)
    return targets, status


def prepare(config):
    out = Path(config["output_dir"]) / "data"
    # A new run automatically accounts for completed/interrupted v3 tests stored
    # alongside it. Its OWN exposure marker must not change its resume identity.
    current = Path(config["output_dir"]).resolve()
    parent = current.parent
    discovered = []
    for pattern in ("*/test_exposure.json", "*/dev_exposure.json", "*/reader/pilot_exposure.json", "*/pairs/exposure.json"):
        discovered.extend(parent.glob(pattern))
    prior_tests = sorted(p for p in discovered if not p.resolve().is_relative_to(current))
    # Imported completed runs need not live alongside output_dir. Their small
    # published exposure ledger must still prevent later reuse as a fresh test.
    problem_sha = file_hash(config["problems"])
    published = [p for p in (Path(config["project_root"]) / "results").rglob("exposure.json")
                 if read_json(p).get("problems_sha256") == problem_sha]
    exposure_files = sorted(set(config["exposure_files"] + [str(p) for p in prior_tests + published]))
    for p in exposure_files:
        if Path(p).suffix == ".json":
            recorded = read_json(p).get("problems_sha256")
            if recorded and recorded != file_hash(config["problems"]):
                raise ValueError("Exposure snapshot belongs to a different problem-table version")
    spec = {"problems": file_hash(config["problems"]), "spans": file_hash(config["span_labels"]),
            "exposures": {p: file_hash(p) for p in exposure_files}, "split": config["split"]}
    marker, state, done = stage_manifest(out, "prepare", spec)
    if done:
        return
    questions = load_problems(config["problems"])
    exposed = set().union(*(exposure_uids(p) for p in exposure_files))
    split = split_questions(questions, exposed, config["split"])
    import pandas as pd
    spans = build_span_table(pd.read_csv(config["span_labels"], keep_default_na=False))
    byuid = {q["uid"]: q for q in questions}
    peruid = defaultdict(list)
    for r in spans.to_dict("records"):
        if r["uid"] not in byuid:
            raise ValueError(f"Label UID absent from problems: {r['uid']}")
        q = byuid[r["uid"]]
        a, b = int(r["span_start"]), int(r["span_end"])
        if r["original_prompt"] != q["original_prompt"] or q["original_prompt"][a:b] != r["span_text"]:
            raise ValueError(f"Label text/offset mismatch: {r['uid']}")
        if not 0 <= a < b <= len(q["original_prompt"]):
            raise ValueError("Out-of-range span")
        if r.get("base_group_id") and r["base_group_id"] != q["base_group_id"]:
            raise ValueError("Span/problem family conflict")
        peruid[r["uid"]].append({"start": a, "end": b, "target": float(r["target"]),
                                 "category": r["span_category"]})
    records = [{**q, "role": split["roles"][q["uid"]], "spans": peruid[q["uid"]]} for q in questions]
    write_jsonl(out / "questions.jsonl", records)
    atomic_json(out / "split.json", split)
    summary = {role: {"questions": sum(q["role"] == role for q in records),
                      "families": len(split["families"][role])} for role in split["families"]}
    atomic_json(out / "summary.json", summary)
    finish_stage(marker, state, [out / "questions.jsonl", out / "split.json", out / "summary.json"])
    log("prepare", partitions=summary, fresh_families=split["fresh_pool_families_before_split"])


def cached_teacher_labels(config):
    out = Path(config["output_dir"]) / "data"
    cfg = config["teacher"]
    marker, state, done = stage_manifest(out, "teacher", {
        "questions": file_hash(out / "questions.jsonl"), "cache": file_hash(config["teacher_cache"]),
        "teacher": cfg})
    if done:
        return
    questions = read_jsonl(out / "questions.jsonl")
    needed = {q["uid"]: q for q in questions if q["role"] in {"train", "val"}}
    teacher = {}
    for r in read_csv(config["teacher_cache"]):
        if r.get("method") != cfg["cache_method"] or float(r["target_ratio"]) != cfg["ratio"]:
            continue
        uid = r["uid"]
        if uid not in needed:
            continue
        if r["original_prompt"] != needed[uid]["original_prompt"]:
            raise ValueError(f"Teacher cache text mismatch: {uid}")
        if uid in teacher and teacher[uid] != r["compressed_prompt"]:
            raise ValueError(f"Conflicting teacher outputs: {uid}")
        teacher[uid] = r["compressed_prompt"]
    if set(needed) - teacher.keys():
        raise ValueError("Missing cached teacher questions; regenerate teacher labels or supply a complete cache")
    rows, counts = [], Counter()
    for uid, q in needed.items():
        labels, status = teacher_word_labels(q["original_prompt"], teacher[uid])
        counts[status] += 1
        counts["valid_words"] += sum(x >= 0 for x in labels)
        counts["masked_words"] += sum(x < 0 for x in labels)
        rows.append({"uid": uid, "role": q["role"], "text_hash": digest(q["original_prompt"]),
                     "labels": labels, "alignment_status": status})
    write_jsonl(out / "teacher.jsonl", rows)
    atomic_json(out / "teacher_audit.json", {"counts": dict(counts), "training_budget": cfg["ratio"],
        "label_scope": "train/val only; no development/test teacher targets in optimization",
        "provenance": "Legacy frozen compression outputs; original teacher revision not established",
        "cache_sha256": file_hash(config["teacher_cache"])})
    finish_stage(marker, state, [out / "teacher.jsonl", out / "teacher_audit.json"])
    log("teacher", **dict(counts))


def local_teacher_labels(config):
    """Optional pinned local teacher; checkpoint each question, never bill a reader."""
    from llmlingua import PromptCompressor
    import torch
    from final_eval.v3.train import resolve_model, download_model
    out = Path(config["output_dir"]) / "data"
    cfg = config["teacher"]
    marker, state, done = stage_manifest(out, "teacher", {
        "questions": file_hash(out / "questions.jsonl"), "teacher": cfg})
    if done:
        return
    source_path = out / "teacher_source.json"
    if source_path.exists():
        source = read_json(source_path)
    else:
        name, identity = resolve_model(cfg["model"], cfg["revision"])
        source = {"name": name, **identity}
        atomic_json(source_path, source)
    location = download_model(source)
    device = config["training"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    teacher = PromptCompressor(model_name=location, use_llmlingua2=True, device_map=device)
    rows, counts = [], Counter()
    for q in read_jsonl(out / "questions.jsonl"):
        if q["role"] not in {"train", "val"}:
            continue
        cache = out / "teacher_raw" / (digest(q["uid"]) + ".json")
        if cache.exists():
            result = read_json(cache)
            if result["text_hash"] != digest(q["original_prompt"]):
                raise ValueError("Teacher cache text changed")
        else:
            answer = teacher.compress_prompt_llmlingua2(q["original_prompt"], rate=cfg["ratio"], return_word_label=True)
            result = {"text_hash": digest(q["original_prompt"]), "output": answer}
            atomic_json(cache, result)
        labels, status = teacher_word_labels(q["original_prompt"], result["output"]["compressed_prompt"])
        # Explicit teacher labels are retained for audit; alignment never guesses
        # an occurrence from a repeated token or assumes teacher whitespace identity.
        rows.append({"uid": q["uid"], "role": q["role"], "text_hash": digest(q["original_prompt"]),
                     "labels": labels, "alignment_status": status})
        counts[status] += 1
        counts["valid_words"] += sum(y >= 0 for y in labels)
        counts["masked_words"] += sum(y < 0 for y in labels)
        if len(rows) % 25 == 0:
            log("teacher_local", questions=len(rows))
    write_jsonl(out / "teacher.jsonl", rows)
    atomic_json(out / "teacher_audit.json", {"counts": dict(counts), "source": source,
        "training_budget": cfg["ratio"], "label_scope": "train/val only; bounded source-occurrence alignment"})
    finish_stage(marker, state, [out / "teacher.jsonl", out / "teacher_audit.json", source_path])
