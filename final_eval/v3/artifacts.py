"""Small, explicit artifacts and process locks for independently resumable stages."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import time

from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.v3 import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def code_hash():
    """Only this package and the shared implementation actually imported by it."""
    paths = [p for p in (ROOT / "final_eval/v3").rglob("*.py")
             if "tests" not in p.relative_to(ROOT / "final_eval/v3").parts]
    for name in ["compressor/budget.py", "compressor/compress.py", "compressor/retention.py",
                 "compressor/evaluate.py", "inference/client.py", "jarvis_spend.py",
                 "scorer/data.py", "scorer/train_scorer.py", "spans/extract.py",
                 "preparation/schema.py", "utils/artifacts.py", "utils/answer_matching.py",
                 "utils/statistics.py"]:
        paths.append(ROOT / "final_eval" / name)
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(paths))})


def versions():
    result = {"python": platform.python_version()}
    for name in ["torch", "transformers", "tokenizers", "numpy", "pandas", "spacy",
                 "tiktoken", "llmlingua", "scikit-learn", "huggingface-hub",
                 "requests", "python-dotenv", "accelerate", "en-core-web-sm"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def stage_sources(name, directory):
    """A reporting-only fix must not invalidate already trained student weights."""
    core = ["v3/artifacts.py", "utils/artifacts.py"]
    training = core + ["v3/train.py", "v3/model.py"]
    preparation = core + ["v3/data.py", "preparation/schema.py", "scorer/data.py", "utils/answer_matching.py"]
    generation = training + ["v3/compression.py", "compressor/budget.py", "compressor/compress.py",
        "compressor/retention.py", "spans/extract.py", "scorer/train_scorer.py", "scorer/data.py", "settings.py"]
    reading = core + ["v3/reader.py", "inference/client.py", "jarvis_spend.py",
                      "compressor/evaluate.py", "utils/answer_matching.py"]
    scopes = {"initialize": training, "train": training, "prepare": preparation,
              "teacher": preparation + ["v3/train.py", "compressor/retention.py", "compressor/budget.py"],
              "compress": generation, "reader": reading, "qualification": reading,
              "model_smoke": generation, "benchmark": generation + ["v3/benchmark.py"]}
    paths = scopes.get(name)
    if paths is None:
        return {"all_v3_and_shared": code_hash()}
    return {p: file_hash(ROOT / "final_eval" / p) for p in sorted(set(paths))}


def stage_manifest(directory, name, specification):
    """Refuse mixed-protocol resume; a completed stage also verifies its outputs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".stage.json")
    spec = {"protocol": PROTOCOL, "stage_sources": stage_sources(name, directory), **specification}
    identity = digest(spec)
    if path.exists():
        previous = read_json(path)
        if previous["identity"] != identity:
            raise ValueError(f"{name}: inputs/code/settings changed. Use a new output_dir; do not mix runs.")
        if previous.get("complete"):
            for item, checksum in previous.get("outputs", {}).items():
                if not Path(item).is_file() or file_hash(item) != checksum:
                    raise ValueError(f"{name}: completed output missing/changed: {item}")
            return path, previous, True
    state = {"identity": identity, "specification": spec, "complete": False}
    atomic_json(path, state)
    return path, state, False


def finish_stage(path, state, outputs, **extra):
    state.update(complete=True, outputs={str(Path(p)): file_hash(p) for p in outputs}, **extra)
    atomic_json(path, state)


@contextmanager
def exclusive_lock(path):
    """OS releases the lock on crashes; presence of a lock file is not ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        # Reading an already-locked byte itself raises PermissionError on
        # Windows. Check file size without touching the protected byte first.
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process owns this run: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resolve_path(value, root):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(root) / p).resolve()


def load_config(path):
    config = read_json(path)
    root = resolve_path(config.get("project_root", "."), Path.cwd())
    config["project_root"] = str(root)
    for field in ["problems", "span_labels", "teacher_cache", "v2_model_dir", "output_dir"]:
        config[field] = str(resolve_path(config[field], root))
    config["exposure_files"] = [str(resolve_path(p, root)) for p in config.get("exposure_files", [])]
    for section, field in [("training", "encoder"), ("teacher", "model"), ("references", "info_model")]:
        candidate = resolve_path(config[section][field], root)
        if candidate.is_dir():
            config[section][field] = str(candidate)
    if not config["exposure_files"]:
        raise ValueError("exposure_files is required: freeze known prior downstream experiments first")
    if len(set(config["ratios"])) != len(config["ratios"]) or any(not 0 < r < 1 for r in config["ratios"]):
        raise ValueError("Distinct ratios strictly between0 and1 required")
    if set(config["policies"]) - {"word", "atomic"} or not config["policies"]:
        raise ValueError("Supported policies: word, atomic")
    if config["training"]["aux_weight"] <= 0 or config["training"].get("ranking_weight", 0) < 0:
        raise ValueError("Combined arm needs a positive auxiliary loss; ranking weight cannot be negative")
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("Positive epochs and batch_size required")
    if config["split"]["dev_families"] < 1 or config["split"]["test_families"] < 2:
        raise ValueError("Need development families and at least2 test families")
    if not 0 < config["split"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie between0 and1")
    if config["teacher"]["mode"] not in {"cached", "local"} or not 0 < config["teacher"]["ratio"] < 1:
        raise ValueError("Teacher mode must be cached/local with one valid retention ratio")
    if config["primary_ratio"] not in config["ratios"]:
        raise ValueError("primary_ratio must be one of the evaluated ratios")
    if config["training"]["save_every_steps"] < 1:
        raise ValueError("save_every_steps must be positive")
    from final_eval.inference.client import public_config
    public_config(config["reader"]["model"])
    for field in ("limit_inr", "guard_input_inr_per_million", "guard_output_inr_per_million"):
        if not math.isfinite(config["reader"][field]) or config["reader"][field] <= 0:
            raise ValueError(f"Positive finite reader {field} required")
    out = Path(config["output_dir"])
    protected = [root / "data", root / "models", root / "src", root / "results"]
    if out == root or any(out == p or p in out.parents for p in protected):
        raise ValueError("output_dir must be a new experiment directory, not historical data/models/results/source")
    return config


def log(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)
