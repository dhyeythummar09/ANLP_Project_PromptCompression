"""Dependency-free helpers for reproducible experiment artifacts."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import importlib.metadata
import math

PROTOCOL_VERSION = "symbiote-v2"

def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "final_eval").rglob("*.py")) + sorted((root / "final_eval").glob("requirements*.txt"))
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in files})

def package_versions():
    result = {}
    for name in ["torch", "transformers", "pandas", "numpy", "scikit-learn", "spacy", "tiktoken", "llmlingua", "selective-context"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def json_ready(value):
    """Undefined scientific metrics use JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Expected explicit boolean, got {value!r}")

def require_columns(frame, columns, name="table"):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def require_unique(frame, columns, name="table"):
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} has duplicate keys {columns}; resolve before joining")

def claim_output(path):
    if Path(path).exists():
        raise FileExistsError(f"Output exists: {path}. Choose a new output path.")

def resume_manifest(output, specification):
    path = Path(str(output) + ".manifest.json")
    specification = {"protocol": PROTOCOL_VERSION, **specification}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != specification:
            raise ValueError(f"Configuration/input mismatch for {output}; use a new output path")
    elif Path(output).exists():
        raise ValueError(f"Existing output has no v2 manifest: {output}; use a new output path")
    else:
        atomic_json(path, specification)
    return digest(specification)
