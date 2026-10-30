"""Staged execution with a predeclared test transition; no manual retuning loop."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

from final_eval.utils.artifacts import atomic_json, file_hash
from final_eval.v3.artifacts import (ROOT, versions, read_json, read_jsonl, exclusive_lock,
                                    code_hash, log)

STAGES = ("prepare", "teacher", "model-smoke", "train", "compress-dev", "evaluate-dev", "freeze",
          "compress-test", "evaluate-test", "report", "benchmark", "bundle")


def preflight(config, offline=False, through="bundle"):
    """No weight download or API call; --offline also skips GPU/optional-library gates."""
    needed = [config["problems"], config["span_labels"], *config["exposure_files"],
              str(Path(config["v2_model_dir"]) / "split.json")]
    if config["teacher"]["mode"] == "cached":
        needed.append(config["teacher_cache"])
    missing = [p for p in needed if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    import torch
    if not offline and STAGES.index(through) >= STAGES.index("train"):
        if config["training"].get("require_cuda", True) and (not torch.cuda.is_available() or config["training"]["device"] == "cpu"):
            raise RuntimeError("GPU required for the full run. Enable Kaggle GPU or explicitly set require_cuda=false for a CPU smoke test.")
    needs_teacher = config["teacher"]["mode"] == "local" or config["references"].get("llmlingua2", True)
    if not offline and needs_teacher and importlib.util.find_spec("llmlingua") is None:
        raise RuntimeError("Install final_eval/v3/requirements.txt before the experiment; LLMLingua-2 is a required reference")
    if STAGES.index(through) >= STAGES.index("evaluate-dev") and not config["reader"].get("enabled", True):
        raise RuntimeError("Reader is disabled; use --through compress-dev for a GPU-only run, or enable the configured reader")
    if config["tokenizer"] == "regex-test-only" and not config.get("test_fixture", False):
        raise ValueError("Scientific runs require a real tokenizer")
    result = {"passed": True, "offline": offline, "cuda_available": torch.cuda.is_available(),
              "packages": versions(), "source": code_hash(), "through": through,
              "note": "No model weights loaded and no API requests made by preflight."}
    log("preflight", **result)
    return result


def run(config, through="bundle", retry_failed=False):
    from final_eval.v3 import data, train, compression, reader, report
    run_dir = Path(config["output_dir"])
    with exclusive_lock(run_dir / "pipeline.lock"):
        preflight(config, through=through)
        resolved = run_dir / "resolved_config.json"
        if resolved.exists() and read_json(resolved) != config:
            raise ValueError("Run configuration changed; use a new output_dir and preserve the previous exposure/ledger")
        atomic_json(resolved, config)
        paid = STAGES.index(through) >= STAGES.index("evaluate-dev")
        try:
            for name in STAGES[:STAGES.index(through) + 1]:
                log("pipeline", entering=name)
                if name == "prepare":
                    data.prepare(config)
                    # Qualify BEFORE lengthy GPU work for an unattended full run.
                    if paid:
                        reader.qualify(config, retry_failed)
                elif name == "teacher":
                    (data.cached_teacher_labels if config["teacher"]["mode"] == "cached" else data.local_teacher_labels)(config)
                elif name == "train":
                    train.train(config)
                elif name == "model-smoke":
                    compression.smoke_dependencies(config)
                elif name.startswith("compress-"):
                    compression.generate(config, name.split("-")[1])
                elif name.startswith("evaluate-"):
                    reader.evaluate(config, name.split("-")[1], retry_failed)
                elif name == "freeze":
                    report.freeze(config)
                elif name == "report":
                    report.summarize(config, "dev")
                    report.summarize(config, "test")
                elif name == "benchmark":
                    if config.get("benchmark", {}).get("enabled", True):
                        from final_eval.v3.benchmark import benchmark
                        benchmark(config)
                elif name == "bundle":
                    failure = run_dir / "failure.json"
                    if failure.exists():
                        previous = read_json(failure)
                        previous["resolved_by_successful_resume"] = True
                        atomic_json(failure, previous)
                    bundle(config)
        except (Exception, KeyboardInterrupt) as exc:
            message = str(exc)
            for variable, value in os.environ.items():
                if value and len(value) >= 8 and (variable.endswith("API_KEY") or variable in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}):
                    message = message.replace(value, "<redacted>")
            atomic_json(run_dir / "failure.json", {"stage": name, "error_type": type(exc).__name__,
                "message": message[:3000], "resolved_by_successful_resume": False,
                "resume": "Preserve this run, its cache and ledger. Resume the same settings; use --retry-failed only after reviewing failed API calls."})
            try:
                bundle(config)
            except Exception as export_error:
                log("partial_export_failed", error_type=type(export_error).__name__)
            raise
        log("pipeline", finished_through=through, output_dir=str(run_dir))


def bundle(config):
    run = Path(config["output_dir"])
    destination = run.parent / (run.name + "_results.zip")
    files = []
    excluded = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.suffix in {".lock", ".tmp", ".env"} or path.name == "bundle.json":
            continue
        relative = path.relative_to(run).as_posix()
        if path.name == "last.pt" or relative == "initialization/common_initial.pt":
            excluded.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash(path)})
            continue
        files.append((relative, path))
    manifest = {"source": code_hash(), "files": [{"path": n, "bytes": p.stat().st_size, "sha256": file_hash(p)} for n, p in files],
                "excluded_large_resume_files": excluded,
                "note": "Best model weights included; optimizer and common initialization remain in the run directory for resume. API credentials are not included."}
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in files:
            archive.write(path, "run/" + name)
        for path in sorted((ROOT / "final_eval").rglob("*")):
            if "tests" in path.relative_to(ROOT).parts and "v3" not in path.relative_to(ROOT).parts:
                continue
            if path.is_file() and (path.suffix == ".py" or
                    ("v3" in path.relative_to(ROOT).parts and path.suffix in {".json", ".md", ".txt"}) or
                    path.name.startswith("requirements")):
                archive.write(path, "implementation/" + path.relative_to(ROOT).as_posix())
        archive.write(ROOT / "requirements.txt", "implementation/requirements.txt")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    os.replace(temporary, destination)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError("Result archive failed its CRC check")
    atomic_json(run / "bundle.json", {"path": str(destination), "sha256": file_hash(destination),
        "bytes": destination.stat().st_size})
    log("bundle", path=str(destination), bytes=destination.stat().st_size)
    return destination
