"""One serial, process-locked cache and spending ledger across all v3 stages."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path

from final_eval.compressor.evaluate import PROMPT_TEMPLATE
from final_eval.inference.client import HostedClient, generation_config, public_config
from final_eval.jarvis_spend import TokenCostGuard
from final_eval.utils.answer_matching import grade_answer, GRADER_VERSION
from final_eval.utils.artifacts import atomic_json, digest, file_hash
from final_eval.v3.artifacts import (read_json, read_jsonl, write_jsonl, exclusive_lock,
                                    stage_manifest, finish_stage, log)


class ReaderStopped(RuntimeError):
    """Outputs/ledger are retained; do not advance to another experimental stage."""


class ValidatedClient:
    """Malformed usage cannot be mistaken for a cheap, fully accounted request."""
    def __init__(self, client):
        self.client = client

    @property
    def attempts(self):
        return self.client.attempts

    def generate(self, prompt):
        result = self.client.generate(prompt)
        usage = result.usage or {}
        if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
            result.usage = None
        return result


class Reader:
    def __init__(self, config, client=None):
        self.config = config
        self.settings = config["reader"]
        self.cfg = public_config(self.settings["model"])
        if self.cfg.get("backend", "hosted") != "hosted" or self.cfg.get("provider") == "google":
            raise ValueError("V3's INR accounting requires the explicit OpenAI-compatible hosted reader")
        if self.cfg.get("max_tokens", 0) < 1 or self.cfg.get("max_attempts", 1) < 1:
            raise ValueError("Explicit positive reader output/attempt limits are required")
        self.root = Path(config["output_dir"]) / "reader"
        self.root.mkdir(parents=True, exist_ok=True)
        self.client_override = client
        self.client = None
        self.identity = digest({"generation": generation_config(self.cfg), "wrapper": PROMPT_TEMPLATE})

    def _initialize(self):
        if self.client is not None:
            return
        if self.client_override is None:
            variable = self.cfg.get("api_key_env", "")
            path = self.settings.get("key_file")
            if variable and not os.environ.get(variable) and path:
                from dotenv import dotenv_values
                p = Path(path)
                p = p if p.is_absolute() else Path(self.config["project_root"]) / p
                value = dotenv_values(p, interpolate=False).get(variable, "") if p.is_file() else ""
                if value:
                    os.environ[variable] = value
            if variable and not os.environ.get(variable):
                raise ReaderStopped(f"Missing {variable}; set the environment/Kaggle secret before API stages")
            base = HostedClient(self.cfg, usage_path=self.root / "attempts.json")
        else:
            base = self.client_override
        self.client = TokenCostGuard(ValidatedClient(base), self.cfg, self.root / "cost.json",
            self.settings["limit_inr"], self.settings["guard_input_inr_per_million"],
            self.settings["guard_output_inr_per_million"])

    def ask(self, text, replicate=None, retry_failed=False):
        query = PROMPT_TEMPLATE.format(question=text)
        key = digest({"reader": self.identity, "query": query, "replicate": replicate})
        path = self.root / "cache" / (key + ".json")
        if path.exists():
            cached = read_json(path)
            if cached["generation"]["status"] == "ok" or not retry_failed:
                return cached
        self._initialize()
        answer = self.client.generate(query)
        if answer.status == "ok" and answer.served_model not in self.settings["allowed_served_models"]:
            answer.status = "served_model_mismatch"
            answer.error = "Served reader identity differs from the frozen allowed list"
        result = {"query_hash": key, "reader_identity": self.identity,
                  "generation": answer.to_dict(), "replicate": replicate}
        atomic_json(path, result)
        return result

    def stop_on(self, result):
        status = result["generation"]["status"]
        if status not in {"ok", "truncated", "empty_response"}:
            raise ReaderStopped(f"Reader stopped: {status}. Cache, responses and spending ledger are retained.")


@contextmanager
def open_reader(config, client=None):
    with exclusive_lock(Path(config["output_dir"]) / "reader/active.lock"):
        yield Reader(config, client)


def qualify(config, retry_failed=False):
    run = Path(config["output_dir"])
    questions = [q for q in read_jsonl(run / "data/questions.jsonl") if q["role"] == "val"]
    from final_eval.v3.data import select_families
    chosen = select_families(questions, config["reader"]["pilot_questions"], config["split"]["seed"] + 10)
    marker, state, done = stage_manifest(run / "reader", "qualification", {
        "reader": config["reader"], "questions": digest(chosen), "grader": GRADER_VERSION})
    if done:
        return
    records = []
    atomic_json(run / "reader/pilot_exposure.json", {
        "exposed_uids": [q["uid"] for q in chosen],
        "exposed_families": sorted({q["base_group_id"] for q in chosen})})
    with open_reader(config) as reader:
        for q in chosen:
            answer = reader.ask(q["original_prompt"], retry_failed=retry_failed)
            reader.stop_on(answer)
            generation = answer["generation"]
            grade = grade_answer(generation["text"], q["ground_truth_answer"], q["original_prompt"])
            records.append({"uid": q["uid"], "family": q["base_group_id"], "status": generation["status"],
                            "correct": bool(grade.correct), "parse_status": grade.status,
                            "query_hash": answer["query_hash"]})
    passed = all(r["status"] == "ok" and r["parse_status"] == "ok" for r in records)
    passed = passed and sum(r["correct"] for r in records) >= math.ceil(len(records) * config["reader"]["pilot_min_accuracy"])
    path = run / "reader/qualification.json"
    atomic_json(path, {"passed": passed, "records": records, "reader": config["reader"]["model"],
        "scope": "reader access/parser/output check on validation originals; not a compression-quality test"})
    if not passed:
        raise ReaderStopped("Reader qualification failed; inspect qualification.json before paying for the main grid")
    finish_stage(marker, state, [path, run / "reader/pilot_exposure.json"])


def evaluation_tasks(compressed):
    originals = {}
    for row in compressed:
        uid = row["uid"]
        signature = [row[k] for k in ("original_prompt", "ground_truth_answer", "base_group_id")]
        if uid in originals and [originals[uid][k] for k in ("original_prompt", "ground_truth_answer", "base_group_id")] != signature:
            raise ValueError("Conflicting original metadata in the compression grid")
        originals[uid] = {**row, "method": "original", "target_ratio": 1.0, "actual_ratio": 1.0,
                          "compressed_prompt": row["original_prompt"], "status": "ok", "budget_compliant": True}
    return list(originals.values()) + compressed


def evaluate(config, role, retry_failed=False):
    run = Path(config["output_dir"])
    out = run / role
    if not read_json(run / "reader/qualification.json")["passed"]:
        raise ReaderStopped("Reader must pass qualification before evaluation")
    if role == "test":
        frozen = read_json(run / "freeze.json")
        if frozen["reader"] != digest(generation_config(config["reader"]["model"])) or frozen["ratios"] != config["ratios"]:
            raise ValueError("Cannot evaluate the sealed test with another reader/ratio protocol")
    source = out / "compressed.jsonl"
    marker, state, done = stage_manifest(out, "reader", {"compression": file_hash(source),
        "reader": config["reader"], "grader": GRADER_VERSION,
        "freeze": file_hash(run / "freeze.json") if role == "test" else None})
    if done:
        return
    tasks = evaluation_tasks(read_jsonl(source))
    path = out / "responses.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    records = {(r["uid"], r["method"], r["target_ratio"]): r for r in previous}
    # Development answers also expose families: a later run cannot call them a
    # fresh test merely because they were not this run's final test.
    exposure = {"exposed_uids": sorted({r["uid"] for r in tasks}),
                "exposed_families": sorted({r["base_group_id"] for r in tasks}),
                "freeze_sha256": file_hash(run / "freeze.json") if role == "test" else None}
    atomic_json(run / (role + "_exposure.json"), exposure)
    with open_reader(config) as reader:
        for i, task in enumerate(tasks):
            key = (task["uid"], task["method"], task["target_ratio"])
            if key in records:
                if records[key]["response_status"] in {"ok", "truncated", "empty_response"} or records[key]["response_status"].startswith("compression_"):
                    continue
                if not retry_failed:
                    raise ReaderStopped("An earlier request failed. Inspect it, then resume with --retry-failed; ledger is preserved.")
            row = {k: task[k] for k in ("uid", "base_group_id", "source_dataset", "method", "target_ratio", "actual_ratio")}
            if task["status"] != "ok" or not task.get("budget_compliant", True):
                row.update(correct=False, response_status="compression_" + task["status"], parse_status="not_queried", query_hash=None)
                answer = None
            else:
                answer = reader.ask(task["compressed_prompt"], retry_failed=retry_failed)
                generation = answer["generation"]
                grade = grade_answer(generation["text"], task["ground_truth_answer"], task["original_prompt"])
                row.update(correct=bool(grade.correct and generation["status"] == "ok"),
                    response_status=generation["status"], parse_status=grade.status,
                    response=generation["text"], query_hash=answer["query_hash"],
                    usage=generation.get("usage"), elapsed_ms=generation.get("elapsed_ms"))
            records[key] = row
            write_jsonl(path, records.values())
            if answer:
                reader.stop_on(answer)
            if (i + 1) % 20 == 0:
                log("reader", role=role, completed=i + 1, total=len(tasks))
    finish_stage(marker, state, [path, run / (role + "_exposure.json")], conditions=len(tasks),
                 technical_failures=sum(r["response_status"] != "ok" for r in records.values()))
