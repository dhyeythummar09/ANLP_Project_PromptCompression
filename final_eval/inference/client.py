"""Provider-configured generation. No hidden model aliases or credential fallback."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
import threading
import time
import requests
from final_eval.utils.artifacts import digest
from final_eval.utils.artifacts import atomic_json
import json
from pathlib import Path

@dataclass
class ModelResponse:
    text: str = ""
    status: str = "ok"
    requested_model: str = ""
    served_model: str = ""
    finish_reason: str = ""
    attempts: int = 0
    elapsed_ms: float = 0
    usage: dict | None = None
    error: str = ""
    reasoning: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return asdict(self)

def public_config(cfg):
    if any(k.lower() in {"api_key", "apikey", "token", "authorization", "headers"} for k in cfg):
        raise ValueError("Put credentials in environment variables; config accepts api_key_env only")
    if "base_url" in cfg:
        from urllib.parse import urlsplit
        parsed = urlsplit(cfg["base_url"])
        if parsed.query or parsed.username or parsed.password:
            raise ValueError("base_url must not contain inline credentials or query parameters")
    return dict(cfg)


def generation_config(cfg):
    """Quota/backoff changes do not change a prompt's generation conditions."""
    return {k:v for k,v in public_config(cfg).items()
            if k not in {"request_budget","max_attempts","sleep_seconds","timeout"}}

class HostedClient:
    def __init__(self, cfg, usage_path=None):
        self.cfg = public_config(cfg)
        self.name = cfg.get("name") or cfg.get("model")
        if not self.name or str(self.name).startswith("SET_"):
            raise ValueError("An exact confirmed model name is required")
        self.url = cfg.get("base_url", "").rstrip("/")
        if not self.url:
            raise ValueError("base_url is required")
        self.env_name = cfg.get("api_key_env", "")
        if self.env_name and not os.environ.get(self.env_name):
            from dotenv import load_dotenv
            load_dotenv()
        self.google = cfg.get("provider") == "google"
        self.lock = threading.Lock()
        self.next_allowed = 0.0
        self.attempts = 0
        self.execution_history = []
        self.stopped = None
        self.usage_path = Path(usage_path) if usage_path else None
        if self.usage_path and self.usage_path.exists():
            saved = json.loads(self.usage_path.read_text(encoding="utf-8"))
            if saved["config_hash"] != digest(generation_config(self.cfg)):
                raise ValueError("Usage ledger configuration mismatch")
            self.attempts = int(saved["attempts"])
            self.execution_history = saved.get("execution_history",[])
        settings={k:self.cfg.get(k) for k in ("request_budget","max_attempts","sleep_seconds","timeout")}
        if not self.execution_history or self.execution_history[-1]["settings"]!=settings:
            self.execution_history.append({"settings":settings,"timestamp_utc":datetime.now(timezone.utc).isoformat()})
        if int(cfg.get("max_attempts", 3)) < 1 or float(cfg.get("sleep_seconds", 0)) < 0:
            raise ValueError("Invalid retry/rate configuration")

    def generate(self, prompt):
        start = time.perf_counter()
        result = ModelResponse(requested_model=self.name)
        key = os.environ.get(self.env_name, "").strip() if self.env_name else ""
        if self.env_name and not key:
            result.status, result.error = "configuration_error", f"Missing environment variable {self.env_name}"
            return result
        if "," in key:
            result.status, result.error = "configuration_error", "Use one credential per configured run; comma-separated key cycling was removed"
            return result
        for attempt in range(int(self.cfg.get("max_attempts", 3))):
            with self.lock:
                limit = self.cfg.get("request_budget")
                if self.stopped or (limit is not None and self.attempts >= int(limit)):
                    result.status = self.stopped or "request_budget_exhausted"
                    break
                delay = self.next_allowed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self.attempts += 1
                result.attempts += 1
                if self.usage_path:
                    atomic_json(self.usage_path, {"config_hash": digest(generation_config(self.cfg)), "attempts": self.attempts,
                        "execution_history":self.execution_history})
                self.next_allowed = time.monotonic() + float(self.cfg.get("sleep_seconds", 1.0))
            try:
                if self.google:
                    url = f"{self.url}/models/{self.name}:generateContent"
                    response = requests.post(url, params={"key": key}, json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": self.cfg.get("temperature", 0),
                                             "maxOutputTokens": self.cfg.get("max_tokens", 512)}},
                        timeout=float(self.cfg.get("timeout", 90)))
                else:
                    url = self.url if self.url.endswith("/chat/completions") else self.url + "/chat/completions"
                    headers = {"Authorization": f"Bearer {key}"} if key else {}
                    response = requests.post(url, headers=headers, json={
                        "model": self.name, "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.cfg.get("temperature", 0),
                        "max_tokens": self.cfg.get("max_tokens", 512)}, timeout=float(self.cfg.get("timeout", 90)))
                if response.status_code in (401, 403, 402):
                    self.stopped = "auth_failed" if response.status_code != 402 else "quota_exhausted"
                    result.status, result.error = self.stopped, f"HTTP {response.status_code}"
                    break
                if response.status_code == 429 or response.status_code >= 500:
                    result.status, result.error = "rate_limit" if response.status_code == 429 else "server_error", f"HTTP {response.status_code}"
                    if attempt + 1 < int(self.cfg.get("max_attempts", 3)):
                        time.sleep(min(30, 2 ** (attempt + 1)))
                        continue
                    break
                if not response.ok:
                    result.status, result.error = "model_not_found" if response.status_code == 404 else "http_error", f"HTTP {response.status_code}"
                    if response.status_code == 404:
                        self.stopped = "model_not_found"
                    break
                data = response.json()
                if self.google:
                    choice = data["candidates"][0]
                    result.text = "".join(p.get("text", "") for p in choice["content"]["parts"] if not p.get("thought"))
                    result.finish_reason = choice.get("finishReason", "")
                    result.served_model = data.get("modelVersion", "")
                    result.usage = data.get("usageMetadata", {})
                else:
                    choice = data["choices"][0]; message = choice["message"]
                    result.text = message.get("content") or ""
                    result.reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    result.finish_reason = choice.get("finish_reason") or ""
                    result.served_model = data.get("model", "")
                    result.usage = data.get("usage", {})
                result.status = ("truncated" if result.finish_reason.lower() in {"length", "max_tokens"} else
                                 "empty_response" if not result.text.strip() else "ok")
                break
            except requests.RequestException:
                result.status, result.error = "network_error", "Request failed; credentials/URLs omitted from error logs"
                if attempt + 1 < int(self.cfg.get("max_attempts", 3)):
                    time.sleep(min(30, 2 ** (attempt + 1)))
            except (ValueError, KeyError, IndexError, TypeError):
                result.status, result.error = "invalid_response", "Unexpected provider response schema"
                break
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

class LocalClient:
    def __init__(self, cfg):
        self.cfg = public_config(cfg)
        self.name = cfg.get("name") or cfg.get("model")
        if not self.name or str(self.name).startswith("SET_"):
            raise ValueError("An exact selected local model name/path is required")
        self.model = self.tokenizer = None

    def generate(self, prompt):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is None:
            revision = self.cfg.get("revision", "main")
            self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=revision)
            dtype_name = self.cfg.get("dtype", "float16" if torch.cuda.is_available() else "float32")
            if dtype_name not in {"float16", "float32", "bfloat16"}:
                raise ValueError("dtype must be float16, float32, or bfloat16")
            self.model = AutoModelForCausalLM.from_pretrained(self.name, revision=revision,
                torch_dtype=getattr(torch, dtype_name), device_map=self.cfg.get("device_map", "auto"))
            self.model.eval()
        start = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                     tokenize=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=int(self.cfg.get("max_tokens", 512)),
                                          pad_token_id=self.tokenizer.eos_token_id)
        generated = outputs[0, inputs.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        stopped = len(generated) > 0 and int(generated[-1]) in eos_ids
        status = "ok" if stopped else "truncated"
        if not text.strip():
            status = "empty_response"
        return ModelResponse(text=text, status=status, requested_model=self.name, served_model=self.name,
            finish_reason="eos" if stopped else "length", attempts=1, elapsed_ms=(time.perf_counter() - start) * 1000,
            usage={"prompt_tokens": int(inputs.shape[1]), "completion_tokens": len(generated)})

def make_client(cfg, usage_path=None):
    return LocalClient(cfg) if cfg.get("backend") == "local" else HostedClient(cfg, usage_path=usage_path)
