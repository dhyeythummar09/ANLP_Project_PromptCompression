"""
Thin wrappers around LLM providers (Groq, Google Gemini, Motapi, HuggingFace, Cerebras, OpenRouter),
with retry + exponential backoff for transient failures (rate limits, timeouts, momentary 5xx errors).

Kept separate from the labeling loop itself so the labeling script only has
to think about "ask this model this question," not HTTP/retry mechanics.
"""

import time
import logging
import requests
import config

logger = logging.getLogger("api_clients")

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2


def _get_motapis_endpoint() -> str:
    base = (config.MOTAPIS_BASE_URL or "https://api.motapis.com/v1").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _call_openai_compatible(url: str, api_key: str, model_name: str, prompt: str) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,  # deterministic decoding
        "max_tokens": 350,  # concise room for reasoning + Final Answer (conserves tokens)
    }
    resp = requests.post(url, headers=headers, json=body, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and ("reasoning" in msg or "reasoning_content" in msg):
        content = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return content.strip()


def _call_google(model_name: str, prompt: str) -> str:
    """Call a Gemini model's generateContent endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={config.GOOGLE_API_KEY}"
    )
    body = {
        "systemInstruction": {
            "parts": [{
                "text": "You are a mathematical and logical reasoning assistant. "
                        "You may reason briefly, but you MUST end your response with a line in exactly this format:\n"
                        "Final Answer: <your answer>"
            }]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }
    resp = requests.post(url, json=body, timeout=45)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_model(model_cfg: dict, prompt: str) -> str | None:
    """
    Call whichever provider `model_cfg` points to, with retry + exponential
    backoff on failure. Returns the model's text response, or None if every
    retry was exhausted.
    """
    provider = model_cfg["provider"].lower()
    model_name = model_cfg["name"]

    # Check key presence upfront so we don't hammer retries if key is not configured
    key_map = {
        "motapis": config.MOTAPIS_API_KEY,
        "groq": config.GROQ_API_KEY,
        "google": config.GOOGLE_API_KEY,
        "huggingface": config.HF_TOKEN,
        "cerebras": config.CEREBRAS_API_KEY,
        "openrouter": config.OPENROUTER_API_KEY,
    }
    if provider in key_map and not key_map[provider]:
        logger.warning("API key for provider %s is not set. Skipping %s.", provider, model_name)
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "groq":
                return _call_openai_compatible(
                    "https://api.groq.com/openai/v1/chat/completions",
                    config.GROQ_API_KEY,
                    model_name,
                    prompt,
                )
            elif provider == "motapis":
                return _call_openai_compatible(
                    _get_motapis_endpoint(),
                    config.MOTAPIS_API_KEY,
                    model_name,
                    prompt,
                )
            elif provider == "huggingface":
                return _call_openai_compatible(
                    "https://router.huggingface.co/hf-inference/v1/chat/completions",
                    config.HF_TOKEN,
                    model_name,
                    prompt,
                )
            elif provider == "cerebras":
                return _call_openai_compatible(
                    "https://api.cerebras.ai/v1/chat/completions",
                    config.CEREBRAS_API_KEY,
                    model_name,
                    prompt,
                )
            elif provider == "openrouter":
                return _call_openai_compatible(
                    "https://openrouter.ai/api/v1/chat/completions",
                    config.OPENROUTER_API_KEY,
                    model_name,
                    prompt,
                )
            elif provider == "google":
                return _call_google(model_name, prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except requests.exceptions.RequestException as e:
            wait = BACKOFF_BASE_SECONDS ** attempt
            logger.warning(
                "Call to %s (%s) failed on attempt %d/%d: %s -- retrying in %ds",
                model_name, provider, attempt, MAX_RETRIES, e, wait,
            )
            if attempt == MAX_RETRIES:
                logger.error("Giving up on %s (%s) after %d attempts", model_name, provider, MAX_RETRIES)
                return None
            time.sleep(wait)
    return None
