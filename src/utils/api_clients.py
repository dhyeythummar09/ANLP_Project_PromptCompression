"""
Thin wrappers around the Groq and Google (Gemini) chat-completion APIs,
with retry + exponential backoff for transient failures (rate limits,
timeouts, momentary 5xx errors).

Kept separate from the labeling loop itself so the labeling script only has
to think about "ask this model this question," not HTTP/retry mechanics.
"""

import time
import logging

import requests

import config

logger = logging.getLogger("api_clients")

# How many times to retry a single call before giving up on it.
MAX_RETRIES = 3
# Base delay for exponential backoff: 2s, 4s, 8s on successive retries.
BACKOFF_BASE_SECONDS = 2


def _call_groq(model_name: str, prompt: str) -> str:
    """Call a Groq-hosted model's chat completion endpoint."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,  # deterministic decoding, per the proposal
        "max_tokens": 300,  # enough room for brief reasoning + final answer
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_google(model_name: str, prompt: str) -> str:
    """Call a Gemini model's generateContent endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={config.GOOGLE_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_model(model_cfg: dict, prompt: str) -> str | None:
    """
    Call whichever provider `model_cfg` points to, with retry + exponential
    backoff on failure. Returns the model's text response, or None if every
    retry was exhausted (the caller is expected to skip this row and move
    on -- see counterfactual_labeling.py).
    """
    provider = model_cfg["provider"]
    model_name = model_cfg["name"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "groq":
                return _call_groq(model_name, prompt)
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
