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

import itertools
import threading

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
        "max_tokens": 1024,  # concise room for reasoning + Final Answer (conserves tokens)
    }
    resp = requests.post(url, headers=headers, json=body, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and ("reasoning" in msg or "reasoning_content" in msg):
        content = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return content.strip()


# def _call_google(model_name: str, prompt: str) -> str:
#     """Call a Gemini model's generateContent endpoint."""
#     url = (
#         f"https://generativelanguage.googleapis.com/v1beta/models/"
#         f"{model_name}:generateContent?key={config.GOOGLE_API_KEY}"
#     )
#     body = {
#         "systemInstruction": {
#             "parts": [{
#                 "text": "You are a mathematical and logical reasoning assistant. "
#                         "You may reason briefly, but you MUST end your response with a line in exactly this format:\n"
#                         "Final Answer: <your answer>"
#             }]
#         },
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
#     }
#     resp = requests.post(url, json=body, timeout=45)
#     resp.raise_for_status()
#     return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

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
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1024},
        # Tell Gemini to stop aggressively blocking text
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    resp = requests.post(url, json=body, timeout=45)
    resp.raise_for_status()
    
    data = resp.json()
    
    # Safely try to extract the text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        # If the structure is broken (safety block or empty response), 
        # trick the script into thinking it was a network error so it retries or gracefully skips.
        raise requests.exceptions.RequestException(f"Gemini returned invalid structure: {data}")


def call_model(model_cfg: dict, prompt: str):
    """
    Call whichever provider `model_cfg` points to, with retry + exponential
    backoff on failure. Returns the model's text response, or None if every
    retry was exhausted.
    """
    provider = model_cfg["provider"].lower()
    model_name = model_cfg["name"]

    # Check key presence upfront so we don't hammer retries if key is not configured
    key_map = {
        "motapis_qwen": config.MOTAPIS_API_KEY_QWEN, 
        "motapis_glm": config.MOTAPIS_API_KEY_GLM,  
        "motapis_deepseek": config.MOTAPIS_API_KEY_DEEPSEEK,
        "groq": config.GROQ_API_KEY,
        "google": config.GOOGLE_API_KEY,
    }
    if provider in key_map and not key_map[provider]:
        logger.warning("API key for provider %s is not set. Skipping %s.", provider, model_name)
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "groq":
                import threading
                import itertools
                if not hasattr(call_model, "_groq_key_lock"):
                    call_model._groq_key_lock = threading.Lock()
                    keys = [k.strip() for k in config.GROQ_API_KEY.split(",") if k.strip()]
                    call_model._groq_key_cycle = itertools.cycle(keys)
                
                with call_model._groq_key_lock:
                    api_key = next(call_model._groq_key_cycle)
                    
                return _call_openai_compatible(
                    "https://api.groq.com/openai/v1/chat/completions",
                    api_key,
                    model_name,
                    prompt,
                )
            elif provider.startswith("motapis"):
                # Dynamically grab the correct key from the key_map based on the specific provider name
                api_key_str = key_map[provider]
                
                # Perfect round-robin to guarantee two workers NEVER use the same key at the same time
                import threading
                import itertools
                
                if not hasattr(call_model, "_key_lock"):
                    call_model._key_lock = threading.Lock()
                    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
                    call_model._key_cycle = itertools.cycle(keys)
                
                with call_model._key_lock:
                    api_key = next(call_model._key_cycle)

                target_model = "qwen3.8-max" if model_name == "qwen3.7-max" else model_name
                return _call_openai_compatible(
                    _get_motapis_endpoint(),
                    api_key,
                    target_model,
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
