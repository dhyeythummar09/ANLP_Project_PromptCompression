"""
Central config for the whole dataset-creation pipeline.
    
  - https://console.groq.com/docs/models
  - https://ai.google.dev/gemini-api/docs/models
"""

import os
from dotenv import load_dotenv
# Load .env into os.environ before reading keys
load_dotenv()

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ---------------------------------------------------------------------------
# The labeling models. Each dict describes how to call it. "provider" is either "groq" or "google" -- api_clients.py branches on this.
# "sleep_seconds" is how long to wait between calls to that provider, to stay under its rate limits. Groq's free tier is generous; Google's is stricter,
# hence the longer sleep for Gemini.
# ---------------------------------------------------------------------------


MODELS = [
    {"name": "openai/gpt-oss-120b", "provider": "groq", "sleep_seconds": 2.2},
    {"name": "gemini-3.6-flash", "provider": "google", "sleep_seconds": 4.2},
    #{"name": "nvidia/nemotron-3-ultra-550b-a55b:free", "provider": "openrouter", "sleep_seconds": 3.2},
    {"name": "qwen-3.8-27b", "provider": "cerebras", "sleep_seconds": 0.2},
]

# ---------------------------------------------------------------------------
# Dataset sizes and splits : These are the numbers of problems to pull from each
# ---------------------------------------------------------------------------
GSM8K_N = 200
GSM_IC_N = 200
BBH_SUBTASKS = {
    "formal_fallacies": 100,
    "date_understanding": 100,
    "tracking_shuffled_objects_three_objects": 100,
    "logical_deduction_three_objects": 100,
    "navigate": 100,
    "object_counting": 100,
}
# 200 + 200 + 600 = 1000 total problems

# GSM8K standard test split, not train -- train is meant for few-shot demonstrations elsewhere in the course project, and using test here keeps us aligned with how GSM8K is normally used as an evaluation set.
GSM8K_SPLIT = "test"

# ---------------------------------------------------------------------------
# Span categories and granularities
# ---------------------------------------------------------------------------
SPAN_CATEGORIES = [
    "negation",
    "number",
    "comparison_conditional",
    "entity",
    "action_verb",
    "distractor_context",  # GSM-IC's inserted irrelevant sentence
    "random_control",
]
SPAN_GRANULARITIES = ["token", "phrase", "clause"]

# Minimum number of models that must have produced a valid label for a span
# before we trust its consensus score. A span "labeled" by only one model
# (because the other three got the original question wrong and were
# skipped) is not a reliable signal
MIN_MODELS_FOR_CONSENSUS = 2

# ---------------------------------------------------------------------------
# File paths : everything lands in ./data/
# ---------------------------------------------------------------------------
DATA_DIR = "data"
LOGS_DIR = "logs"

RAW_PROBLEMS_FILE = f"{DATA_DIR}/raw_problems.csv"
SPANS_FILE = f"{DATA_DIR}/candidate_spans.csv"
LABELS_FILE = f"{DATA_DIR}/per_model_labels.csv"
CONSENSUS_FILE = f"{DATA_DIR}/consensus_dataset.csv"
LABELING_LOG_FILE = f"{LOGS_DIR}/labeling_errors.log"
