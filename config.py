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
# API keys (loaded from .env)
# ---------------------------------------------------------------------------
MOTAPIS_API_KEY_QWEN = os.environ.get("MOTAPIS_API_KEY_QWEN", "")
MOTAPIS_API_KEY_GLM = os.environ.get("MOTAPIS_API_KEY_GLM", "")
MOTAPIS_BASE_URL = os.environ.get("MOTAPIS_BASE_URL", "https://api.motapis.com/v1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

WORKER_ID = int(os.environ.get("WORKER_ID", 1))

# ---------------------------------------------------------------------------
# Labeling models (4 diverse architectures across providers)
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "qwen3.7-max", "provider": "motapis_qwen", "sleep_seconds": 0.5},
    {"name": "glm-5.3-flash", "provider": "motapis_glm", "sleep_seconds": 0.5},
    {"name": "openai/gpt-oss-120b", "provider": "groq", "sleep_seconds": 2.2},
    #{"name": "gemini-3.6-flash", "provider": "google", "sleep_seconds": 4.2},
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
