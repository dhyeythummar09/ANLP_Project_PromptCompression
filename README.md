# Reasoning-Critical Prompt Compression
**Cross-Model Counterfactual Identification of Answer-Critical Spans**

This repository implements the end-to-end dataset creation, counterfactual labeling, consensus aggregation, and evaluation pipeline for reasoning-critical prompt compression (GSM8K, GSM-IC, and BBH).

---

## 1. Environment Setup

### Install Dependencies
Activate your virtual environment and install the required packages:
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure `.env` (No terminal exports needed)
Copy `.env.example` to `.env` (or open the existing `.env` file in the root folder):
```powershell
cp .env.example .env
```
Fill in your API keys in `.env`:
```dotenv
# --- MOTAPIS (https://motapis.com) ---
# Used for Qwen 3.7 Max and GLM 5.3 Flash (2M free tokens)
MOTAPIS_API_KEY=your_motapis_key_here
MOTAPIS_BASE_URL=https://motapis.com/v1

# --- GROQ (https://console.groq.com) ---
# Used for OpenAI GPT-OSS 120B (Free tier)
GROQ_API_KEY=your_groq_key_here

# --- GOOGLE AI STUDIO (https://aistudio.google.com) ---
# Used for Gemini 3.6 Flash
GOOGLE_API_KEY=your_google_key_here

# --- WORKER ID (1, 2, 3, or 4) ---
WORKER_ID=1
```

---

## 2. Models & Architectures

To satisfy the project proposal's requirement of cross-model counterfactual necessity testing across distinct architectures, four model families are configured in `config.py`:

| # | Model | Provider | Architecture / Family | Quota / Cost |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `qwen3.7-max` | **Motapis** | Alibaba Qwen Architecture | Free tier (~480k / 2M tokens per person) |
| **2** | `glm-5.3-flash` | **Motapis** | Zhipu AI GLM Architecture | Free tier (~480k / 2M tokens per person) |
| **3** | `openai/gpt-oss-120b` | **Groq** | OpenAI Open-Weights MoE | Free tier (Request-based ceiling) |
| **4** | `gemini-3.6-flash` | **Google** | Google DeepMind Transformer | Google AI Studio |

### Token Consumption Feasibility
- 1,000 problems = **15,024 candidate spans**.
- Divided among 4 teammates = **250 problems** and **~3,750 candidate spans** per teammate.
- Each call is capped at `max_tokens: 350`, producing concise mathematical reasoning (~150 tokens/call).
- **Both `qwen3.7-max` and `glm-5.3-flash` combined require ~960k tokens per teammate**, which is well below the **2,000,000 free token allowance** on Motapis (leaving >1M tokens safety buffer).

---

## 3. Team Workload Distribution (Workers 1 to 4)

The 1,000 problems are deterministically partitioned into 4 non-overlapping splits of 250 problems each:
- **Worker 1**: `WORKER_ID=1` (Problems 0, 4, 8, ...)
- **Worker 2**: `WORKER_ID=2` (Problems 1, 5, 9, ...)
- **Worker 3**: `WORKER_ID=3` (Problems 2, 6, 10, ...)
- **Worker 4**: `WORKER_ID=4` (Problems 3, 7, 11, ...)

Each teammate simply sets their assigned `WORKER_ID` in their local `.env` file before running Step 3.

> [!NOTE]
> If a teammate does not have a Google API key, the script will log an informational message and gracefully proceed with Motapis and Groq models without crashing.

---

## 4. End-to-End Execution Pipeline

### Optional Smoke Test (Verify API Keys)
To verify that your API key works without consuming many tokens (tests only 1 question per dataset family = ~500 tokens total):
```powershell
python src/dataset_creation/test_models.py --provider motapis
```

---

### Step 1: Load Raw Problems
Loads 200 GSM8K, 200 GSM-IC, and 600 BBH problems (1,000 problems total):
```powershell
python src/dataset_creation/01_load_data.py
```
*Output: `data/raw_problems.csv`*

---

### Step 2: Extract Candidate Spans
Extracts reasoning-critical spans (negation, numbers, comparisons, entities, action verbs, and length-matched random controls):
```powershell
python src/dataset_creation/02_extract_spans.py
```
*Output: `data/candidate_spans.csv` (15,024 candidate spans)*

---

### Step 3: Counterfactual Labeling (Split among 4 Workers)
Runs the baseline check and counterfactual masking loop for the worker's assigned 250 problems:
```powershell
python src/dataset_creation/03_counterfactual_labeling.py
```
- Automatically checkpoints every 25 labels to `data/per_model_labels_part{WORKER_ID}.csv`.
- Completely resumable if paused or interrupted (re-running continues from the last completed span).
- Tracks daily request caps and caches original prompt answers permanently in `data/03_original_answers_cache_part{WORKER_ID}.json`.

---

### Step 4: Build Consensus Dataset
Once all 4 workers have completed their labeling runs:
1. Place all four files in `data/`:
   - `data/per_model_labels_part1.csv`
   - `data/per_model_labels_part2.csv`
   - `data/per_model_labels_part3.csv`
   - `data/per_model_labels_part4.csv`
2. Run the consensus aggregation script:
```powershell
python src/dataset_creation/04_build_consensus.py
```
*Output: `data/consensus_dataset.csv`*

This computes the cross-model agreement score for every span, applies the consensus threshold (`agreement_score > 0.5`), and filters out spans below `MIN_MODELS_FOR_CONSENSUS`.

---

## 5. Directory Structure

```
ANLP_Project_PromptCompression/
├── .env                                  # API keys & WORKER_ID (git-ignored)
├── .env.example                          # Template for environment configuration
├── config.py                             # Central project configuration & model definitions
├── requirements.txt                      # Project dependencies
├── README.md                             # Documentation & pipeline execution guide
├── data/
│   ├── raw_problems.csv                  # 1,000 raw benchmark questions
│   ├── candidate_spans.csv               # 15,024 extracted candidate spans
│   ├── per_model_labels_part{1..4}.csv   # Partitioned labels per worker
│   └── consensus_dataset.csv             # Final aggregated consensus dataset
├── logs/
│   └── labeling_errors_part{1..4}.log    # Detailed run logs per worker
└── src/
    ├── dataset_creation/
    │   ├── 01_load_data.py               # Step 1: Download & normalize data
    │   ├── 02_extract_spans.py            # Step 2: Extract linguistic spans
    │   ├── 03_counterfactual_labeling.py  # Step 3: Multi-worker counterfactual labeling
    │   ├── 04_build_consensus.py          # Step 4: Merge worker parts & score consensus
    │   └── test_models.py                # Smoke testing utility
    ├── utils/
    │   ├── api_clients.py                # Unified LLM API client with backoff
    │   └── answer_matching.py            # Robust correctness & option matching logic
    ├── scorer/                           # Phase 2: Lightweight DistilBERT span scorer
    └── compressor/                       # Phase 3: Token budget prompt compressor
```
