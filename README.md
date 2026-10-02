# Reasoning-Critical Prompt Compression: Cross-Model Counterfactual Identification of Answer-Critical Spans

**Team Symbiote** | Advanced Natural Language Processing (ANLP), Monsoon 2024  
International Institute of Information Technology, Hyderabad (IIIT-H)  
GitHub Repository: [https://github.com/dhyeythummar09/ANLP_Project_PromptCompression](https://github.com/dhyeythummar09/ANLP_Project_PromptCompression)

**Team Members**:
- Pranav Srivastava
- Vikhyath Pattipaty
- Chatrathi Abhinav
- Dhyey Thummar

---

## Overview

Existing prompt compression frameworks (such as LLMLingua and Selective-Context) rely on surrogate model perplexity or self-information to discard "redundant" tokens. While effective for verbose documents and multi-shot prompts, these proxy signals often fail on compact reasoning tasks (such as math word problems and deductive logic), where predictable or low-information tokens (e.g., small numbers, negations, units) are strictly necessary to arrive at the correct answer.

This project investigates **reasoning-critical prompt compression** through offline counterfactual necessity testing:
1. **Counterfactual Labeling**: Masking candidate linguistic spans across three architecturally diverse LLMs (GPT-OSS 120B, Qwen 3.7-Max, GLM 5.3-Flash) over 1,000 reasoning problems (GSM8K, GSM-IC, and BigBench Hard) to measure whether removing a span flips an answer from correct to incorrect.
2. **Consensus Modeling**: Aggregating multi-model votes into a consensus criticality score for 14,162 spans.
3. **Lightweight Neural Scorer**: Training a DistilBERT-base span classifier (5-fold cross-validation) to predict criticality on unseen prompts without runtime LLM calls.
4. **Budgeted Compression & Evaluation**: Compressing prompts to strict token budgets (e.g., 50%, 25%) and evaluating downstream accuracy on both seen and unseen target LLMs (DeepSeek-V4-Flash), as well as benchmarking against SOTA compressors (LLMLingua, LLMLingua-2, and Selective-Context).

---

## Repository Structure

```
ANLP_Project_PromptCompression/
├── config.py                 # Central configurations (benchmark splits, span types, API endpoints)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── src/
│   ├── dataset_creation/     # Pipeline for dataset curation and counterfactual labeling
│   │   ├── 01_load_data.py               # Downloads and standardizes GSM8K, GSM-IC, and BBH
│   │   ├── 02_extract_spans.py            # Extracts candidate linguistic spans via spaCy
│   │   ├── 03_counterfactual_labeling.py  # Masks spans and queries labeling LLMs with checkpoints
│   │   ├── 04_build_consensus.py          # Aggregates multi-model labels into consensus scores
│   │   └── test_models.py                # API connectivity and sanity testing utility
│   │
│   ├── scorer/               # Neural criticality scoring
│   │   └── train_scorer.py               # Trains DistilBERT span classifier with 5-fold CV
│   │
│   ├── compressor/           # Budgeted compression and evaluation
│   │   ├── compress.py                   # Implements our hybrid compressor and baselines
│   │   └── evaluate.py                   # Evaluates compressed prompts on target LLMs
│   │
│   └── utils/                # Shared utilities
│       ├── api_clients.py                # Unified LLM provider client with backoff and retry
│       └── answer_matching.py            # Extraction and answer validation for math & MCQ
│
└── notebooks/
    └── 01_eda_counterfactuals.ipynb      # Analysis of span distributions, agreement, and failure rates
```

*Note: Raw problem caches, generated datasets (`data/`), model checkpoints (`models/`), and run logs (`logs/`) are excluded from version control via `.gitignore`.*

---

## Setup & Installation

### 1. Environment Setup
Clone the repository and install dependencies in a Python 3.10+ virtual environment:
```bash
git clone https://github.com/dhyeythummar09/ANLP_Project_PromptCompression.git
cd ANLP_Project_PromptCompression

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Environment Variables
Create a `.env` file in the root directory following `.env.example`:
```env
MOTAPIS_API_KEY_QWEN=your_key_here
MOTAPIS_API_KEY_GLM=your_key_here
GROQ_API_KEY=your_key_here
MOTAPIS_BASE_URL=https://api.motapis.com/v1
```

---

## Execution Workflow

### Step 1: Benchmark Loading & Span Extraction
Fetch the 1,000 reasoning problems and extract candidate linguistic spans (numbers, entities, verbs, negations, conditionals, and length-matched random controls):
```bash
python src/dataset_creation/01_load_data.py
python src/dataset_creation/02_extract_spans.py
```

### Step 2: Counterfactual Labeling & Consensus Aggregation
Evaluate span necessity across the labeling panel (Qwen, GLM, GPT-OSS) and aggregate consensus labels:
```bash
python src/dataset_creation/03_counterfactual_labeling.py
python src/dataset_creation/04_build_consensus.py
```

### Step 3: Train the Criticality Scorer
Train the DistilBERT-base span classifier across 5 folds:
```bash
python src/scorer/train_scorer.py --epochs 5 --batch-size 32
```

### Step 4: Budgeted Prompt Compression
Compress prompts at target token retention budgets (e.g., 50% and 25%) across our methods and baselines:
```bash
# Run our hybrid compressor and baselines
python src/compressor/compress.py --methods ours ours_scorer ours_signal random --ratios 0.5 0.25

# Run external baselines (LLMLingua, LLMLingua-2, Selective-Context)
python src/compressor/compress.py --methods llmlingua llmlingua2 selective_context --ratios 0.5 0.25
```

### Step 5: Downstream & Intrinsic Evaluation
Evaluate exact-match answer accuracy on target models (seen Groq vs. unseen DeepSeek) and compute reference-free intrinsic metrics (Perplexity, Critical Span Recall, ROUGE-L):
```bash
python src/compressor/evaluate.py --model deepseek --mode summarize
```

---

## Key Experimental Results

- **Math Word Problems (GSM8K & GSM-IC)**: At 50% token compression, our hybrid compressor retains **40.6% to 43.3% downstream accuracy** on an unseen model (DeepSeek-V4-Flash), outperforming random deletion (**4.6%**) by roughly 10x.
- **Cross-Model Generalization**: Performance transfers near-identically between seen (Groq 120B: 43.5%) and unseen (DeepSeek: 43.3%) target models, confirming that arithmetic answer-criticality is an intrinsic property of the problem text rather than model-specific artifact.
- **Distractor Filtering**: On GSM-IC (injected irrelevant sentences), our hybrid method discards 100% of distractor tokens while retaining over 88% of task-critical numbers.
- **Baseline Budget Analysis**: In intrinsic evaluations across 1,000 problems, LLMLingua-1 exhibits severe budget failure on short reasoning questions (mean ratio error of 0.43 at 0.50 target, retaining ~93% of tokens and 98.9% of distractors). At 25% budget, our method retains **67.7% of numeric spans** compared to 43.7% for LLMLingua-2 and 26.6% for Selective-Context.
