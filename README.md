# Reasoning-Critical Prompt Compression — Dataset Creation

This covers the dataset-generation and labeling pipeline only (Phase 1-2 of
the project). The scorer-training and final-compressor code will live in
`src/scorer/` and `src/compressor/` respectively, added in later steps.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="AIza..."
```

## Run order

Each step reads the previous step's output from `data/` and is safe to
re-run (steps 1-2 are idempotent; step 3 resumes from where it left off if
interrupted).

```bash
python src/dataset_creation/01_load_data.py            # -> data/raw_problems.csv
python src/dataset_creation/02_extract_spans.py         # -> data/candidate_spans.csv
python src/dataset_creation/03_counterfactual_labeling.py  # -> data/per_model_labels.csv
python src/dataset_creation/04_build_consensus.py       # -> data/consensus_dataset.csv
```

Step 3 is the slow, API-call-heavy one — expect it to take a while and run
it in the background (e.g. `nohup python src/dataset_creation/03_counterfactual_labeling.py &`).
Check `logs/03_labeling_errors.log` afterward for anything that failed even after retries.

## Layout

- `config.py` — every path, model name, dataset size, and category list.
  Change things here, not inside the individual scripts.
- `src/utils/` — shared logic (answer matching, API calls) used by more than
  one script, so it can't silently drift out of sync between them.
- `src/dataset_creation/` — the 4 pipeline steps above.
- `src/scorer/`, `src/compressor/` — not yet built; next steps.
- `data/`, `logs/`, `outputs/` — generated artifacts, not source code.
