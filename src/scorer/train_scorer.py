"""
STEP 7: Train a DistilBERT span classifier to predict consensus criticality.

Each row in the consensus dataset is a (problem, span) pair with character-level
start/end offsets into original_prompt and a binary consensus_label.

We frame this as span classification:
  Input  → [CLS] original_prompt [SEP]  (tokenized with offset mapping)
  Target → 1 (critical) or 0 (non_critical) for the span
  Model  → DistilBERT encoder → mean-pool span tokens → linear head → sigmoid

Train/test split is done at the PROBLEM level (by example_id) to prevent
data leakage across spans from the same problem.

Run:
    python src/scorer/train_scorer.py
    python src/scorer/train_scorer.py --epochs 5 --lr 3e-5 --test-size 0.3

Outputs:
    models/span_scorer/          ← saved HuggingFace model + tokenizer
    data/scorer_test_predictions.csv  ← test-set predictions for evaluation
"""

import os
import sys
import argparse
import random
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT_DIR = "models/span_scorer"
PREDICTIONS_FILE = os.path.join(config.DATA_DIR, "scorer_test_predictions.csv")
MAX_SEQ_LEN = 512
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_scorer")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SpanDataset(Dataset):
    """
    Each item is one (problem, span) pair.
    We tokenize the full original_prompt and find which sub-tokens correspond
    to [span_start, span_end) using HuggingFace's offset_mapping.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.records = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        text = str(row["original_prompt"])
        char_start = int(row["span_start"])
        char_end = int(row["span_end"])
        label = 1 if str(row["consensus_label"]) == "critical" else 0

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)  # (seq_len, 2)

        # Build a boolean mask over tokens that overlap with the span
        span_mask = torch.zeros(self.max_len, dtype=torch.bool)
        for i, (tok_start, tok_end) in enumerate(offsets.tolist()):
            # Skip special tokens (offset == [0,0])
            if tok_start == 0 and tok_end == 0:
                continue
            # Token overlaps with span if intervals intersect
            if tok_start < char_end and tok_end > char_start:
                span_mask[i] = True

        # Fallback: if span was truncated away, mark position 1 (first real token)
        if not span_mask.any():
            span_mask[1] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_mask": span_mask,
            "label": torch.tensor(label, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SpanScorer(nn.Module):
    """
    DistilBERT encoder with a span-pooling head.
    Mean-pools the hidden states of tokens inside the span, then
    passes through a single linear layer → scalar logit.
    """

    def __init__(self, encoder_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # 768 for distilbert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, span_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, seq_len, H)

        # Mean-pool over span tokens
        mask_f = span_mask.unsqueeze(-1).float()         # (B, seq_len, 1)
        span_repr = (hidden * mask_f).sum(dim=1)          # (B, H)
        n_tokens = mask_f.sum(dim=1).clamp(min=1e-9)     # (B, 1)
        span_repr = span_repr / n_tokens                  # (B, H)

        span_repr = self.dropout(span_repr)
        logit = self.classifier(span_repr).squeeze(-1)    # (B,)
        return logit


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            span_mask=batch["span_mask"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            span_mask=batch["span_mask"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
    return total_loss / len(loader), np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Problem-level train/test split
# ---------------------------------------------------------------------------
def problem_level_split(df: pd.DataFrame, test_size: float = 0.3, seed: int = SEED):
    """
    Split by unique example_id so spans from the same problem never bleed
    across train and test sets.
    """
    problem_ids = df["example_id"].unique()
    train_ids, test_ids = train_test_split(
        problem_ids, test_size=test_size, random_state=seed
    )
    train_df = df[df["example_id"].isin(train_ids)].copy()
    test_df = df[df["example_id"].isin(test_ids)].copy()
    logger.info(
        "Train: %d problems / %d spans | Test: %d problems / %d spans",
        len(train_ids), len(train_df), len(test_ids), len(test_df),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT span scorer")
    parser.add_argument("--consensus-file", default=config.CONSENSUS_FILE,
                        help="Path to the consensus CSV from Step 4")
    parser.add_argument("--output-dir", default=MODEL_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Fraction of problems held out for testing")
    parser.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    _set_seed(SEED)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s", device)

    # --- Load data ---
    if not os.path.exists(args.consensus_file):
        logger.error("Consensus file not found: %s. Run 04_build_consensus.py first.", args.consensus_file)
        sys.exit(1)

    df = pd.read_csv(args.consensus_file).fillna("")
    # Keep only one row per (problem, span) — drop the per-model duplicates
    # since consensus_label is the same for all of them
    span_df = df.drop_duplicates(
        subset=["example_id", "span_text", "span_start", "span_end"]
    ).copy()
    logger.info("Loaded %d unique spans from %s", len(span_df), args.consensus_file)
    logger.info("Label distribution:\n%s", span_df["consensus_label"].value_counts().to_string())

    # --- Train / test split at problem level ---
    train_df, test_df = problem_level_split(span_df, test_size=args.test_size)

    # --- Tokenizer + Datasets ---
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds = SpanDataset(train_df, tokenizer, max_len=args.max_len)
    test_ds = SpanDataset(test_df, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # --- Model ---
    model = SpanScorer(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # --- Training ---
    best_auc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_probs, val_labels = evaluate(model, test_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_preds = (val_probs >= 0.5).astype(int)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_AUC=%.4f",
            epoch, args.epochs, train_loss, val_loss, val_auc,
        )
        print(classification_report(
            val_labels, val_preds, target_names=["non_critical", "critical"]
        ))

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            model.encoder.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(model.classifier.state_dict(),
                       os.path.join(args.output_dir, "classifier_head.pt"))
            logger.info("  ↑ Best model saved (AUC=%.4f)", best_auc)

    # --- Save test predictions for analysis ---
    _, final_probs, final_labels = evaluate(model, test_loader, device)
    pred_df = test_df.copy()
    pred_df["predicted_prob"] = final_probs
    pred_df["predicted_label"] = (final_probs >= 0.5).astype(int)
    pred_df["true_label"] = final_labels.astype(int)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info("Test predictions saved to %s", PREDICTIONS_FILE)
    logger.info("Final test AUC: %.4f", roc_auc_score(final_labels, final_probs))


if __name__ == "__main__":
    main()
