"""
STEP 7 (revised): Train a DistilBERT span scorer to predict counterfactual criticality.

What changed vs. the first version
  * Problem-level K-FOLD cross-validation (default 5). Every problem gets an
    out-of-fold (OOF) prediction from a scorer that never saw it. With only
    300-500 problems a single 15% test split is too small/noisy to evaluate on.
  * Inside each fold: train / val (checkpoint selection) / test (touched once).
  * The BEST-val checkpoint is reloaded before test predictions (old bug: the
    saved model and the reported test numbers came from different epochs).
  * Soft targets (fraction of labeling LLMs that flipped) + pos_weight for the
    class imbalance. Use --hard-labels / --no-pos-weight to turn them off.
  * Each problem is encoded ONCE and all of its spans are pooled from that one
    forward pass (same thing the compressor does at inference).
  * Dynamic padding (no more padding everything to 512).
  * Spans truncated away by max_len are DROPPED, not silently relabeled.
  * uid = source_dataset|example_id, so ids never collide across GSM8K / BBH.
  * Report: ROC-AUC, PR-AUC, F1(critical), problem-level bootstrap 95% CIs,
    per-category and per-dataset metrics, and a CATEGORY-PRIOR BASELINE
    (predict criticality from span category alone). If the scorer barely beats
    the prior, it is only learning "numbers/negations matter".
  * Optional cross-model test: --holdout-model NAME trains on the consensus of
    the OTHER labeling models and evaluates against NAME's own labels.

Run:
    python src/scorer/train_scorer.py
    python src/scorer/train_scorer.py --folds 5 --epochs 4 --lr 3e-5
    python src/scorer/train_scorer.py --holdout-model openai/gpt-oss-120b

Outputs:
    models/span_scorer/fold{k}/scorer.pt   one scorer per fold
    models/span_scorer/split.json          uid -> fold (compress.py needs this)
    data/scorer_oof_predictions.csv        out-of-fold predictions for every span
    data/scorer_report.json                all metrics (overall / category / dataset / CIs)
"""

import os
import sys
import json
import copy
import random
import logging
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT_DIR = "models/span_scorer"
MAX_SEQ_LEN = 512          # dynamic padding makes this free for short prompts
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_scorer")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_uid(source_dataset, example_id) -> str:
    """Globally unique problem id (example_id alone can collide across datasets)."""
    return f"{source_dataset}|{example_id}"


# ---------------------------------------------------------------------------
# Span <-> token helpers (shared with compress.py)
# ---------------------------------------------------------------------------
def span_token_mask(offsets: np.ndarray, char_start: int, char_end: int) -> np.ndarray:
    """Boolean mask over tokens whose character range overlaps [char_start, char_end)."""
    real = offsets[:, 1] > offsets[:, 0]  # excludes [CLS]/[SEP] (offset 0,0)
    return real & (offsets[:, 0] < char_end) & (offsets[:, 1] > char_start)


# ---------------------------------------------------------------------------
# Span table (one row per unique labeled span, with soft + hard targets)
# ---------------------------------------------------------------------------
def build_span_table(df: pd.DataFrame, holdout_model=None) -> pd.DataFrame:
    """
    Collapse the consensus CSV into one row per (problem, span).

    Columns produced: uid, example_id, source_dataset, original_prompt, span_text,
    span_start, span_end, span_category, target (soft, 0-1), label (hard 0/1, used
    for pos_weight + checkpoint selection), eval_label (what metrics are computed on).

    Soft target priority: a consensus score column if present, else the mean of the
    per-model binary labels (needs `labeling_model` + `label` columns), else hard label.
    With holdout_model: target/label come from the OTHER models, eval_label from the
    held-out model only.
    """
    df = df.copy()
    if "source_dataset" not in df.columns:
        df["source_dataset"] = ""
    df["uid"] = df["source_dataset"].astype(str) + "|" + df["example_id"].astype(str)
    df["span_start"] = df["span_start"].astype(int)
    df["span_end"] = df["span_end"].astype(int)
    key = ["uid", "span_start", "span_end"]

    base = df.drop_duplicates(key).set_index(key)
    out = base[["example_id", "source_dataset", "original_prompt", "span_text", "span_category"]].copy()
    hard = base["consensus_label"].astype(str).eq("critical").astype(float)

    per_model = None
    if {"labeling_model", "label"}.issubset(df.columns):
        df["_y"] = df["label"].astype(str).eq("critical").astype(float)
        per_model = df.pivot_table(index=key, columns="labeling_model", values="_y", aggfunc="max")

    if holdout_model:
        if per_model is None or holdout_model not in per_model.columns:
            raise SystemExit(f"--holdout-model {holdout_model!r} needs `labeling_model`/`label` "
                             f"columns and that model present. Found: "
                             f"{None if per_model is None else list(per_model.columns)}")
        others = [c for c in per_model.columns if c != holdout_model]
        out["target"] = per_model[others].mean(axis=1)
        out["eval_label"] = per_model[holdout_model]
        out = out.dropna(subset=["target", "eval_label"])
        out["label"] = (out["target"] >= 0.5).astype(int)
        out["eval_label"] = out["eval_label"].astype(int)
    else:
        soft = None
        for col in ("consensus_score", "consensus_criticality", "criticality_score"):
            if col in base.columns:
                soft = pd.to_numeric(base[col], errors="coerce")
                break
        if soft is None and per_model is not None:
            soft = per_model.mean(axis=1)
        out["target"] = (soft if soft is not None else hard).reindex(out.index).fillna(hard)
        out["label"] = hard.astype(int)
        out["eval_label"] = out["label"]

    return out.reset_index()


# ---------------------------------------------------------------------------
# Data: one item per PROBLEM (all its spans), dynamic padding collate
# ---------------------------------------------------------------------------
def build_items(span_df: pd.DataFrame, tokenizer, max_len: int) -> list[dict]:
    """Tokenize each problem once; build a token mask per span. Drops truncated spans."""
    span_df = span_df.reset_index(drop=True)
    items, dropped = [], 0
    for uid, g in span_df.groupby("uid", sort=False):
        text = str(g["original_prompt"].iloc[0])
        enc = tokenizer(text, max_length=max_len, truncation=True, return_offsets_mapping=True)
        offsets = np.array(enc["offset_mapping"])
        masks, rows = [], []
        for ridx, r in zip(g.index, g.itertuples()):
            m = span_token_mask(offsets, r.span_start, r.span_end)
            if m.any():
                masks.append(m)
                rows.append(ridx)
            else:
                dropped += 1
        if masks:
            items.append({
                "uid": uid,
                "input_ids": enc["input_ids"],
                "masks": np.stack(masks),
                "targets": g.loc[rows, "target"].values.astype(np.float32),
                "rows": rows,
            })
    if dropped:
        logger.warning("Dropped %d spans that fall outside max_len=%d", dropped, max_len)
    return items


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    S = max(len(b["targets"]) for b in batch)
    ids = torch.zeros(B, L, dtype=torch.long)       # DistilBERT [PAD] id = 0
    att = torch.zeros(B, L, dtype=torch.long)
    span_masks = torch.zeros(B, S, L)
    targets = torch.zeros(B, S)
    valid = torch.zeros(B, S, dtype=torch.bool)
    for i, b in enumerate(batch):
        n, k = len(b["input_ids"]), len(b["targets"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
        span_masks[i, :k, :n] = torch.from_numpy(b["masks"].astype(np.float32))
        targets[i, :k] = torch.from_numpy(b["targets"])
        valid[i, :k] = True
    return {"input_ids": ids, "attention_mask": att, "span_masks": span_masks,
            "targets": targets, "valid": valid, "rows": [b["rows"] for b in batch]}


# ---------------------------------------------------------------------------
# Model (imported by compress.py, so there is exactly one definition)
# ---------------------------------------------------------------------------
class SpanScorer(nn.Module):
    """DistilBERT encoder -> mean-pool each span's tokens -> linear -> logit."""

    def __init__(self, encoder_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, span_masks):
        """span_masks: (B, S, L) float. Returns logits (B, S)."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        num = torch.einsum("bsl,blh->bsh", span_masks, h)
        den = span_masks.sum(-1, keepdim=True).clamp(min=1.0)
        return self.classifier(self.dropout(num / den)).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def _masked_bce(logits, targets, valid, pos_weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")
    return (bce * valid).sum() / valid.sum().clamp(min=1)


def train_epoch(model, loader, optimizer, scheduler, device, pos_weight):
    model.train()
    total = 0.0
    for b in loader:
        optimizer.zero_grad()
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["span_masks"].to(device))
        loss = _masked_bce(logits, b["targets"].to(device), b["valid"].to(device), pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (row_indices, probs, mean_val_loss)."""
    model.eval()
    rows, probs, total = [], [], 0.0
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["span_masks"].to(device))
        total += _masked_bce(logits, b["targets"].to(device), b["valid"].to(device)).item()
        p = torch.sigmoid(logits).cpu().numpy()
        for i, r in enumerate(b["rows"]):
            for j, ridx in enumerate(r):
                rows.append(ridx)
                probs.append(float(p[i, j]))
    return rows, probs, total / max(len(loader), 1)


def safe(fn, y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, p))


def summarize_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": safe(roc_auc_score, y, p),
        "pr_auc": safe(average_precision_score, y, p),
        "f1_critical": float(f1_score(y, (p >= thr).astype(int), zero_division=0)),
    }


def train_one(train_df, val_df, tokenizer, args, device, out_dir) -> SpanScorer:
    train_items = build_items(train_df, tokenizer, args.max_len)
    val_items = build_items(val_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_items, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_items, batch_size=args.batch_size, collate_fn=collate)

    pos_weight = None
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - pos)
    if not args.no_pos_weight and pos > 0:
        pos_weight = torch.tensor(min(neg / pos, 10.0), device=device)
    logger.info("  train spans=%d (pos=%d) | val spans=%d | pos_weight=%s",
                len(train_df), int(pos), len(val_df),
                None if pos_weight is None else round(pos_weight.item(), 2))

    model = SpanScorer(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_score, best_state = -float("inf"), None
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, pos_weight)
        rows, probs, vloss = predict(model, val_loader, device)
        vy = val_df.loc[rows, "label"].values  # selection uses the TRAINING-side label
        ap = safe(average_precision_score, vy, probs)
        auc = safe(roc_auc_score, vy, probs)
        score = ap if np.isfinite(ap) else -vloss
        logger.info("  epoch %d/%d | train_loss=%.4f val_loss=%.4f val_AUC=%.3f val_PR-AUC=%.3f",
                    epoch, args.epochs, tr, vloss, auc, ap)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)  # <-- reload the BEST epoch before anything else
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "scorer.pt"))
    return model


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def group_report(oof: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for k, g in oof.groupby(col):
        m = summarize_metrics(g["eval_label"], g["prob"])
        m["prior_roc_auc"] = safe(roc_auc_score, g["eval_label"], g["prior_prob"])
        m[col] = k
        rows.append(m)
    return rows


def bootstrap_report(oof: pd.DataFrame, B: int, seed: int = SEED) -> dict:
    """Problem-level bootstrap (resample whole problems, not spans)."""
    oof = oof.reset_index(drop=True)
    y, ps, pp = oof["eval_label"].values, oof["prob"].values, oof["prior_prob"].values
    groups = list(oof.groupby("uid").indices.values())
    rng = np.random.default_rng(seed)
    acc = {"scorer_roc_auc": [], "scorer_pr_auc": [], "prior_roc_auc": [], "roc_auc_gain_over_prior": []}
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        a = safe(roc_auc_score, y[idx], ps[idx])
        b = safe(roc_auc_score, y[idx], pp[idx])
        acc["scorer_roc_auc"].append(a)
        acc["scorer_pr_auc"].append(safe(average_precision_score, y[idx], ps[idx]))
        acc["prior_roc_auc"].append(b)
        acc["roc_auc_gain_over_prior"].append(a - b)
    return {k: {"mean": float(np.nanmean(v)),
                "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
            for k, v in acc.items()}


def print_report(rep: dict):
    print("\n" + "=" * 78)
    print("SCORER REPORT (out-of-fold; every problem scored by a model that never saw it)")
    print("=" * 78)
    o = rep["overall"]
    print(f"Overall  n={o['n']}  critical-rate={o['pos_rate']:.3f}  ROC-AUC={o['roc_auc']:.3f}  "
          f"PR-AUC={o['pr_auc']:.3f}  F1(critical)={o['f1_critical']:.3f}")
    for k, v in rep["bootstrap"].items():
        print(f"  {k:26s} {v['mean']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"\nWithin-category ROC-AUC (weighted): {rep['within_category_roc_auc']:.3f}   "
          f"(category prior is 0.500 by construction)")
    print("\nBy category:")
    for r in rep["by_category"]:
        print(f"  {r['span_category']:24s} n={r['n']:5d} crit-rate={r['pos_rate']:.3f} "
              f"AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} F1={r['f1_critical']:.3f}")
    print("\nBy dataset:")
    for r in rep["by_dataset"]:
        print(f"  {r['source_dataset']:24s} n={r['n']:5d} AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")
    print("\nPer-fold ROC-AUC:", [round(x, 3) for x in rep["per_fold_roc_auc"]])
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train DistilBERT span scorer (K-fold, problem-level)")
    ap.add_argument("--consensus-file", default=config.CONSENSUS_FILE)
    ap.add_argument("--output-dir", default=MODEL_OUTPUT_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of each fold's train problems used for validation")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--hard-labels", action="store_true", help="train on binary consensus instead of soft score")
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--holdout-model", default=None,
                    help="train on consensus of the OTHER labeling models, evaluate vs this model's labels")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    _set_seed(SEED)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    logger.info("Device: %s", device)

    tag = "" if not args.holdout_model else "_holdout_" + args.holdout_model.replace("/", "_")
    out_root = args.output_dir + tag
    oof_file = os.path.join(config.DATA_DIR, f"scorer_oof_predictions{tag}.csv")
    report_file = os.path.join(config.DATA_DIR, f"scorer_report{tag}.json")

    if not os.path.exists(args.consensus_file):
        logger.error("Consensus file not found: %s", args.consensus_file)
        sys.exit(1)
    raw = pd.read_csv(args.consensus_file).fillna("")
    span_df = build_span_table(raw, args.holdout_model)
    if args.hard_labels:
        span_df["target"] = span_df["label"].astype(float)
    logger.info("%d unique spans over %d problems", len(span_df), span_df["uid"].nunique())
    logger.info("Label distribution (train-side):\n%s", span_df["label"].value_counts().to_string())
    logger.info("Critical rate by category:\n%s",
                span_df.groupby("span_category")["label"].agg(["mean", "count"]).round(3).to_string())

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    uids = np.array(sorted(span_df["uid"].unique()))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    fold_of, oof_parts = {}, []
    for fold, (rest_idx, test_idx) in enumerate(kf.split(uids)):
        test_uids = set(uids[test_idx])
        tr_uids, val_uids = train_test_split(uids[rest_idx], test_size=args.val_frac, random_state=SEED)
        for u in test_uids:
            fold_of[u] = fold
        logger.info("=== Fold %d/%d: %d train / %d val / %d test problems ===",
                    fold + 1, args.folds, len(tr_uids), len(val_uids), len(test_uids))

        train_df = span_df[span_df["uid"].isin(set(tr_uids))].reset_index(drop=True)
        val_df = span_df[span_df["uid"].isin(set(val_uids))].reset_index(drop=True)
        test_df = span_df[span_df["uid"].isin(test_uids)].reset_index(drop=True)

        model = train_one(train_df, val_df, tokenizer, args, device,
                          os.path.join(out_root, f"fold{fold}"))

        # test set is touched exactly once, with the best-val checkpoint
        test_loader = DataLoader(build_items(test_df, tokenizer, args.max_len),
                                 batch_size=args.batch_size, collate_fn=collate)
        rows, probs, _ = predict(model, test_loader, device)
        part = test_df.loc[rows].copy()
        part["prob"] = probs
        part["fold"] = fold
        prior = train_df.groupby("span_category")["label"].mean()
        part["prior_prob"] = part["span_category"].map(prior).fillna(train_df["label"].mean())
        oof_parts.append(part)
        logger.info("  fold %d test ROC-AUC=%.3f", fold,
                    safe(roc_auc_score, part["eval_label"], part["prob"]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(oof_parts, ignore_index=True)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "split.json"), "w") as f:
        json.dump({"n_folds": args.folds, "holdout_model": args.holdout_model, "folds": fold_of}, f)
    oof.drop(columns=["original_prompt"]).to_csv(oof_file, index=False)

    by_cat = group_report(oof, "span_category")
    valid_cat = [r for r in by_cat if np.isfinite(r["roc_auc"])]
    within = (sum(r["roc_auc"] * r["n"] for r in valid_cat) / max(sum(r["n"] for r in valid_cat), 1)
              if valid_cat else float("nan"))
    report = {
        "holdout_model": args.holdout_model,
        "overall": summarize_metrics(oof["eval_label"], oof["prob"]),
        "prior_baseline_roc_auc": safe(roc_auc_score, oof["eval_label"], oof["prior_prob"]),
        "bootstrap": bootstrap_report(oof, args.bootstrap),
        "within_category_roc_auc": within,
        "by_category": by_cat,
        "by_dataset": group_report(oof, "source_dataset"),
        "per_fold_roc_auc": [safe(roc_auc_score, g["eval_label"], g["prob"]) for _, g in oof.groupby("fold")],
    }
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    logger.info("Saved: %s | %s | %s", oof_file, report_file, os.path.join(out_root, "split.json"))


if __name__ == "__main__":
    main()