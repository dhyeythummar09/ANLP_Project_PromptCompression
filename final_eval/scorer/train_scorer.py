"""Train DistilBERT span sensitivity scores with base-family-safe folds.

Soft vote fractions and unweighted BCE are defaults. The data loader is shared
with diagnostics; held-out-labeler targets exclude that labeler's votes.
--dry-run validates data/partitions without loading weights or writing outputs.
--resume reuses complete folds only when source/data/environment metadata match.
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
from final_eval import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT_DIR = "models/span_scorer_v2"
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
from final_eval.scorer.data import build_span_table
from final_eval.preparation.schema import attach_groups, grouped_folds
from final_eval.utils.artifacts import atomic_json, file_hash, source_fingerprint, package_versions, PROTOCOL_VERSION


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
            if m.any() and offsets[m, 0].min() <= r.span_start and offsets[m, 1].max() >= r.span_end:
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

    def __init__(self, encoder_name: str = MODEL_NAME, dropout: float = 0.1, encoder_config=None):
        super().__init__()
        self.encoder = DistilBertModel(encoder_config) if encoder_config is not None else DistilBertModel.from_pretrained(encoder_name)
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
        "brier_hard_eval": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
    }


def train_one(train_df, val_df, tokenizer, args, device, out_dir) -> SpanScorer:
    train_items = build_items(train_df, tokenizer, args.max_len)
    val_items = build_items(val_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_items, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_items, batch_size=args.batch_size, collate_fn=collate)

    pos_weight = None
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - pos)
    if args.pos_weight and pos > 0:
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
    model.encoder.config.save_pretrained(out_dir)
    atomic_json(os.path.join(out_dir, "selection.json"), {"best_validation_ap_or_negative_loss": best_score})
    temporary_checkpoint = os.path.join(out_dir, "scorer.pt.tmp")
    torch.save(best_state, temporary_checkpoint)
    os.replace(temporary_checkpoint, os.path.join(out_dir, "scorer.pt"))
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
    groups = list(oof.groupby("base_group_id").indices.values())
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
          f"(category prior has no within-category context features)")
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
    ap.add_argument("--consensus-file", default=settings.CONSENSUS_FILE)
    ap.add_argument("--output-dir", default=MODEL_OUTPUT_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of each fold's train problems used for validation")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--hard-labels", action="store_true", help="train on binary consensus instead of soft score")
    ap.add_argument("--no-pos-weight", action="store_true", help="Legacy alias; unweighted loss is now default")
    ap.add_argument("--pos-weight", action="store_true", help="Named imbalance-weighted ablation; probabilities are not calibrated")
    ap.add_argument("--problems-file", default=None, help="Needed to reconstruct base groups from an unprepared legacy consensus")
    ap.add_argument("--dry-run", action="store_true", help="Validate targets and partitions without model initialization or writes")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true", help="Reuse completed fold checkpoints only if training metadata matches")
    ap.add_argument("--holdout-model", default=None,
                    help="train on consensus of the OTHER labeling models, evaluate vs this model's labels")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.bootstrap < 1 or args.max_len < 8:
        ap.error("Positive epochs, batch size, learning rate, bootstrap count and max_len >= 8 required")
    if args.pos_weight and args.no_pos_weight:
        ap.error("Choose either --pos-weight or --no-pos-weight")
    _set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    logger.info("Device: %s", device)

    tag = "" if not args.holdout_model else "_holdout_" + args.holdout_model.replace("/", "_")
    out_root = args.output_dir + tag
    oof_file = os.path.join(out_root, "scorer_oof_predictions.csv")
    report_file = os.path.join(out_root, "scorer_report.json")

    if not os.path.exists(args.consensus_file):
        logger.error("Consensus file not found: %s", args.consensus_file)
        sys.exit(1)
    raw = pd.read_csv(args.consensus_file).fillna("")
    span_df = build_span_table(raw, args.holdout_model)
    problems = pd.read_csv(args.problems_file, keep_default_na=False) if args.problems_file else None
    span_df = attach_groups(span_df, problems)
    if args.hard_labels:
        span_df["target"] = span_df["label"].astype(float)
    logger.info("%d unique spans over %d problems", len(span_df), span_df["uid"].nunique())
    logger.info("Label distribution (train-side):\n%s", span_df["label"].value_counts().to_string())
    logger.info("Critical rate by category:\n%s",
                span_df.groupby("span_category")["label"].agg(["mean", "count"]).round(3).to_string())

    partitions = list(grouped_folds(span_df, args.folds, args.val_frac, args.seed))
    if args.dry_run:
        print(json.dumps({"spans": len(span_df), "questions": span_df.uid.nunique(),
                          "base_groups": span_df.base_group_id.nunique(),
                          "target_values": sorted(span_df.target.unique().tolist()),
                          "fold_sizes": [{k: len(v) for k, v in roles.items()} for _, roles in partitions]}, indent=2))
        return
    metadata = {"protocol": PROTOCOL_VERSION, "encoder": MODEL_NAME,
                "source": source_fingerprint(), "packages": package_versions(),
                "target_kind": "hard" if args.hard_labels else "soft",
                "loss": "weighted_bce" if args.pos_weight else "unweighted_bce",
                "consensus_sha256": file_hash(args.consensus_file),
                "problems_sha256": file_hash(args.problems_file) if args.problems_file else None,
                "arguments": {k: v for k, v in vars(args).items() if k not in {"resume", "dry_run"}},
                "group_count": span_df.base_group_id.nunique()}
    if os.path.exists(out_root):
        meta_path = os.path.join(out_root, "training_metadata.json")
        if not args.resume or not os.path.exists(meta_path) or json.load(open(meta_path)) != metadata:
            raise FileExistsError(f"Use a new output directory, or --resume with identical configuration: {out_root}")
    else:
        os.makedirs(out_root)
        atomic_json(os.path.join(out_root, "training_metadata.json"), metadata)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(os.path.join(out_root, "tokenizer"))
    fold_of, oof_parts = {}, []
    for fold, roles in partitions:
        _set_seed(args.seed + fold)
        test_uids = roles["test"]
        tr_uids, val_uids = roles["train"], roles["val"]
        for u in test_uids:
            fold_of[u] = fold
        logger.info("=== Fold %d/%d: %d train / %d val / %d test problems ===",
                    fold + 1, args.folds, len(tr_uids), len(val_uids), len(test_uids))

        train_df = span_df[span_df["uid"].isin(set(tr_uids))].reset_index(drop=True)
        val_df = span_df[span_df["uid"].isin(set(val_uids))].reset_index(drop=True)
        test_df = span_df[span_df["uid"].isin(test_uids)].reset_index(drop=True)

        fold_dir = os.path.join(out_root, f"fold{fold}")
        if args.resume and os.path.exists(os.path.join(fold_dir, "scorer.pt")):
            from transformers import DistilBertConfig
            model = SpanScorer(encoder_config=DistilBertConfig.from_pretrained(fold_dir)).to(device)
            model.load_state_dict(torch.load(os.path.join(fold_dir, "scorer.pt"), map_location=device, weights_only=True))
        else:
            model = train_one(train_df, val_df, tokenizer, args, device, fold_dir)

        # test set is touched exactly once, with the best-val checkpoint
        test_loader = DataLoader(build_items(test_df, tokenizer, args.max_len),
                                 batch_size=args.batch_size, collate_fn=collate)
        rows, probs, _ = predict(model, test_loader, device)
        part = test_df.loc[rows].copy()
        part["prob"] = probs
        part["fold"] = fold
        prior = train_df.groupby("span_category")["target"].mean()
        part["prior_prob"] = part["span_category"].map(prior).fillna(train_df["target"].mean())
        atomic_json(os.path.join(out_root, f"fold{fold}", "category_prior.json"), prior.to_dict())
        oof_parts.append(part)
        logger.info("  fold %d test ROC-AUC=%.3f", fold,
                    safe(roc_auc_score, part["eval_label"], part["prob"]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(oof_parts, ignore_index=True)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "split.json"), "w") as f:
        json.dump({"protocol": PROTOCOL_VERSION, "n_folds": args.folds, "holdout_model": args.holdout_model, "folds": fold_of, "base_groups": dict(zip(span_df.uid, span_df.base_group_id)), "partitions": {str(k): {role: sorted(values) for role, values in roles.items()} for k, roles in partitions}}, f, indent=2)
    oof["brier_soft"] = (oof["prob"] - oof["target"]) ** 2
    oof.drop(columns=["original_prompt"]).to_csv(oof_file, index=False)

    by_cat = group_report(oof, "span_category")
    valid_cat = [r for r in by_cat if np.isfinite(r["roc_auc"])]
    within = (sum(r["roc_auc"] * r["n"] for r in valid_cat) / max(sum(r["n"] for r in valid_cat), 1)
              if valid_cat else float("nan"))
    report = {
        "holdout_model": args.holdout_model,
        "overall": summarize_metrics(oof["eval_label"], oof["prob"]),
        "prior_baseline_roc_auc": safe(roc_auc_score, oof["eval_label"], oof["prior_prob"]),
        "bootstrap": bootstrap_report(oof, args.bootstrap, args.seed),
        "within_category_roc_auc": within,
        "by_category": by_cat,
        "by_dataset": group_report(oof, "source_dataset"),
        "per_fold_roc_auc": [safe(roc_auc_score, g["eval_label"], g["prob"]) for _, g in oof.groupby("fold")],
    }
    report["mean_brier_soft"] = float(oof["brier_soft"].mean())
    atomic_json(report_file, report)
    print_report(report)
    logger.info("Saved: %s | %s | %s", oof_file, report_file, os.path.join(out_root, "split.json"))


if __name__ == "__main__":
    main()