"""Contextual word retention with an independently supervised span head."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
from torch import nn


class DenseStudent(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.keep_head = nn.Linear(hidden, 1)
        self.span_head = nn.Linear(hidden, 1)

    @staticmethod
    def pool(hidden, masks):
        return torch.bmm(masks, hidden) / masks.sum(-1, keepdim=True).clamp_min(1)

    def forward(self, input_ids, attention_mask, word_masks, span_masks):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        keep = self.keep_head(self.dropout(self.pool(hidden, word_masks))).squeeze(-1)
        spans = self.span_head(self.dropout(self.pool(hidden, span_masks))).squeeze(-1)
        return keep, spans


def encode_question(question, tokenizer, labels=None, max_length=512, stride=64):
    """Overflow windows; every fully covered word/span is owned exactly once.

    Pick the window offering the most balanced context, never silently train on
    a truncated span. Uncoverable words are fatal at inference, masked in training.
    """
    from final_eval.compressor.budget import words
    text = question["original_prompt"]
    units = words(text)
    if labels is not None and len(labels) != len(units):
        raise ValueError("Teacher word count changed")
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.num_special_tokens_to_add() != 2:
        raise ValueError("This DistilBERT experiment requires a CLS/SEP tokenizer")
    if not 0 <= stride < max_length - 2:
        raise ValueError("Invalid overflow stride")
    # Tokenize without truncation, THEN create windows. Some fast-tokenizer
    # versions pre-truncate before producing overflow windows; relying on them
    # can silently omit most of a long question.
    full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    capacity = max_length - 2
    ids_windows, offsets = [], []
    start = 0
    while start < len(full["input_ids"]):
        end = min(start + capacity, len(full["input_ids"]))
        ids_windows.append([tokenizer.cls_token_id] + full["input_ids"][start:end] + [tokenizer.sep_token_id])
        offsets.append(np.asarray([(0, 0)] + full["offset_mapping"][start:end] + [(0, 0)]))
        if end == len(full["input_ids"]):
            break
        start = end - stride
    if not ids_windows:
        raise ValueError("Question has no encodable source tokens")

    def owners(ranges):
        result = []
        for a, b in ranges:
            # Annotation spans may include leading/trailing whitespace (notably
            # random controls). Whitespace has no encoder token; it must not make
            # an otherwise fully covered label look truncated.
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            options = []
            for k, off in enumerate(offsets):
                mask = (off[:, 1] > off[:, 0]) & (off[:, 0] < b) & (off[:, 1] > a)
                if mask.any() and off[mask, 0].min() <= a and off[mask, 1].max() >= b:
                    indices = np.flatnonzero(mask)
                    real = np.flatnonzero(off[:, 1] > off[:, 0])
                    context = min(indices[0] - real[0], real[-1] - indices[-1])
                    options.append((context, -k, k, mask))
            result.append(max(options, key=lambda x: x[:2])[2:] if options else None)
        return result

    word_owners = owners([(a, b) for a, b, _ in units])
    spans = question.get("spans", [])
    span_owners = owners([(s["start"], s["end"]) for s in spans])
    windows = []
    for k, ids in enumerate(ids_windows):
        ww = [(i, v[1]) for i, v in enumerate(word_owners) if v is not None and v[0] == k]
        ss = [(i, v[1]) for i, v in enumerate(span_owners) if v is not None and v[0] == k]
        windows.append({"input_ids": ids, "word_indices": [i for i, _ in ww],
            "word_masks": [m.tolist() for _, m in ww],
            "teacher": [labels[i] if labels is not None else -1 for i, _ in ww],
            "span_masks": [m.tolist() for _, m in ss],
            "span_targets": [spans[i]["target"] for i, _ in ss]})
    return {"uid": question["uid"], "source": question["source_dataset"], "windows": windows,
            "n_words": len(units), "uncovered_words": [i for i, v in enumerate(word_owners) if v is None],
            "uncovered_spans": [i for i, v in enumerate(span_owners) if v is None]}


def collate_questions(items, pad_id=0):
    windows = [(q, w) for q, item in enumerate(items) for w in item["windows"]]
    length = max(len(w["input_ids"]) for _, w in windows)
    n_words = max(1, max(len(w["word_masks"]) for _, w in windows))
    n_spans = max(1, max(len(w["span_masks"]) for _, w in windows))
    b = len(windows)
    result = {"input_ids": torch.full((b, length), pad_id, dtype=torch.long),
              "attention_mask": torch.zeros(b, length, dtype=torch.long),
              "word_masks": torch.zeros(b, n_words, length),
              "span_masks": torch.zeros(b, n_spans, length),
              "teacher": torch.zeros(b, n_words), "teacher_valid": torch.zeros(b, n_words, dtype=torch.bool),
              "span_targets": torch.zeros(b, n_spans), "span_valid": torch.zeros(b, n_spans, dtype=torch.bool),
              "question_index": torch.tensor([q for q, _ in windows]), "n_questions": len(items),
              "word_indices": [w["word_indices"] for _, w in windows]}
    for i, (_, w) in enumerate(windows):
        n, nw, ns = len(w["input_ids"]), len(w["word_masks"]), len(w["span_masks"])
        result["input_ids"][i, :n] = torch.tensor(w["input_ids"])
        result["attention_mask"][i, :n] = 1
        if nw:
            result["word_masks"][i, :nw, :n] = torch.tensor(w["word_masks"], dtype=torch.float32)
            result["teacher"][i, :nw] = torch.tensor(w["teacher"])
            result["teacher_valid"][i, :nw] = torch.tensor(w["teacher"]) >= 0
        if ns:
            result["span_masks"][i, :ns, :n] = torch.tensor(w["span_masks"], dtype=torch.float32)
            result["span_targets"][i, :ns] = torch.tensor(w["span_targets"])
            result["span_valid"][i, :ns] = True
    return result


def question_loss(logits, targets, valid, question_index, n_questions):
    """Equal weight per labelled question, independent of word/window count."""
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets.clamp(0, 1), reduction="none")
    sums = torch.zeros(n_questions, device=logits.device).scatter_add_(0, question_index, (losses * valid).sum(-1))
    counts = torch.zeros(n_questions, device=logits.device).scatter_add_(0, question_index, valid.sum(-1).float())
    active = counts > 0
    return (sums / counts.clamp_min(1))[active].mean() if active.any() else logits.sum() * 0


def ranking_loss(logits, targets, valid, question_index, max_pairs=256):
    """Optional within-question ordering, weighted by target separation; default off."""
    terms = []
    for q in question_index.unique():
        mask = (question_index == q)[:, None] & valid
        p, t = logits[mask], targets[mask]
        if p.numel() < 2:
            continue
        difference = t[:, None] - t[None, :]
        a, b = torch.where(difference > 0)
        if len(a):
            # Fixed lexical order cap; never creates arbitrary labels for tied targets.
            a, b = a[:max_pairs], b[:max_pairs]
            weight = difference[a, b]
            terms.append((nn.functional.softplus(-(p[a] - p[b])) * weight).sum() / weight.sum())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


@torch.inference_mode()
def predict_words(model, item, device):
    model.eval()
    if item["uncovered_words"]:
        raise ValueError(f"Tokenizer could not cover {len(item['uncovered_words'])} whole words: {item['uid']}")
    # One overflow window at a time bounds inference memory on long questions.
    output = np.full(item["n_words"], np.nan)
    for window in item["windows"]:
        batch = collate_questions([{**item, "windows": [window]}])
        args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "word_masks", "span_masks")]
        keep, _ = model(*args)
        output[window["word_indices"]] = torch.sigmoid(keep[0, :len(window["word_indices"])]).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Incomplete dense prediction")
    return output
