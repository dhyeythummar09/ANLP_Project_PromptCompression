"""
Intrinsic (reference-free, no-LLM-call) metrics for compressed prompts.

Use this when you don't have time/budget to run `evaluate.py` (which queries a
target LLM) for every method. These metrics are computed directly from
`compressed_prompts.csv` and are standard in the compression literature
(LLMLingua/LLMLingua-2 papers report perplexity and ratio-fidelity too), so
they're a legitimate thing to report -- just label them clearly as INTRINSIC
quality metrics, not downstream task accuracy. They tell you "is the compressed
text coherent and does it keep the labeled-critical content", not "does the
LLM still get the right answer".

Metrics computed, per (method, ratio[, source_dataset]):
  1. critical_span_retention   - fraction of counterfactually-critical spans
                                  (from your consensus labels) still present
                                  (pure string matching, already in your CSV
                                  if present, else recomputed here)
  2. category_retention        - same, broken out by span_category
  3. ratio_fidelity             - how close actual_ratio is to target_ratio
                                  (mean abs error); baselines sometimes miss
                                  their stated budget
  4. perplexity                - fluency of the compressed text under a small
                                  causal LM (distilgpt2 by default). High
                                  perplexity ~ garbled/incoherent text.
                                  Reported as perplexity itself AND as a ratio
                                  to the ORIGINAL prompt's perplexity, since
                                  absolute perplexity isn't comparable across
                                  different text lengths.
  5. rouge_l_vs_original       - longest-common-subsequence-based overlap
                                  between compressed and original text, a
                                  standard compression-fidelity metric.

Run:
    python intrinsic_metrics.py --input data/compressed_prompts.csv
    python intrinsic_metrics.py --input data/compressed_prompts.csv --methods ours llmlingua2 selective_context random
    python intrinsic_metrics.py --input data/compressed_prompts.csv --sample 5 --dump-samples samples.txt

Outputs:
    data/intrinsic_metrics_summary.csv   (one row per method x ratio)
    data/intrinsic_metrics_by_dataset.csv (one row per method x ratio x dataset, if source_dataset present)
"""

from __future__ import annotations

import os
import re
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("intrinsic_metrics")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
class Perplexity:
    def __init__(self, model_name: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, text: str) -> float:
        text = text.strip()
        if not text:
            return float("nan")
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=1000).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan")
        loss = self.lm(ids, labels=ids).loss
        return float(torch.exp(loss).item())


# ---------------------------------------------------------------------------
# ROUGE-L (LCS-based F1), no external dependency
# ---------------------------------------------------------------------------
def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(compressed: str, original: str) -> float:
    c, o = compressed.split(), original.split()
    if not c or not o:
        return 0.0
    lcs = _lcs_len(c, o)
    p = lcs / len(c)
    r = lcs / len(o)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Critical-span / category retention (recomputed if not already in the CSV)
# ---------------------------------------------------------------------------
def retention_for_row(compressed: str, spans: pd.DataFrame) -> tuple:
    comp = compressed.lower()
    cat_kept, cat_tot = {}, {}
    crit_total = crit_kept = 0
    for r in spans.itertuples():
        t = str(r.span_text).strip().lower()
        if not t:
            continue
        kept = re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", comp) is not None
        cat_tot[r.span_category] = cat_tot.get(r.span_category, 0) + 1
        cat_kept[r.span_category] = cat_kept.get(r.span_category, 0) + int(kept)
        if int(r.label) == 1:
            crit_total += 1
            crit_kept += int(kept)
    return crit_kept, crit_total, cat_kept, cat_tot


def load_label_spans(labels_file: str) -> dict:
    # intrinsic_metrics.py lives in src/compressor/, same as compress.py;
    # train_scorer.py lives in src/scorer/ -- go up to the project root first,
    # same pattern compress.py already uses for this import.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.scorer.train_scorer import build_span_table  # reuse the exact same logic as training/compression
    tbl = build_span_table(pd.read_csv(labels_file).fillna(""))
    return {u: g for u, g in tbl.groupby("uid")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Intrinsic (no-LLM-call) metrics on compressed prompts")
    ap.add_argument("--input", default="data/compressed_prompts.csv")
    ap.add_argument("--labels-file", default="data/consensus.csv",
                    help="consensus CSV for span retention; pass '' to skip if compressed_prompts.csv "
                         "already has crit_total/crit_kept/cat_retention_json columns")
    ap.add_argument("--output-summary", default="data/intrinsic_metrics_summary.csv")
    ap.add_argument("--output-by-dataset", default="data/intrinsic_metrics_by_dataset.csv")
    ap.add_argument("--methods", nargs="+", default=None, help="restrict to these methods (default: all present)")
    ap.add_argument("--ppl-model", default="distilgpt2")
    ap.add_argument("--sample", type=int, default=None, help="use only first N rows per method/ratio (fast preview)")
    ap.add_argument("--dump-samples", default=None,
                    help="write K qualitative examples per method/ratio to this text file for eyeballing")
    ap.add_argument("--dump-k", type=int, default=3)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    df = pd.read_csv(args.input).fillna("")
    if args.methods:
        df = df[df["method"].isin(args.methods)]
    if "source_dataset" not in df.columns:
        df["source_dataset"] = ""
    logger.info("Loaded %d rows. Methods present: %s", len(df), sorted(df["method"].unique()))

    need_retention = not {"crit_total", "crit_kept", "cat_retention_json"}.issubset(df.columns) or df["crit_total"].isna().all()
    label_spans = {}
    if need_retention:
        if not args.labels_file or not os.path.exists(args.labels_file):
            logger.warning("No retention columns in CSV and no --labels-file given/found; "
                           "critical-span retention metrics will be skipped.")
        else:
            logger.info("Recomputing span retention from %s ...", args.labels_file)
            label_spans = load_label_spans(args.labels_file)

    logger.info("Loading perplexity model: %s", args.ppl_model)
    ppl = Perplexity(args.ppl_model, device)

    # cache original-prompt perplexity per uid (expensive to repeat per method)
    orig_ppl_cache: dict = {}

    rows = []
    groups = df.groupby(["method", "target_ratio"])
    for gi, ((method, ratio), g) in enumerate(groups, 1):
        if args.sample:
            g = g.head(args.sample)
        logger.info("[%d/%d] %s @ ratio=%s (%d rows)", gi, len(groups), method, ratio, len(g))

        crit_kept_sum = crit_tot_sum = 0
        cat_kept_sum: dict = {}
        cat_tot_sum: dict = {}
        ratio_err, comp_ppl_list, ppl_ratio_list, rouge_list = [], [], [], []

        for r in g.itertuples():
            uid = r.uid
            comp_text = str(r.compressed_prompt)
            orig_text = str(r.original_prompt)

            # --- retention ---
            if label_spans and uid in label_spans:
                ck, ct, catk, catt = retention_for_row(comp_text, label_spans[uid])
                crit_kept_sum += ck
                crit_tot_sum += ct
                for k, v in catk.items():
                    cat_kept_sum[k] = cat_kept_sum.get(k, 0) + v
                for k, v in catt.items():
                    cat_tot_sum[k] = cat_tot_sum.get(k, 0) + v
            elif "crit_total" in df.columns and pd.notna(getattr(r, "crit_total", np.nan)):
                crit_kept_sum += float(r.crit_kept)
                crit_tot_sum += float(r.crit_total)

            # --- ratio fidelity ---
            target = float(ratio)
            actual = float(getattr(r, "actual_ratio", np.nan))
            if np.isfinite(actual):
                ratio_err.append(abs(actual - target))

            # --- perplexity ---
            cp = ppl(comp_text)
            comp_ppl_list.append(cp)
            if uid not in orig_ppl_cache:
                orig_ppl_cache[uid] = ppl(orig_text)
            op = orig_ppl_cache[uid]
            if np.isfinite(cp) and np.isfinite(op) and op > 0:
                ppl_ratio_list.append(cp / op)

            # --- rouge-L vs original ---
            rouge_list.append(rouge_l_f1(comp_text, orig_text))

        rows.append({
            "method": method, "target_ratio": ratio, "n": len(g),
            "critical_span_retention": round(crit_kept_sum / crit_tot_sum, 4) if crit_tot_sum else np.nan,
            "mean_abs_ratio_error": round(float(np.nanmean(ratio_err)), 4) if ratio_err else np.nan,
            "mean_perplexity": round(float(np.nanmean(comp_ppl_list)), 2),
            "median_perplexity": round(float(np.nanmedian(comp_ppl_list)), 2),
            "mean_perplexity_ratio_vs_original": round(float(np.nanmean(ppl_ratio_list)), 3) if ppl_ratio_list else np.nan,
            "mean_rouge_l_vs_original": round(float(np.nanmean(rouge_list)), 4),
        })
        for cat in cat_tot_sum:
            rows[-1][f"retention__{cat}"] = round(cat_kept_sum.get(cat, 0) / cat_tot_sum[cat], 4)

    summary = pd.DataFrame(rows).sort_values(["target_ratio", "method"])
    summary.to_csv(args.output_summary, index=False)
    print("\n" + "=" * 100)
    print("INTRINSIC METRICS SUMMARY (no LLM-answer calls -- fluency/retention/ratio-fidelity only)")
    print("=" * 100)
    base_cols = ["method", "target_ratio", "n", "critical_span_retention", "mean_abs_ratio_error",
                "mean_perplexity", "mean_perplexity_ratio_vs_original", "mean_rouge_l_vs_original"]
    print(summary[[c for c in base_cols if c in summary.columns]].to_string(index=False))
    logger.info("Saved %s", args.output_summary)

    # --- by-dataset breakdown, if source_dataset is populated ---
    if df["source_dataset"].astype(str).str.len().gt(0).any():
        by_ds_rows = []
        for (method, ratio, ds), g in df.groupby(["method", "target_ratio", "source_dataset"]):
            if not ds:
                continue
            if args.sample:
                g = g.head(args.sample)
            comp_ppl = [ppl(str(t)) for t in g["compressed_prompt"]]
            rouge = [rouge_l_f1(str(c), str(o)) for c, o in zip(g["compressed_prompt"], g["original_prompt"])]
            by_ds_rows.append({
                "method": method, "target_ratio": ratio, "source_dataset": ds, "n": len(g),
                "mean_perplexity": round(float(np.nanmean(comp_ppl)), 2),
                "mean_rouge_l_vs_original": round(float(np.nanmean(rouge)), 4),
            })
        pd.DataFrame(by_ds_rows).sort_values(["source_dataset", "target_ratio", "method"]).to_csv(
            args.output_by_dataset, index=False)
        logger.info("Saved %s", args.output_by_dataset)

    # --- qualitative dump ---
    if args.dump_samples:
        with open(args.dump_samples, "w") as f:
            for (method, ratio), g in df.groupby(["method", "target_ratio"]):
                f.write(f"\n{'=' * 80}\n{method} @ ratio={ratio}\n{'=' * 80}\n")
                for r in g.head(args.dump_k).itertuples():
                    f.write(f"\nORIGINAL:   {r.original_prompt}\nCOMPRESSED: {r.compressed_prompt}\n")
        logger.info("Saved qualitative samples to %s", args.dump_samples)


if __name__ == "__main__":
    main()