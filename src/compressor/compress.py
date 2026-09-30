"""
STEP 8 (revised): Budgeted compression with our scorer + baselines, all on the same problems.

What changed vs. the first version
  * FIXED TOKEN BUDGET (--ratios = fraction of tokens KEPT) instead of a score
    threshold. The proposal compares methods at fixed compression ratios; a
    threshold gives every prompt a different, uncontrolled ratio.
  * Deletion units are ALL words, ranked by priority = alpha*criticality +
    (1-alpha)*self-information rank. Words outside candidate spans get
    criticality 0, so they are deleted first (ordered by the info signal);
    critical spans are only deleted when the budget forces it.
    (Old version could only delete candidate spans, so it could not hit a ratio.)
  * The scorer is applied ONLY to held-out problems: each problem is compressed
    by the fold model that never saw it (uses split.json from train_scorer.py).
    The old default compressed all of raw_problems.csv, i.e. training problems too.
  * Scorer class is imported from train_scorer.py (no duplicate definition to drift).
  * Baselines in the same script, same budgets, same token counter:
      llmlingua, llmlingua2, selective_context, random (sanity floor)
  * Ablations of our method: ours_scorer (alpha=1), ours_signal (alpha=0), ours (alpha=0.5)
  * Per output row: actual ratio, latency, critical-span retention, per-category
    retention (needs the consensus/labels file).

Run:
    python src/compressor/compress.py                                   # everything
    python src/compressor/compress.py --methods ours random --ratios 0.5 0.33 0.25
    python src/compressor/compress.py --prompt "John has 5 apples..."   # quick demo

Output:  data/compressed_prompts.csv  (long format: one row per problem x method x ratio)
"""

import os
import re
import sys
import json
import time
import zlib
import bisect
import argparse
import logging

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import DistilBertTokenizerFast, AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.scorer.train_scorer import (
    SpanScorer, span_token_mask, build_span_table, make_uid,
    MODEL_NAME, MODEL_OUTPUT_DIR, MAX_SEQ_LEN,
)

COMPRESSED_FILE = os.path.join(config.DATA_DIR, "compressed_prompts.csv")
OURS_ALPHA = {"ours": 0.5, "ours_scorer": 1.0, "ours_signal": 0.0}
BASELINES = ["random", "llmlingua", "llmlingua2", "selective_context"]
DEFAULT_METHODS = list(OURS_ALPHA) + BASELINES
DEFAULT_RATIOS = [0.5, 0.33, 0.25]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compressor")


# ---------------------------------------------------------------------------
# Token counting (ONE counter for every method, so ratios are comparable)
# ---------------------------------------------------------------------------
_ENC = None


def count_tokens(s: str) -> int:
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False
    if _ENC:
        return len(_ENC.encode(s))
    return len(re.findall(r"\w+|[^\w\s]", s))


# ---------------------------------------------------------------------------
# Candidate span extraction
# NOTE: this MUST match the extraction used when the counterfactual labels were
# generated (your 02_extract_spans). If that file exposes a function, import it
# and use it here instead. The `span_extract_coverage` column in the output
# tells you how many labeled spans this extractor reproduces exactly.
# ---------------------------------------------------------------------------
def extract_spans(text: str, nlp) -> list[dict]:
    doc = nlp(text)
    spans = []
    for token in doc:
        cat = None
        if token.like_num or token.pos_ == "NUM":
            cat = "number"
        elif token.dep_ == "neg" or token.lower_ in {
            "not", "no", "never", "neither", "nor", "n't", "without",
            "nobody", "nothing", "nowhere", "none",
        }:
            cat = "negation"
        elif token.pos_ == "VERB" and token.dep_ not in {"aux", "auxpass"}:
            cat = "action_verb"
        elif token.pos_ == "PROPN" or token.ent_type_:
            cat = "entity"
        elif token.lower_ in {
            "than", "more", "less", "fewer", "if", "unless", "whether",
            "greater", "smaller", "most", "least", "better", "worse",
        }:
            cat = "comparison_conditional"
        if cat:
            spans.append({"span_text": token.text, "span_start": token.idx,
                          "span_end": token.idx + len(token.text), "span_category": cat})
    for ent in doc.ents:
        spans.append({"span_text": ent.text, "span_start": ent.start_char,
                      "span_end": ent.end_char, "span_category": "entity"})
    seen, unique = set(), []
    for s in spans:
        k = (s["span_start"], s["span_end"])
        if k not in seen:
            seen.add(k)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Scorer loading + scoring (one encoder pass per prompt, all spans pooled)
# ---------------------------------------------------------------------------
def load_scorer(fold_dir: str, device: torch.device):
    model = SpanScorer(MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(fold_dir, "scorer.pt"), map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def score_spans(text, spans, model, tokenizer, device, max_len=MAX_SEQ_LEN):
    """Adds `criticality_score` in [0,1]. Spans that fall outside max_len get 1.0 (kept)."""
    for s in spans:
        s["criticality_score"] = 1.0
    if not spans:
        return spans
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].numpy()
    masks = [span_token_mask(offsets, s["span_start"], s["span_end"]) for s in spans]
    keep = [i for i, m in enumerate(masks) if m.any()]
    if not keep:
        return spans
    m = torch.tensor(np.stack([masks[i] for i in keep]), dtype=torch.float32, device=device)[None]
    logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), m)[0]
    for i, p in zip(keep, torch.sigmoid(logits).cpu().tolist()):
        spans[i]["criticality_score"] = p
    return spans


# ---------------------------------------------------------------------------
# Standard importance signal: self-information from a small causal LM
# ---------------------------------------------------------------------------
class InfoScorer:
    def __init__(self, name: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def word_info(self, text: str, words: list[tuple]) -> np.ndarray:
        """Sum of token surprisals per whitespace word."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False,
                       truncation=True, max_length=1000)
        bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        x = torch.tensor([[bos] + enc["input_ids"]], device=self.device)
        logp = torch.log_softmax(self.lm(x).logits[0, :-1].float(), dim=-1)
        surprisal = -logp.gather(1, x[0, 1:, None])[:, 0].cpu().numpy()
        starts = [w[0] for w in words]
        info = np.zeros(len(words))
        for (s, e), v in zip(enc["offset_mapping"], surprisal):
            wi = bisect.bisect_right(starts, max(e - 1, s)) - 1
            info[min(max(wi, 0), len(words) - 1)] += v
        return info


def rank01(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(x, kind="stable")
    r = np.empty(n)
    r[order] = np.arange(n) / (n - 1)
    return r


# ---------------------------------------------------------------------------
# Budgeted deletion
# ---------------------------------------------------------------------------
def delete_to_budget(words: list[tuple], priority: np.ndarray, keep_ratio: float) -> str:
    """Delete lowest-priority words until <= keep_ratio of the original tokens remain."""
    toks = np.array([count_tokens(" " + w[2]) for w in words])
    budget = max(1, int(np.ceil(keep_ratio * toks.sum())))
    keep = np.ones(len(words), dtype=bool)
    total = toks.sum()
    for i in np.argsort(priority, kind="stable"):
        if total <= budget:
            break
        keep[i] = False
        total -= toks[i]
    return " ".join(w[2] for w, k in zip(words, keep) if k)


def prepare_ours(text, model, tokenizer, nlp, info, device) -> dict:
    """Everything our method needs for one prompt (shared across ratios/ablations)."""
    t0 = time.perf_counter()
    spans = score_spans(text, extract_spans(text, nlp), model, tokenizer, device)
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    crit = np.zeros(len(words))
    prio_bonus = np.zeros(len(words))
    
    for sp in spans:
        for i, (ws, we, _) in enumerate(words):
            if ws < sp["span_end"] and we > sp["span_start"]:
                crit[i] = max(crit[i], sp["criticality_score"])
                if sp.get("span_category") == "number":
                    prio_bonus[i] = max(prio_bonus[i], 0.4)
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "nummod":
            hs, he = token.head.idx, token.head.idx + len(token.head.text)
            for i, (ws, we, _) in enumerate(words):
                if ws < he and we > hs:
                    prio_bonus[i] = max(prio_bonus[i], 0.4)
                    
    target_sent = None
    for sent in reversed(list(doc.sents)):
        text_lower = sent.text.lower()
        if "?" in text_lower or any(wh in [t.lower_ for t in sent] for wh in ["how", "what", "where", "when", "who", "which", "why", "calculate", "find"]):
            target_sent = sent
            break
    if target_sent:
        hs, he = target_sent.start_char, target_sent.end_char
        for i, (ws, we, _) in enumerate(words):
            if ws < he and we > hs:
                prio_bonus[i] = max(prio_bonus[i], 0.2)
                
    # Airtight number fallback: any word containing a digit (e.g., $6, 7:11)
    for i, (ws, we, wtext) in enumerate(words):
        if any(char.isdigit() for char in wtext):
            prio_bonus[i] = max(prio_bonus[i], 0.4)
                
    info_rank = rank01(info.word_info(text, words)) if len(words) else np.zeros(0)
    return {"spans": spans, "words": words, "crit": crit, "info_rank": info_rank,
            "prio_bonus": prio_bonus, "prep_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------------------
# Baselines (lazy imports: only needed if you request them)
# ---------------------------------------------------------------------------
class BaselineRunner:
    def __init__(self, device: str, llmlingua_model: str, llmlingua2_model: str):
        self.device = device
        self.cfg = {"llmlingua": llmlingua_model, "llmlingua2": llmlingua2_model}
        self._obj = {}

    def run(self, method: str, text: str, keep_ratio: float) -> str:
        if method in ("llmlingua", "llmlingua2"):
            if method not in self._obj:
                from llmlingua import PromptCompressor
                self._obj[method] = PromptCompressor(
                    model_name=self.cfg[method], use_llmlingua2=(method == "llmlingua2"),
                    device_map=self.device)
            kw = {"force_tokens": ["\n", "?"]} if method == "llmlingua2" else {}
            out = self._obj[method].compress_prompt(text, rate=keep_ratio, **kw)  # rate = fraction KEPT
            return out["compressed_prompt"]
        if method == "selective_context":
            if method not in self._obj:
                from selective_context import SelectiveContext
                self._obj[method] = SelectiveContext(model_type="gpt2", lang="en")
            compressed, _ = self._obj[method](text, reduce_ratio=1.0 - keep_ratio,  # ratio REMOVED
                                              reduce_level="phrase")
            return compressed
        raise ValueError(method)


# ---------------------------------------------------------------------------
# Retention statistics against the counterfactual labels
# ---------------------------------------------------------------------------
def retention(compressed: str, spans: pd.DataFrame) -> dict:
    """
    Which labeled spans survive in the compressed text? String-containment check
    (word-boundary, case-insensitive) so it works identically for every method,
    including black-box baselines. Caveat: a span like "5" also counts as kept if
    another "5" survives elsewhere in the prompt.
    """
    comp = compressed.lower()
    cat = {}
    crit_total = crit_kept = 0
    for r in spans.itertuples():
        t = str(r.span_text).strip().lower()
        if not t:
            continue
        kept = re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", comp) is not None
        c = cat.setdefault(r.span_category, [0, 0])
        c[0] += int(kept)
        c[1] += 1
        if int(r.label) == 1:
            crit_total += 1
            crit_kept += int(kept)
    return {"crit_total": crit_total, "crit_kept": crit_kept,
            "all_critical_kept": int(crit_kept == crit_total) if crit_total else np.nan,
            "cat_retention_json": json.dumps(cat)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Budgeted prompt compression: ours + baselines")
    ap.add_argument("--model-dir", default=MODEL_OUTPUT_DIR, help="dir with fold*/scorer.pt and split.json")
    ap.add_argument("--input-csv", default=config.RAW_PROBLEMS_FILE)
    ap.add_argument("--labels-file", default=config.CONSENSUS_FILE,
                    help="consensus CSV, used only for retention stats (pass '' to skip)")
    ap.add_argument("--output-csv", default=COMPRESSED_FILE)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    ap.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS,
                    help="fraction of tokens KEPT (0.5 = 2x compression)")
    ap.add_argument("--alpha", type=float, default=None, help="override the scorer weight of `ours`")
    ap.add_argument("--info-model", default="distilgpt2")
    ap.add_argument("--llmlingua-model", default="microsoft/phi-2")
    ap.add_argument("--llmlingua2-model",
                    default="microsoft/llmlingua-2-xlm-roberta-large-meta-llama-3-8b-instruct-mbert")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", type=str, default=None, help="demo: compress one prompt with the fold-0 scorer")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    alpha = dict(OURS_ALPHA)
    if args.alpha is not None:
        alpha["ours"] = args.alpha
    nlp = spacy.load("en_core_web_sm")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    info = InfoScorer(args.info_model, device)

    # ---- single-prompt demo ----
    if args.prompt:
        model = load_scorer(os.path.join(args.model_dir, "fold0"), device)
        prep = prepare_ours(args.prompt, model, tokenizer, nlp, info, device)
        rng = np.random.default_rng(0)
        for r in args.ratios:
            prio = alpha["ours"] * prep["crit"] + (1 - alpha["ours"]) * prep["info_rank"] + 1e-6 * rng.random(len(prep["words"]))
            out = delete_to_budget(prep["words"], prio, r)
            print(f"\n[keep={r}] {count_tokens(args.prompt)} -> {count_tokens(out)} tokens\n{out}")
        return

    # ---- held-out problems only ----
    split_path = os.path.join(args.model_dir, "split.json")
    if not os.path.exists(split_path):
        logger.error("%s not found. Run train_scorer.py first.", split_path)
        sys.exit(1)
    fold_of = json.load(open(split_path))["folds"]

    probs = pd.read_csv(args.input_csv).fillna("")
    if "source_dataset" not in probs.columns:
        probs["source_dataset"] = ""
    probs["uid"] = [make_uid(s, e) for s, e in zip(probs["source_dataset"], probs["example_id"])]
    probs = probs[probs["uid"].isin(fold_of)].copy()
    probs["fold"] = probs["uid"].map(fold_of)
    if args.limit:
        probs = probs.head(args.limit)
    logger.info("Compressing %d held-out problems | methods=%s | keep-ratios=%s",
                len(probs), args.methods, args.ratios)

    label_spans = {}
    if args.labels_file and os.path.exists(args.labels_file):
        tbl = build_span_table(pd.read_csv(args.labels_file).fillna(""))
        label_spans = {u: g for u, g in tbl.groupby("uid")}

    done = set()
    if os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        done = set(zip(prev["uid"], prev["method"], prev["target_ratio"].round(4)))
    baselines = BaselineRunner(str(device), args.llmlingua_model, args.llmlingua2_model)
    failed = set()

    n_done = 0
    for fold in sorted(probs["fold"].unique()):
        model = None
        if any(m in OURS_ALPHA for m in args.methods):
            model = load_scorer(os.path.join(args.model_dir, f"fold{fold}"), device)
        for row in probs[probs["fold"] == fold].itertuples():
            text, uid = str(row.original_prompt), row.uid
            rng = np.random.default_rng(zlib.crc32(uid.encode()))
            prep, out_rows = None, []
            orig_tokens = count_tokens(text)
            for method in args.methods:
                if method in failed:
                    continue
                for ratio in args.ratios:
                    if (uid, method, round(ratio, 4)) in done:
                        continue
                    try:
                        t0 = time.perf_counter()
                        if method in OURS_ALPHA:
                            if prep is None:
                                prep = prepare_ours(text, model, tokenizer, nlp, info, device)
                            a = alpha[method]
                            prio = a * prep["crit"] + (1 - a) * prep["info_rank"] + 1e-6 * rng.random(len(prep["words"]))
                            prio += prep["prio_bonus"]
                            comp = delete_to_budget(prep["words"], prio, ratio)
                            latency = prep["prep_ms"] + (time.perf_counter() - t0) * 1000
                        elif method == "random":
                            words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
                            comp = delete_to_budget(words, rng.random(len(words)), ratio)
                            latency = (time.perf_counter() - t0) * 1000
                        else:
                            comp = baselines.run(method, text, ratio)
                            latency = (time.perf_counter() - t0) * 1000
                    except Exception as ex:  # missing package / OOM / API change
                        logger.error("Method %s failed (%s: %s). Skipping it for the rest of the run.",
                                     method, type(ex).__name__, ex)
                        failed.add(method)
                        break

                    comp_tokens = count_tokens(comp)
                    rec = {
                        "uid": uid, "example_id": row.example_id, "source_dataset": row.source_dataset,
                        "fold": fold, "method": method, "target_ratio": round(ratio, 4),
                        "original_prompt": text, "compressed_prompt": comp,
                        "ground_truth_answer": getattr(row, "ground_truth_answer", ""),
                        "orig_tokens": orig_tokens, "comp_tokens": comp_tokens,
                        "actual_ratio": round(comp_tokens / max(orig_tokens, 1), 4),
                        "latency_ms": round(latency, 2),
                        # defaults keep the CSV columns identical across appended chunks
                        "crit_total": np.nan, "crit_kept": np.nan, "all_critical_kept": np.nan,
                        "cat_retention_json": "{}", "span_extract_coverage": np.nan,
                    }
                    ls = label_spans.get(uid)
                    if ls is not None:
                        rec.update(retention(comp, ls))
                    out_rows.append(rec)

            if prep is not None and uid in label_spans:  # extractor vs labeled spans
                lab = set(zip(label_spans[uid]["span_start"], label_spans[uid]["span_end"]))
                ext = {(s["span_start"], s["span_end"]) for s in prep["spans"]}
                cov = len(lab & ext) / max(len(lab), 1)
                for r in out_rows:
                    r["span_extract_coverage"] = round(cov, 3)

            if out_rows:
                pd.DataFrame(out_rows).to_csv(args.output_csv, mode="a",
                                              header=not os.path.exists(args.output_csv), index=False)
            n_done += 1
            if n_done % 25 == 0:
                logger.info("  %d / %d problems", n_done, len(probs))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.read_csv(args.output_csv)
    logger.info("Saved %d rows -> %s", len(out), args.output_csv)
    print("\nActual kept-ratio (target vs achieved) and latency:")
    print(out.groupby(["method", "target_ratio"]).agg(
        actual_ratio=("actual_ratio", "mean"), latency_ms=("latency_ms", "mean"),
        n=("uid", "count")).round(3).to_string())
    if "span_extract_coverage" in out.columns:
        logger.info("Mean span-extractor coverage of labeled spans: %.1f%% "
                    "(low => compress.extract_spans differs from your labeling extractor)",
                    100 * out["span_extract_coverage"].mean())


if __name__ == "__main__":
    main()