"""Versioned whole-question compression; no model/API execution during dry runs."""
from __future__ import annotations
import argparse
import bisect
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_eval.utils.artifacts import source_fingerprint, package_versions, digest, file_hash, resume_manifest, require_unique
from final_eval.preparation.schema import prepare_problems
from final_eval.scorer.data import build_span_table
from final_eval.compressor.budget import TokenCounter, select_to_budget, words
from final_eval.compressor.retention import retention

OURS_ALPHA = {"ours": 0.5, "ours_scorer": 1.0, "ours_signal": 0.0,
              "grouped_hybrid": 0.5, "grouped_signal": 0.5}
NEURAL = {"ours", "ours_scorer", "grouped_hybrid"}
BASELINES = ["random", "surprisal_word", "llmlingua", "llmlingua2", "selective_context"]
BASELINES += [m+"_budgeted" for m in ("llmlingua","llmlingua2","selective_context")]
DEFAULT_METHODS = list(OURS_ALPHA) + BASELINES
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT_DIR = "models/span_scorer_v2"
MAX_SEQ_LEN = 512

def count_tokens(text):
    # Legacy import compatibility, with no silent fallback.
    return TokenCounter()(text)

def extract_spans(text, nlp):
    from final_eval.spans.extract import find_spans
    return [{"span_text": t, "span_start": s, "span_end": e, "span_category": c}
            for t, s, e, c, _ in find_spans(text, nlp=nlp, include_controls=False)]

def load_scorer(fold_dir, device):
    import torch
    from final_eval.scorer.train_scorer import SpanScorer
    from transformers import DistilBertConfig
    configuration = Path(fold_dir) / "config.json"
    model = SpanScorer(encoder_config=DistilBertConfig.from_pretrained(fold_dir) if configuration.exists() else DistilBertConfig())
    model.load_state_dict(torch.load(Path(fold_dir) / "scorer.pt", map_location=device, weights_only=True))
    return model.to(device).eval()

def score_spans(text, spans, model, tokenizer, device, max_len=MAX_SEQ_LEN):
    import torch
    from final_eval.scorer.train_scorer import span_token_mask
    for span in spans:
        span["criticality_score"] = 1.0
    if not spans:
        return spans
    enc = tokenizer(text, max_length=max_len, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].numpy()
    masks = [span_token_mask(offsets, s["span_start"], s["span_end"]) for s in spans]
    valid = [i for i, mask in enumerate(masks) if mask.any() and offsets[mask, 0].min() <= spans[i]["span_start"]
             and offsets[mask, 1].max() >= spans[i]["span_end"]]
    if valid:
        mask = torch.tensor(np.stack([masks[i] for i in valid]), dtype=torch.float32, device=device)[None]
        with torch.inference_mode():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), mask)[0]
        for i, probability in zip(valid, torch.sigmoid(logits).cpu().tolist()):
            spans[i]["criticality_score"] = probability
    return spans

class InfoScorer:
    def __init__(self, name: str, device: torch.device):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device

    def word_info(self, text: str, words: list[tuple]) -> np.ndarray:
        """Sum of token surprisals per whitespace word."""
        import torch
        if len(self.tok(text, add_special_tokens=False)["input_ids"]) > 1000:
            raise ValueError("Surprisal context exceeds 1000 tokens; choose a compatible longer-context signal instead of silently truncating")
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False,
                       truncation=True, max_length=1000)
        bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        x = torch.tensor([[bos] + enc["input_ids"]], device=self.device)
        with torch.inference_mode():
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



def prepare_ours(text, model, tokenizer, nlp, info, device, prior=None):
    spans = extract_spans(text, nlp)
    if model is not None:
        spans = score_spans(text, spans, model, tokenizer, device)
    else:
        for span in spans:
            span["criticality_score"] = float((prior or {}).get(span["span_category"], 0.0))
    units = words(text)
    crit = np.zeros(len(units)); bonuses = np.zeros(len(units))
    for span in spans:
        for i, (s, e, token) in enumerate(units):
            if s < span["span_end"] and e > span["span_start"]:
                crit[i] = max(crit[i], span["criticality_score"])
                if span["span_category"] == "number":
                    bonuses[i] = max(bonuses[i], 0.4)
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "nummod":
            for i, (s, e, _) in enumerate(units):
                if s < token.head.idx + len(token.head.text) and e > token.head.idx:
                    bonuses[i] = max(bonuses[i], 0.4)
    query_ranges = []
    for sent in reversed(list(doc.sents)):
        if "?" in sent.text or any(t.lower_ in {"how", "what", "which", "calculate", "find"} for t in sent):
            query_ranges.append((sent.start_char, sent.end_char)); break
    for i, (s, e, token) in enumerate(units):
        if any(c.isdigit() for c in token):
            bonuses[i] = max(bonuses[i], 0.4)
        if any(s < b and e > a for a, b in query_ranges):
            bonuses[i] = max(bonuses[i], 0.2)
    groups = [(s["span_start"], s["span_end"]) for s in spans
              if s["span_category"] in {"number", "comparison_conditional", "negation"}]
    for token in doc:
        if token.dep_ == "neg":
            groups.append((min(token.idx, token.head.idx), max(token.idx + len(token.text), token.head.idx + len(token.head.text))))
    options = [(m.start(), m.end()) for m in re.finditer(r"(?m)^\s*\([A-Z]\)[^\n]*", text)]
    return {"spans": spans, "words": units, "crit": crit, "bonus": bonuses,
            "info": rank01(info.word_info(text, units)) if units else np.zeros(0),
            "groups": groups, "interface": query_ranges + options}

class BaselineRunner:
    def __init__(self, device, model1, model2):
        self.device, self.models, self.objects = device, {"llmlingua": model1, "llmlingua2": model2}, {}

    def run(self, method, text, ratio, budget):
        if method in {"llmlingua", "llmlingua2"}:
            if method not in self.objects:
                from llmlingua import PromptCompressor
                self.objects[method] = PromptCompressor(model_name=self.models[method],
                    use_llmlingua2=method == "llmlingua2", device_map=self.device)
            settings = {}
            if method == "llmlingua":
                # A whole-question native window avoids both negative window
                # offsets on short inputs and legacy KV-cache reuse on TF5.
                # This policy is limited to questions fitting Phi-2's context.
                compressor = self.objects[method]
                length = compressor.get_token_length(text, add_special_tokens=False)
                if length > compressor.max_position_embeddings:
                    raise ValueError("LLMLingua whole-question window exceeds the model context; no silent truncation")
                settings = {"iterative_size": max(1, length)}
            out = self.objects[method].compress_prompt(text, rate=ratio, target_token=budget, **settings)
            return out["compressed_prompt"]
        if method == "selective_context":
            if method not in self.objects:
                from selective_context import SelectiveContext
                self.objects[method] = SelectiveContext(model_type="gpt2", lang="en")
            output, _ = self.objects[method](text, reduce_ratio=1 - ratio)
            return output
        raise ValueError(method)

    def run_budgeted(self, method, text, ratio, budget, counter):
        """Predeclared black-box budget calibration; never cuts returned text.

        Keep the native attempt and try progressively lower public keep/target
        controls, then three bracket refinements. No gold, labels or outcomes
        enter this controller. All attempted outputs/controls are recorded.
        """
        base=method.removesuffix("_budgeted")
        native=self.run(base,text,ratio,budget)
        attempts=[{"factor":1.,"text":native,"tokens":counter(native)}]
        valid=[a for a in attempts if a["text"].strip() and a["tokens"]<=budget]
        low=None; high=1.
        def attempt(factor):
            try:
                output=self.run(base,text,max(.0001,ratio*factor),max(1,int(budget*factor)))
                row={"factor":factor,"text":output,"tokens":counter(output)}
            except Exception as exc:
                row={"factor":factor,"text":"","tokens":None,"error":type(exc).__name__}
            attempts.append(row)
            if row["text"].strip() and row["tokens"]<=budget:
                valid.append(row)
                return True
            return False
        if not valid and budget>0:
            for factor in (.9,.75,.5,.25,.05):
                if attempt(factor):
                    low=factor; break
                high=factor
            if low is not None:
                for _ in range(3):
                    midpoint=(low+high)/2
                    if attempt(midpoint):
                        low=midpoint
                    else:
                        high=midpoint
        chosen=max(valid,key=lambda a:a["tokens"]) if valid else attempts[0]
        details={"baseline_base_method":base,"native_compressed_prompt":native,"native_comp_tokens":counter(native),
                 "budget_controller":"official-controls-bracket-v1",
                 "budget_search_json":json.dumps(attempts)}
        return chosen["text"],"ok" if valid else "baseline_budget_infeasible",details

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--model-dir", default=MODEL_OUTPUT_DIR)
    ap.add_argument("--labels-file", default=None)
    ap.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=["ours", "grouped_hybrid", "grouped_signal", "random"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.75, 0.5])
    ap.add_argument("--tokenizer", default="cl100k_base")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-bonuses", action="store_true")
    ap.add_argument("--protect-interface", action="store_true", help="Named own-method diagnostic; not shared with external baselines")
    ap.add_argument("--info-model", default="distilgpt2")
    ap.add_argument("--llmlingua-model", default="microsoft/phi-2")
    ap.add_argument("--llmlingua2-model", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    ap.add_argument("--allow-legacy-checkpoint", action="store_true", help="Historical analysis only; does not certify old splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.alpha <= 1 or any(not 0 < r <= 1 for r in args.ratios):
        ap.error("alpha in [0,1] and retention ratios in (0,1] required")
    if len(set(args.methods)) != len(args.methods) or len(set(args.ratios)) != len(args.ratios):
        ap.error("Duplicate methods/ratios")
    problems = prepare_problems(pd.read_csv(args.input_csv, keep_default_na=False, dtype={"ground_truth_answer": str}))
    if args.limit:
        problems = problems.head(args.limit).copy()
    need_folds = bool(set(args.methods) & (NEURAL | {"grouped_signal"}))
    split = None
    if need_folds:
        split = json.loads((Path(args.model_dir) / "split.json").read_text(encoding="utf-8"))
        if split.get("protocol") != "symbiote-v2" and not args.allow_legacy_checkpoint:
            raise ValueError("Legacy split cannot certify family-safe evaluation. Retrain or explicitly allow historical analysis.")
        missing = set(problems.uid) - set(split["folds"])
        if missing:
            raise ValueError(f"{len(missing)} questions lack held-out fold assignments; select the scorer cohort explicitly")
        if split.get("base_groups"):
            if any(split["base_groups"].get(r.uid) != r.base_group_id for r in problems.itertuples()):
                raise ValueError("Input base groups disagree with training groups")
    if split:
        problems = problems.assign(scorer_fold=problems.uid.map(split["folds"])).sort_values(["scorer_fold", "uid"])
    print(json.dumps({"questions": len(problems), "base_groups": problems.base_group_id.nunique(),
                      "tasks": len(problems) * len(args.methods) * len(args.ratios),
                      "methods": args.methods, "ratios": args.ratios, "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return
    counter = TokenCounter(args.tokenizer)
    if args.tokenizer == "regex-test-only":
        raise ValueError("regex-test-only is only for unit tests, not scientific compression runs")
    import torch
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    labels = {}
    if args.labels_file:
        table = build_span_table(pd.read_csv(args.labels_file, keep_default_na=False))
        labels = {u: g for u, g in table.groupby("uid")}
    spec = {"stage": "compression", "input": file_hash(args.input_csv),
            "source": source_fingerprint(), "packages": package_versions(), "arguments": {k: v for k, v in vars(args).items() if k != "dry_run"},
            "labels": file_hash(args.labels_file) if args.labels_file else None,
            "split": file_hash(Path(args.model_dir) / "split.json") if need_folds else None,
            "checkpoints": {str(p.relative_to(args.model_dir)): file_hash(p) for p in Path(args.model_dir).glob("fold*/*") if p.is_file()} if need_folds else {}}
    run_hash = resume_manifest(args.output_csv, spec)
    previous = pd.read_csv(args.output_csv, keep_default_na=False) if Path(args.output_csv).exists() else pd.DataFrame()
    key = ["uid", "method", "target_ratio"]
    if not previous.empty:
        require_unique(previous, key, "compression output")
    done = set(previous[key].itertuples(index=False, name=None)) if not previous.empty else set()
    nlp = info = tokenizer = None
    if set(args.methods) & (set(OURS_ALPHA) | {"surprisal_word"}):
        info = InfoScorer(args.info_model, device)
    if set(args.methods) & set(OURS_ALPHA):
        import spacy
        nlp = spacy.load("en_core_web_sm")
    if set(args.methods) & NEURAL:
        from transformers import DistilBertTokenizerFast
        location = Path(args.model_dir) / "tokenizer"
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(location) if location.exists() else MODEL_NAME)
    baseline = BaselineRunner(str(device), args.llmlingua_model, args.llmlingua2_model)
    current_fold, model, prior = None, None, None
    for row in problems.itertuples():
        fold = split["folds"][row.uid] if split else None
        if fold != current_fold or (need_folds and current_fold is None):
            if model is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            model = load_scorer(Path(args.model_dir) / f"fold{fold}", device) if set(args.methods) & NEURAL else None
            prior_path = Path(args.model_dir) / f"fold{fold}" / "category_prior.json"
            prior = json.loads(prior_path.read_text()) if "grouped_signal" in args.methods else None
            current_fold = fold
        text = row.original_prompt
        preparations = {}
        for method in args.methods:
            for ratio in args.ratios:
                if (row.uid, method, ratio) in done:
                    continue
                started = time.perf_counter()
                record = {"uid": row.uid, "example_id": row.example_id, "source_dataset": row.source_dataset,
                          "base_group_id": row.base_group_id, "fold": fold, "method": method, "target_ratio": ratio,
                          "original_prompt": text, "ground_truth_answer": row.ground_truth_answer,
                          "run_hash": run_hash, "tokenizer": counter.name, "orig_tokens": counter(text),
                          "budget_tokens": int(ratio * counter(text)), "crit_total": None, "crit_kept": None,
                          "crit_kept_lower": None, "crit_kept_upper": None, "all_critical_kept": None,
                          "cat_retention_json": "{}", "alignment_status": "not_annotated",
                          "baseline_base_method":"","native_compressed_prompt":"","native_comp_tokens":None,
                          "budget_controller":"none","budget_search_json":"[]"}
                try:
                    selection = None
                    baseline_status="ok"
                    rng = np.random.default_rng(int(digest([row.uid, method, ratio, args.seed])[:16], 16))
                    if method in OURS_ALPHA:
                        kind = "prior" if method == "grouped_signal" else "neural" if method in NEURAL else "signal"
                        prep_start = time.perf_counter()
                        if kind not in preparations:
                            preparations[kind] = prepare_ours(text, model if kind == "neural" else None,
                                                tokenizer, nlp, info, device, prior if kind == "prior" else None)
                            preparations[kind]["prep_ms"] = (time.perf_counter() - prep_start) * 1000
                            preparations[kind]["built_now"] = True
                        prep = preparations[kind]
                        a = args.alpha if method in {"ours", "grouped_hybrid", "grouped_signal"} else OURS_ALPHA[method]
                        priority = a * prep["crit"] + (1-a) * prep["info"]
                        if not args.no_bonuses:
                            priority = priority + prep["bonus"]
                        selection = select_to_budget(text, priority, ratio, counter,
                            prep["groups"] if method.startswith("grouped_") else (),
                            prep["interface"] if args.protect_interface else ())
                        compressed = selection.text
                        elapsed = (time.perf_counter() - started) * 1000
                        # Every variant's standalone cost includes its preparation exactly once.
                        if not prep.pop("built_now", False):
                            elapsed += prep["prep_ms"]
                    elif method in {"random", "surprisal_word"}:
                        units = words(text)
                        priority = rng.random(len(units)) if method == "random" else info.word_info(text, units)
                        selection = select_to_budget(text, priority, ratio, counter)
                        compressed = selection.text
                        elapsed = (time.perf_counter() - started) * 1000
                    else:
                        if method.endswith("_budgeted"):
                            compressed,baseline_status,details=baseline.run_budgeted(method,text,ratio,record["budget_tokens"],counter)
                            record.update(details)
                        else:
                            compressed = baseline.run(method, text, ratio, record["budget_tokens"])
                        elapsed = (time.perf_counter() - started) * 1000
                    status = selection.status if selection else baseline_status
                    tokens = counter(compressed)
                    record.update(compressed_prompt=compressed, comp_tokens=tokens,
                        actual_ratio=tokens / max(record["orig_tokens"], 1), latency_ms=elapsed,
                        status=status, budget_compliant=tokens <= record["budget_tokens"],
                        kept_ranges_json=json.dumps(selection.kept_ranges) if selection else "")
                    if row.uid in labels and status == "ok":
                        record.update(retention(text, compressed, labels[row.uid], selection.kept_ranges if selection else None))
                except Exception as exc:
                    record.update(status="method_failed", error=f"{type(exc).__name__}: {exc}", compressed_prompt="",
                                  comp_tokens=None, actual_ratio=None, latency_ms=None, budget_compliant=False, kept_ranges_json="")
                # Fixed columns across successes/failures, with immediate per-task checkpointing.
                record.setdefault("error", "")
                ordered = sorted(record)
                pd.DataFrame([record], columns=ordered).to_csv(args.output_csv, mode="a", index=False,
                                                            header=not Path(args.output_csv).exists())
                done.add((row.uid, method, ratio))
    print(f"Saved compression records to {args.output_csv}; failures remain explicit. Use a new run for changed configurations.")

if __name__ == "__main__":
    from final_eval.utils.profiling import profile_cli
    profile_cli(main, "--output-csv")
