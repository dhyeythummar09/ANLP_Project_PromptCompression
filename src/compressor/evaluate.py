"""
STEP 9 (revised): Evaluate every compressor on a target LLM and produce the paper's tables.

What changed vs. the first version
  * Long format: every (problem, method, ratio) is one query; the UNCOMPRESSED
    prompt is queried once per problem (method="none") as the upper bound.
  * Correctness is checked against the ORIGINAL question, never the compressed one
    (options / units that were deleted must not change how the answer is matched).
  * Failed / empty API responses are retried and, if still failing, NOT recorded
    (old version counted them as "incorrect", which silently penalised whichever
    method happened to hit a rate-limit). Re-running fills them in.
  * Thread pool (--workers) + resume by key, since 300 problems x ~7 methods x 3 ratios
    is thousands of calls.
  * `--mode summarize` produces:
      - accuracy per (model, method, ratio) with bootstrap 95% CIs, accuracy drop vs
        uncompressed, and P(correct | uncompressed was correct)
      - PAIRED comparison ours-vs-each-other-method (bootstrap CI on the difference)
      - the core hypothesis test: accuracy when ALL counterfactually-critical spans
        survive vs when some were dropped, per method
      - category retention + critical-span retention + latency + seen/unseen model role
  * `--mode deletion-check` implements proposal section 3.4: how often does true
    deletion give the same critical/non-critical outcome as the placeholder labels?

Run order:
    python src/compressor/evaluate.py --model deepseek --limit 20        # smoke test
    python src/compressor/evaluate.py --model deepseek                   # full run (resumable)
    python src/compressor/evaluate.py --mode summarize
    python src/compressor/evaluate.py --mode deletion-check --n 100
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.utils.api_clients import call_model
from src.utils.answer_matching import answers_match, extract_final_answer

COMPRESSED_FILE = os.path.join(config.DATA_DIR, "compressed_prompts.csv")
RESULTS_FILE = os.path.join(config.DATA_DIR, "evaluation_results.csv")
SUMMARY_FILE = os.path.join(config.DATA_DIR, "evaluation_summary.csv")
PAIRED_FILE = os.path.join(config.DATA_DIR, "paired_comparisons.csv")
CRITKEPT_FILE = os.path.join(config.DATA_DIR, "critical_kept_effect.csv")
CATRET_FILE = os.path.join(config.DATA_DIR, "category_retention.csv")
DELCHECK_FILE = os.path.join(config.DATA_DIR, "deletion_check.csv")

PROMPT_TEMPLATE = """Solve this problem. You may reason briefly, but you MUST end your \
response with a line in exactly this format:
Final Answer: <your answer>

Problem: {question}
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

# Models used to GENERATE labels ("seen") vs. genuinely unseen targets.
LABEL_MODEL_NAMES = {"qwen3.7-max", "glm-5.3-flash", "openai/gpt-oss-120b"}
EVAL_MODELS = {
    "qwen": {"name": "qwen3.7-max", "provider": "motapis_qwen", "sleep_seconds": 0.5},
    "glm": {"name": "glm-5.3-flash", "provider": "motapis_glm", "sleep_seconds": 0.5},
    "groq": {"name": "openai/gpt-oss-120b", "provider": "groq", "sleep_seconds": 2.0},
    # Unseen target. TODO: set `provider` to whatever your api_clients.py expects for this endpoint.
    "deepseek": {"name": "DeepSeek-V4-Flash-0731", "provider": "motapis_deepseek", "sleep_seconds": 0.5},
    "groq-llama": {"name": "llama-3.1-70b-versatile", "provider": "groq", "sleep_seconds": 1.0},
}


# ---------------------------------------------------------------------------
# One model call (shared by the main run and the deletion check)
# ---------------------------------------------------------------------------
def ask_model(cfg: dict, question: str, gt: str, reference_question: str, retries: int = 3):
    """Returns (correct: bool | None, answer, response). None => API failure, don't record."""
    query = PROMPT_TEMPLATE.format(question=question)
    resp = None
    for _ in range(retries):
        resp = call_model(cfg, query)
        time.sleep(cfg.get("sleep_seconds", 0.5))
        if resp:
            break
    if not resp:
        return None, "", ""
    return bool(answers_match(resp, gt, reference_question)), extract_final_answer(resp), resp


def _key(uid, model, method, ratio) -> str:
    return f"{uid}|{model}|{method}|{float(ratio):.4f}"


def _load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return {_key(u, m, me, r) for u, m, me, r in zip(d["uid"], d["eval_model"], d["method"], d["target_ratio"])}


def _append(rows: list[dict], path: str):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ---------------------------------------------------------------------------
# MODE: run
# ---------------------------------------------------------------------------
def run_eval(args):
    cfg = EVAL_MODELS[args.model]
    comp = pd.read_csv(args.input_csv).fillna("")
    comp["target_ratio"] = comp["target_ratio"].astype(float)
    uids = comp["uid"].drop_duplicates().tolist()
    if args.limit:
        uids = uids[:args.limit]
    comp = comp[comp["uid"].isin(set(uids))]
    if args.methods:
        comp = comp[comp["method"].isin(args.methods)]
    if args.ratios:
        comp = comp[comp["target_ratio"].round(4).isin([round(r, 4) for r in args.ratios])]

    tasks = []
    for r in comp.drop_duplicates("uid").itertuples():  # uncompressed upper bound, once per problem
        tasks.append(dict(uid=r.uid, example_id=r.example_id, source_dataset=r.source_dataset,
                          method="none", target_ratio=1.0, actual_ratio=1.0,
                          text=str(r.original_prompt), original=str(r.original_prompt),
                          gt=str(r.ground_truth_answer)))
    for r in comp.itertuples():
        tasks.append(dict(uid=r.uid, example_id=r.example_id, source_dataset=r.source_dataset,
                          method=r.method, target_ratio=r.target_ratio, actual_ratio=r.actual_ratio,
                          text=str(r.compressed_prompt), original=str(r.original_prompt),
                          gt=str(r.ground_truth_answer)))

    done = _load_done(args.output_csv)
    todo = [t for t in tasks if _key(t["uid"], cfg["name"], t["method"], t["target_ratio"]) not in done]
    logger.info("%s: %d queries total, %d already done, %d to run (%d workers)",
                cfg["name"], len(tasks), len(tasks) - len(todo), len(todo), args.workers)

    def work(t):
        try:
            # match against the ORIGINAL question, not the compressed text
            correct, ans, resp = ask_model(cfg, t["text"], t["gt"], t["original"])
        except Exception as ex:
            logger.warning("query failed (%s): %s", t["uid"], ex)
            return None
        if correct is None:
            return None
        return {"uid": t["uid"], "example_id": t["example_id"], "source_dataset": t["source_dataset"],
                "eval_model": cfg["name"], "method": t["method"], "target_ratio": t["target_ratio"],
                "actual_ratio": t["actual_ratio"], "ground_truth": t["gt"], "answer": ans,
                "correct": correct, "response": resp[:1500]}

    buf, ok, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, t) for t in todo]), 1):
            res = fut.result()
            if res is None:
                failed += 1
            else:
                buf.append(res)
                ok += 1
            if len(buf) >= args.checkpoint_every:
                _append(buf, args.output_csv)
                buf = []
            if i % 50 == 0:
                logger.info("  %d/%d done (%d failed, will be retried on re-run)", i, len(todo), failed)
    if buf:
        _append(buf, args.output_csv)
    logger.info("Finished: %d recorded, %d failed. Re-run the same command to retry failures.", ok, failed)


# ---------------------------------------------------------------------------
# MODE: summarize
# ---------------------------------------------------------------------------
def boot_ci(x, B: int = 1000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(args):
    res = pd.read_csv(args.output_csv)
    comp = pd.read_csv(args.input_csv)
    res["target_ratio"] = res["target_ratio"].astype(float).round(4)
    comp["target_ratio"] = comp["target_ratio"].astype(float).round(4)
    res["correct"] = res["correct"].astype(bool)

    orig = (res[res["method"] == "none"][["eval_model", "uid", "correct"]]
            .rename(columns={"correct": "orig_correct"}))
    keep_cols = [c for c in ["uid", "method", "target_ratio", "latency_ms", "crit_total", "crit_kept",
                             "all_critical_kept", "actual_ratio"] if c in comp.columns]
    d = (res[res["method"] != "none"].drop(columns=["actual_ratio"], errors="ignore")
         .merge(orig, on=["eval_model", "uid"], how="left")
         .merge(comp[keep_cols], on=["uid", "method", "target_ratio"], how="left"))

    # ---- 1. accuracy table ----
    rows = []
    for (model, method, ratio), g in d.groupby(["eval_model", "method", "target_ratio"]):
        lo, hi = boot_ci(g["correct"].astype(float))
        oc = g["orig_correct"].dropna().astype(bool)
        given = g[g["orig_correct"] == True]  # noqa: E712
        rows.append({
            "eval_model": model,
            "role": "seen(label model)" if model in LABEL_MODEL_NAMES else "UNSEEN",
            "method": method, "target_ratio": ratio, "n": len(g),
            "actual_ratio": round(g["actual_ratio"].mean(), 3),
            "accuracy": round(g["correct"].mean(), 4), "acc_ci_lo": round(lo, 4), "acc_ci_hi": round(hi, 4),
            "uncompressed_acc_same_problems": round(oc.mean(), 4) if len(oc) else np.nan,
            "acc_drop": round(oc.mean() - g["correct"].mean(), 4) if len(oc) else np.nan,
            "P(correct|orig_correct)": round(given["correct"].mean(), 4) if len(given) else np.nan,
            "latency_ms": round(g["latency_ms"].mean(), 1) if "latency_ms" in g else np.nan,
            "critical_span_retention": (round(g["crit_kept"].sum() / g["crit_total"].sum(), 4)
                                        if g["crit_total"].fillna(0).sum() > 0 else np.nan),
        })
    summ = pd.DataFrame(rows).sort_values(["eval_model", "target_ratio", "accuracy"], ascending=[True, False, False])
    summ.to_csv(args.summary_csv, index=False)
    base = res[res["method"] == "none"].groupby("eval_model")["correct"].agg(["mean", "count"])
    print("\nUncompressed accuracy (upper bound):\n", base.round(4).to_string())
    print("\nACCURACY BY METHOD x RATIO\n", summ.drop(columns=["eval_model", "role"]).to_string(index=False))

    # ---- 2. paired comparison: focus method vs every other ----
    pr = []
    for (model, ratio), g in d.groupby(["eval_model", "target_ratio"]):
        piv = g.pivot_table(index="uid", columns="method", values="correct", aggfunc="mean")
        ar = g.groupby("method")["actual_ratio"].mean()
        if args.focus not in piv.columns:
            continue
        for other in [c for c in piv.columns if c != args.focus]:
            p = piv[[args.focus, other]].dropna()
            if p.empty:
                continue
            diff = (p[args.focus] - p[other]).values
            lo, hi = boot_ci(diff)
            pr.append({"eval_model": model, "target_ratio": ratio, "focus": args.focus, "other": other,
                       "n_paired": len(p), "acc_focus": round(p[args.focus].mean(), 4),
                       "acc_other": round(p[other].mean(), 4), "diff": round(diff.mean(), 4),
                       "diff_ci_lo": round(lo, 4), "diff_ci_hi": round(hi, 4),
                       "significant_95": bool(lo > 0 or hi < 0),
                       "focus_only_correct": int(((p[args.focus] == 1) & (p[other] == 0)).sum()),
                       "other_only_correct": int(((p[args.focus] == 0) & (p[other] == 1)).sum()),
                       "actual_ratio_focus": round(ar.get(args.focus, np.nan), 3),
                       "actual_ratio_other": round(ar.get(other, np.nan), 3)})
    if pr:
        paired = pd.DataFrame(pr)
        paired.to_csv(PAIRED_FILE, index=False)
        print(f"\nPAIRED: {args.focus} minus other (positive = {args.focus} better)\n",
              paired.drop(columns=["focus"]).to_string(index=False))
        print("NOTE: compare actual_ratio_focus vs actual_ratio_other; a method that keeps more tokens is not a fair win.")

    # ---- 3. core hypothesis: does keeping ALL critical spans predict correctness? ----
    if "all_critical_kept" in d.columns:
        sub = d[(d["orig_correct"] == True) & d["all_critical_kept"].notna()]  # noqa: E712
        if len(sub):
            eff = (sub.groupby(["eval_model", "method", "all_critical_kept"])["correct"]
                   .agg(accuracy="mean", n="count").round(4).reset_index())
            eff.to_csv(CRITKEPT_FILE, index=False)
            print("\nACCURACY (problems the model solved uncompressed) BY WHETHER ALL CRITICAL SPANS SURVIVED"
                  "\n(all_critical_kept=1 should be clearly higher if criticality labels mean anything)\n",
                  eff.to_string(index=False))

    # ---- 4. category retention (from compression output) ----
    if "cat_retention_json" in comp.columns:
        recs = []
        for r in comp.itertuples():
            for cat, (kept, tot) in json.loads(r.cat_retention_json if isinstance(r.cat_retention_json, str) else "{}").items():
                recs.append((r.method, r.target_ratio, cat, kept, tot))
        if recs:
            cr = pd.DataFrame(recs, columns=["method", "target_ratio", "category", "kept", "total"])
            cr = cr.groupby(["method", "target_ratio", "category"]).sum().reset_index()
            cr["retention"] = (cr["kept"] / cr["total"]).round(4)
            cr.to_csv(CATRET_FILE, index=False)
            print("\nCATEGORY RETENTION (fraction of labeled spans still present)\n",
                  cr.pivot_table(index=["target_ratio", "method"], columns="category", values="retention").to_string())

    logger.info("Saved: %s, %s, %s, %s", args.summary_csv, PAIRED_FILE, CRITKEPT_FILE, CATRET_FILE)


# ---------------------------------------------------------------------------
# MODE: deletion-check  (proposal section 3.4)
# ---------------------------------------------------------------------------
def apply_deletion(text: str, s: int, e: int) -> str:
    out = text[:s] + " " + text[e:]
    out = re.sub(r" {2,}", " ", out).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", out)


def deletion_check(args):
    from sklearn.metrics import cohen_kappa_score
    cons = pd.read_csv(config.CONSENSUS_FILE).fillna("")
    need = {"labeling_model", "model_specific_label", "span_start", "span_end", "original_prompt", "ground_truth_answer"}
    if need - set(cons.columns):
        sys.exit(f"Consensus file is missing columns: {need - set(cons.columns)}")
    name2cfg = {c["name"]: c for c in EVAL_MODELS.values()}

    tasks = []
    for mname, g in cons.groupby("labeling_model"):
        if mname not in name2cfg:
            continue
        crit, non = g[g["model_specific_label"] == "critical"], g[g["model_specific_label"] != "critical"]
        half = args.n // 2
        samp = pd.concat([crit.sample(min(len(crit), half), random_state=0),
                          non.sample(min(len(non), half), random_state=0)])
        for r in samp.to_dict("records"):
            tasks.append((name2cfg[mname], r))
    logger.info("Deletion check on %d (span, model) pairs", len(tasks))

    def work(item):
        cfg, r = item
        text = str(r["original_prompt"])
        deleted = apply_deletion(text, int(r["span_start"]), int(r["span_end"]))
        correct, _, _ = ask_model(cfg, deleted, str(r["ground_truth_answer"]), text)
        if correct is None:
            return None
        return {"labeling_model": cfg["name"], "example_id": r.get("example_id", ""),
                "span_text": r.get("span_text", ""), "span_category": r.get("span_category", ""),
                "placeholder_critical": int(r["model_specific_label"] == "critical"),
                "deletion_critical": int(not correct)}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in tasks]):
            if fut.result():
                rows.append(fut.result())
    df = pd.DataFrame(rows)
    df.to_csv(DELCHECK_FILE, index=False)
    df["agree"] = df["placeholder_critical"] == df["deletion_critical"]
    print("\nPLACEHOLDER vs TRUE-DELETION agreement")
    for m, g in df.groupby("labeling_model"):
        kappa = cohen_kappa_score(g["placeholder_critical"], g["deletion_critical"]) if len(g) else np.nan
        print(f"  {m}: n={len(g)} agreement={g['agree'].mean():.3f} kappa={kappa:.3f} "
              f"| placeholder-critical -> still critical under deletion: "
              f"{g[g.placeholder_critical == 1]['deletion_critical'].mean():.3f} "
              f"| placeholder-non-critical -> critical under deletion: "
              f"{g[g.placeholder_critical == 0]['deletion_critical'].mean():.3f}")
    print("By category:\n", df.groupby("span_category")["agree"].agg(["mean", "count"]).round(3).to_string())
    logger.info("Saved %s", DELCHECK_FILE)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate compressors on a target LLM")
    ap.add_argument("--mode", choices=["run", "summarize", "deletion-check"], default="run")
    ap.add_argument("--input-csv", default=COMPRESSED_FILE)
    ap.add_argument("--output-csv", default=RESULTS_FILE)
    ap.add_argument("--summary-csv", default=SUMMARY_FILE)
    ap.add_argument("--model", default="deepseek", choices=list(EVAL_MODELS))
    ap.add_argument("--methods", nargs="+", default=None, help="restrict to these methods (default: all in the CSV)")
    ap.add_argument("--ratios", nargs="+", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--focus", default="ours", help="method compared against all others in --mode summarize")
    ap.add_argument("--n", type=int, default=100, help="spans per labeling model for --mode deletion-check")
    args = ap.parse_args()

    {"run": run_eval, "summarize": summarize, "deletion-check": deletion_check}[args.mode](args)


if __name__ == "__main__":
    main()