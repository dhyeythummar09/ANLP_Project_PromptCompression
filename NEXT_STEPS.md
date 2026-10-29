# What to do next — short execution checklist

Verified from local checkpoints, final CSVs and archived summaries on **26 October 2026**. Start here; [FOLLOWUP_EXPERIMENTS.md](FOLLOWUP_EXPERIMENTS.md) is background research.

## First: what is actually done?

**The v2 scorer is trained. No dataset recollection or retraining is needed below.** Missing downstream evaluation means generating compressed questions and checking the reader's answers.

| Name | What it is | Current status |
| --- | --- | --- |
| v2 span scorer | Our trained DistilBERT predicts sensitivity to masking a span | Five trained checkpoints exist. Its label-prediction AUC is .9074. |
| `ours_scorer` | Use that scorer to rank deletions; default also includes bonuses | **The standalone v2 configuration is missing from the final downstream run.** Its neural weights are already trained. |
| `ours_signal` | Rank using pretrained DistilGPT2 surprisal; default also includes bonuses | **Missing from the final downstream run.** Its pretrained weights need no training by us. |
| `ours` | Mix neural criticality and surprisal | v2 downstream evaluation completed. |
| `grouped_hybrid` | That mixture with selected phrases kept/deleted together | v2 downstream evaluation completed. |
| `grouped_signal` | Grouping + category prior + surprisal | v2 downstream evaluation completed. **This is different from `ours_signal`.** |

**Yes, downstream accuracy already exists for the v2 study.** The GPT-OSS .75 rows in RESULTS.md are `ours` **73.125%**, `grouped_hybrid` **73.125%**, and `grouped_signal` **71.250%**. These are the completed configurations you may remember. `grouped_signal` is not the missing standalone `ours_signal`.

The final v2 compression and reader CSVs contain seven methods, **excluding `ours_scorer` and `ours_signal`**. Only those standalone configurations are missing; this does not mean v2 lacks downstream results.

**AUC measures span-label prediction. Downstream accuracy measures answers to compressed questions.** These are different results.

## 27 October update: GPT-OSS removal does not block follow-up

The user reports GPT-OSS is unavailable on Jarvislabs and Qwen is still available. **Use Jarvislabs Qwen for the new follow-up**, subject to a brief access/output check. It was already evaluated in the completed study; GLM Flash is a fallback requiring its own qualification. The [current public catalog](https://jarvislabs.ai/products/token-factory) lists `qwen-3.8-27b-fp8` and `glm-5.3-flash`; account access/current rates must still be checked.

- **Step 2:** ask the same Qwen reader to answer both historical-checkpoint and v2 compressed prompts, plus originals, on the exposure-safe common cohort. Prefer one fresh paired run to avoid confounding a checkpoint change with serving changes over time. For N questions, two checkpoints and two ceilings mean 5N logical conditions including originals, before identical-query reuse/retries. Do not repeat the entire seven-method grid for this comparison.
- **Reuse computation:** existing labels/weights remain valid. Reuse saved v2 compressed strings where cohort, compressor variant and budgets match; generate old-checkpoint compression if it is missing. There is no need to retrain either scorer or recollect labels. The answers, not the compressor weights, depend on the reader.
- **New reader config:** `backend=hosted`, `provider=openai-compatible`, `name=qwen-3.8-27b-fp8`, `base_url=https://models.jarvislabs.net/v1`, `api_key_env=JARVISLABS_API_KEY`, `temperature=0`, `max_tokens=8192`, `reasoning_effort=low`. These are the saved Qwen settings; check current request support and representative original/compressed outputs before the main run.
- **Runner changes:** select the Qwen config; use a new output/run ID and a `target_qwen` response directory; update summary/cache references, requested/served reader identity, current prices and spending limits. Keep distinct old/v2 method IDs. Do not overwrite `target_gptoss`, reuse its answer cache or resume its ledger under Qwen. Preserve historical reports and manifests.
- **Steps 3–9:** use the selected reader consistently within each comparison. If completing component ablations on Qwen, compare all required arms on Qwen, not new Qwen component scores against old GPT-OSS hybrid scores. Historical saved-Qwen reuse needs matching query/settings/provenance and explicit disclosure; a fresh matched reference arm avoids serving-drift ambiguity. New-reader baseline comparisons require new answers for each compared method, but existing baseline compression text can be reused.

The shared HTTP client already supports an explicit OpenAI-compatible reader config. Historical `jarvis_main.py` is intentionally tied to the original frozen grids and pilots: adapt the teammate's new follow-up runner, not that historical launcher. Their Step-1 changes have not been shared in this checkout, so the exact new-script patch remains to be reviewed there.

GPT-OSS on [Groq](https://console.groq.com/docs/model/openai/gpt-oss-120b) or [Cerebras](https://inference-docs.cerebras.ai/models/overview) is optional. Switching provider is a new serving configuration, even with the same model family; do not fill missing Jarvislabs responses from another provider and call them the same run. No external provider switch is necessary while the Jarvislabs Qwen path works.

## 28 October status: Step 2 on Qwen

- **Compression done (Kaggle T4):** historical and v2 checkpoints, `ours` + `grouped_hybrid`, .50/.75, on the 739-question / 736-family exposure-safe common cohort (`outputs/followup/old_v2/{historical,v2}/compressed.csv`, 2,956 rows each, all `ok` and budget-compliant).
- **GPT-OSS pilot:** first request returned `HTTP 404 model_not_found`; kept unchanged in `target_pilots/gptoss/`.
- **Qwen pilot passed:** 16 disjoint GSM-IC families, both checkpoints, 144/144 `ok`, originals 16/16, served model `qwen-3.8-27b-fp8`, ₹5.23 (`target_pilots/qwen/pilot_check.json`).
- **Main paired run complete:** one run, distinct `*_historical` / `*_v2` method IDs, originals included → 6,651 logical conditions, 2,842 unique requests (64–82% of prompts are identical across checkpoints). Output `outputs/followup/old_v2/target_qwen/`; 6,649 `ok`, 2 `truncated` (`bbh_navigate_57`, `grouped_hybrid` .50, both arms; excluded from that pair). All fresh answers served as `qwen-3.8-27b-fp8`.
- **Cost:** provider-reported token usage ≈ ₹393. The guard's tracked ₹726 includes worst-case reservations for 171 failed attempts during a local network outage (08:26 UTC); the cap was raised 700 → 850 with user approval, recorded in `responses.csv.cost.json` → `limit_history`.
- **Result (Qwen, 739 questions / 736 families, originals 99.46%): no detectable difference between checkpoints.**

| Method | Ratio | Historical | v2 | Δ v2−hist [95% family CI] | Δ on differing-text pairs (n) |
| --- | --- | --- | --- | --- | --- |
| `ours` | .50 | 41.54% | 41.00% | −0.54 pp [−2.72, +1.77] | −1.49 pp (268) |
| `ours` | .75 | 71.31% | 71.85% | +0.54 pp [−0.94, +2.04] | +3.03 pp (132) |
| `grouped_hybrid` | .50 | 42.14% | 41.19% | −0.95 pp [−3.12, +1.36] | −2.69 pp (260) |
| `grouped_hybrid` | .75 | 71.45% | 71.99% | +0.54 pp [−0.95, +2.04] | +3.05 pp (131) |

  Report: `outputs/followup/old_v2/reports/qwen/old_vs_v2.{json,summary.csv,paired.csv}`. These are Qwen numbers; do not compare them with GPT-OSS rows in RESULTS.md. Step 2 is done; next is Step 3.

Tracked copies of the step-2 results and provenance: `results/followup/old_vs_v2_qwen/` (index: `results/followup/README.md`). Kaggle compression notebook: `notebooks/04_followup_old_v2_compression_kaggle.ipynb`.

## Step 3 status: scorer-only / signal-only on Qwen

- **Code ready:** `final_eval/component_ablations.py` (`pilot-cohort`, `prepare`, `run`, `summarize`) with shared guards in `final_eval/followup_reader.py`; commands in README, "Scorer-only and signal-only ablations".
- **Compression done locally (CPU, no Kaggle needed):** `ours_scorer`/`ours_signal` (640 rows) plus a fresh `ours` reference arm (320 rows) on the frozen 160 questions / 125 families. The fresh `ours` matches the main study's saved `ours` text in **320/320** conditions.
- **Why a fresh `ours` arm:** GPT-OSS is gone and the saved Qwen `ours` answers are not in this checkout, so all arms are answered in one matched Qwen run (1,120 logical conditions, 1,071 unique requests).
- **New pilot required:** the step-2 Qwen pilot shares 5 families with this cohort. A new disjoint 16-family pilot (2 per dataset) is compressed and prepared.
- **Qwen pilot passed:** 112/112 `ok`, originals 16/16 across all 8 datasets, ₹15.39.
- **Main matched run complete:** 1,120/1,120 `ok` after one retry pass for 2 truncations; 1,071 fresh answers, all served as `qwen-3.8-27b-fp8`; provider-reported cost ≈ ₹138 (guard ₹145.8, cap ₹450).
- **Serving-drift check:** fresh `ours` 46.25% (.50) / 71.25% (.75), originals 98.75%, versus the completed study's Qwen `ours` 46.25% / 71.875%, originals 98.75%.
- **Result (160 questions / 125 families): no component is detectably better or worse than the mixture; every 95% family-clustered CI includes zero.**

| Ratio | `ours` | `ours_scorer` | `ours_signal` | scorer − ours [CI] | signal − ours [CI] | scorer − signal [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| .50 | 46.25% | 48.75% | 45.62% | +2.50 pp [−6.4, +11.9] | −0.62 pp [−9.6, +8.1] | +3.12 pp [−6.2, +14.1] |
| .75 | 71.25% | 70.62% | 74.38% | −0.62 pp [−10.5, +8.7] | +3.12 pp [−4.9, +11.1] | −3.75 pp [−13.4, +6.6] |

  Actual retained ratios match (.4956 / ~.744). Per-dataset patterns are large but rest on 15–16 questions each (exploratory only): scorer-only scores 0/16 on `bbh_object_counting` at both ratios (signal-only 3/16 and 14/16), but 16/16 on logical deduction and 15/15 on tracking at .75. `ours_signal` equals `ours` text in 20 (.50) / 26 (.75) conditions. Reports: `outputs/followup/components_v2/reports/qwen/`; tracked copies in `results/followup/components_qwen/`. Step 3 is done; next is Step 4 (bonuses on/off on development families).

## Do these in this order

GPU work here means compression/scoring. Use one already-qualified reader first.

| Order | Actual work | Brief purpose and execution |
| --- | --- | --- |
| 1 | **Prepare the follow-up code and inputs.** | Add old-checkpoint/fold handling, named variants, separate outputs, persistent caching and spending limits. Check current credit/settings before paid calls. |
| 2 | **Compare old versus v2 downstream.** | Compress using old/new weights inside the same pipeline, then evaluate on a common exposure-safe cohort. This answers “did the checkpoint improve?”; whole-system changes are a separate comparison. |
| 3 | **Evaluate v2 scorer-only and signal-only.** | Generate `ours_scorer` / `ours_signal` at .50/.75 on the frozen 160 questions; obtain reader answers and compare with saved `ours`. Reuse only verified identical queries/settings. |
| 4 | **Compare bonuses on/off on development families.** | Keep other settings fixed and measure accuracy/deletion changes. This tests whether bonuses help; it is not fixing an established bug. |
| 5 | **Run mask versus deletion validation.** | For e.g. 120 spans, ask original / masked / deleted questions on the same reader. It tests the labeling assumption, independently of the three compressor variants. |
| 6 | **Summarize and benchmark.** | Produce paired results, failures and controlled runtime/memory including parser, models and budget selection. These complete the main missing checks; keep negative outcomes. |

**Step 1 needs implementation.** Existing flags are available, but the combined runner is not ready. Before paid use, improve `deletion-check` resume/reporting. Preserve executed code/artifacts; use new versioned outputs.

For step 2, use the **published old split**, which differs from the local one; audit related-question exposure before choosing the cohort. Older accuracy exists; the **controlled old-versus-v2 comparison** is missing.

Step 3 adds **640 compressed conditions per reader** before caching, plus any needed original controls/retries. Step 5 has at most **360 logical conditions** before shared-original reuse; retries add attempts. These are not cost quotations.

**Actual deletion was already used in the completed compression evaluation.** Step 5 checks whether masking a particular span and deleting that same span cause the same answer outcome. It is not a switch from masked evaluation to real compression.

## If we also want to improve the compressor, continue here

This branch is optional for reporting existing results. Its diagnostic/rule-based version needs no retraining.

| Order | Actual work | Brief purpose and execution |
| --- | --- | --- |
| 7 | **Test structured joint deletions.** | Original / delete A / delete B / delete both, with related and matched random pairs. New diagnostic outcomes test whether individually safe deletions fail together. |
| 8 | **Develop at most two candidates; freeze one.** | Test targeted relation/interface grouping and supported ranking changes on development data. Include the same structure with a cheap prior to isolate neural benefit. |
| 9 | **Confirm on a fresh test.** | Compare v2, candidate, matched cheap variant and LLMLingua-2. Report accuracy, actual tokens, failures and runtime; a second reader can confirm the frozen result. |

**Freeze development/test families before steps 4, 7 and 8.** Simplest design: use one existing fold's checkpoint; develop within its training families and test on fresh families it never trained on. Audit sample size first. Larger multi-fold tuning needs nested development; details are in the longer assessment. Step 3 is a retrospective comparison.

Then update RESULTS.md with tables/figures, reconcile billing and finish report/slides. If improvement is absent, report that finding; do not tune on the final test.

**Park for now:** distillation, adaptive 90% controller, RL, expanded-label retraining and new domains/languages. None is required for this checklist.

## Coverage of the screenshots and mid-eval commitments

All suggestions are assessed in the longer document; the checklist selects work rather than making every suggestion mandatory.

| Suggestion / commitment | Where it stands |
| --- | --- |
| Extractor mismatch | Main duplication addressed; remaining coverage audit belongs in step 1. |
| Bonus dominance and missing components | Steps 3–4; dominance remains an unproven explanation. |
| Word deletion versus semantic units | Selected grouping exists; stronger grouping is step 8. |
| Baseline downstream evaluation | Completed on two readers; genuinely unseen-family transfer is a separate optional confirmation, not established by the current readers. |
| Conditional/quantifier/causal taxonomy | Partial existing coverage; targeted structural additions can be considered in step 8. New supervised labels/training are deferred. |
| GSM-IC sentence-level distractor pruning | Optional step-8 candidate if the development audit supports it; not implemented/evaluated yet. |
| Placeholder versus true deletion | Step 5. Mid-eval slide 16 explicitly listed this as still being validated / a next step. |
| Multi-span masking | Step 7 tests structured joint deletion; scaling set labels and retraining remain conditional, not automatically required. |
| LLMLingua-2 distillation | Assessed and deferred. |
| Adaptive accuracy target / RL | Assessed and deferred. |
| Curves, uncertainty, category analysis, runtime and reporting | Existing evidence in RESULTS.md; steps 6/9 extend it. Broader curves and causal category ablations remain optional, not silently completed. |

Retained mid-eval slide text and speaker notes are `outputs/planning_audit/Symbiote_MidEval.txt` and `outputs/planning_audit/Symbiote_MidEval_pptx.txt`. The original proposal and mid-draft PDFs, plus historical result MDs, remain in `outputs/repo_cleanup_audit/20261022T061303Z/pre_cleanup_sources_and_documents.zip`. The original PPTX itself is not in that archive.

## Why other AI still flags things

- **Implemented versus tested:** bonuses exist; bonus-on/off evidence is missing. Maximum bonus is .4, not additive .8–1.0.
- **Grouping/extraction:** selected phrases are grouped and runtime extraction is shared. Broader semantic protection remains a hypothesis; training consumes saved spans, so coverage still needs auditing.
- **The pasted AI:** matching mask/delete outcomes does not validate all labels or joint deletion. Original-answer failures need coverage reporting.
- **Its counts/readiness:** two variants mean 640 conditions, not 320. The assay defaults to 30 spans; 120 needs `--n 120`. Current output is raw records plus simple agreement, not the full planned table. Cost depends on tokens/retries.

My earlier wording mixed historical results, training, implementation and v2 evaluation. The file-checked statuses above correct that ambiguity.

Evidence: [final accuracy table](results/tables/full_cohort/accuracy.csv), `outputs/final_eval/main_gptoss_v1/compressed.csv`, both final `target_*/responses.csv`, `models/span_scorer_v2_20261021/training_metadata.json`, five local `fold*/scorer.pt` files, and the historical archive named above.
