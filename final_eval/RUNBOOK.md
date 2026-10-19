# Team execution runbook

Use the updated code from this handoff, from the repository root on Kaggle/server. Original submitted files under `src/` and root configuration are preserved. **Do not run the old numbered dataset-collection scripts. Do not collect labels again.**

Run A, B, then C in order. A uses saved labels; B validates the real GPU/packages/target on at most 16 questions; C is the main experiment. If B passes, continue to C without waiting for approval. A failed check means stop the expensive sweep and send the saved artifacts/error. Offline tests cannot guarantee GPU/provider compatibility.

The core code now includes agreement, a feature scorer baseline, retention prediction analysis, plots, process profiling, and a claims ledger. Optional interaction studies, extra targets, hard-label fits and adaptive compression are outside this core run. They are not prerequisites for completing it, and are not implied by its claims.

## A. Reuse data, train once, run offline analyses

### A1. Environment and existing exports

Install the platform's appropriate PyTorch/CUDA build first, then:

```bash
python -m pip install -r final_eval/requirements.txt
python -m pip install -r final_eval/requirements-baselines.txt
python -m spacy download en_core_web_sm
python -m unittest discover -s final_eval/tests -v
python -m final_eval.handoff environment --output outputs/final_eval/environment.json
```

Place the **existing team exports** in ignored `data/legacy/team/`:

- `raw_problems.csv`: exact collected questions and gold answers.
- `per_model_labels.csv`: complete merged raw labels, with full original and perturbed responses. Confirm that all four workers' exports are included; software cannot infer completeness from a partial file.
- `GSM-IC_2step.json`: original GSM-IC source export/cache, needed if clean base questions are missing.

Names below are explicit example filenames; substitute the real ones consistently. Keep originals. Do not push data or credentials. Old Markdown result summaries cannot replace raw responses.

### A2. Recover families and prepare analysis tables, offline

```bash
python -m final_eval.preparation.recover_gsmic --problems data/legacy/team/raw_problems.csv --source-json data/legacy/team/GSM-IC_2step.json --output data/processed/problems_recovered.csv
python -m final_eval.preparation.prepare --labels data/legacy/team/per_model_labels.csv --problems data/processed/problems_recovered.csv --output-dir data/processed/team_v2 --dry-run
python -m final_eval.preparation.prepare --labels data/legacy/team/per_model_labels.csv --problems data/processed/problems_recovered.csv --output-dir data/processed/team_v2
python -m final_eval.analysis.agreement --votes data/processed/team_v2/per_model_labels.csv --output-dir outputs/final_eval/agreement
```

If the problem table already has verified `clean_question` or `base_group_id`, skip recovery and pass that table to preparation. A source/offset/gold/duplicate mismatch must be resolved from the real export; do not guess. For separate shards, replace the merged file after `--labels` with the complete explicit shard list.

Inspect `data/processed/team_v2/manifest.json`: reliable spans/questions, vote and parse statuses, base groups and input hashes. Regrading can change old counts. Nonempty unparseable masked answers count as operational failures; missing/failed provider responses do not become valid votes. Agreement is measured on common eligible occurrences, not asserted for all models/questions.

### A3. Corrected scorer, then a cheap feature baseline

```bash
python -m final_eval.scorer.train_scorer --consensus-file data/processed/team_v2/consensus_dataset.csv --output-dir models/span_scorer_v2 --folds 5 --batch-size 8 --epochs 4 --dry-run
python -m final_eval.scorer.train_scorer --consensus-file data/processed/team_v2/consensus_dataset.csv --output-dir models/span_scorer_v2 --folds 5 --batch-size 8 --epochs 4
python -m final_eval.analysis.feature_baseline --consensus-file data/processed/team_v2/consensus_dataset.csv --model-dir models/span_scorer_v2 --output-dir outputs/final_eval/feature_baseline
```

Check the dry-run's fractional `target_values` and fold sizes before training. This is **one five-fold training job**, using existing labels, with soft unweighted BCE. Batch size 8 is a starting setting, not a hardware guarantee. If it cannot fit, lower it before a successful full fit and retain the recorded setting. Leave `--hard-labels`, `--pos-weight`, and `--holdout-model` out of the core run.

An interrupted identical training invocation can add `--resume`; completed folds are reused. No automatic full retraining is needed for a later CPU analysis. If valid v2 preparation/training is already completed, use those artifacts rather than repeating A2/A3. A source/configuration guard failure is a reason to return metadata and the error, not to delete weights or recollect data.

If the environment inventory has no GPU, fix the server/Kaggle GPU session before A3; do not accidentally launch a long CPU training job.

Outputs include `training_metadata.json`, `split.json`, `scorer_oof_predictions.csv`, `scorer_report.json`, tokenizer and five checkpoints. **Keep the checkpoints on the execution machine.** The feature report fits only training folds and compares feature/neural/category-prior scores on shared evaluated occurrences. It is a sensitivity-prediction baseline, not another compressor or an extra API sweep.

The older Hugging Face checkpoint is historical: its known family overlap cannot be repaired by renaming folds. Do not substitute it for this corrected job. If historical reproduction is needed separately: `python -m final_eval.scorer.hf_artifact --output-dir models/legacy_hf --download`.

## B. Small server pilot before the main sweep

### B1. Compression and packages

```bash
python -m final_eval.preparation.cohort --problems data/processed/team_v2/problems.csv --split models/span_scorer_v2/split.json --output outputs/final_eval/pilot/cohort.csv --max-questions 16
python -m final_eval.batch --pilot --cohort outputs/final_eval/pilot/cohort.csv --work-dir outputs/final_eval/pilot --model-dir models/span_scorer_v2 --labels-file data/processed/team_v2/consensus_dataset.csv
```

`batch` runs four internal/random methods, then `llmlingua_budgeted`, `llmlingua2_budgeted` and `selective_context_budgeted` in separate processes at 0.75 and 0.50 retention. It merges the seven-method grid, computes intrinsic metrics and stops on failed or over-budget conditions **before target calls**. `--plan` prints commands without running them.

The `_budgeted` suffix is deliberate: these are official compressors plus our predeclared **budget controller**, not unchanged native algorithms. First keep the native output, then, only if needed, try lower official rate/target-token settings at fixed factors (.9, .75, .5, .25, .05) and three bracket refinements. Choose the longest compliant attempted output. The controller uses no gold answers or target outcomes, never truncates returned text, and can still report infeasibility. LLMLingua uses explicit `iterative_size=50`. Fields `native_compressed_prompt`, `native_comp_tokens` and `budget_search_json` retain the evidence. Native methods remain available individually for secondary analysis; native output is not a matched-budget result if it exceeds the cap. See the [official LLMLingua API](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py).

Report exact method names, actual ratios, budget-search overhead and underfilling. A win against a budget-controlled variant is limited to that stated implementation; do not attribute the controller to the original authors. No substitute algorithm or arbitrary final truncation is allowed for a failed baseline.

Actual package APIs, baseline weights and short-input budget behavior are validated here. If the pilot fails, return its CSVs/manifests and the exact traceback. Fix the specific stage before a main sweep; do not train the scorer again merely because a baseline fails. Identical compression commands reuse completed conditions. Failed conditions remain recorded; return them before choosing a corrected stage output.

### B2. Confirm target settings and test inference

Copy **one** example config to `outputs/final_eval/target.json`, then fill its confirmed fields:

- `final_eval/configs/target.local.example.json`: actual model/path, pinned revision, supported dtype and device settings.
- `final_eval/configs/target.hosted.example.json`: exact endpoint/model, environment-variable name for one credential, output-token limit, rate and request ceiling. Never put the key in JSON. The example ceiling of 20 requests is just a placeholder, not a main-run budget.

Do not guess a model, endpoint, quota or GPU dtype. A local target is available as a route once a suitable model is selected and validated. Both pilot and main use the same target, wrapper, decoding and output-token settings. With 160 main questions the grid has at most 2,400 logical generations, plus at most 240 for the pilot, before caching/retries. Actual HTTP attempts count against the configured ceiling; shared key quota must be coordinated across people/runs.

```bash
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/pilot/compressed.csv --output-csv outputs/final_eval/pilot/responses.csv --model-config outputs/final_eval/target.json --dry-run
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/pilot/compressed.csv --output-csv outputs/final_eval/pilot/responses.csv --model-config outputs/final_eval/target.json
python -m final_eval.handoff check --compressed outputs/final_eval/pilot/compressed.csv --responses outputs/final_eval/pilot/responses.csv --output outputs/final_eval/pilot/check.json
```

The check requires complete successful conditions, budget compliance, at least 12 original responses and at least 50% original accuracy. These are operational floors, not proof of statistical adequacy. Inspect per-dataset originals, parser behavior, output truncation and the measured profile as well. If the target is unsuitable, fix it on this pilot before C. Do not lower gates after seeing main outcomes. The main cohort excludes **all pilot base families**.

If no target is confirmed yet, complete A and save the archive in D. Send confirmed GPU, model/provider, available quota and context/output limits as text; no API keys.

## C. Main run once the pilot passes

```bash
python -m final_eval.preparation.cohort --problems data/processed/team_v2/problems.csv --split models/span_scorer_v2/split.json --exclude-cohort outputs/final_eval/pilot/cohort.csv --output outputs/final_eval/main/cohort.csv --max-questions 160
python -m final_eval.batch --cohort outputs/final_eval/main/cohort.csv --work-dir outputs/final_eval/main --model-dir models/span_scorer_v2 --labels-file data/processed/team_v2/consensus_dataset.csv --pilot-check outputs/final_eval/pilot/check.json
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --model-config outputs/final_eval/target.json --pilot-check outputs/final_eval/pilot/check.json --dry-run
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --model-config outputs/final_eval/target.json --pilot-check outputs/final_eval/pilot/check.json
python -m final_eval.compressor.evaluate --mode summarize --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --report-dir outputs/final_eval/main/reports --focus grouped_hybrid
python -m final_eval.analysis.results --compressed outputs/final_eval/main/compressed.csv --responses outputs/final_eval/main/responses.csv --reports outputs/final_eval/main/reports --output-dir outputs/final_eval/main/analysis
```

The cap can yield fewer than 160 questions because families stay intact. If resources require a smaller cap, choose it uniformly before inspecting main outcomes and report wider uncertainty. Keep the default alpha 0.5 and seven methods; do not add method searches after looking at main results.

Inference checks pilot/main family separation and matching generation settings. Repeat an identical inference command to continue missing conditions; add `--retry-failed` to retry recorded failures. Original and identical-query generations are cached; successful answers need not be regenerated to make new reports. Request ceilings/backoff/timeouts can be adjusted when access changes without losing saved answers; attempt counts and settings history remain recorded. Changing the target/model, wrapper, decoding, checkpoint or input is a different experiment.

Reports/plots read saved files and make **no inference calls**. Use a new report output directory if it already exists; this does not require repeating training, compression or generation. Later analysis-only edits do not change core experiment fingerprints.

Read `coverage.csv` before accuracy or paired differences. Missing/failed responses and budget exclusions are explicit. Main accuracy is on common complete questions; native over-budget outputs cannot support a fixed-budget paired win. Retention analysis is a five-fold, family-disjoint predictive diagnostic within target/task/budget, with method, length, actual ratio and original-correctness controls. Sparse strata are explicitly skipped. It is not a causal proof. Plot data and a scope-limited claims ledger are saved alongside figures.

Process profiles (`*.profiles/` and model `profiles/`) record wall time including cold loads, peak PyTorch GPU allocation/reservation and CPU peak RSS where supported. These are measured process costs, not end-to-end speedup guarantees or pure warmed latency. Do not combine timings from different hardware as one speed comparison.

## Optional bounded reliability diagnostic

Only if access remains after the core, use a separate output and explicit quota:

```bash
python -m final_eval.compressor.evaluate --mode deletion-check --input-csv data/processed/team_v2/per_model_labels.csv --output-csv outputs/final_eval/reliability/check.csv --model-config outputs/final_eval/target.json --n 12 --dry-run
```

This plans at most 36 original/mask/delete conditions before cache/retries. Remove `--dry-run` only to run it on the execution machine. It is a fresh same-target assay, not validation of historical labeler identity or repeated-sampling reliability. If it is skipped, mark mask-to-deletion validity unverified. Joint-deletion interaction experiments are not part of this core run; do not claim they were demonstrated.

## D. Send one archive, including failures

Run after A if target access is pending, after a failed pilot, or after C completes. Choose a fresh ZIP filename each time:

```bash
python -m final_eval.handoff bundle --run-dir outputs/final_eval --prepared-dir data/processed/team_v2 --model-dir models/span_scorer_v2 --output outputs/handoffs/final_eval_results_01.zip
```

Send that ZIP and any failing command/traceback. It includes prepared vote/consensus tables, agreement and feature reports, training metadata/OOF/split, any pilot/main CSVs and manifests, full responses and usage ledger, target config without keys, environment inventory, profiles, plots, claims ledger, source snapshot and checksums. It excludes `.env` and checkpoint weights. Keep the weights and original exports backed up on the execution machine; persist/download results before Kaggle ends. Nothing is uploaded by this command.

I will use the archive to check coverage, grading/eligibility, learned-versus-cheap scores, matched-budget accuracy, uncertainty and failure explanations, then finalize claims/report content. Weak or negative results do not trigger a new dataset campaign or an automatic training rerun. Invalid or interrupted stages may need a targeted fix; no implementation can promise that untested GPU/providers never fail.

No Git commit, push or model upload is performed by these commands. The research background remains in [PROJECT_COMPLETION_PLAN.md](../PROJECT_COMPLETION_PLAN.md).
