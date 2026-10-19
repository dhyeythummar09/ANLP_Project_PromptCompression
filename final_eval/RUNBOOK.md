# Execution runbook

Use this document for commands; use [PROJECT_COMPLETION_PLAN.md](../PROJECT_COMPLETION_PLAN.md) for research decisions. Commands below are for the **server/Kaggle**, from the repository root. No training, real model inference, or LLM API evaluation has been run during implementation.

The original submitted files are restored to commit `97dd6e7`. All commands below use the separate `final_eval` package. **Reuse the collected dataset; do not rerun dataset collection.** Preparation below is offline conversion/regrading of saved exports. New scorer training and target-model evaluation are distinct later jobs.

**Current status:** the core v2 pipeline and lightweight checks are implemented. GPU loading, official-baseline package compatibility, throughput, and new provider settings still need server smoke tests. Extra experiments from the plan remain separate work; this document does not certify completion of the research.

## 1. Obtain the team's existing artifacts

Keep original exports in an ignored directory such as `data/legacy/team/`. The commands use these example filenames; substitute the actual files:

- `raw_problems.csv`: the exact question table used during collection.
- `per_model_labels.csv`: the full merged raw-label table with original and masked responses. Alternatively pass every worker shard explicitly with `--labels file1.csv file2.csv ...`.
- `GSM-IC_2step.json`: the source JSON from the collection cache, for recovering clean base questions.
- Existing full evaluation responses and original references, if recoverable. Markdown result summaries cannot replace them.

Ignored datasets do not need to be pushed to Git. Transfer an archive separately, retain originals, and keep the generated hash manifests with the results. Do not run the new importer on only one worker and call its output the team dataset.

The public legacy model contains all five `foldN/scorer.pt` files and a 924-question split at revision `c24d8ce9ef2cfbdc7b11185958dee5810327e432`. That split has confirmed cross-fold GSM-IC variants. Its training-target type is unrecorded. Keep it for historical reproduction; train new weights for the repaired protocol.

Optional download, on the server:

```bash
python -m final_eval.scorer.hf_artifact --output-dir models/legacy_hf --download
```

This is a custom PyTorch state-dict bundle, not a standard `AutoModel` checkpoint. New scientific compression rejects its old split by default. `--allow-legacy-checkpoint` is only for explicitly labeled historical analysis, and the old bundle lacks the new category-prior artifacts.

## 2. Prepare the execution environment

Use the platform's appropriate PyTorch/CUDA installation. Install other dependencies into an isolated environment:

```bash
python -m pip install -r final_eval/requirements.txt
python -m pip install -r final_eval/requirements-baselines.txt
python -m spacy download en_core_web_sm
python -m unittest discover -s final_eval/tests -v
```

Do not assume the laptop's package versions are a validated Linux/GPU lockfile. After the server smoke tests pass, export `pip freeze` and the actual GPU/VRAM information with the run. A scientific run requires `tiktoken`; there is no silent regex fallback. Unit tests use an explicitly fabricated tokenizer and mocked generations.

Use one GPU-heavy baseline per process as shown below. An A100 is optional. Start the scorer at batch size 8; lower it if needed. Do not change precision/model/decoding settings between compression methods in one target comparison.

## 3. Recover clean GSM-IC questions, then import/regrade

The old loader saved distractor text, but five local annotations do not match the question verbatim. This recovery verifies exact distracted-question identity against the source JSON before using its clean question:

```bash
python -m final_eval.preparation.recover_gsmic --problems data/legacy/team/raw_problems.csv --source-json data/legacy/team/GSM-IC_2step.json --output data/processed/problems_recovered.csv
```

If the team's question table already contains correct `clean_question` or verified `base_group_id`, use that table directly. If recovery reports a source mismatch, stop and retrieve the matching source export. Do not infer a family from a possibly incorrect subtraction.

Audit the complete import without writing files, then write separate corrected analysis tables from the saved responses. These commands make no API calls:

```bash
python -m final_eval.preparation.prepare --labels data/legacy/team/per_model_labels.csv --problems data/processed/problems_recovered.csv --output-dir data/processed/team_v2 --dry-run
python -m final_eval.preparation.prepare --labels data/legacy/team/per_model_labels.csv --problems data/processed/problems_recovered.csv --output-dir data/processed/team_v2
```

Outputs: `problems.csv`, `per_model_labels.csv`, `consensus_dataset.csv`, `manifest.json` in `data/processed/team_v2/`.

Check source hashes, question/base-family counts, original-wrong/missing-response counts, and reliable span counts. Counts can differ from 924/14,162 after regrading. Conflicting duplicates and prompt/gold mismatches fail explicitly. Resolve them from source records; do not choose the more favorable answer.

Operational labels count a nonempty returned but unparseable masked answer as a failure, with its parse status retained. Missing/failed/truncated provider responses are excluded from valid votes. These distinctions must appear in the final methods section.

Optional regrade of old downstream responses, without new calls:

```bash
python -m final_eval.preparation.regrade_evaluation --results data/legacy/team/evaluation_results.csv --problems data/processed/team_v2/problems.csv --output outputs/final_eval/legacy_regrade/responses.csv
```

The old evaluator truncated stored raw responses to 1,500 characters. Potentially cut-off finals are flagged as unverifiable; recover full outputs when possible. Regrading cannot establish an unrecorded historical model identity.

## 4. Validate training inputs and train a new scorer

```bash
python -m final_eval.scorer.train_scorer --consensus-file data/processed/team_v2/consensus_dataset.csv --output-dir models/span_scorer_v2 --folds 5 --batch-size 8 --epochs 4 --dry-run
python -m final_eval.scorer.train_scorer --consensus-file data/processed/team_v2/consensus_dataset.csv --output-dir models/span_scorer_v2 --folds 5 --batch-size 8 --epochs 4
```

The dry-run checks fractional targets and family partitions without initializing/downloading encoder weights. Check that fractional vote scores survive when the panel disagrees. Soft targets, strict-majority ties, and unweighted BCE are the defaults. `--hard-labels` and `--pos-weight` are explicitly named ablations, each requiring a different output directory.

Outputs: `training_metadata.json`, `tokenizer/`, `foldN/scorer.pt`, `foldN/config.json`, `foldN/category_prior.json`, `split.json`, `scorer_oof_predictions.csv`, `scorer_report.json` under `models/span_scorer_v2/`.

After an interruption, repeat the exact training command with `--resume`. Completed folds are reused only if the input/configuration/source/environment metadata matches. This resumes at fold boundaries, not halfway through an epoch. A changed configuration needs a new directory.

Read held-out category-prior versus scorer results before claiming learned value. Current F1 uses a stated 0.5 threshold; a tuned threshold, a richer feature baseline, calibration plots, or extra seeds are additional analyses, not automatically completed by this command.

## 5. Freeze one cohort for every method

```bash
python -m final_eval.preparation.cohort --problems data/processed/team_v2/problems.csv --split models/span_scorer_v2/split.json --output outputs/final_eval/main/cohort.csv --max-questions 160
```

This selects whole label-eligible families in dataset round-robin order, without target outcomes. The cap is an upper bound, not a promised exact sample size. Inspect `cohort.csv.cohort.json` for per-dataset and independent-family counts. Questions without a held-out scorer assignment are counted as excluded; they are not silently treated as trained-model test examples.

For a server smoke test, first create a separate small cohort (for example cap 16) and separate output paths. Do not promote that smoke cohort to a final result after tuning on it. Keep alpha at the declared 0.5 unless a separate validation protocol has been fixed.

## 6. Compress, one GPU stage at a time

First inspect the task count:

```bash
python -m final_eval.compressor.compress --input-csv outputs/final_eval/main/cohort.csv --output-csv outputs/final_eval/main/ours.csv --model-dir models/span_scorer_v2 --labels-file data/processed/team_v2/consensus_dataset.csv --methods ours grouped_hybrid grouped_signal random --ratios 0.75 0.5 --dry-run
```

On the server, execute that command without `--dry-run`. Then run each official baseline in a fresh process:

```bash
python -m final_eval.compressor.compress --input-csv outputs/final_eval/main/cohort.csv --output-csv outputs/final_eval/main/llmlingua.csv --labels-file data/processed/team_v2/consensus_dataset.csv --methods llmlingua --ratios 0.75 0.5
python -m final_eval.compressor.compress --input-csv outputs/final_eval/main/cohort.csv --output-csv outputs/final_eval/main/llmlingua2.csv --labels-file data/processed/team_v2/consensus_dataset.csv --methods llmlingua2 --ratios 0.75 0.5
python -m final_eval.compressor.compress --input-csv outputs/final_eval/main/cohort.csv --output-csv outputs/final_eval/main/selective_context.csv --labels-file data/processed/team_v2/consensus_dataset.csv --methods selective_context --ratios 0.75 0.5
python -m final_eval.preparation.merge_compressions --inputs outputs/final_eval/main/ours.csv outputs/final_eval/main/llmlingua.csv outputs/final_eval/main/llmlingua2.csv outputs/final_eval/main/selective_context.csv --expected-methods ours grouped_hybrid grouped_signal random llmlingua llmlingua2 selective_context --output outputs/final_eval/main/compressed.csv
```

The baseline adapters use official packages. Package loading, version-specific arguments, and short-prompt budget behavior still need real server validation. A failed method stays a recorded failure; it is never replaced by another algorithm.

`grouped_signal` uses train-fold category priors with the same grouping/bonus machinery as `grouped_hybrid`. `ours_signal` and `ours_scorer` remain available for controlled ablations; all legacy `ours` variants share numeric/unit/query bonuses unless `--no-bonuses` is specified. These names describe ranking components, not established quality.

Whole-question budgets use the final string and floor rounding. Native external outputs are kept intact and their actual budgets recorded. `--protect-interface` is an explicitly different own-method diagnostic; it is not a shared-interface external baseline comparison. Protection infeasibility is recorded, never silently overshot.

Repeating an identical compression command skips saved conditions. Changes to source, configuration, input, checkpoints, or packages require a new output. Failed compression conditions currently need a new stage output after the cause is fixed; do not merge the failed and corrected copies as duplicate keys.

## 7. Run diagnostics and target inference

```bash
python -m final_eval.compressor.metrics --input outputs/final_eval/main/compressed.csv --output-dir outputs/final_eval/main/intrinsic
```

Inspect `summary.csv`, `per_question.csv`, and `categories.csv`. Retention is occurrence-aware; black-box ambiguities have bounds and unresolved alignments have missing metrics. Distractor token mass differs from full-sentence survival. Legacy compression CSVs need `--labels-file ... --recompute-retention`; their old substring metrics are rejected as current evidence.

Before target inference, copy one example target config to a run-specific JSON and fill its **confirmed** fields:

- Hosted: `final_eval/configs/target.hosted.example.json`; put one key in the configured environment variable, not in JSON. Set the exact model ID, endpoint, request budget, and rate limit after access is confirmed. The request budget counts HTTP attempts across resumes of that run.
- Local: `final_eval/configs/target.local.example.json`; choose the actual model/path, pin a revision, and set supported dtype/device settings after the GPU pilot.

Assuming the confirmed config is saved as `outputs/final_eval/main/target.json`:

```bash
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --model-config outputs/final_eval/main/target.json --dry-run
python -m final_eval.compressor.evaluate --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --model-config outputs/final_eval/main/target.json
```

Use the small separate smoke cohort first to measure uncompressed accuracy, parse/truncation rates, memory, and throughput. The full-input reference is reused once per question. Identical query text under identical target settings can share a generation. Full responses, served model IDs where provided, finish reasons, timing, and usage are retained.

Repeat an identical evaluation command to continue unfinished conditions. Add `--retry-failed` to retry recorded failures. Authentication/quota/budget/model-not-found errors stop the run. The run-level usage ledger is conservative across restarts; separate processes/accounts still need team-level quota coordination.

## 8. Summarize, then interpret

```bash
python -m final_eval.compressor.evaluate --mode summarize --input-csv outputs/final_eval/main/compressed.csv --output-csv outputs/final_eval/main/responses.csv --report-dir outputs/final_eval/main/reports --focus grouped_hybrid
```

Read `coverage.csv` first. Main accuracy uses a common complete cohort across methods at each target/budget, with the full-input reference present. Intervals resample base families. `paired.csv` excludes over-budget conditions from paired fixed-budget comparisons and reports those exclusions and actual ratios. A sparse or empty compliant comparison cannot establish a win. CIs are unadjusted; predeclare primary tests and treat the rest as exploratory.

For the fresh same-model original/mask/deletion diagnostic, use its own output and bounded quota:

```bash
python -m final_eval.compressor.evaluate --mode deletion-check --input-csv data/processed/team_v2/per_model_labels.csv --output-csv outputs/final_eval/reliability/check.csv --model-config outputs/final_eval/main/target.json --n 30 --dry-run
```

Remove `--dry-run` only on the execution machine. This is a new same-target assay, not proof that a new model validates old-model mask labels. Its category-balanced sampling is diagnostic, not population-representative. The output includes complete responses and an agreement summary. Broader reliability statistics, interaction experiments, labeler-agreement reports, feature baselines, final figures, and claim-ledger automation remain follow-up work from the plan.

## 9. What to return, and when to stop

Return the preparation manifest, training metadata/report/OOF predictions/split, cohort manifest, compression CSVs and manifests, target config **without keys**, full response CSV and usage ledger, report CSVs, environment export, GPU details, and error summaries. Persist/download these before a Kaggle session ends.

- Source/duplicate/family mismatch: resolve the data export; do not continue to training.
- Model unavailable, API schema incompatible, or unsupported GPU dtype: adjust the explicit backend config and run a new smoke test.
- Truncated/poor uncompressed answers: resolve output budget/target adequacy before comparing compressors.
- Baseline failure or budget miss: investigate settings and record native results; do not manufacture equal-budget outputs by arbitrary truncation.
- Weak or negative results: keep the frozen comparison and bring back the paired tables. Do not choose a favorable metric, task subset, or new threshold from the test results.
- Extra server/quota unavailable: use the local-model path and narrow the scope according to the plan.

No Git commit, push, model upload, or overwrite of the legacy model is required by these steps.
