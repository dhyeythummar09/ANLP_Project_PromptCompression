# Final evaluation work

The submitted pipeline is preserved at its original paths, as recorded in commit `97dd6e7`. All 15 previously edited tracked files have been restored: the numbered dataset scripts, scorer, compressor, evaluator, utilities, README, configuration, requirements, environment example, and Git ignore rules.

This folder contains the separate correction and experiment code. Commands use `python -m final_eval...` from the repository root. The original `python -m src...` commands continue to refer to the submitted code.

## What this work requires

**Do not recollect the dataset.** Obtain the existing merged raw-label/response exports from the teammate who has them. The preparation commands read those saved files, regrade the answers, check duplicates and question families, and write separate analysis tables. They do not call an API. Original data, responses, model weights, and reported results stay available for historical reproduction.

The corrected experiment includes fractional vote targets and family-disjoint folds. Training a scorer for that experiment is a separate server job using the existing collected responses. It does not require new labeling calls. Existing checkpoint files remain available for historical runs; the corrected protocol needs its own weights and output directory.

Target-model evaluation is a later step that does generate new answers. Choose its backend, model, and bounded quota after server/API access is confirmed. No server hardware or API key is required to review this code or run the offline tests.

## What is in this folder

| Location | Purpose |
| --- | --- |
| `preparation/` | Import/regrade saved responses, recover verified question families, freeze a cohort, merge compression outputs. |
| `scorer/` | Corrected training targets and grouped folds; inspect/download the historical checkpoint explicitly. |
| `compressor/` | Budget checks, compression adapters, retention metrics, response evaluation, paired reports. |
| `inference/`, `utils/`, `spans/` | Helpers used by this experiment, isolated from the original pipeline. |
| `configs/` | Placeholder target configurations; fill confirmed settings in an ignored run directory. |
| `requirements*.txt` | Extra dependencies for this experiment; root requirements remain the submitted version. |
| `tests/` | Offline regression checks with fabricated fixtures and mocked generation. |

Start with the [execution runbook](RUNBOOK.md). It gives the order of server commands, required inputs, outputs to return, and checks that can stop an invalid run. The broader [research plan](../PROJECT_COMPLETION_PLAN.md) contains proposed experiments; that proposal does not mean every experiment has been implemented or validated.

```bash
python -m unittest discover -s final_eval/tests -v
```

GPU model loading, official baseline compatibility, provider settings, throughput, and research outcomes still require execution-machine checks. Offline tests do not establish that the new method beats a baseline.

## Which files belong to which work

- `final_eval/`: the new correction and experiment implementation, including its runbook and extra dependencies.
- `PROJECT_COMPLETION_PLAN.md`: the requested research plan.
- Root PDFs, PPTX, `final_results_by_dataset.md`, `final_results_old.md`, and `intrinsic_metrics_report.md`: your existing local reference material, left untouched.
- Ignored `data/`, `models/`, and existing results: historical artifacts, left untouched. New results use separate paths documented in the runbook.

The superseded broad edits were archived locally under ignored `outputs/repo_repair_backup/` before restoration. The new dataset-collection command was archived there rather than included in this folder. No commit, pull, push, training run, or new labeling campaign is needed to make this folder reviewable.
