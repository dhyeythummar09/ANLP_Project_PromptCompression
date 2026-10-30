# Symbiote: reasoning-critical prompt compression

**Final research model: combined v3 (`teacher_cf`). Experimental work is complete.**
Report/slides, actual billing reconciliation and v3 release packaging remain.

We study whether masking a reasoning span changes a reader's correct answer,
combine eligible labeler votes, and learn those soft sensitivity scores. The
final student combines these labels with dense LLMLingua-2 retention supervision.
It compresses a new question without calling a reader/API.

**Conclusion:** criticality labels are predictable (held-out span AUC .9074),
but their added downstream benefit is not established. V3 improves point accuracy
over the earlier hybrid on its paired test; it does not establish overall
superiority to LLMLingua-2. This is a completed empirical finding, not proof
that criticality can never help. [RESULTS.md](RESULTS.md) contains the full
methods, comparisons, conclusions and report-ready figures.

## Repository structure

```text
src/                         Original submitted collection/scorer/compressor
config.py                    Original collection configuration
requirements.txt             Original submitted dependency list
environments/                Recorded audit/Kaggle dependency versions
notebooks/01_*.ipynb          Historical exploratory analysis
notebooks/02_*.ipynb          Corrected v2 scorer workflow
notebooks/03_*.ipynb          Corrected v2 compression workflow
notebooks/04_*.ipynb          Historical-checkpoint/v2 follow-up compression
notebooks/05_*.ipynb          Completed v3 Kaggle run: frozen payload required
final_eval/
  preparation/, scorer/      Corrected data and v2 span scorer
  spans/, compressor/        Shared extraction, exact token budgets, baselines
  inference/, utils/         Reader client, grading, statistics, artifacts
  analysis/                  Rebuild earlier study tables/figures
  v3/                        Final student and its controlled experiment
    model.py                 DistilBERT encoder, word/span heads, pooling/loss
    train.py                 Matched initialization, training, epoch selection
    data.py                  Saved-label alignment and family-disjoint splits
    compression.py           Word selection, inference, evaluated references
    reader.py                Hosted evaluation, exact-query cache, spend guard
    report.py                 Shared selector freeze and paired evaluation
    pipeline.py               Ordered training/evaluation/export stages
    audit_return.py          Verify saved outputs; regenerate public exhibits
    artifacts.py             Hashes, stage identity, resume locks
    benchmark.py             Warm compressor measurements
    build_kaggle.py           Prepare a new transfer, not the model algorithm
    kaggle_*.py               Install/recover/run the transferred experiment
    configs/                  Declared experiment and exposure records
    tests/                    Implementation checks, not extra research models
  *.py, Start-*.ps1           Executed v2/follow-up/reporting/qualification tools
results/                     Saved report tables, provenance and figures
  v3/20261030/                Final test, ablation, training metadata and plots
  project_summary/            Cross-study comparison CSV, source hashes
  exploratory_20261029/       Earlier diagnostic outcomes; not neural v3
  tables/, metadata/          Earlier seven-method/two-reader study
  followup*/, profiles/       Checkpoint/component/reliability/runtime evidence
docs/CODE_MAP.csv            File-by-file purpose index
README.md                    Code guide, reproduction and push instructions
RESULTS.md                   Scientific record and report/PPT handoff
```

`data/`, `models/`, `outputs/`, `logs/`, `.env` and `.venv/` are **local,
Git-ignored artifacts**. A plain Git clone does not include datasets, raw reader
answers or trained weights. The artifact addresses/checksums and replay roles
are listed in [results/reproducibility.json](results/reproducibility.json).

The repository contains more than the final network because the report compares
several executed approaches. The earlier reader checks, baseline repairs,
follow-ups and their tests support those measurements. They are not additional
models to train or files to explain individually in a 20-minute presentation.
Original `src/`, model-producing v2/v3 code, data and checkpoints are preserved.
The unused v3 pair-assay implementation and redundant planning documents were
removed; the measured exploratory outcomes and exact executed snapshots remain.
Retired notes and the old sync backup are retained only in ignored local
`outputs/` storage; they are not part of the pushable repository.

## A 20-minute code explanation

| Time | Open | Explain |
| --- | --- | --- |
| 0-2 min | This README and RESULTS | Research question; single-span answer sensitivity versus many simultaneous deletions |
| 2-5 min | `src/dataset_creation/03_counterfactual_labeling.py`, `final_eval/v3/data.py` | Original-correct labeler eligibility, soft vote fractions; cached teacher alignment; no sibling-family leakage |
| 5-10 min | `final_eval/v3/model.py`, `train.py` | Contextual DistilBERT, pooled word/span heads; matched initialization; teacher BCE versus teacher BCE + .2 span BCE; validation-only epoch selection |
| 10-14 min | `final_eval/v3/compression.py` | `infer_text`, `select`, `generate`: rank retained words, preserve order, respect the actual tokenizer ceiling; span head is auxiliary training supervision |
| 14-17 min | `final_eval/v3/reader.py`, `report.py` | Frozen common selector, original controls, failures-as-wrong, strict grading, paired family intervals; no test-guided model selection |
| 17-20 min | RESULTS figures and `audit_return.py` | Main results, teacher-only ablation, timing scope, what improved and what was not demonstrated |

For a filename outside that route, use [CODE_MAP.csv](docs/CODE_MAP.csv).
The actual neural network is `DenseStudent`; CLI, transfer, resume, billing and
plotting code are experiment infrastructure. Tests with `fixture`/`failed` in
their names intentionally simulate faults to verify fixes; they are not broken
runners used to obtain reported answers.

## Versions and final model

| Version | What changed | Where it belongs |
| --- | --- | --- |
| Original / mid-eval | Multi-model masking, consensus, DistilBERT span scorer, surprisal/rule hybrid; historical source already supports soft labels | Historical evidence in `src/` and the earlier public checkpoint |
| Corrected v2 | Strict regrading/deduplication/eligibility, reliable soft fractions, unweighted BCE, family-safe folds, shared extraction, grouping and checked budgets | Executed baseline/follow-up study; keep its real chronology |
| V3 teacher-only | Dense retention supervision from saved LLMLingua-2 outputs | Matched experimental control |
| Final v3 combined | Same student and initialization, plus .2 x corrected criticality-span BCE | One final research artifact, with the ablation disclosed |

V1/v2 scorer, signal and hybrid are configurations, not three separately trained
networks. V3 has two separately trained checkpoints; only the combined one is the
intended final research release. Selecting it for that purpose does not establish
that it is statistically better than the teacher-only control.

Both v3 arms use 66,364,418 parameters, four epochs, batch 8, learning rate 3e-5,
seed 42, and minimum validation teacher BCE for checkpoint selection. Teacher-only
development results choose the shared word policy. The span head influences the
shared encoder during training; final selection ranks the word head's outputs.
Do not claim that every compressed output keeps annotated spans indivisible.

Current Qwen test: 64 held-out families, one question each; original 59/64 correct.

| Method | .50 retained-question-token ceiling | .75 ceiling |
| --- | ---: | ---: |
| Final v3 combined | 27/64 (42.19%) | 55/64 (85.94%) |
| Teacher-only control | 24/64 (37.50%) | 53/64 (82.81%) |
| V2 grouped hybrid | 17/64 (26.56%) | 49/64 (76.56%) |
| LLMLingua-2 | 31/64 (48.44%) | 53/64 (82.81%) |

Primary combined-minus-control contrast at .50: **+4.69 pp, 95% CI
[-4.69, +15.63]**. No added criticality gain established. At .75 the combined
model preserves 55/59 originally correct answers; that is not 90% overall
accuracy or a per-prompt guarantee. Earlier tables use other cohorts/readers and
must remain separate panels. All rows, actual ratios and caveats are in RESULTS.

## Use the saved final model locally

From the repository root, using the existing project environment:

```powershell
.\.venv\Scripts\python.exe -B -m final_eval.v3 infer --run-dir outputs/imports/v3_20261030_completed/run --ratio .75 --text "Alice has 8 apples and gives away 3. How many apples remain?"
```

This uses `run/models/teacher_cf/best.pt`, checks its identity, and makes no API
calls. It returns compressed text, not a guaranteed correct downstream answer.
Dependencies for the model/experiment are in `final_eval/v3/requirements.txt`;
observed GPU versions are recorded in the returned environment metadata.

## Reproduce the completed study

There are three different operations; none requires recollecting the data.

1. **Reproduce reported scores/figures from saved answers.** Keep the original
   494 MB return ZIP and status file under the addresses below, then run:

   ```powershell
   .\.venv\Scripts\python.exe -B -m final_eval.v3.audit_return --archive outputs/imports/v3_20261030_completed/symbiote_v3_results.zip --status outputs/imports/v3_20261030_completed/v3_status.json
   ```

   This checks archive/source/stage/checkpoint provenance, family splits, token
   ceilings, grades and the primary interval. It makes no training/API calls.
   A public tokenizer encoding must be cached or downloadable.

2. **Replay training/evaluation on Kaggle.** Use the matching frozen pair in
   `outputs/handoffs/v3_20261030/`: upload `symbiote_v3_kaggle.zip` as a private
   Kaggle dataset, import `Symbiote_V3_Run_All.ipynb`, attach that dataset, enable
   Internet and T4 GPU. Add the Jarvis key through Kaggle Secrets only if new
   hosted answering is intended. The versioned notebook 05 is the matching
   launcher; it sets `HF_HUB_DISABLE_XET=1`. Use saved compatible output to resume
   with its cache/ledger. The frozen input contains the exact executed source,
   corrected data and reference metadata; pinned v2 weights download separately.
   Running against the live reader can incur spend and need not reproduce
   identical answers. GPU retraining is not promised bit-identical.

3. **Inspect/reproduce the earlier experiments.** `src/` is the submitted original
   workflow; notebooks 02-04 and the corresponding `final_eval` modules reproduce
   v2 and the old-checkpoint comparison. `final_eval.analysis.audit` verifies the
   earlier saved study. `compare_downstream`, `component_ablations`,
   `followup_summary`, `benchmark_runtime` and `prepare_step5_spans` expose their
   existing `--help` interfaces. Their input/response manifests, paired outcomes,
   costs and profiles remain under `results/` and the ignored original output
   paths. The exploratory archive includes its exact prototype scripts and
   `REPRODUCE.txt`; these are not installed as final model code.

The final return includes both best checkpoints, tokenizer/config initialization,
all raw answers/cache/usage and the exact executed implementation. Kaggle can be
closed. Full optimizer resume additionally requires its saved `last.pt` files,
which were intentionally excluded from the downloadable return.

The final 64 families are now exposed. Current split preparation reads published
exposure ledgers; only nine old-pool families remain unexposed. Use the frozen
input for a historical replay. A new improvement study requires a newly declared
independent evaluation cohort, not relabeling these families fresh.

## Artifacts and publication

| Artifact | Address |
| --- | --- |
| Final return and extracted run | `outputs/imports/v3_20261030_completed/` |
| Matching frozen Kaggle input/notebook | `outputs/handoffs/v3_20261030/` |
| Earlier exploratory evidence archive | `outputs/imports/exploratory_20261029/` |
| Existing corrected data / v2 models | `data/processed/team_v2_20261021/`, `models/span_scorer_v2_20261021/` |
| Report/slides inputs | RESULTS and `results/` tables, figures, metadata and manifests |
| File purposes / cleanup verification | `docs/CODE_MAP.csv`, `results/repository_cleanup.json` |

Earlier public models: [original/mid-eval](https://huggingface.co/flashstep/reasoning-critical-span-scorer/tree/c24d8ce9ef2cfbdc7b11185958dee5810327e432)
and [corrected v2](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2/tree/70cc329f10ef6415e152c0bcdcef8894f8dc0d69).
**V3 is not yet uploaded to HF.** Package its correct two-head loader, tokenizer,
config, weights, hashes and model card; the old v2 release loader is incompatible.
Keep one combined final release, with the teacher-only control in the experiment
archive. Actual Jarvis billing remains to reconcile; estimates are not invoices.

## What to push

The Git changes contain code, two current Markdown documents, the code index and
report evidence. Many deletions remove obsolete plans; they do not remove dataset
collection, old results or trained models. Review the complete diff, not just the
file count. Required historical experiment helpers and regression tests remain.

```powershell
git add -A
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
# After reviewing the staged list:
git commit -m "Finalize Symbiote v3 code, verified results and reproducibility guide"
git push origin main
```

The reviewed tree excludes `.env`, datasets, model weights, raw output archives,
virtual environments and caches. Do not use `git add -f` on those. Keep the listed
private input/result archives when transferring machines; Git alone cannot
restore ignored data. No commit or push has been performed by the assistant.
