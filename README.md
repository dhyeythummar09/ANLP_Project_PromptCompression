# Symbiote: critical-span scoring and reasoning-prompt compression

ANLP course project, IIIT Hyderabad, Monsoon 2026. Team: Pranav Srivastava,
Vikhyath Pattipaty, Chatrathi Abhinav and Dhyey Thummar.

**Checkpoint: 22 October 2026.** Preparation, five-fold training, seven-method
compression and two hosted-reader runs are finished. No dataset recollection,
retraining, GPU session or paid inference is needed to report these results.
Qwen has two recorded empty outputs; its technical completion gate remains false.

The scorer predicts counterfactual labels better than a cheap feature baseline.
Our compression loses to LLMLingua-2 on both readers/budgets. Treat this as an
empirical study with a negative compression result. Final numbers, figures,
limitations and report/PPT assignments are in [RESULTS.md](RESULTS.md).

Published models: [corrected v2 scorer](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2)
and [historical mid-evaluation scorer](https://huggingface.co/flashstep/reasoning-critical-span-scorer).
Both links and exact revisions are recorded in section 7. **HF publication is complete.**

## 1. Repository map

| Location | Purpose |
| --- | --- |
| `src/dataset_creation/`, `config.py`, `.env.example`, root `requirements.txt` | Original submitted collection pipeline/configuration, unchanged |
| `src/scorer/`, `src/compressor/`, `src/utils/` | Historical submitted implementation, unchanged |
| `final_eval/preparation/` | Recover source families, regrade saved responses, deduplicate votes, prepare consensus and freeze cohorts |
| `final_eval/scorer/` | Soft-target scorer, family-safe partitions, checkpoint loader/export support |
| `final_eval/spans/`, `final_eval/compressor/` | Shared extraction, grouping, exact budgets, baseline adapters, target evaluation and intrinsic diagnostics |
| `final_eval/inference/`, `final_eval/utils/` | Explicit provider settings, resumable responses, strict grading, provenance, timing and statistics |
| `final_eval/jarvis_*.py`, `final_eval/Start-*.ps1` | Executed access, calibration, pilot and main orchestration; later stages depend on diagnostic evidence |
| `final_eval/kaggle_*.py`, `build_kaggle_main.py`, `repair_llmlingua.py` | GPU setup/transfer/compression and the actual compatibility repair |
| `final_eval/analysis/` | Agreement, feature baseline, retention checks, full-cohort comparison, offline audit and exhibit publication |
| `final_eval/tests/` | Offline regressions, including the intentionally failed historical repair fixture |
| `notebooks/01_eda_counterfactuals.ipynb` | Original EDA |
| `notebooks/02_train_scorer.ipynb` | Executed training workflow pinned to the original starter payload |
| `notebooks/03_main_compression.ipynb` | Four-cell main workflow pinned to the executed transfer |
| `environments/` | Audit requirements and observed Kaggle constraints |
| `results/` | Small verified tables, plots and metadata for Git/report sharing |
| `data/`, `models/`, `outputs/` | Ignored inputs, full weights, raw responses, failures, transfer archives and audits |

`final_eval/` stays separate because manifests, notebook bootstraps and imports
use those executed paths. Moving it into `src/` would alter reproduction.
Historical diagnostic scripts are retained evidence, not alternate next steps.

## 2. Back up artifacts before moving machines

A fresh Git clone has code and `results/`, **not** labels, raw responses, weights
or frozen transfer archives. Copy these separately, retaining relative paths:

1. `data/legacy/team/`, `data/per_model_labels.csv`, `data/processed/`.
2. `models/span_scorer_v2_20261021/`: five full checkpoints (about 1.33 GB),
   tokenizer, splits, metadata, OOF predictions and profiles.
3. `outputs/final_eval/`, `outputs/offline_preparation_20261021/`,
   `outputs/imports/`, `outputs/handoffs/`, `outputs/repo_cleanup_audit/`.
4. `outputs/hf_release/remote_release_verification.json`, the published-model
   verification record. The extra local upload staging copy is optional to retain;
   the verified release can also be restored from HF as described in section 7.

Actual `.env` and `final_eval/jarvis.env` stay private and are unnecessary for
saved-result verification. `final_eval/jarvis.env.example` is the hosted-provider
template; the original root template belongs to historical collection. The public
v2 release is a verified copy of the local five-fold model. The older HF repo
remains a separate historical artifact; do not substitute it for the corrected weights.

## 3. Verify the completed study now: CPU, no new inference

Run from the repository root, using the existing `.venv`. For a fresh audit-only
environment install `python -m pip install -r environments/requirements-audit.txt`.
The full test suite also needs the project's ML dependencies because some tests
import scorer/compression modules; the audit itself never loads model weights.

PowerShell (on Linux use the environment's `python`):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s final_eval/tests -v
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location) 'outputs/repo_cleanup_audit/tiktoken_cache'
.\.venv\Scripts\python.exe -m final_eval.analysis.audit --output-dir outputs/reproduction_check_01 --retokenize --checkpoints
```

Use a **new directory** for each audit. It recounts original/compressed tokens,
checks the 2,240-condition grid, pilot separation, prepared inputs and held-out
fold assignments, recomputes shared scorer metrics, regrades all 4,800 answers,
checks query/gold/model provenance, reproduces both summary policies and agreement
tables, totals cost estimates and hashes five checkpoints. First use of
`cl100k_base` may download the small public tokenizer encoding.

Expected: 160 main questions/125 families, 11 excluded pilot families, 2,240
compressed prompts, 2,400 response records per reader and zero provenance/grade
mismatches. The two Qwen empty outputs remain failures.

To rebuild shareable exhibits after a passing audit:

```powershell
.\.venv\Scripts\python.exe -m final_eval.analysis.publish --audit-dir outputs/reproduction_check_01 --output-dir outputs/report_exhibits_01
```

`results/` is already published. The command requires a new destination and
never replaces originals. [results/manifest.json](results/manifest.json) maps
every exhibit to its source/hash/size; [results/verification.json](results/verification.json)
records the checks. Saved tables can be reproduced; fresh training or hosted
generations cannot be promised bitwise identical, including at temperature zero.

## 4. Executed workflow, in progressive order

These are completed stages, not instructions to repeat the project.

### A. Original collection and mid-evaluation

The numbered `src/dataset_creation/` scripts loaded 1,000 problems, extracted
spans, distributed counterfactual API labeling across the team and built the
historical consensus. Original training/compression code and EDA remain for
explaining the submission. **Do not rerun collection.** Older intrinsic-only
summaries and proposal claims are historical and not directly comparable to
the corrected final downstream experiment.

### B. Recovery and strict regrading on CPU

The team's export was called `consensus_dataset.csv` but retained per-model
original/perturbed answers. It was already consensus-filtered; it is not the
complete four-worker raw collection. Combine recoverable responses with the
existing local shard without inventing missing votes. GSM-IC family recovery
used the original public source, pinned to revision
`a7c848ae516f8df9712abc5425aefc553489d720`; all 200 prompts/clean pairs/golds matched.

Historical preparation command, reconstructed into a **new** output directory:

```bash
python -m final_eval.preparation.prepare --labels data/legacy/team/smoke_20261021/consensus_dataset.csv data/per_model_labels.csv --problems data/processed/team_20261021/problems_recovered.csv --output-dir data/processed/reproduction_01 --minimum-models 2
```

Completed canonical outputs: `data/processed/team_v2_20261021/`. There are 38,603
unique model/span votes, 37,979 eligible and 624 original-answer failures;
13,966 reliable span positions over 912 questions/778 families. The complete
problem table has 1,000 questions/848 families. Strict regrading changes 1,125
shared perturbed-outcome labels. Agreement/coverage and the source audit are
under `outputs/offline_preparation_20261021/`.

### C. Five-fold scorer training on free Kaggle T4

The [training notebook](notebooks/02_train_scorer.ipynb) contains the actual
bootstrap, commands and export. Its starter archive is
`outputs/handoffs/symbiote_kaggle_start_20261021.zip`, preserving **training-time**
source before later comparator fixes. All checkpoints are already local: restore
them when moving machines. Do not train again merely because a GPU session ended.

Actual settings: soft targets, unweighted BCE, seed 42, five family-disjoint
folds, .15 validation fraction, four epochs, batch 8, LR 3e-5, max length 512.

```bash
python -m final_eval.scorer.train_scorer --consensus-file data/processed/team_v2_20261021/consensus_dataset.csv --output-dir models/span_scorer_v2_20261021 --folds 5 --epochs 4 --batch-size 8 --bootstrap 500
python -m final_eval.analysis.feature_baseline --consensus-file data/processed/team_v2_20261021/consensus_dataset.csv --model-dir models/span_scorer_v2_20261021 --output-dir outputs/final_eval/feature_baseline
```

Those historical paths are occupied. For an intentional training reproduction,
restore the starter archive into an isolated workspace. The notebook's resume
reuses only metadata-compatible folds. Original training fingerprints/weights
remain intact despite later overall inference-source changes.

### D. Compression pilot and repairs

The frozen pilot had 16 questions/11 families and 224 conditions. Initially
198 passed: six methods passed 32/32 each; LLMLingua had 12 cache errors and
14 budget misses. Modern Transformers and short iterative windows caused the
observed incompatibility. The native whole-question `iterative_size` repair
in `final_eval/repair_llmlingua.py` is guarded by the model's context limit.
Budget control searches native settings; it does not arbitrarily truncate text.

The first repair launcher failed on Windows CRLF/Linux LF hashing. Corrected
component/provenance checks were regression-tested. Only 32 LLMLingua conditions
were recomputed; the other 192 remained unchanged. Final valid pilot:
`outputs/final_eval/pilot_llmlingua_window_v2/`, 224/224 passing conditions.
Failed outputs/source snapshots/fixtures remain preserved.

### E. Reader qualification and main compression

Access checks did not qualify a reader. DeepSeek's 4,096-output pilot completed
115/240 conditions and truncated 125; two bounded settings checks also failed,
so it was stopped. GPT-OSS's fallback and full pilot passed. Qwen's initial
4,096 pilot had three truncations. An 8,192-cap diagnostic completed the queries,
including a still-wrong compressed answer, then a new **uniform** 8,192 pilot
completed all 240 conditions. Both successful pilots had 16/16 correct originals.

Main selection used dataset round-robin strata and complete eligible families,
without outcome-based selection or overlap with the 11 pilot families. It froze
160 questions/125 families. Kaggle main compression produced seven methods at
.50/.75 retained-question-token ceilings: 2,240 valid outputs.

The [main notebook](notebooks/03_main_compression.ipynb) uses the preserved
`outputs/handoffs/symbiote_main_stage_c_v1.zip` and verifies restored checkpoints.
`python -m final_eval.build_kaggle_main` renders its four cells from that frozen
archive without replacing executed source. Its setup/run/export module is
`final_eval.kaggle_main`. Never overwrite a historical transfer version.

Observed environment: Python 3.13.15, CUDA Torch 2.11.0+cu128, Transformers
5.16.1, LLMLingua .2.2, Selective Context .1.4 and spaCy 3.8.16. See
[Kaggle constraints](environments/kaggle-20261021.constraints.txt).
Preserve the image's CUDA Torch build. Selective Context's obsolete spaCy/click
pins were explicitly overridden using its official wheel with `--no-deps`;
the setup helper records this exception. Installing only the old broad baseline
requirements is not the reproduced setup.

### F. Hosted main answering runs from the laptop

Both used the **same saved** `outputs/final_eval/main_gptoss_v1/compressed.csv`.
The laptop made HTTP calls and recorded results; it did not run the target model.

| Reader | Fixed settings | Evidence |
| --- | --- | --- |
| `gpt-oss-120b` | Temperature 0, output cap 4,096, recorded default reasoning controls | `outputs/final_eval/main_gptoss_v1/target_gptoss/` |
| `qwen-3.8-27b-fp8` | Temperature 0, output cap 8,192, reasoning effort `low` | `outputs/final_eval/main_qwen8192_v1/target_qwen/` |

Historical launchers: `Start-JarvisMain.ps1 -Model gptoss` and `-Model qwen`.
They are **not current commands to launch**: retries can make new requests.
Main ceilings were 3,000 cumulative HTTP attempts; Qwen also had a persistent
INR 500 conservative token-cost guard. Exact public configs and response manifests
remain under outputs. Secrets are needed only for an intentional new hosted run.

Each reader records 160 originals + 2,240 compressed conditions. Exact-query
cache reuse avoids duplicate calls. One GPT-OSS and two Qwen truncated logical
rows were filled offline from already saved identical successful queries;
their answers remained incorrect. Paid attempts and pre-resume files were
archived. Two Qwen empty LLMLingua outputs had no saved successful equivalent
and remain failures. Never select a saved response because it is correct.

### G. Statistics and final exhibits

The original summary uses the common provider-success cohort across methods:
Qwen .50 has 158 questions, other cells 160. Invalid final parses count wrong.
An explicit full-160 companion was added after failure review, counting provider
failures as incorrect. Both policies remain; empties are never called successes.

The audit in section 3 reruns those functions and compares tables to originals
at tolerance 1e-12. Individual entry points: `final_eval.compressor.evaluate
--mode summarize`, `final_eval.analysis.strict_results` and
`final_eval.analysis.results` (`--help`). The last module's retention diagnostics
were evaluated for GPT-OSS, not an equivalent Qwen retention experiment.

## 5. Viva reading guide

Read `src/dataset_creation/03_counterfactual_labeling.py`, then these files
under `final_eval/`: `preparation/schema.py`, `scorer/data.py`,
`scorer/train_scorer.py`, `compressor/compress.py`, `compressor/budget.py`,
`compressor/evaluate.py`.

- An eligible labeler answers the original correctly. Its vote measures whether
  masking one span causes an operational answer failure under the grader. The
  soft target is the fraction of eligible critical votes (minimum two labelers),
  not universal necessity.
- DistilBERT contextual tokens are pooled over each span, then scored with a
  sigmoid head and unweighted BCE on fractional targets. Binary evaluation uses
  target > .5. Related clean/distracted questions share train/validation/test families.
- Compression uses each question's held-out scorer. `ours` mixes criticality and
  normalized DistilGPT2 word surprisal at alpha .5 plus structural bonuses.
  `grouped_hybrid` adds phrase grouping. `grouped_signal` substitutes a training-fold
  category prior while retaining the other machinery: the matched cheap ablation.
- Remove low-priority groups, preserving order, until the rendered string meets
  `floor(ratio * original_cl100k_tokens)`. Group priority is its maximum word score.
  External methods use official implementations with disclosed native budget search.
  Common ceilings do not imply equal realized counts or equal request costs.
- Single-span masking sensitivity need not imply sufficiency after many deletions.
  Redundant facts can each look dispensable alone. This is a plausible explanation
  for the scorer/compression gap; causal validation was not performed.

## 6. Cleanup record and remaining deliverables

All old active sources/documents/notebooks and local proposal/mid-report/result
references were saved before cleanup in
`outputs/repo_cleanup_audit/20261022T061303Z/pre_cleanup_sources_and_documents.zip`.
It excludes actual secret env files. Its source manifest/checksums and adjacent
inventories preserve original artifact hashes and 14 Python blocks extracted
from Markdown. Include this ignored archive in the separate project backup
for the full historical notes and old intrinsic result summaries.

Original submitted `src/`, root config, requirements and environment template
remain byte-for-byte. Executed core logic, data, responses and weights are
preserved. Training/main notebooks moved into `notebooks/` with code cells
unchanged. Active documentation is this guide and RESULTS.md.

Cleanup verification passed 119 offline tests and confirmed unchanged hashes
for 618 original artifacts, 19 historical source/config files and 29 executed
core files. The detailed record is
`outputs/repo_cleanup_audit/20261022T061303Z/final_verification.json`.

Finish report/PPT and reconcile actual Jarvislabs billing. The v2 scorer/model
card is published and verified (section 7). Existing results suffice for the scoped study.
Additional experiments should use separately versioned outputs and disclose
their selection; they are not mandatory prerequisites to writing up this work.

### Optional seven-day extension, proposed after cleanup

This extension is **not executed**. Preserve the completed study above as v1.
There are 736 reliably labeled questions / 642 families outside its main and
pilot families, so no labeling campaign or scorer retraining is required for
a structural-compression follow-up. Each question must still use its held-out scorer.

1. Days 1–2: inspect matched failures and score/bonus scales; audit whether query,
   answer choices, negation and relations survive. Test at most two changes to
   ranking/structural grouping on a small, separately declared development pool.
   Existing scorer-only/bonus/interface flags are ablations, not measured improvements.
2. Days 3–4: freeze one candidate and a fresh 160-question, family-disjoint test
   outside both development and the old study. Compare old hybrid, new hybrid,
   matched new non-neural ablation and LLMLingua-2 at both ceilings on one fixed
   reader first. Preserve exact budgets, failures and all negative results.
   Do not tune the candidate on this new test or rebrand the old main as untouched.
3. Day 5, only if time/confirmed quota permits: a small predeclared four-arm
   deletion diagnostic (original, delete A, delete B, delete A+B) can investigate
   the single-span/joint-deletion mismatch. This requires some new target calls,
   not a replacement training dataset. An additional reader is optional.
4. Days 6–7: stop experiments, finish report/slides and reconcile actual billing.

Success criteria are a reproducible gain over the old compressor, evidence for
or against the neural contribution, and a clearer failure explanation. A win
over LLMLingua-2 remains an unconfirmed hypothesis. Check actual remaining
credit before setting a bounded request/token allowance for any new calls.

## 7. Hugging Face models: published and verified

**Complete:** the user uploaded the corrected five-fold v2 scorer, and read-only
Hub metadata verification confirmed all **67 release files**, including all five
checkpoints, match the local prepared package. Both repositories below are public.

| Version | Model repository | Fixed revision |
| --- | --- | --- |
| Corrected final-study v2 | [Pranav-Sriv/reasoning-critical-span-scorer-v2](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2) | [70cc329f10ef6415e152c0bcdcef8894f8dc0d69](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2/tree/70cc329f10ef6415e152c0bcdcef8894f8dc0d69) |
| Historical mid-evaluation model | [flashstep/reasoning-critical-span-scorer](https://huggingface.co/flashstep/reasoning-critical-span-scorer) | [c24d8ce9ef2cfbdc7b11185958dee5810327e432](https://huggingface.co/flashstep/reasoning-critical-span-scorer/tree/c24d8ce9ef2cfbdc7b11185958dee5810327e432) |

The older URL was recovered from the archived original project README, and its
recorded revision was checked on the Hub. Preserve it for the historical submission;
use v2 for reproducing the corrected final study.
`final_eval.scorer.hf_artifact` intentionally remains the legacy inspection
helper; use the fixed v2 download command below for the current model.

The authoritative local model remains `models/span_scorer_v2_20261021/`: 27
unchanged model files, about 1.33 GB. The published release additionally contains
a custom loader, model card, inference requirements, exact source snapshot,
comparison evidence and checksum manifest. It contains no private env files or
provider keys. The local upload copy at
`outputs/hf_release/span_scorer_v2_20261021/` is kept unchanged so its checksums
continue to match the published revision.

Verification: [results/metadata/huggingface_release.json](results/metadata/huggingface_release.json).
Remote LFS SHA256 values and Git blob IDs/sizes were checked against the local
release without downloading 1.33 GB again or running inference. The only extra
remote file is HF's `.gitattributes`. Packaging-time records remain historical;
the separate publication verification records the completed upload.

Restore the fixed v2 release in a fresh checkout with the documented
[HF download CLI](https://huggingface.co/docs/huggingface_hub/guides/download):

```powershell
hf download Pranav-Sriv/reasoning-critical-span-scorer-v2 --revision 70cc329f10ef6415e152c0bcdcef8894f8dc0d69 --local-dir models/span_scorer_v2_20261021
```

The model is public, so authentication is not required for download. Do not
download over different existing artifacts. The fold/tokenizer/split layout
matches existing project commands. After restoring, verify payload hashes:

```powershell
python models/span_scorer_v2_20261021/load_scorer.py
```

The same verifier can run on the local upload copy. The downloaded model card
includes direct scoring examples and inference dependencies. These are custom
span-scoring models, not a standard Transformers AutoModel/token classifier.
Use each study question's held-out assignment from split.json; choosing a best
fold or silently ensembling would change the reported protocol. A new-input
fold/ensemble policy is a separate unmeasured deployment choice.

The upload initially stalled in Xet; the completed user upload used the HTTP
fallback with `HF_HUB_DISABLE_XET=1`. No repository-creation or upload command
remains to run for this release. Further analysis/reporting needs no HF changes.
Changed checkpoints or loader/source artifacts would require a separately
recorded release revision; preserve the fixed versions above.

The local packaging helper is retained for future use:
`python -m final_eval.model_release --output-dir outputs/hf_release/new_version_name`.
It verifies inputs and copies to a new folder; it never uploads or retrains.
Portable loading passed a tiny synthetic-checkpoint roundtrip and the actual
saved tokenizer loaded offline. Publication verification adds no claim of a
new real-checkpoint forward pass; the original Kaggle experiment remains unchanged.
