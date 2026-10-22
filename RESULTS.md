# Final results and report/presentation handoff

Verified from saved artifacts on 22 October 2026. Reproduction and code/viva
guide: [README.md](README.md). Exhibits below are small copies suitable for Git;
raw generations, training outputs and failures remain at their original paths.

Published models: [corrected final-study v2](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2)
and [historical mid-eval model](https://huggingface.co/flashstep/reasoning-critical-span-scorer).
HF publication is complete; exact revisions and verification are in section 9.

**Assessment:** the implemented, audited empirical study is complete enough to
write up. Critical-span prediction works well on the recovered labels. The
proposed compressor does **not** beat LLMLingua-2, and an advantage from adding
the neural scorer over the matched cheap compressor is not established.
Present this outcome honestly; a successful experiment can reject its hypothesis.

## 1. Data and scorer: the positive result

| Quantity | Final scope |
| --- | ---: |
| Original questions / canonical families | 1,000 / 848 |
| Recoverable deduplicated model/span votes | 38,603 |
| Eligible votes / original-answer failures | 37,979 / 624 |
| Consensus positions / questions / families | 13,966 / 912 / 778 |
| Shared held-out scored positions | 13,855 |
| Excluded positions | 111 random controls without full tokenizer-offset coverage |
| Changed shared perturbed labels after strict regrading | 1,125 |

The supplied team export was **already consensus-filtered**. These counts do
not establish complete pre-filter coverage from all workers. No votes were
recollected or fabricated. All 200 GSM-IC clean/distracted pairs and golds were
verified against the pinned original source. Pairwise common-eligible labeler
agreement is approximately 82.1–85.0%, Cohen's kappa .487–.578; consult coverage
before extrapolating beyond the recovered observations.

Five family-disjoint folds; soft vote-fraction targets; DistilBERT;
unweighted BCE; best validation checkpoint. All three scorers below use the
**same 13,855 held-out occurrences**.

| Scorer | ROC AUC | Average precision | Critical F1 | Soft-target Brier (lower better) |
| --- | ---: | ---: | ---: | ---: |
| Neural scorer | .9074 | .7133 | .6344 | .06059 |
| Feature ridge | .8217 | .5487 | .5422 | .08617 |
| Training-fold category prior | .7515 | .3574 | .0000 | .09976 |

Neural family-bootstrap AUC interval: [.8968, .9164], 500 replicates. The
.0857 AUC point gain over ridge is descriptive; no paired ridge-significance
test is claimed. Label prediction metrics do not demonstrate downstream utility.
Do not mix soft-target Brier here with hard-label Brier elsewhere in the scorer report.

Evidence: [feature comparison](results/metadata/feature_report.json),
[full scorer report](results/metadata/scorer_report.json),
[agreement tables](results/tables/label_agreement/pairwise.csv),
[preparation manifest](results/metadata/preparation.json).

## 2. Frozen downstream design and completion

160 questions / 125 families, excluding 11 pilot families. Eight datasets:
GSM-IC 50, GSM8K 15, BBH date understanding 16, formal fallacies 16,
logical deduction (three objects) 16, navigation 16, object counting 16 and
tracking shuffled objects (three objects) 15. Dataset sizes differ, so report
both pooled accuracy and the supplied dataset-macro/per-dataset results.

Seven methods × two retained-question-token ceilings (.50/.75) × 160 questions
= 2,240 compressed conditions, plus 160 original controls per answering model.
All compressed strings pass the recorded `cl100k_base` ceilings, independently
recounted. Methods use identical saved compressed text across readers.

| Answering model | Original correct | Final response statuses | Technical gate |
| --- | ---: | --- | --- |
| GPT-OSS-120B | 159/160 (99.375%) | 2,400 completed | Passed |
| Qwen-3.8-27B-FP8 | 158/160 (98.750%) | 2,398 completed, 2 empty | **Failed; empties retained** |

GPT-OSS uses temperature 0/output cap 4,096. Qwen uses temperature 0/output
cap 8,192/reasoning effort `low`. Settings are uniform across methods within
each reader; cross-reader differences cannot be attributed to model identity alone.

Exact-query cache fills resolved one GPT-OSS and two Qwen truncations using
already saved identical successful generations, without new calls. Reused
answers remained incorrect; original paid attempts/pre-resume files are retained.
The two Qwen empties are LLMLingua .50 navigation conditions with no saved
successful equivalent. A recorded condition is not proof of provider success.

## 3. Downstream accuracy: all methods, both readers

**Full planned-cohort companion:** every cell below has n=160. Provider failures
and invalid final-answer parses count incorrect. This offline policy was added
after reviewing failures and is shown alongside the original common-success
analysis, not silently substituted for it. Values are percentages.

| Method | GPT-OSS .50 | GPT-OSS .75 | Qwen .50 | Qwen .75 |
| --- | ---: | ---: | ---: | ---: |
| `ours` (ungrouped neural hybrid) | 36.250 | 73.125 | 46.250 | 71.875 |
| `grouped_hybrid` | 36.875 | 73.125 | 46.250 | 73.125 |
| `grouped_signal` (matched cheap ablation) | 31.250 | 71.250 | 40.625 | 72.500 |
| `llmlingua2_budgeted` | **60.000** | **90.000** | **69.375** | **95.625** |
| `llmlingua_budgeted` | 11.250 | 26.250 | 12.500 | 35.625 |
| `random` | 21.250 | 36.875 | 17.500 | 42.500 |
| `selective_context_budgeted` | 30.000 | 74.375 | 31.250 | 77.500 |

Original **common provider-success cohort** policy, retained separately:

| Reader | Ceiling | Common n | Grouped hybrid | LLMLingua-2 | Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS | .50 | 160 | 36.875 | 60.000 | 21.250 |
| GPT-OSS | .75 | 160 | 73.125 | 90.000 | 36.875 |
| Qwen | .50 | 158 | 45.570 | 69.620 | 17.089 |
| Qwen | .75 | 160 | 73.125 | 95.625 | 42.500 |

Invalid parses remain wrong in both policies. Complete seven-method common-cohort
tables: [GPT-OSS](results/tables/common_success/gptoss/accuracy.csv) and
[Qwen](results/tables/common_success/qwen/accuracy.csv).
Full-cohort [per-dataset accuracy/conditional preservation](results/tables/full_cohort/accuracy.csv),
[dataset-macro accuracy](results/tables/full_cohort/macro_accuracy.csv), and
[failure coverage](results/tables/full_cohort/coverage.csv) are also available.

### Paired differences and defensible interpretation

Full-cohort grouped hybrid minus LLMLingua-2, percentage points:

| Reader | Ceiling | Difference | 95% family-bootstrap interval |
| --- | ---: | ---: | --- |
| GPT-OSS | .50 | −23.125 | [−36.55, −7.97] |
| GPT-OSS | .75 | −16.875 | [−28.84, −6.49] |
| Qwen | .50 | −23.125 | [−35.76, −10.46] |
| Qwen | .75 | −22.500 | [−33.84, −13.33] |

Against `grouped_signal`, hybrid gains are +5.625/+1.875 points on GPT-OSS and
+5.625/+.625 on Qwen. **All four intervals include zero.** No reliable neural
contribution to compression is established. The hybrid beats random descriptively
and LLMLingua-1 in these cells; those findings do not overturn the LLMLingua-2 loss.
All paired results, including random/ungrouped/Selective Context, are in
[paired.csv](results/tables/full_cohort/paired.csv). Intervals use 1,000 family
bootstrap replicates, seed 42, and are unadjusted for multiple comparisons.

## 4. Budget, baseline and validity disclosures

- External names ending `_budgeted` denote official implementations plus our
  native-setting budget controller, not untouched published-paper protocols.
  Search factors .9/.75/.5/.25/.05 plus three refinements select the longest
  compliant native output. Actual search histories/native text remain in raw CSVs.
- .50 and .75 are token **ceilings**, not equal realized lengths or 50%/75% token
  savings. Mean hybrid retention is .4956/.7439; LLMLingua-2 is .4689/.7331.
  Report actual ratios and note the stronger baseline often uses fewer tokens.
  Accounting concerns question text, not the whole wrapped request or output tokens.
- GPT-OSS was a labeler; an older Qwen model was a labeler. This does not prove
  transfer to an unseen model family. Family-held-out scorer evaluation is a
  different claim from unseen-reader transfer.
- Masking one span differs from deleting it; multiple individually dispensable
  facts can be necessary jointly. The method does not certify compressed-input
  sufficiency. Joint-deletion/mask-to-deletion causal validation was not completed.
- Nonempty invalid parses count as operational failures under the declared strict
  grader. They are not all proven semantic reasoning errors. GPT-OSS has 408
  unparseable compressed outputs; Qwen has 242 unparseable/unresolved outputs,
  including the two empties. All 320 original controls parse. No manual semantic
  adjudication or lenient regrading is claimed.
- Retention-versus-correctness checks are GPT-OSS-only, conditional/predictive
  diagnostics with small strata and mixed results. They are not causal evidence.
  See [retention prediction](results/tables/gptoss_diagnostics/retention_prediction.csv).

## 5. Runtime, memory and compute credit

Kaggle showed two T4 devices; these processes used `cuda:0`, not distributed
two-GPU training. Training CLI wall time: **197.90 s**, peak allocated/reserved
PyTorch GPU memory 1,645,785,600 / 1,975,517,184 bytes. Main compression setup:
61.09 s; main runner: 339.30 s. These are observed process times, not complete
Kaggle session duration or billable GPU hours.

| Main compression process | Wall seconds | Peak allocated GPU bytes |
| --- | ---: | ---: |
| Own-method batch (four methods together) | 45.75 | 677,436,416 |
| Budgeted LLMLingua-1 | 31.32 | 5,683,676,672 |
| Budgeted LLMLingua-2 | 106.88 | 2,268,694,528 |
| Budgeted Selective Context | 139.72 | 565,846,016 |

These are separate cold CLI processes with different batching/search work;
do not claim a controlled per-method speedup from this table. Use
[condition timing](results/tables/gptoss_diagnostics/compression_timing.csv)
and [raw process profiles](results/profiles/) with these caveats.

Course funding: **INR 5,880**, a grant, not expenditure. Recorded returned-token
estimates include access checks, failed work, pilots, main runs and archived
paid attempts:

| Stage | Estimated INR |
| --- | ---: |
| Three access checks combined | .048364791 |
| Failed DeepSeek pilot + calibration | 16.870882000 |
| GPT-OSS fallback + pilot | 4.255590000 |
| Qwen 4,096 pilot + cap check + uniform 8,192 pilot | 61.968843959 |
| GPT-OSS main, including archived failure | 39.646680000 |
| Qwen main, including two archived failures | 264.152354991 |
| **Cumulative returned-token estimate** | **386.942715741** |

GPT-OSS main: 1,897 HTTP attempts, 278,388 input + 921,570 output tokens;
Qwen main: 1,899 attempts, 239,335 input + 1,155,534 output tokens. Active
records and archived paid failures are deduplicated; logical cache copies
are not extra billable generations. Historical quoted rates used: GPT-OSS
INR 10/40 per million input/output; Qwen INR 28.431/222.709. These are recorded
run assumptions, not a current pricing quotation. Qwen's INR 500 guard accounted
264.152354991 with no unresolved reservations.

**Actual Jarvislabs billing is not reconciled.** Do not call this estimate the
provider bill or subtract it to claim a verified remaining balance. Kaggle was
free; no paid Jarvislabs GPU rental is claimed. Stage evidence is in
[compute_estimates.csv](results/tables/compute_estimates.csv); the original
ledger remains `outputs/final_eval/cloud_usage.csv`.

Suggested sponsorship paragraph: “The INR 5,880 course grant supported hosted
downstream validation and bounded reader-qualification experiments. We reused
existing labels, trained/compressed on free Kaggle T4 compute, and controlled
API work through pilots, caching and request limits. Returned-token estimates
total INR 386.94, including failed attempts; actual provider charges and the
remaining balance are [fill only after reconciliation].” Add the actual
statement/export before submission; do not replace the estimate with an invented total.

## 6. Ready-to-use figures and tables

| Exhibit | Use |
| --- | --- |
| [Two-reader accuracy vs actual retained tokens](results/figures/two_reader_accuracy.png) | Main comparative result; full-160 failure-as-incorrect companion |
| [Scorer ROC](results/figures/scorer_roc.png) | Neural versus feature/prior prediction, shared support |
| [Labeler agreement](results/figures/label_agreement.png) | Recovered common-eligible agreement, with coverage caveat |
| [GPT-OSS accuracy/token plot](results/figures/gptoss_accuracy_tokens_0.png) | Original common-cohort descriptive plot |
| [GPT-OSS retention .50](results/figures/gptoss_retention_accuracy_0.png) | Descriptive retention diagnostic |
| [GPT-OSS retention .75](results/figures/gptoss_retention_accuracy_1.png) | Descriptive retention diagnostic |
| [Full-cohort paired table](results/tables/full_cohort/paired.csv) | All comparisons and uncertainty |
| [Dataset-macro table](results/tables/full_cohort/macro_accuracy.csv) | Equal-weight dataset sensitivity |
| [Grading/failure table](results/tables/full_cohort/grading_coverage.csv) | Parse failures versus provider failures |

Use CSV/JSON values for tables; do not transcribe scores from screenshots.
[results/manifest.json](results/manifest.json) includes exact source addresses
and checksums for every copied exhibit. Prepared labels, full OOF predictions,
compressed text and raw responses stay in the directories listed in README.md.
The two original response hashes and all five checkpoint hashes are recorded
in [verification.json](results/verification.json).

## 7. Report/PPT work packages and claims

| Work package | Material to use |
| --- | --- |
| Data/method/viva | README workflow/reading guide; preparation, regrading and agreement metadata; original collection code |
| Quantitative results | Sections 1–4; full seven-method/two-reader tables; scorer ROC; paired intervals; both cohort policies |
| Compute and reproducibility | Section 5; token estimates; original ledgers/profiles; obtain actual Jarvislabs statement |
| Discussion and slides | Negative main result; scorer/compression gap; limitations; measured contributions below |

For a full-project report: motivation → original collection → corrected label
scope/folds/scorer → budget-controlled baselines → two-reader design/results →
failure analysis/limitations → compute justification and reproduction.
For a post-mid-eval presentation: one recap slide → recovered/regraded data and
family-safe training → scorer result → fair budget protocol/repairs → two-reader
main result → negative finding/limitations → credit use and conclusion.
Keep historical mid-eval numbers explicitly labeled; they are not a comparable
before/after accuracy experiment under the corrected protocol.

Suggested central claim: “On recovered multi-model counterfactual labels, a
family-held-out DistilBERT scorer predicts span sensitivity well. Under common
question-token ceilings, this does not translate into an established neural
compression gain, and budgeted LLMLingua-2 is substantially more accurate on
both evaluated readers.”

Defensible contributions are a recoverable multi-labeler sensitivity analysis,
an evaluated scorer/cheap ablation, a reproducible budget-controlled comparison,
and a documented scorer-to-compression mismatch. Do not claim first-ever
novelty, SOTA, proven necessity/sufficiency, causal validation, universal
model transfer or success at beating the strongest baseline.

Remaining deliverables: final report/slides and actual billing reconciliation.
The v2 model release is published and verified; the historical model is preserved.
The executed study needs no additional paid run to support these bounded claims.

## 8. Post-cleanup diagnostic observations and extension scope

A read-only comparison of exact strings in the frozen `compressed.csv` found:

| Exact-text equality, out of 160 questions | .50 ceiling | .75 ceiling |
| --- | ---: | ---: |
| `grouped_hybrid` equals ungrouped `ours` | 148 | 150 |
| `grouped_hybrid` equals cheap `grouped_signal` | 86 | 116 |

Thus current grouping changes few outputs, and the learned scorer frequently
does not change the selected text. These are descriptive output-equality counts,
not a causal attribution of the accuracy gap. Inspected losing examples also
delete relations or answer-option content while retaining isolated numbers.
Those examples motivate a systematic audit; they do not establish prevalence.

The .50 date-understanding slice is an exploratory exception: hybrid is correct
on 8/16 versus LLMLingua-2's 2/16 for GPT-OSS, and 9/16 versus 3/16 for Qwen.
This is a small subgroup noticed after seeing outcomes, not an independent
confirmed win or a substitute for the negative overall result. Any focused
follow-up needs fresh held-out families and should retain all task results.

README.md records a proposed seven-day extension. No new candidate has been
implemented/evaluated as part of these diagnostic observations. Use the 736
reliably labeled questions / 642 families outside the old main/pilot families
for separately frozen development/test pools; preserve the old study unchanged.

## 9. Model distribution: complete

| Version | Link | Revision for report/reproduction |
| --- | --- | --- |
| Corrected final-study v2 | [Pranav-Sriv/reasoning-critical-span-scorer-v2](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2) | `70cc329f10ef6415e152c0bcdcef8894f8dc0d69` |
| Historical mid-evaluation | [flashstep/reasoning-critical-span-scorer](https://huggingface.co/flashstep/reasoning-critical-span-scorer) | `c24d8ce9ef2cfbdc7b11185958dee5810327e432` |

Both repositories are public. All **67 v2 release files**, including the five
`scorer.pt` checkpoints, were verified against the prepared upload through
remote LFS SHA256/Git blob IDs and sizes: zero missing files or mismatches.
HF's `.gitattributes` is the only additional remote file. Full proof:
[huggingface_release.json](results/metadata/huggingface_release.json).

The release includes tokenizer/configs, split/OOF/training metadata, a custom
loader, source snapshot, model card, comparison evidence and checksums. Original
local weights remain unchanged. The older URL was recovered from the original
archived README, and its fixed revision was checked on the Hub.

README.md section 7 now contains the actual pinned download command; placeholder
usernames, pending-upload instructions and publication to-do items are removed.
No further HF upload is required for this version. Changed weights or packaged
loader/source artifacts should receive a separately recorded release revision;
new analysis/reporting alone does not require re-uploading these checkpoints.
Publication verification did not download or run the full checkpoints again.
