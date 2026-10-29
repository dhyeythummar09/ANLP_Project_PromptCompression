# Follow-up assessment: what is done, what is worth doing, and what remains speculative

Reviewed 25 October 2026 against the current source, saved artifacts, all six supplied screenshots, the supplied proposal, and primary research sources. This is an experiment decision document, not an executable runbook. **None of the proposed experiments below has been run by writing this document.**

The existing reproduction guide is [README.md](README.md); measured results and report figures are in [RESULTS.md](RESULTS.md). Preserve that completed study. New work should receive new manifests, output directories and, only if weights change, a new model release.

## 1. What I recommend doing

There is enough remaining work for a stronger project without rebuilding the dataset or starting a different project. The most useful package is:

1. **Evaluate the historical HF scorer downstream**, with a careful audit of its actual saved folds. Also compare old versus v2 weights inside the same compression machinery, so checkpoint changes and algorithm changes are distinguishable.
2. **Complete the missing component ablations** and inspect score scales, bonuses and deletion units. Develop at most two structural candidates, then freeze one. An ablation flag existing in code does not mean its experiment has been completed.
3. **Run the promised mask-versus-deletion assay**, followed by a small, structured pair-deletion diagnostic. Use it to test the proposed explanation for the prediction/compression gap before buying thousands of training labels.
4. **Confirm the selected candidate on fresh downstream test families**, against v2, its matched non-neural ablation and LLMLingua-2. Add controlled compressor runtime/memory measurements.

An adaptive accuracy-target controller is the most useful optional product extension. Teacher distillation and an RL pilot are separate optional branches, not prerequisites for completing the four items above. Doing every idea in this file would spread the project too thin.

**Current judgment:** the project has a valid, reproducible empirical result and a strong sensitivity predictor. It has not demonstrated a competitive overall compressor against LLMLingua-2, an established downstream neural contribution, or improvement over the historical checkpoint. New experiments may change those conclusions; no proposed fix guarantees that they will.

## 2. What has already been completed

Status meanings: **Done** = executed evidence exists; **Partial** = only part of the claim is supported; **Code only** = implemented but no completed result established; **New** = needs implementation.

| Item | Status | What the evidence actually supports |
| --- | --- | --- |
| Original collection pipeline and mid-eval prototype | Done | Original `src/`, root config and collection requirements remain unchanged. Do not recollect the dataset. |
| Historical baseline compression / intrinsic comparison | Done historically | Older compressed outputs and intrinsic summaries exist. These are not a matched old-versus-v2 downstream comparison. |
| Recover, deduplicate and strictly regrade saved labels | Done | 38,603 unique model/span votes; 13,966 reliable span positions, 912 questions and 778 families. The recovered team export was already filtered, not the full unfiltered four-person collection. |
| Correct soft-target training and prevent related-family split leakage | Done | Five family-disjoint v2 folds, with held-out scorer selection for evaluation. |
| Labeler agreement | Done | Common-eligible pairwise agreement approximately 82.1–85.0%, with kappa approximately .487–.578. This is not mask/deletion agreement. |
| Cheap scorer baselines | Done | Neural, feature ridge and category prior compared on 13,855 shared held-out occurrences. Neural AUC .9074 versus ridge .8217. This is not an old-versus-new checkpoint comparison. |
| Shared runtime span extraction | Done, with qualifications | Final runtime imports the shared extractor; saved training spans, controls and oracle distractor annotations still need to be distinguished from deployable candidates. |
| Phrase grouping and exact token ceilings | Done | Selected number/comparison/negation groups are atomic. Not every entity, relation or clause is atomic. |
| Downstream external baselines | Done | Seven methods, 160 questions / 125 families, two ceilings, two readers. This includes LLMLingua, LLMLingua-2 and Selective Context. |
| Hybrid versus matched category-prior compression | Done | Small positive point differences; all four paired intervals include zero. Neural downstream benefit remains unestablished. |
| Original scorer-only and surprisal-only downstream ablations | Code only | `ours_scorer` and `ours_signal` were omitted from the executed final seven-method grid. Both retain structural bonuses by default. |
| Critical/category retention and correctness diagnostics | Partial | Retention measurements and GPT-OSS predictive analyses exist. They do not prove causal importance or that retention is sufficient for correctness. Equivalent Qwen predictive analysis was not established. |
| Confidence intervals, plots and two-reader comparison | Done, bounded scope | Two compression ceilings, not a dense trade-off curve. Qwen retains two empty outputs and a failed technical completion gate. |
| Genuinely unseen-reader-family transfer | Not established | GPT-OSS was a labeler; an older Qwen was a labeler. Two readers do not establish this particular transfer claim. |
| Mask versus true deletion validation | Code only | `final_eval/compressor/evaluate.py --mode deletion-check` exists. No completed assay is established. |
| Dedicated quantifier/causal taxonomy and sentence pruning | New / partial existing coverage | Conditionals, comparisons and some temporal/negation markers already exist. The proposed expanded taxonomy and sentence-pruning evaluation do not. |
| Joint-deletion labels / interaction-aware training | New | Single-span votes cannot be relabeled into joint effects without new observations. |
| Controlled latency/memory comparison | Partial | Process profiles exist; they are not a matched per-method speed benchmark. |
| HF release, artifacts, tables and reproducibility audit | Done | Both old and v2 models are public; v2 release files were verified. No re-upload is needed for analysis alone. |
| Actual sponsorship billing reconciliation | Outstanding | Returned-token estimate is INR 386.94. This is not a verified invoice or wallet balance. |

The downstream gap is substantial: grouped hybrid versus LLMLingua-2 is 36.875% versus 60.000% at .50 and 73.125% versus 90.000% at .75 on GPT-OSS; 46.250% versus 69.375% and 73.125% versus 95.625% on Qwen. The paired intervals for those losses exclude zero. Full tables, failure policies and intervals remain in [RESULTS.md](RESULTS.md).

## 3. Assessment of every screenshot and teammate suggestion

### 3.1 Training spans differed from compression spans

**Verdict: a real historical implementation issue; the final code already addresses the main duplication.**

The historical `src/compressor/compress.py` has its own simpler extractor. Final compression calls `final_eval.spans.extract.find_spans(..., include_controls=False)`, retaining definitions such as “at least” and adjacent number/unit combinations.

Remaining useful work is an offset/category coverage audit on actual prompts, not another collection pass. Count saved labeled spans recovered by runtime candidates; separately identify random controls, unavailable `inserted_context` annotations, parser differences and tokenizer truncation. Do not promise identical training/runtime distributions merely because functions are shared. A new taxonomy also creates spans outside the trained distribution and must be treated accordingly.

### 3.2 Structural bonuses drowned out the scorer

**Verdict: a plausible ranking concern, supported by incorrect arithmetic and a misquoted equality count in the screenshot.**

The historical and final implementations set each word's bonus with `max`, not addition:

`priority = 0.5 * criticality + 0.5 * surprisal_rank + bonus`

The bonus is at most **.4**, not .8–1.0. At the default blend, the neural contribution can reach .5 and the surprisal contribution can reach .5. Bounds alone do not show which term decides marginal keep/drop choices: observed distributions, ties and group aggregation matter.

Recounted directly from the frozen compressed strings:

| Exact text equality, out of 160 questions | .50 | .75 |
| --- | ---: | ---: |
| `grouped_hybrid` versus ungrouped neural `ours` | 148 | 150 |
| `grouped_hybrid` versus prior-based `grouped_signal` | 86 | 116 |

Thus **148/160 is not the neural-versus-prior count**. The real concern remains worth testing: the neural scorer often does not change selected text, and its downstream contribution has not been established.

Better experiment: log the three score terms at selection boundaries, then compare bonus on/off and a small predeclared alpha set on development families. The existing `rank01` assigns distinct ranks to ties using position order; if adding score normalization, use tie-aware ranks and document the new rule. Normalizing a category prior with arbitrary tie-breaking can introduce position bias. No change should silently rewrite the executed v2 results.

### 3.3 Span predictions were converted into individual-word deletions

**Verdict: true for the historical compressor and current ungrouped `ours`; partially addressed by `grouped_hybrid`.**

Final grouping joins selected number, comparison/conditional and negation ranges, plus negation/head ranges. The budget selector deletes and refills whole groups and recounts the rendered string. It still starts from whitespace words; action/argument relations, many entities, entire conditions and answer interfaces are not comprehensively protected.

Grouping changes few outputs and changes accuracy by only 0–1.25 points across current cells, with intervals including zero. “Already fixed and proven effective” would be wrong. Expanding every dependency subtree is also risky: overlapping ranges can merge into huge groups, consume the budget and preserve irrelevant numbers. Test targeted semantic units and a matched non-neural variant, not indiscriminate protection.

### 3.4 “We did not do phases 2 and 3”

**Verdict: the document being numbered matters.** The screenshot's taxonomy/validation phases are mid-evaluation remaining-work phases. They are not phases 2 and 3 of the supplied original proposal. Section 4 maps both numbering systems; original proposal phases 2 and 3 have largely been completed.

The screenshot wording also differs from the archived local draft: that draft's mid-eval Phase 1 explicitly promises alpha/component ablations, which remain incomplete. Both versions are covered here rather than assuming the screenshot and local draft are identical.

### 3.5 Distill LLMLingua-2 into DistilBERT

**Verdict: feasible optional efficiency experiment; the “never done” and guaranteed speed/quality claims are unsupported.**

LLMLingua-2 itself uses distilled compression supervision to train token classifiers and already evaluates a smaller mBERT variant. Distilling its particular decisions into DistilBERT can be useful engineering, but is not the first use of distillation for prompt compression. [LLMLingua-2, ACL 2024](https://aclanthology.org/2024.findings-acl.57/).

The actual large checkpoint used here is roughly **559M parameters**, versus approximately **66.36M** for the span scorer: about 8.4 times as many parameters, not a measured fivefold model reduction. This comparison excludes the hybrid's additional DistilGPT2 model. It also does not imply proportional GPU memory or latency. The teacher's official config is XLM-RoBERTa-large; its Hub page rounds its size to .6B. [Official teacher model/config](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank/blob/main/config.json).

Further corrections:

- 160 evaluated questions are too small and already exposed to analysis; do not train on that test and claim held-out performance on it. Generate teacher supervision on declared training families only.
- Two retained ratios do not automatically supply consistent binary labels for one unconditional classifier: a word can be kept at .75 and dropped at .50. Use a budget-conditioned model, teacher salience/ranking targets, or one declared training budget.
- Compressed strings do not uniquely identify original word positions when text repeats. Prefer the official returned word labels, retain offsets, and audit alignment. Do not label every matching repeated word as kept. [Official word-label interface](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank).
- This needs a new token-level objective/head or an explicitly defined aggregation into span targets. The current span-sensitivity head does not become a token-retention classifier by changing a CSV column.
- No hosted calls are needed to generate teacher labels locally; downstream evaluation still needs a reader. Teacher inference, training, alignment and benchmarking still take compute and implementation effort.
- A student can match, improve or lose to the teacher on held-out tasks. Small data and teacher imitation do not imply comparable quality. A cheaper model that loses some accuracy can still yield a useful measured trade-off.

If chosen, compare teacher, student-only and existing criticality methods at matched ceilings and controlled runtime. Combining teacher and criticality supervision is a further ablation, not an automatic novelty claim. Do not pursue this before the direct project gaps unless lightweight deployment becomes the central objective.

### 3.6 Random pairs/triples as new compositional labels

**Verdict: the mismatch is real; a structured diagnostic is worthwhile; 9,000 random labels and immediate v3 retraining are premature.**

The scorer learns sensitivity to masking one span in an otherwise intact question. Compression removes many parts. Two facts can each be dispensable individually while jointly necessary. Conversely, a single deletion failure need not remain a failure after another edit; observed correctness is not monotone in retained text.

Your teammate is right to ask for consistency. Random sampling is useful as a matched control, but random sets alone do not isolate the interaction. Use paired intervention arms and predetermined semantic strata, as in experiment E below. Exhaustive combinations are unnecessary.

**Do not assign a failed pair's label to both component spans.** Failure identifies the selected set's outcome, not which member caused it. Training needs a pair/set predictor, a conditional deletion scorer, candidate reranker, or policy objective. The existing single-span BCE training cannot absorb set labels unchanged.

The “nobody does this” claim is not established. Interaction attribution has prior literature, and prompt-compression policies have already been trained from whole-compression rewards. Our possible contribution is a carefully controlled reasoning-span interaction study and its demonstrated utility, not ownership of joint feature effects. [Shapley–Taylor interaction attribution](https://proceedings.mlr.press/v119/sundararajan20a.html), [TACO-RL](https://aclanthology.org/2025.findings-acl.81/).

The budget claim also needs qualification: 9,000 calls can be inexpensive or exceed the grant depending on reader, output length, retries and controls. Section 8 gives transparent scenarios. Begin with a few hundred diagnostic conditions, not a new dataset campaign.

## 4. Proposal commitments: completed work versus actual gaps

The supplied PDF matches the proposal previously archived in this repo: SHA256 `9e0a66d105d95a27d7d03c88a2604f5b24c4ab8beb03af62cf5e653690b2527c`. Its extracted text is `outputs/planning_audit/Symbiote-Proposal.txt`; the full historical documents are backed up in `outputs/repo_cleanup_audit/20261022T061303Z/pre_cleanup_sources_and_documents.zip`.

### Original proposal, section 5

| Original phase | Commitment | Current assessment |
| --- | --- | --- |
| Phase 1 | Pipeline pilot and baseline critical-span analysis | Pipeline and historical intrinsic comparison completed; final baseline retention evidence also exists. Earlier intrinsic claims retain their original limitations. |
| Phase 2 | Full collection, clean data, train small classifier | Completed in substantive scope, including recovered/regraded data and v2 training. Missing unfiltered team votes limit population claims; they do not require recollection to report this study. |
| Phase 3 | Hybrid integration and downstream comparison to three baselines | Completed: seven methods, two ceilings and two readers. Baseline superiority was a research question; a negative answer is a valid result. |
| Phase 4 | Trade-off curves, category recall, latency and category ablations | Partial: plots, retention and process profiles exist. Two points are a sparse curve; controlled speed tests and causal category ablations remain. |

### Mid-evaluation remaining-work phases, section 10 / supplied screenshot

| Mid-eval phase | Current assessment | Useful action |
| --- | --- | --- |
| Phase 1: baseline evaluation / ablations | Downstream baseline comparison done; pure component/alpha ablations not completed under the final protocol | Run the missing component experiment, not all baseline collection again. |
| Phase 2: taxonomy expansion and sentence pruning | Existing conditional/comparison coverage is partial prior work; dedicated quantifier/causal expansion and sentence pruning remain | Add only categories implicated by a systematic development audit; test a non-oracle sentence-pruning candidate if GSM-IC failure patterns justify it. |
| Phase 3: mask/deletion validation and reporting / unseen readers | Validation code exists but assay not run. Two-reader plots and intervals exist. Genuinely unseen-family transfer and dense curves remain unsupported | Complete the small validation assay; add one unseen-family reader only after qualification and budget confirmation. |

### Other proposal promises and bold claims

| Promise or claim | Honest status / action |
| --- | --- |
| No repeated counterfactual LLM calls for a new prompt | Achieved for the current compressor: local scorer/surprisal passes, then the normal reader call. RL training would not itself violate this deployment property. |
| Agreement establishes model-independent criticality | Too strong. Agreement on eligible recovered spans is evidence of consistency, not universality. Report coverage, positive/negative agreement and kappa. |
| Ground-truth necessity / neutral masking avoids artifacts | Operational sensitivity is established, not logical necessity. Placeholder artifacts are possible; run the planned assay. |
| Unseen-target transfer | Outstanding as a clean model-family claim. A newer Qwen endpoint is not a new model family. |
| Initial random controls are length matched | Audit rather than repeat literally. The extractor samples 1–3-token controls; this does not ensure matching the full length distribution of curated spans. Use explicit length matching in new diagnostics. |
| Consensus supervision is better than single-model labels | Not tested by labeler agreement. A matched training comparison is required, using common support and identical folds/hyperparameters. |
| Human spot-checks validate the pipeline | Not documented as a completed blinded study. A small audit is useful; human judgments of importance are not proof of model-specific necessity. |
| Third domain / Indian-language evaluation | Stretch goals, not core omissions. New grading, extraction and data coverage make these poor first extensions now. |
| End-to-end cost/latency savings | Not proven by question-token savings or process profiles. Measure complete requests, output/reasoning tokens and compressor overhead. |
| Near-zero accuracy loss, universal criticality, superior baselines | Unsupported by the corrected final evidence. Remove or rewrite these stronger mid-draft claims regardless of whether another experiment is run. |

The original proposal explicitly says not to assume model independence. The archived mid-eval conclusion is stronger than that proposal and stronger than the final evidence. Also, its method text mentions DeBERTa/TF-IDF/alpha .6, inconsistent with the executed DistilBERT/DistilGPT2/default .5 implementation. Write the final method from executed code and artifacts.

## 5. Experiments worth implementing, in dependency order

These are experiment specifications, not instructions to launch unimplemented commands. **A–F form the recommended package; G–K are optional branches.** Implementation should reuse the existing loaders, budget accounting, grading and artifact conventions rather than generating another collection of alternate launchers.

### A. Historical model downstream comparison — highest priority

**Question:** Did the checkpoint and/or complete compression system improve after mid-evaluation?

Use the pinned historical model [flashstep/reasoning-critical-span-scorer](https://huggingface.co/flashstep/reasoning-critical-span-scorer), revision `c24d8ce9ef2cfbdc7b11185958dee5810327e432`, and v2 [Pranav-Sriv/reasoning-critical-span-scorer-v2](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2), revision `70cc329f10ef6415e152c0bcdcef8894f8dc0d69`.

**Before running:** restore old weights/configuration in a separate directory and validate the loader. Use the old model's published `split.json`, not the local historical split: they differ, with 924 versus 1,000 question entries. The prior audit found 41 GSM-IC families spanning multiple old folds, covering 178 rows; five unresolved GSM-IC rows were excluded from that audit. Reconstruct train/validation/test exposure conservatively and identify an evaluable intersection. Never assign an unknown old question to an arbitrary fold or ensemble all five folds for a supposedly held-out result.

Report two different comparisons:

1. **System comparison:** historical compressor plus historical weights versus executed v2 system, using the same reader, answer wrapper/grader and declared ceiling protocol. Preserve any native historical budget violations as historical behavior; if adding an adapter for fair ceilings, label that adapted variant explicitly.
2. **Checkpoint comparison:** old versus v2 weights inside an identical extraction/ranking/grouping/budget pipeline, on questions held out from the selected checkpoint in both versions. This isolates checkpoint choice better than the system comparison, though training data and split differences remain confounders.

Use matched questions, two retained ceilings, all failures, paired family-bootstrap intervals and actual token counts. Start with one qualified reader. The existing 160-question set can provide a transparent retrospective comparison after exposure filtering; it is already inspected, not a fresh confirmation set. Confirm any new selection decision separately.

**No relabeling or old-model retraining is needed.** Unknown legacy target metadata prevents claiming that the published checkpoint was definitely hard-label trained. Even a v2 win cannot be attributed solely to soft labels. A causal hard-versus-soft claim needs matched retraining on the same corrected labels/folds, with a declared tie policy; that is optional, not a prerequisite for comparing the released models.

**Deliverable:** old-model exposure audit, common-question manifest, both comparison tables, retained-token counts and paired intervals. If exposure cannot be certified, mark the affected legacy result descriptive and exclude it from generalization claims.

### B. Missing component and ranking ablations — high value, no new labels

**Question:** Is criticality useful when its influence is actually allowed to change selection?

First finish `ours_scorer`, `ours_signal` and `ours` under the final wrapper/budgets. Clearly label that their defaults still include bonuses. Include a strict bonus-free comparison and the category-prior ablation, keeping extraction and grouping identical within each causal comparison. `grouped_signal` is a prior-plus-surprisal method, not pure `ours_signal`.

On development data only, inspect score histograms, ties, word coverage, marginal deletion decisions, output equality and group sizes. Compare a small fixed alpha set such as {0, .5, 1}; compare bonuses on/off before introducing new scaling. If testing calibrated probabilities or tie-aware ranks, treat them as explicit candidates, not corrections guaranteed to help. Uncovered words currently receive zero criticality; audit whether this systematically discards relation words.

Select at most two candidates on declared development outcomes. A reproducible no-bonus win would support a ranking diagnosis; a tie or loss would not support removing bonuses, though an imprecise result cannot rule the explanation out. Neither outcome alone resolves interactions between deletions.

**Code status:** original component flags, alpha and bonus removal exist. A bounded comparative runner and diagnostic reporting need to be assembled; a fully matched grouped pure-component factorial is not already an executed experiment. No scorer retraining is required.

### C. Semantic units and answer-interface preservation — plausible improvement, test it

**Question:** Can we retain relationships rather than isolated salient words without wasting the budget?

Candidate units include number–unit–entity relations, a negation plus its predicate, a comparison plus operands, a conditional plus its necessary clause, and an entity/action/argument link. Prefer limited, inspectable ranges over entire parse subtrees. Test cost-aware group selection because assigning a long group its maximum word score can overvalue it. Always recount the full rendered string; BPE costs are not simply additive across pieces.

Audit the actual requested answer and option identifiers. `--protect-interface` exists, but its current option recognizer targets uppercase `(A)`-style lines and protection is not a completed result. Hard protection can exceed .50 on short multiple-choice prompts. Count that as infeasible; do not increase the ceiling silently. Any automatic fallback to a larger ratio belongs in a separately evaluated adaptive policy.

Use three matched variants: existing v2, structural candidate with neural scores, and the same candidate with the category prior. If the rule-only/prior version matches the neural version, credit the structural change rather than the trained model.

**Code status:** grouping and protection primitives exist; broader units and controlled comparisons need implementation. No new labels are required for a rule-based candidate. Feeding new span types into the old neural scorer is a distribution shift, not evidence that it learned their criticality.

### D. Mask versus true deletion — fulfill the reliability promise

**Question:** Does placeholder sensitivity predict the behavior of actual removal?

Use fresh original/mask/delete conditions on the same reader/settings, sampled across tasks, categories and span lengths. Condition sensitivity analysis on a correct original; retain ineligible originals and technical/parse failures in coverage tables. Report the 2×2 mask/deletion outcome table, disagreement directions, positive/negative agreement and family-clustered uncertainty, not just overall agreement dominated by non-critical spans.

A practical diagnostic is 120 sampled spans: at most 360 logical conditions before shared-original caching, for one reader. This is a proposed cap, not a power guarantee. Use length-matched random controls and optionally a second placeholder on a small subset if artifacts look important. Do not reuse a historical mask outcome generated by another model/settings as one arm of a fresh paired assay.

**Code status:** the three-condition assay exists. Before paid use, improve persistent per-arm caching, cumulative request/cost limits and the report: current resume saves completed span records, while its within-run response cache is in memory. An interruption can repeat unfinished calls. Keep the original assay/source version reproducible.

### E. Structured joint-deletion diagnostic — best extension of the core hypothesis

**Question:** Do single-span labels miss combinations that break reasoning?

Choose disjoint spans A and B with predetermined strata: related quantity/action pairs, negation/scope or comparison/operand pairs, apparently redundant facts, and length/distance-matched random pairs. Use task-balanced sampling and save the seed/selection rationale before calling the reader. Random pairs remain a control, not the whole design.

For each pair run **original, delete A, delete B, delete both**. Add mask A/mask B/mask both only for a smaller declared subset if mask/delete differences need testing. Shared originals can be cached. For example, 80 pairs mean at most 320 four-arm conditions or 560 seven-arm conditions before deduplication.

For eligible originals define `F(S)=1` when deleting set S causes an operational wrong answer, otherwise 0. Report all patterns. The especially informative pattern is `F(A)=0, F(B)=0, F(A,B)=1`: both single deletions survive but the pair fails. The contrast `F(A,B)-F(A)-F(B)+F(empty)` is a local interaction diagnostic, not a full Shapley value or a theorem about language semantics. Keep rescues and non-monotone outcomes rather than discarding them.

Only if a useful, sufficiently frequent pattern appears should we scale targeted collection. Better next targets are deletions actually selected by the compressor or conditional marginal effects after another deletion, not arbitrary triples everywhere. A binary downstream observation from one generation remains noisy despite temperature zero.

**Possible training:** predict a set's risk, rerank a few budget-compliant candidates, or learn conditional deletion decisions. Span-level supervision can initialize those models. No result here licenses attaching one set label to every member or claiming zero downstream risk.

**Code status:** new sampler, per-condition ledger and interaction report required; no need to alter the old collection scripts. **Go/no-go:** if the diagnostic shows little relevant interaction or only parser damage, stop the new-label campaign and report that evidence.

### F. Fresh confirmation and controlled efficiency — required before an improvement claim

Freeze one candidate after development. Predeclare a new family-disjoint test and primary comparison. There are 736 reliably labeled questions / 642 families outside old main/pilot families; this is a pool before reserving development, calibration and test, not a guarantee that every subset supports a large balanced study.

For a 160-question example, four methods × two ceilings plus originals is at most **1,440 logical conditions per reader**: existing v2, selected candidate, its matched prior/rule variant and LLMLingua-2. The exposure audit must determine the feasible test size first; a clean single-fold subset may be smaller than 160. Reuse exact queries only with matching settings/provenance and a fixed cache policy. Do not choose responses based on correctness. Start with one reader; a second can confirm a frozen candidate rather than repeat every development search.

Report accuracy, original-correct conditional preservation, actual retention, dataset macro/per-task results, failures and paired family intervals. Common token ceilings do not imply identical realized lengths or total request costs; show both, and include matched-length sensitivity checks if length differences affect the interpretation. Predeclare the primary contrast; label other comparisons exploratory or adjust for multiplicity. At n=160, one extra correct question is .625 percentage points, and clustering reduces precision. A few questions or a CI spanning zero is not an established win.

For efficiency use the same hardware, precision, tokenizer settings, batch sizes, lengths and budgets. Separate cold load from warm compression; repeat timings and report median/tail latency, throughput and incremental allocated/reserved GPU memory. Include spaCy, tokenization, DistilGPT2 and the actual budget search. Compare end-to-end methods as deployed, not just a 66M scorer forward pass against the full baseline. GPU timing requires synchronization; process-level peaks are not interchangeable with live inference memory.

**Leakage rule for all new work:** “not in the previous downstream test” does not mean “unseen by every scorer.” Use each question's proper held-out v2 fold. Never use an ensemble containing checkpoints trained on its family for a held-out claim. For updated neural weights, teacher students, outcome rerankers or RL, declare outer training/development/calibration/test families first; exclude test families from every update and policy choice.

The simplest rigorous follow-up reuses **one fixed existing outer-fold checkpoint**, draws final test families only from its held-out fold, and confines development/calibration and any updates to its training families. This also prevents indirect leakage when choosing structural rules: development predictions from other checkpoints may have been trained on proposed test-family labels. A larger multi-fold confirmation needs appropriately nested, fold-specific development or a newly trained outer-split model; simply assigning each test question its old OOF checkpoint is not sufficient to audit the entire new selection procedure. Cold-starting a student on a freshly fixed global family split is another valid route. Choose the exposure-safe design before promising sample size; none requires rebuilding the original labels.

### G. Quantifier/causal taxonomy and sentence pruning — conditional, targeted

Extend scope-bearing units such as “all,” “exactly,” “only if,” “because,” and temporal relations only after measuring coverage and errors on development prompts. Existing comparisons/conditionals and “before/after/until” are not new work. Lexical matches alone miss scope, attachment and causal direction; new category names do not solve that.

Start with inference-only structure and compare against the same rules without neural scores. If needing learned scores for new categories, acquire a small, separately versioned label subset with length-matched controls; preserve old labels and report incremental coverage. Historical votes cannot supply labels for never-tested spans.

For GSM-IC, a sentence selector may remove distractors more coherently than word deletion, followed by compression inside retained sentences. It must infer irrelevance from the available question. Using dataset `inserted_context` or the clean counterpart to identify the distractor is **oracle pruning**, suitable only as a labeled diagnostic ceiling. Evaluate deployable pruning on both clean and distracted variants within their shared families, including failures where necessary facts are removed.

Rule-based linguistic compression is also an active research area; do not claim first-ever novelty for adding syntactic groups or sentence rules. [Ma et al., updated October 2026 preprint](https://arxiv.org/abs/2607.25335).

### H. Where LLMLingua-2 might leave room — hypotheses, not established weaknesses

The sensible angle is preservation of reasoning relationships and task/answer interfaces at tight budgets, with a lightweight deployment trade-off. It is not “LLMLingua-2 ignores context”: its classifier is contextual, and task-aware compression already has prior work. [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/).

Candidate tests:

- Measure how often each method splits a labeled relation or loses a required option identifier, and whether those differences predict errors within task and budget. Aggregate all methods; do not inspect only baseline failures.
- Test shared interface/semantic protection for **both** our method and LLMLingua-2, alongside their raw versions. If both improve, the contribution is the wrapper/structure, not proof that our learned scores are superior.
- Generate a few teacher/own budget-compliant candidates and rank using a local set-risk model trained on separate outcome data. Reader calls used to train that ranker must be counted; ground-truth correctness cannot select candidates at deployment.
- Investigate the observed date-understanding exception on fresh preselected temporal tasks. The current .50 slice is 8/16 versus 2/16 on GPT-OSS and 9/16 versus 3/16 on Qwen, discovered after looking at results. It is a lead, not a confirmed specialty win.

Recovering output format may reduce the many compressed-answer parse failures. Separate malformed/empty responses from semantically wrong valid answers. A small blinded manual audit can clarify this, but never relax grading only for our method. A new answer template must be compared symmetrically and must not overwrite old scores.

### I. Your “retain 90% accuracy and choose compression itself” idea

**This can remain a span-based project.** Predict span/set risk, choose retained spans and choose how aggressively to compress. It becomes an outcome-aware selection policy built on the current scorer, rather than abandoning criticality.

First define which target is intended:

- **Absolute accuracy:** at least 90% correct on the deployment population.
- **Conditional preservation:** at least 90% correct after compression among cases the fixed reader answers correctly uncompressed.
- **Relative accuracy:** compressed accuracy / original accuracy at least .90. This is not identical to conditional preservation because compression can rescue some originally wrong cases.

Neither a sigmoid span score nor a model saying it is confident certifies any of these. Real users usually do not supply gold answers. Agreement with an uncompressed answer is also not correctness. No policy can promise an individual prompt's correct answer from the current evidence, and it cannot generally guarantee accuracy above its uncompressed reader.

**Feasible first version, without RL:** on separate development/calibration families, evaluate a small ratio grid, for example {.50, .65, .75, .85, 1.00}, with a fixed reader. Fit a small risk model from deployable features: removed-score mass, broken groups, query/interface loss, task cues, actual length and budget. Choose the shortest permitted candidate under a calibrated risk threshold, with full-prompt fallback when needed. At deployment the controller should use local features; generating several candidates has a local cost that must be measured. Testing candidate answers against gold is an evaluation oracle, not a deployable controller.

Compare to fixed .75/.85 policies, a globally calibrated fixed ratio and an analogous LLMLingua-2 adaptive policy. Report achieved held-out quality, average tokens, fallback rate, coverage, failure rate and uncertainty. Reaching 90% by returning almost every full prompt is possible but a weak compression result; the savings at that quality are the point.

For context, the existing LLMLingua-2 .75 results already exceed or equal 90% as sample point estimates on both readers, while our grouped hybrid is 73.125% on both. Our controller would need less aggressive compression, better selection, or both. Those observed baseline percentages are not population guarantees, and an adaptive policy still needs separate calibration and confirmation.

No assumption of monotone answer accuracy with increasing ratio is justified. A threshold fitted on calibration data is an empirical policy, not automatically a conformal guarantee. Formal risk control needs its actual loss, sampling assumptions and policy family checked; ordinary monotone-loss conformal results cannot simply be attached to arbitrary non-monotone LLM correctness. Family dependence and multiple-policy selection also matter. [Conformal Risk Control](https://arxiv.org/abs/2208.02814).

**Code status:** new controller/calibration/evaluation needed. No scorer retraining is needed for the first feature-based version. This is the best optional practical extension after A–F if enough separate calibration families and quota remain.

### J. Outcome-reward training / RL — compatible, but a larger branch

Your proposed penalty is a legitimate learning signal. Keep the reader fixed; use training questions with gold answers, not validation/test outcomes for gradient updates. A span policy samples whole semantic-unit keep/drop decisions, and a reward can combine correctness, actual retained tokens, interface validity and budget feasibility. For example:

`reward = correct - lambda * retained_fraction - mu * invalid_or_infeasible`

This scalar reward trades quality against compression; it does **not** by itself enforce a 90% constraint. A constrained objective would minimize expected retained tokens subject to an estimated population error constraint, with separate calibration/held-out testing. Lambda, policy selection and stopping must be chosen without consulting the test.

Start with existing scorer features or weights, a small policy/reranker head and a frozen encoder. Full model retraining is not necessary for a pilot. Then consider fine-tuning only if useful. REINFORCE needs the probability of the actions actually executed; sampling words and then silently repairing them into unrelated grouped decisions gives the wrong policy objective. Use a defined constrained sampler or an explicitly modeled repair policy. Binary outcome rewards can have high variance, so use a baseline/control variate, bounded exploration and a fixed rollout cap.

A cheaper intermediate is an **offline candidate reranker**: generate a small fixed menu per training question, obtain outcomes, then learn which locally observable candidate features predict success. That avoids continual on-policy API sampling, though it has limited coverage of unseen candidate types and still needs honest family splits. Prefer it to launching a general PPO pipeline.

Training a compressor from downstream outcomes has clear precedents: PCRL learns discrete edits against a black-box reader; TACO-RL fine-tunes an encoder token-selection policy using task rewards and REINFORCE. Their existence rules out a generic “first RL compressor” claim, without making this a useless project extension. [PCRL](https://arxiv.org/abs/2308.08758), [TACO-RL](https://aclanthology.org/2025.findings-acl.81/).

**Budget example:** 100 training questions × 4 rollouts × 3 policy rounds = 1,200 compressed-reader calls, plus original controls, development, final tests and retries. This is a cost cap scenario, not evidence that 1,200 calls suffice to learn a competitive policy. Use a small pilot and stop if reward/held-out behavior does not improve. No reader fine-tuning or reader gradients are required.

**Recommendation:** retain this as a future extension, or a bounded pilot only after the diagnostic/confirmation package. Mixing teacher distillation, new taxonomy, RL and adaptive guarantees into one unablated v3 would make any gain uninterpretable.

### K. Other useful bounded additions

| Addition | Value and conditions |
| --- | --- |
| Consensus versus single-labeler / hard versus soft targets | Useful to substantiate supervision claims; same family splits, matched label support and identical training settings. Separate from old-checkpoint comparison. No recollection needed where saved votes exist. |
| One genuinely unseen-family reader | Closes an original promise; qualify it on a small disjoint pilot, freeze settings and keep it out of policy tuning. Endpoint availability/cost are not assumed. A cheaper local reader is possible, but reader weakness/context limits must be measured. |
| Held-out-labeler training | `--holdout-model` exists but was not executed. Useful for a specific transfer question; it does not by itself create an unseen target reader. |
| Budget/length-normalized critical retention | Use existing span lengths and random controls to ask whether retained critical content exceeds what length/category alone predicts. Report alignment lower/upper bounds for external outputs. Additional intrinsic scores do not replace downstream accuracy. |
| Small blinded human/parse audit | Select a fixed sample across methods/tasks, hide method labels, classify interface loss/relation damage/true reasoning error and report disagreements. State if only one annotator is available. |
| Denser accuracy–retention curves | Add a few declared ratios around a useful operating point for finalists, not another full seven-method grid by default. Include original controls and actual request costs. |
| Clean-versus-distracted robustness | Leverage the GSM-IC family mapping and clean prompts, with correct exposure checks. Useful to distinguish distractor pruning from simple number preservation. |
| Targeted minimal-pair challenge set | Optional small diagnostic of negation, comparison or temporal direction, with audited gold answers and explicit synthetic scope. Do not treat templated variants as independent families or use a handpicked challenge score as a broad benchmark win. |

## 6. A defensible contribution and novelty position

The best current claim is an empirical one: **cross-model single-span sensitivity is learnable, but that predictive success does not establish the sufficiency of a jointly compressed question.** The data/reliability limitations and negative LLMLingua-2 comparison are part of the finding.

If D/E yield evidence, a stronger contribution would be a measured account of **where mask/deletion disagreement and span interactions limit sensitivity-based compression**, with controlled semantic-unit or outcome-aware selection tested on fresh families. If C/F succeed, report the actual structural improvement and whether the neural component adds value. If I succeeds, report the held-out savings at a declared empirical quality target and its fallback behavior.

Those are concrete contributions; none is established now and none is certified first-ever by this literature check. Avoid “first counterfactual compressor,” “novel distillation,” “first RL method,” “guaranteed 90% accuracy,” or “model-independent criticality.” A course project can be strong without inventing a new general paradigm or beating the strongest baseline overall.

## 7. What I would defer or reject

- **Recollecting the existing full dataset:** unnecessary for A–D/F and most structural/controller work. Targeted new joint/taxonomy outcomes are different experiments, not replacement collection.
- **9,000 arbitrary pair/triple masks immediately:** costly evidence with ambiguous credit assignment. Diagnose and target first.
- **Changing all files or moving `final_eval/` into `src/`:** unnecessary and risks recorded imports/source hashes. Preserve historical and executed code; add one coherent follow-up entry point/config when implementation is requested.
- **Training on the 160 main questions and presenting the same set as a new test:** leakage, including teacher distillation, reranking and RL.
- **Blindly increasing bonuses or protecting every number:** GSM-IC distractors also contain numbers, and long protected groups can make the ceiling infeasible.
- **Changing reader, grader, answer wrapper or budget only for our method:** cannot establish a fair method gain. Symmetric new protocols must remain separate from old results.
- **Adding BLEU/ROUGE/BERTScore to make the project look stronger:** possible descriptive checks, but lexical/semantic similarity does not establish reasoning correctness. More metrics are not a substitute for the missing controls.
- **New huge encoders, broad hyperparameter searches, many new domains/languages, full reader fine-tuning, general PPO:** poor fit before direct gaps are closed. No need to rent a top GPU merely because credits exist.
- **Selecting only the date slice or a favorable reader as the final headline:** exploratory subgroup evidence needs confirmation, and the overall negative result remains reportable.
- **Saying no improvement means failure:** the scientific question can be answered negatively. Conversely, calling the existing compressor competitive overall would misrepresent the evidence.

## 8. Compute, API and stopping decisions

The recorded INR 386.9427 token estimate is about **6.58% of the INR 5,880 grant**. That suggests modest recorded usage, but does not establish the current wallet balance, pricing, credit expiry or availability of a particular endpoint. Reconcile the actual account statement before allocating paid runs. No exact remaining-credit claim is made here.

Historical recorded rates, INR per million tokens: GPT-OSS input/output 10/40; Qwen 28.431/222.709. Use current confirmed rates when executing. These examples are transparent arithmetic scenarios, not predictions of actual generation lengths or a current provider quote:

| Scenario for 9,000 requests | Assumed input/output tokens per request | Estimated INR at historical rates |
| --- | --- | ---: |
| GPT-OSS, moderate outputs | 400 / 1,000 | 396.00 |
| Qwen, moderate outputs | 400 / 1,000 | 2,106.73 |
| GPT-OSS, every output reaches 4,096 | 400 / 4,096 | 1,510.56 |
| Qwen, every output reaches 8,192 | 400 / 8,192 | 16,522.24 |

Formula: `requests * (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`. Add controls, retries, other experiments and any paid GPU rental. Repeated calls to the same original can be cached only under the fixed query/settings policy. Output/reasoning length can dominate the bill; short compressed inputs do not guarantee cheap generations.

Proposed initial caps, conditional on confirmed access:

| Work | Local GPU? | Additional hosted conditions before exact-query reuse |
| --- | --- | --- |
| Old-model/offset/exposure audit and saved-result analysis | Mostly CPU; GPU for old compression/scoring | Zero until reader evaluation |
| Component/structural development | Kaggle T4 or available server; no large reader hosted locally required | Example: 40 dev questions × 4 variants × 2 ceilings + 40 originals = 360 |
| D: mask/deletion | None for hosted reader | At most 360 for 120 spans on one reader |
| E: pair diagnostic | None for hosted reader | At most 320 for 80 pairs with four arms |
| F: fresh finalist test | GPU for compression, laptop for HTTP | At most 1,440 for 160 questions, four methods, two ceilings, one reader |
| Teacher distillation | Teacher inference + small-model training GPU | Zero for teacher label generation; separate downstream evaluation |
| Adaptive calibration | GPU for candidate compression | Scales as questions × candidate ratios, plus controls and independent test |
| RL pilot | Small policy training GPU; hosted reader optional | Explicit rollout cap, such as the 1,200-call example, plus all controls/evaluation |

These rows are overlapping planning examples, not an instruction to spend on all of them. The old-model retrospective can reuse saved v2/baseline responses for identical questions/configurations; no reason to repay the whole seven-method grid. A new snapshot of a provider endpoint is not necessarily bitwise identical to an old run, even at temperature zero; record served versions/settings and disclose temporal drift.

Use small dry runs, then a qualified reader pilot, then bounded execution with persistent reservations/cost accounting and a hard cumulative request ceiling. Stop on auth/quota/configuration failures; do not use retries as an unlimited budget. Never choose settings after seeing final test accuracy.

## 9. What a later implementation should produce

Keep `src/dataset_creation/`, original `src/` training/compression and existing frozen artifacts unchanged. Do not overwrite the published v2 model package. The current executed inference-core fingerprint is `1b089e04b7205aeeaae1be06ebafa4e92dfc300e657803c9b153942cb9339825`; future behavior changes require new provenance.

For an implemented follow-up, use a new versioned root such as `outputs/followup_v3_20261025/` and store:

1. Family/exposure audit and frozen train/development/calibration/test manifests, including exclusions and reasons.
2. Experiment specification: method IDs, source revision/fingerprint, package versions, checkpoint revisions/hashes, seeds, candidates, budgets, reader/grader/settings, primary contrasts and stop rules.
3. Original/compressed texts, kept ranges/groups, actual token counts, native baseline attempts and infeasibility/failure records.
4. Persistent raw response/usage ledger, every paid attempt, exact cache provenance and provider settings.
5. Paired comparisons, uncertainty, task/macro summaries, failure tables, process/controlled timing and final figures.
6. A claim-to-evidence table, including null/negative outcomes and a record of development versus confirmation decisions.

Keep executable instructions in the existing README/runbook workflow after code exists, and add final measured exhibits to RESULTS.md with an explicit new-study heading. This assessment can remain the one decision document; do not create a new MD per failed command. The proposed names above are future destinations, not claims that files already exist.

## 10. Second-pass checks and final decision criteria

This assessment was checked a second time for the mistakes most likely to mislead implementation or reporting:

| Check | Conclusion |
| --- | --- |
| Screenshot bonus arithmetic | Maximum .4, not additive .8–1.0; verified in both historical and final code. |
| Screenshot 148/160 claim | Recounted exact outputs: it is grouped versus ungrouped neural; neural versus prior is 86/160 at .50. |
| “We skipped proposal phases 2/3” | False for the original proposal; taxonomy/reliability phases belong to the mid-eval remaining-work list. |
| Shared extractor means full distribution match | Too strong; controls, unavailable distractor annotations, saved spans and parser/tokenizer coverage remain distinct. |
| AUC proves downstream or old-to-new improvement | False; label prediction, neural contribution and historical comparison are different questions. |
| Old-model evaluation can use any split/checkpoint | False; use the actual published split and audit related-family exposure. |
| Set failures are per-span labels | False; use set/conditional outcomes or an explicitly defined policy target. |
| Distillation/RL novelty | Generic claims refuted by primary prior work; specific extensions need evidence and narrower positioning. |
| A compact scorer guarantees proportional speedup | False; whole-system dependencies, batching, precision and budget search must be measured. |
| Accuracy must improve with longer prompts | Not guaranteed for generated answers; avoid assuming monotonic correctness in calibration or diagnostics. |
| 90% target implies individual guarantees | False; define the population metric, separate calibration/test and report uncertainty/fallback. |
| Unused downstream families are unseen by all models | False; proper held-out-fold use and outer splits are required for every model update. |
| 9,000 calls fit the budget | Depends on confirmed pricing, generated tokens, retries and controls; the supplied scenarios differ by over an order of magnitude. |
| More experiments are mandatory for a defensible submission | No. Existing scoped findings are reportable; A–F would answer important open questions rather than manufacture a win. |

**Success criteria for follow-up:** we can establish a fair old-versus-v2 comparison; identify whether the neural scorer changes useful compression decisions; measure rather than speculate about masking/deletion and joint effects; and either demonstrate a held-out improvement/efficiency trade-off or explain a well-tested negative result. A narrow empirical contribution with reliable evidence is worth pursuing. An unvalidated novelty claim or an endless search for one favorable score is not.
