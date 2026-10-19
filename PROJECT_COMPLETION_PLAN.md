# Project completion plan: Team Symbiote

Prepared 19 October 2026 from the local repository, proposal, draft mid-evaluation report, slides, reference result summaries, and primary research sources.

**Implementation location after repository repair:** the submitted tracked files are restored to commit `97dd6e7`. New correction code is isolated in [final_eval/README.md](final_eval/README.md), with commands in [final_eval/RUNBOOK.md](final_eval/RUNBOOK.md). The text below preserves the original planning audit; proposed paths and implementation-status statements describe that planning stage. Use the runbook for current commands. Reuse the existing collected responses; no full dataset recollection is required.

**Recommendation:** finish a reproducible study of cross-model counterfactual sensitivity on short reasoning questions, repair the evaluation, and test one structure-aware compression extension. Treat improvement over strong compressors as a research question. A well-supported negative or mixed result can still complete the project.

This is the first requested document. It specifies what to fix, implement, measure, and claim. The second document is now [final_eval/RUNBOOK.md](final_eval/RUNBOOK.md). The proposals below are a research roadmap, not a claim that all experiments are implemented or complete.

## 1. Decisions and constraints

- Plan for **nine days**, as requested; the evaluation date is not fixed.
- Use the **Kaggle free-tier T4** as the baseline execution environment, subject to session availability. Server GPUs such as the reported 2080 Ti, 4070, or A100 are opportunities to accelerate the same experiment, not prerequisites. The reported "a400" model is unconfirmed; detect actual hardware at execution time.
- Additional GLM/DeepSeek quota and more generous server access may be clarified tomorrow. Provide both a local-model route and a hosted-model route; do not make the core project depend on that access.
- Preserve collected responses. Do not repeat the entire labeling campaign or silently replace historical artifacts.
- This laptop is for implementation, inspection, and lightweight checks. Training, model downloads for experiments, compressor sweeps, and target-model inference belong on Kaggle/server.
- No application code is changed in this planning turn. No model/API experiments were run. Inspection included small CSV summaries and deterministic examples of answer-matching behavior.
- Existing untracked PDFs, PPTX, and result Markdown files are reference material, as you specified. They are not treated as shared, authoritative team outputs and were not committed or edited.

**Can the code be completed before the experiments run?** Yes: the pipeline, adapters, validation, tests with fabricated fixtures, and runbook can be implemented locally. Actual model loading, GPU behavior, API compatibility, runtime, and scientific outcomes still require server smoke tests. No responsible implementation can guarantee zero runtime issues or a baseline win without those checks.

## 2. What was inspected, and what actually exists

Read all tracked Python files, `config.py`, `requirements.txt`, `README.md`, `.gitignore`, `.env.example`, and the notebook's source and saved text outputs. Read all pages of `Symbiote-Proposal.pdf` (5), `MidSubdraft.pdf` (6), and `Symbiote_MidEval.pdf` (17), plus the PPTX's 17 slides and speaker notes. Read the three untracked results summaries. Inspected local data schemas, counts, offsets, repeated problem families, saved folds, and artifact availability.

The secret `.env` contents were not needed for the audit and were not displayed. Source/reference hashes and extracted document text are under ignored `outputs/planning_audit/`.

### 2.1 Local artifacts versus team-wide statements

| Item | Observed locally | Interpretation |
| --- | --- | --- |
| Raw questions | 1,000: GSM8K 200, GSM-IC 200, six BBH subtasks of 100 each | Full raw question collection exists here. |
| Candidate spans | 15,024 rows; 15,020 distinct `(example_id, start, end)` positions | Four positions have multiple category annotations. IDs must separate a span occurrence from its category memberships. |
| Consolidated raw labels | 9,189 model-span rows; 3,663 distinct positions; 247 questions | This is a partial local collection, not all four teammates' labels. |
| Local labelers | Qwen-labelled 3,642 rows, GPT-OSS 3,570, GLM 1,977 | These correspond to the local worker-1 shards. Model identity still needs the alias audit below. |
| Local consensus | 9,125 model-span rows; 3,599 distinct positions; 243 questions | Only 565 of these distinct spans have hard consensus `critical`. Do not substitute these counts for the team's merged dataset. |
| Local consensus coverage | 1,927 positions have three labelers; 1,672 have two | Report eligible labeler counts rather than assuming every span has three votes. |
| Saved split | `models/span_scorer/split.json` assigns all 1,000 questions to five folds | It cannot be validated against missing training outputs/checkpoints. |
| Local scorer weights/report | No `fold*/scorer.pt`, scorer report, or OOF predictions | Training may have been completed elsewhere, as you explained; it is not independently reproducible from this checkout's artifacts alone. |
| Local compressed prompts | 4,000 rows: LLMLingua and LLMLingua-2, each at 0.50 and 0.25, over 1,000 questions | No local hybrid/random compression outputs in this CSV. Only 486 rows per baseline have retention annotations, reflecting 243 locally labeled questions times two budgets. |
| Local downstream outputs | No `evaluation_results.csv` or `evaluation_summary.csv` | Reported Groq/DeepSeek accuracy cannot be regraded or independently verified here yet. |
| Slides/notebook | Report 924 questions and 14,162 distinct labeled spans | Plausible merged-team figures, but the corresponding complete CSVs are not present locally. These are not the raw candidate-extraction counts. |

The notebook's saved 924/14,162 counts differ from both current raw extraction and local partial consensus. Use the artifact handover to reconcile these versions before quoting team-wide counts.

### 2.2 Important observations from the actual data

1. Removing the stored distractor and normalizing punctuation/whitespace reduces the 200 GSM-IC rows to **51 clean base questions**. The existing saved split places variants from **41 base families, containing 187 rows, in multiple folds**. This is train/test leakage through variants even when `example_id` differs. This audit found no exact canonical overlap between these GSM-IC bases and the selected GSM8K questions; variants within GSM-IC are already enough to require a repair. Verify families against upstream IDs during import.
2. Of 3,000 random-control spans, **1,171 overlap a curated span**. The code excludes only identical boundaries, despite describing non-overlapping controls. These controls are also not explicitly length-matched to the targeted spans.
3. All 15,024 candidate span texts match their saved character offsets. This part of the local extraction is consistent.
4. All 200 GSM-IC rows have distractor text, but only **195 distractor spans** were extracted. Recover exact insertion boundaries before claiming complete distractor annotation coverage.
5. The local LLMLingua outputs do show mean saved retention ratios of about **0.932 at target 0.50** and **0.897 at target 0.25**. That is a real result for this run/configuration. It does not establish that LLMLingua generally cannot compress short questions.
6. `tiktoken` is absent in this virtual environment. Current token counting therefore falls back silently to a regex, while the saved CSV uses another counting regime. Recomputing LLMLingua-2 ratios with the fallback gives materially different ratios. Preserve and explicitly name the tokenizer used for every run.

## 3. Research framing: what to keep and what to change

The proposal is more defensible than parts of the draft: it explicitly asks whether criticality transfers, rather than assuming universality, and includes deletion validation and problem-level splitting. Finish those commitments.

**Suggested title:** *Counterfactual Span Sensitivity for Short Reasoning Prompt Compression: Transfer, Structural Preservation, and Evaluation Limits.*

Recommended research questions:

1. Which sampled span categories show the greatest measured failure sensitivity, conditional on a model solving the original question?
2. How much of that signal is shared across labelers, after accounting for label prevalence and eligibility?
3. Does learned multi-model supervision add value beyond category priors, simple features, surprisal, and structural protections?
4. How well does held-out critical-span retention predict downstream correctness at matched budgets? Where does it fail?
5. Does grouping related words during deletion improve the accuracy/token trade-off over the corrected independent-word compressor and external baselines?

These questions remain useful if the learned method loses. **At present, downstream inferiority to LLMLingua-2 has not been established either:** the available summaries do not contain the needed paired downstream baseline evaluation.

### 3.1 Literature and novelty boundaries

Sources checked on 19 October 2026. This is a targeted literature check, not proof that every related paper has been found. Statements below concern the cited work's scope, not an independent verification of its reported experimental results.

| Prior work | What it already establishes or proposes | Consequence for this project |
| --- | --- | --- |
| [LLMLingua, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.825/) | Coarse-to-fine prompt compression with a budget controller and iterative token removal. | Do not describe it as merely an unordered perplexity threshold. |
| [Selective Context, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.391/) | Self-information-based context reduction. | Use the official implementation or clearly name our approximation. |
| [LLMLingua-2, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.57/) | Distilled encoder-based token classification; its evaluation includes GSM8K, BBH, and transfer across LLMs. | Learned compression, these benchmarks, and cross-model evaluation are not individually novel. Its exact prompt scope may differ from compressing only the question here. |
| [PCRL](https://arxiv.org/abs/2308.08758) and [TACO-RL, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.81/) | Learned discrete/task-aware compression, including downstream task reward. | Do not claim to introduce task-aware or answer-aware compression in general. |
| [AGORA, May 2026 preprint](https://arxiv.org/abs/2605.26596) | Uses offline counterfactual action-change supervision, a small scorer, and structural retention for agent history compression. | Counterfactual supervision plus a small inference-time scorer is already present in closely related work. Our question-level, multi-labeler assay differs in scope, not in all primitives. |
| [Relevant but Incomplete, August 2026 preprint](https://arxiv.org/abs/2608.04569) | Studies dependent evidence split by independent hard compression and restoration of missing support in multi-hop QA. | Dependency preservation and the failure of independent ranking are also existing ideas. Relate our short-question experiment to this work. |
| [Causal Sufficiency and Necessity Improves CoT Reasoning](https://arxiv.org/abs/2506.09853) | Studies necessity and sufficiency of generated reasoning steps. | The general distinction is not new; input-question deletion and output-CoT optimization are different settings. |

**Recommended contribution claim, before results:**

> We investigate multi-model, correctness-conditioned span sensitivity in short reasoning questions, audit its reliability as a compression signal, and compare independent-word and structure-aware deletion under common token budgets.

Possible empirical contributions, contingent on verified experiments:

- A reproducible dataset with model-specific outcomes, explicit eligibility/missingness, occurrence offsets, and base-question group IDs.
- An empirical test of whether consensus supervision improves held-out performance beyond inexpensive priors.
- A measurement of when critical-span recall does and does not track reasoning accuracy.
- A controlled comparison of a modest structural extension, including its non-neural counterpart.

Do not call these a first-ever method or a new theory. The particular combination and evidence can make a strong course project without an invention-of-the-field claim.

### 3.2 Why individually non-critical spans may be unsafe to delete together

The current labels ask a local question: does masking **one** span in the otherwise complete prompt make the answer wrong? The compressor performs a different intervention: deleting many words simultaneously, including words outside labeled spans.

For example, if two sentences each state the same required quantity, deleting either alone may be harmless. Deleting both loses the quantity. Separately, preserving `3`, `4`, and `12` is insufficient if deletion destroys who owns which quantity or what `per day` modifies. A `[MASKED]` token can also signal a missing slot that plain deletion does not signal.

Therefore:

- Single-span masking sensitivity is neither a proof of intrinsic linguistic necessity nor a certificate that a jointly compressed prompt remains solvable.
- High critical-span recall alongside low accuracy can be an informative finding, after the retention metric and grading are repaired.
- Adding more surface similarity metrics will not solve this mismatch. A controlled structural intervention and downstream evaluation address it more directly.

## 4. Mandatory repairs before interpreting new results

The locations below refer to the audited source snapshot. Historical server code may differ; recover that version before attributing any bug to an old run.

### P0-A. Protect artifacts and make worker merging explicit

Evidence: `03_counterfactual_labeling.py:279` assigns work by question order modulo four. `04_build_consensus.py:31` reads matching `_part*.csv` files whenever they exist and overwrites consolidated labels/consensus.

Required work:

- Import teammates' outputs into immutable, named legacy folders with hashes, worker IDs, source commit/version where known, and dataset/prompt hashes.
- Use an explicit list/manifest of input shards. A local worker shard must not automatically replace a more complete team aggregate.
- Validate identical prompt, gold, span offsets, perturbation text, and labeling configuration for the same identity across workers. Reject conflicting duplicates with a conflict report; do not silently keep the first row.
- Give each occurrence a stable ID from dataset revision, problem identity, prompt hash, and offsets. Keep category memberships separately when boundaries coincide.
- Count distinct eligible models, not arbitrary rows. Separate `original_wrong`, `api_failed`, `parse_failed`, `not_attempted`, and valid perturbed outcomes.
- Check that the four assignments cover the intended common manifest without overlap or omissions. Persist assignments instead of recomputing them from mutable CSV order.
- Validate worker IDs and handle an empty `WORKER_ID` safely; `.env.example` currently leaves it blank while `int("")` would fail.

### P0-B. Repair answer grading and regrade saved responses

Evidence: `src/utils/answer_matching.py:97` accepts any matching number in the extracted answer; the choice-text fallback at line 124 accepts empty substrings.

Lightweight examples run against the current function:

| Response | Gold | Current result | Required behavior |
| --- | --- | --- | --- |
| `Final Answer: 42 or 17` | `42` | Correct | Ambiguous; do not accept as a unique numeric answer. |
| `Final Answer: 1/2` | `1` | Correct | Wrong; parse the fraction as one quantity. |
| Empty output, with `(A) Alice` in the original options | `(A)` | Correct | Invalid/empty answer. |
| `Final Answer: no, actually yes` | `yes` | Correct | Require a documented, task-specific rule; do not match by a loose suffix. |

Implement task-specific numeric, option-letter, yes/no, and valid/invalid parsers. Parse a unique final answer; handle signs, decimals, fractions, currency, and units deliberately. An explanation containing the gold number must not count as a correct final answer. Empty, non-finite, truncated, or ambiguous responses need explicit statuses. Use original question options as the reference schema, while recording if the compressor destroyed the options.

Regrade existing raw responses before purchasing or spending any new calls. Recompute original-answer eligibility and consensus from those grades. Some previously valid labels may disappear. Conversely, questions whose earlier original response was graded wrong may have unqueried spans; report that missingness rather than inventing labels. Exact regrading may be impossible for responses truncated by the old evaluator (`response[:1500]`); recover full outputs where available and flag unrecoverable cases.

### P0-C. Repair base-family splits and evaluation cohorts

Evidence: training at `train_scorer.py:425` splits `uid`, not clean base-question families; the local GSM-IC audit demonstrates cross-fold variants.

- Recover upstream IDs, clean GSM-IC questions, and exact insertion positions; retain clean and distracted versions under one `base_group_id`.
- Use group-aware train/validation/test partitions, with dataset/subtask balance where feasible. All spans and variants of a base stay together. Record group counts and row counts.
- Keep training eligibility separate from downstream evaluation eligibility. A question without two competent labelers can still be part of a downstream benchmark; its retention labels are simply unavailable.
- Compare methods on the same frozen question manifests. The existing intrinsic report mixes 1,000, 924, and 100 question cohorts.
- Existing questions have already been inspected historically. A newly locked split is a repaired retrospective evaluation, not a pristine never-seen benchmark. If time permits, add a small fresh, deduplicated inference-only holdout; it needs no expensive counterfactual labels.
- Retrain when fixing a leaked split. An old checkpoint cannot be made leak-free by renaming its folds.

### P0-D. Make soft targets and held-out-labeler tests real

Evidence: `train_scorer.py:124` looks for per-model `label`, but the pipeline emits `model_specific_label`. At line 141 it checks several soft-score names but omits `agreement_score`, the column emitted by consensus construction.

As written, the current trainer falls back to hard consensus targets on the current schema. It also cannot execute its advertised leave-one-labeler-out path on that schema. In the local consensus, **747 of 3,599 positions have intermediate scores** that this default path discards.

- Normalize legacy names into an explicit schema and use the actual vote counts/soft fraction. Add a regression fixture with scores 0, 1/3, 1/2, 2/3, and 1.
- Define strict-majority ties consistently. Consensus uses `> 0.5`; the held-out-labeler path currently uses `>= 0.5`.
- For a held-out labeler, construct training targets from the other labelers only, and prevent its labels from influencing feature/model selection. Compare on shared eligible occurrences; include a complete-panel sensitivity analysis.
- Use unweighted soft BCE/binomial supervision as the calibration-oriented default. Current `pos_weight` changes probability calibration; weighted-loss scores cannot automatically be interpreted as probabilities of failure. Keep imbalance weighting as a named ablation or calibrate on validation data.
- Drop or window spans that are only partly represented after truncation. Checking `mask.any()` handles fully missing spans but still accepts partly truncated spans.
- Save tokenizer, encoder revision, preprocessing version, training parameters, best epoch, group partitions, and checkpoints together. Keep DistilBERT initially; a larger encoder is not a prerequisite.

### P0-E. Unify extraction and separate controls from deployment features

Evidence: `02_extract_spans.py` extracts multi-token quantities and comparison phrases; `compress.py:90` implements a different mostly token-level extractor. Random-control exclusion tests exact range equality, not overlap.

- Share one deterministic inference extractor between training and compression. Preserve historical span definitions when reusing labels; version new definitions rather than pretending old labels apply to new boundaries.
- Report coverage by category. Never treat absent criticality predictions as evidence that a word is safe to delete.
- Preserve repeated occurrence offsets and overlapping category membership.
- Repair controls prospectively using actual overlap exclusion and a matched length distribution. Existing mislabeled-as-matched controls can still be analyzed as random windows, with their overlaps disclosed. Use existing non-overlapping, length-stratified subsets before requesting new labels.
- Do not feed `inserted_context`, gold answers, counterfactual labels, or test outcome metadata to a deployed compressor. Known distractor boundaries are evaluation annotations and an explicitly named oracle diagnostic only.
- Conditional/comparison categories already exist. The draft's statement that the taxonomy contains no conditionals is incorrect. Missing items include richer scopes, quantifiers, temporal and relational constructions, not the entire category.

### P0-F. Fix token budgets, baseline adapters, and ablation names

Evidence: `compress.py:70` silently changes token counters; `delete_to_budget` sums independently tokenized words with added spaces and uses a ceiling; reconstruction joins words with spaces and removes line structure. `BaselineRunner` implements its own surprisal baseline under the name `selective_context`.

- Require one explicitly configured, versioned accounting tokenizer, with no regex fallback in scientific runs. Separately report the target model's prompt-token usage when available.
- Define the whole-question budget as `floor(r * T(original_question))`. Count the final reconstructed string, including any retained query/options, and enforce that budget. Do not claim a strict whole-question budget from additive per-word costs.
- Preserve original order, useful punctuation, and newlines in reconstruction. Log requested and actual ratios per example; report underfilling as well as overflow.
- Rename the current Selective-Context substitute to `surprisal_word`. It may remain a useful ablation. Add the [official Selective Context adapter](https://github.com/liyucheng09/Selective_Context), verifying its removal-ratio semantics on tiny examples.
- Pin the official LLMLingua package/revision and supported checkpoints. The current LLMLingua-2 default name does not match the documented [official example](https://github.com/microsoft/LLMLingua), which uses `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` or the documented smaller mBERT checkpoint. Verify actual resolution before running.
- Investigate LLMLingua's short-input behavior with the official API, explicit `target_token`, suitable iteration settings, and token filtering enabled. Its [documented API](https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md) exposes these controls. The specific cause of the old budget misses is not established by this audit.
- Baselines must run without our scorer checkpoints, fold files, or unnecessary neural models. Current `main` initializes tokenizer/surprisal machinery and requires `split.json` even for unrelated methods.
- Preserve native external-baseline outputs. If a separate budget-adjusted variant is needed, label the adjustment, choose it on validation data, and report it separately. Do not truncate a baseline arbitrarily and attribute the resulting method to its authors.
- All current `ours` variants share numeric/unit/query bonuses (`compress.py:417`). Call them accordingly; an apparent scorer improvement may be the protection heuristic. Add clean component toggles.
- Assign random seeds by immutable `(uid, method, ratio, seed)` so method ordering, resumptions, or adding a new baseline cannot change existing outputs.

### P0-G. Replace substring retention with occurrence-aware retention

Evidence: `compress.py:296` and `metrics.py:109` check whether `span_text` appears anywhere. If three occurrences say `5`, retaining one may credit all three. A whole distractor sentence counts as removed after just one word is missing.

- Our extractive methods should emit retained source-character/token intervals, with a mapping to the reconstructed output.
- For external methods, use their retention/word labels when available, or a monotone token/character alignment. Repeated text and subword fragments can make alignment ambiguous; mark uncertainty and report bounds or exclude unresolved annotations with coverage. An LCS alignment alone is not unique ground truth for repeated occurrences.
- Report exact full-span survival and fractional source-token coverage separately.
- Compute distractor **token/character mass retained**, not just survival of the full sentence. A reported 0% sentence-survival rate does not establish that 100% of distractor tokens were discarded.
- Compute both macro per-question retention and pooled micro retention, with denominators, no-critical-span handling, and category supports stated.
- Recompute old intrinsic summaries under the corrected metric; preserve the old ones as legacy, not silently edited results.

### P0-H. Fix intrinsic reporting and statistical joins

Evidence: `metrics.py` fills missing values with empty strings before checking missing retention; this can prevent recomputation and cause `float("")`. Existing `cat_retention_json` is not aggregated when using saved retention columns. Its default labels path is `data/consensus.csv`, not the configured consensus file. `evaluate.py` joins summaries without enforcing key uniqueness and pools the critical-kept effect over budgets.

- Use typed missingness and explicit booleans, schema validation, unique experiment keys, and merge cardinality checks. Never let a duplicate multiply evaluated rows.
- Recompute retention only for missing/invalid rows, or explicitly recompute all rows under the new metric version.
- Include retention, budget compliance, supports, and uncertainty in by-dataset outputs, not only PPL and ROUGE.
- Compare downstream methods on paired common examples; report missing and failed outputs separately. Do not turn an API outage into a wrong reasoning answer, or silently compare different successful subsets.
- Analyze critical retention versus correctness within dataset and budget, controlling for question length and baseline difficulty. The old pooled `all_critical_kept` table is associative and confounded, not a causal test.
- Treat a confidence interval excluding zero as a CI result. Do not invent an exact `p < 0.05` value from it. Predeclare primary comparisons and correct families of secondary tests where formal significance is claimed.

### P0-I. Make execution resumable and identify actual models

Evidence: `api_clients.py:169` silently rewrites requested `qwen3.7-max` to `qwen3.8-max`. A single Motapis key cycle is shared across its provider variants. Retry layers can multiply attempts; daily tracking does not count each underlying HTTP attempt. `evaluate.py` saves truncated responses and sleeps per thread rather than enforcing one provider-wide request rate.

- Remove silent model aliases. Record requested ID, effective request ID, returned model ID where provided, provider, endpoint, date, generation parameters, token usage, finish reason, and configuration hash. Historical model identity that cannot be recovered must remain uncertain.
- Do not repeat the draft's architecture/parameter-size claims for provider aliases without authoritative evidence. Three provider/model names do not establish architectural diversity.
- Use one response/result abstraction for local and hosted targets. Preserve full outputs, parsed answers, parse status, provider failure status, timing, and usage separately. Reasoning-only content after truncation must not become a valid final answer by accident.
- Keep cache keys tied to exact prompt text, wrapper, model/revision, decoding settings, method/checkpoint, and tokenizer configuration. Existing keys based only on method name and ratio can mix incompatible reruns.
- Use atomic checkpointing/append records and guaranteed flushing on interruption. Save failure records too; retry only retryable failures with bounded provider-aware backoff.
- Apply provider-wide throttling and explicit attempt/token budgets. Stop a provider on exhausted quota/auth failure instead of repeatedly querying it. Maintain credential isolation; do not share one initialized key cycle across model providers.
- Budget each confirmed provider/key separately and expose its remaining allowance. Local inference is the durable fallback.
- Measure cold load separately from warmed compression latency. Fix double counting of preparation time in the first hybrid variant, and time only the components each variant uses.

## 5. The one recommended algorithmic extension

Implement **structure-aware grouped deletion**, retaining the existing DistilBERT scorer. This is an engineering hypothesis to test, not a guaranteed novelty or accuracy gain.

### 5.1 Concrete behavior

1. Parse the question into ordered source ranges. Identify query and answer-option ranges, plus local constructions such as quantity-unit phrases, negation-governor phrases, and multiword comparisons. Use explicit rules with coverage diagnostics rather than claiming a complete logical parser.
2. Merge overlapping protected local constructions into atomic groups; preserve uncovered text as selectable units too. Every input range must have a known fate. Option/query handling must also cover BBH tasks without a question mark.
3. Score units from predicted criticality and the existing surprisal signal, optionally using a small penalty for breaking dependency links. Tune a small, declared grid on validation only. Include the query/options in all token accounting.
4. Select a budget-feasible sequence of groups, reconstruct it from source ranges, and recount the entire output. Use a deterministic greedy/knapsack-style approximation with exact final budget checking; do not claim a globally optimal solution because tokenizer costs are non-additive.
5. Save the retained ranges, group decisions, score components, requested/achieved budget, and reason for any infeasibility.

The minimum extension is local phrase grouping and robust query/options handling. Full coreference resolution, a learned graph network, and a new labeling campaign are outside the nine-day core.

**Handle infeasibility explicitly.** If protected query/options alone exceed a strict budget, report that the protected variant is infeasible at that budget. Do not overshoot silently or drop those examples from its accuracy denominator. Report infeasibility rate and, if useful, a separately named relaxed-protection variant. An adaptive policy that keeps the full prompt is a different operating point, with its actual token cost reported.

### 5.2 Comparisons that make this informative

The extension must have a counterpart with **identical groups, protection rules, budget solver, and reconstruction**, but scores from train-derived category priors/simple features rather than the neural scorer. This isolates what the trained component adds.

Also compare corrected independent-word hybrid versus grouped hybrid. If groups help both heuristic and learned methods equally, the finding is structural preservation, not a benefit of counterfactual training. If the heuristic is best, report that and use it as the practical recommendation.

Use two explicitly separated settings where feasible:

- **Whole-question compression:** every method receives the same complete question, and output length is measured against that question. Our protections count against its budget.
- **Shared protected interface diagnostic:** every method gets the same query/options scaffold, with only the remaining context compressible. Use the same decomposition and whole-question accounting for all methods. This tests whether answer-interface damage explains losses.

Do not compare our protected setting with a baseline's unprotected setting and attribute the entire difference to learned scoring. Keep this diagnostic small rather than doubling the entire experiment.

### 5.3 Optional inference-time policy

If moderate budgets are consistently better, fit a simple validation-only policy that chooses among a small set of retention levels using input length and predicted risk. Evaluate its average tokens, accuracy, and fallback rate against fixed-budget baselines at comparable aggregate cost.

This is a stretch item. A sum of single-span failure scores is not a certified bound on joint deletion risk; any such score must be empirically validated. Never present "skip compression" as a win at the same 50% budget.

## 6. Experiment program: priority, cost, and interpretation

The core result should be a fair downstream comparison, supported by a few decisive diagnostics. More metrics cannot substitute for missing downstream evidence.

### 6.1 Freeze the experimental contract

- **Question scope:** compress the benchmark question only, as proposed. Keep the external solve instruction identical for every method. Count both question tokens and complete wrapped input tokens, and report the distinction.
- **Training population:** recovered label-eligible occurrences from training groups. Never expose held-out gold, known distractor ranges, or intervention outcomes to compression.
- **Evaluation population:** frozen group-aware question manifest, including original-wrong examples for unconditional accuracy. Report original-correct conditional results separately.
- **Primary budgets:** retain **0.75 and 0.50** of question tokens. Existing evidence motivates adding moderate compression; it does not prove a safe floor. Use 0.90 and 0.25 as diagnostic operating points if compute permits; 0.33 is lower priority.
- **Primary local target:** select an available open-weight instruction model using original-question accuracy, parsing reliability, and runtime on a small validation pilot. Candidate cards include [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) and [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct). These are fallback candidates, not claims about which will perform adequately here.
- **Target-model gate:** if a small target has near-floor uncompressed performance on a subtask, it cannot give a useful preservation test there. Try a stronger available target or narrow that subtask's inference claims, keeping the failure visible. Do not select the target according to which compressor wins.
- **Additional hosted target:** use the newly available DeepSeek/GLM access only after the actual model ID, quota, and throughput are confirmed. A new Qwen checkpoint is unseen as an exact model but still belongs to a labeling family; distinguish model holdout from family holdout.
- **Primary methods:** official LLMLingua, official LLMLingua-2, official Selective Context, corrected legacy hybrid, grouped hybrid, the same grouped method with non-neural scores, and seeded random deletion. The uncompressed reference is queried once per question. If an official adapter fails, report it as unavailable; do not replace it silently with a different algorithm.
- **Ablation methods:** pure surprisal, category prior, a small feature model, scorer without bonuses, legacy signal/scorer with bonuses, and hard versus soft supervision. Run most downstream ablations on validation/diagnostic subsets, not the full test sweep.
- Freeze method parameters, primary comparisons, cohort identities, and evaluator settings before the final sweep. Historical exploration must be disclosed.

### 6.2 Ordered experiment list

| ID / priority | Experiment | Uses existing outputs? | New compute/calls | Question answered |
| --- | --- | --- | --- | --- |
| E0 / mandatory | Import, deduplicate, regrade, reconstruct eligibility and consensus | Yes; needs complete teammate artifacts | CSV/CPU only | Are labels and reported accuracy trustworthy? |
| E1 / mandatory | Labeler agreement on common eligible occurrences; prevalence, positive/negative agreement, Cohen kappa, category/subtask breakdown | Yes | CPU only | What is actually shared across models? |
| E2 / mandatory | Group-safe scorer training, with category prior and simple feature baselines | Existing labels | Small encoder training on T4/server | Does context learning beat cheap category recognition? |
| E3 / mandatory | Corrected compression and occurrence-aware intrinsic audit, including budget and alignment coverage | Some existing prompts reusable | Compressors on GPU; metrics mostly CPU | Are comparisons matched and are critical occurrences retained? |
| E4 / mandatory | Paired downstream comparison at 0.75 and 0.50 on one frozen cohort and one adequate target | Reuse only configuration-compatible full responses | Local GPU inference, or confirmed hosted quota | Does any proposed method improve task accuracy at matched cost? |
| E5 / high value | Same groups/bonuses with neural versus non-neural scores; independent versus grouped selection | Reuse E4 where applicable | Small additional validation inference | Is the benefit from supervision, structure, or heuristics? |
| E6 / high value | Predictive value of occurrence-aware critical retention for correctness, within dataset/budget | E3/E4 | CPU only | Is the intrinsic metric a useful proxy? |
| E7 / required to claim mask-to-deletion validity | Stratified single-span mask versus true-deletion check, with fresh originals/repeats on a small subset | Old mask responses reusable only if model/config identity matches | Bounded additional inference | Do labeling and deployment interventions agree? |
| E8 / optional | Leave one labeler out; train using only the others, evaluate its own labels on held-out question groups | Existing per-model labels | Additional small scorer fits; no new API labels | Does consensus predict another model's sensitivity? |
| E9 / optional | Selected jointly deleted low-sensitivity spans versus their single deletions, with matched-length random groups | Some single interventions reusable | Small targeted inference | Can joint removal fail despite benign single interventions? |
| E10 / optional | Second target, fresh question holdout, extra budgets, adaptive policy, or broader BBH relations | Partly | Extra inference/training | How far do the findings generalize? |

**Core completion:** E0-E6 plus an honestly bounded E7, if the necessary exact model access exists. If E7 cannot be completed, keep mask-to-deletion validity explicitly unverified and narrow the claim; do not fabricate equivalence. E8-E10 are ordered stretch work, not simultaneous obligations.

### 6.3 Scorer evaluation and cheap baselines

Use a training-only category prior, and a lightweight feature model using category, span length, position, digit/unit flags, and query overlap. Do not include source gold or distractor metadata in deployable features.

Report held-out average precision/PR-AUC, ROC-AUC, critical-class precision/recall/F1, Brier score or log loss for soft targets, and calibration. F1 thresholds come from validation data. Include within-category results: high pooled AUC may merely identify that numbers receive higher criticality labels than verbs.

Run unweighted soft supervision versus hard consensus under the **same** fixed partitions and training budget. If training time is tight, start with one group-aware train/validation/test split and one seed. Five-fold cross-validation already exists as a design, but do not require five-fold retraining for every ablation. Add grouped OOF results or extra seeds only after the core pipeline passes. State seed and split limitations.

For consensus versus single-labeler comparisons, match training occurrences and number of examples. Do not give consensus more data and call the entire improvement an aggregation benefit.

### 6.4 Mask/deletion and interaction checks

For E7, sample across label, category, dataset, and span length. A balanced diagnostic sample is not representative of overall prevalence; state its sampling scheme or use inclusion weights for population estimates.

Compare mask and deletion **on the same model configuration** with a correct original baseline. Report a 2x2 outcome table, agreement, kappa, both directional disagreements, and sample sizes. Include a small repeated-original control: temperature zero does not guarantee hosted determinism. Do not compare an old Qwen mask result to a new DeepSeek deletion result as a mask/deletion agreement test.

If the original hosted model is unavailable, a fresh local-model three-way assay (original, mask, delete) is still useful, but call it a new-model assay rather than validation of all historical labels.

For E9, choose a small predeclared sample of pairs/groups of individually low-sensitivity spans, measure the joint-deletion outcome, and compare with length-matched deletions. This supports a concrete joint-intervention finding if observed. Keeping all labeled critical spans and still failing is suggestive alone: unannotated content, noisy labels, mask artifacts, and solver errors are alternative explanations.

### 6.5 Compute and quota budgeting

For `N` questions, `M` compressed methods, `R` retention levels, and one target, the planned generations are:

`N * (1 + M * R)`

The `1` is the shared uncompressed reference, not one per method. Identical query texts under identical target settings can share a cached generation without being counted as independent samples.

Example planning envelopes, **not runtime promises**:

| Tier | Questions / compressed methods / ratios | Generations per target | Use |
| --- | --- | --- | --- |
| Smoke | 8-16 / selected methods / 1 | Small bounded batch | Validate all answer formats, no full sweep. |
| Minimum defensible | 80 / 7 / 2 | 1,200 | A broad pilot with wide uncertainty and limited per-subtask claims. |
| Recommended initial main sweep | 160 / 7 / 2 | 2,400 | Feasible starting scope pending measured tokens/second or confirmed API quota. |
| Expanded | 240 / 7 / 2 | 3,600 | Only if the pilot and remaining compute support it. |

These are example cohort sizes; the final size must respect actual group counts, dataset coverage, and labeled versus unlabeled membership. A 160-row set with many variants does not contain 160 independent GSM-IC problems. If hosted quota is too small, evaluate all chosen methods on the same smaller predefined cohort or only use the hosted model as an explicitly limited transfer check. Do not selectively omit competitors after seeing outcomes.

Benchmark a short validation batch to estimate model-load time, compression throughput, generated tokens, generation time, and truncation frequency. Budget **tokens and retries**, not just successful requests. A 2,400-generation sweep can be cheap or slow depending on output length and hardware; no estimate here assumes an A100.

T4/server execution should load the scorer, each large compressor, and target model in separate stages rather than retaining them all on GPU. Use supported FP16 on older cards; choose BF16 only after device capability checks. Batch conservatively and checkpoint per stage. Optional quantization needs a fixed, recorded configuration; do not change target precision between methods.

For Kaggle, cache required model/dataset files and persist outputs/checkpoints outside the ephemeral session through the team's normal export workflow. The runbook must explain how to restore them. Do not make correctness depend on continuously available internet or one uninterrupted session.

## 7. Metrics that strengthen the project, and metrics that do not

### 7.1 Primary downstream measures

- **Unconditional exact-match accuracy**, by dataset/subtask, on common questions.
- **Paired accuracy difference** against the strongest external baseline and against the matched non-neural method, with 95% confidence intervals and independent group counts.
- **Conditional preservation:** `P(compressed correct | original correct)`, on a correctly graded original baseline. Also show correct-to-wrong losses and wrong-to-correct recoveries.
- Accuracy against **actual** retained-token fraction, with budget compliance. Plot 0.50 and 0.75 plus optional points and the full-input reference. Two measured budgets support two operating-point comparisons, not a universal optimum.
- Pooled micro accuracy and an explicitly defined dataset/subtask macro average. BBH contributes six hundred raw rows and should not silently dominate an alleged general reasoning score.

Uncompressed accuracy is a **reference**, not a mathematical upper bound: removing distracting content may improve an answer. Random deletion is a sanity comparator, not a formal lower bound.

### 7.2 Useful diagnostic measures

| Measure | Why it helps | Limit |
| --- | --- | --- |
| Exact budget-compliance rate, mean absolute error, signed error, actual-ratio quantiles | Detects methods using more or fewer tokens than intended | A mean close to the target can hide large individual misses. |
| Occurrence-aware critical-span recall and soft-vote-weighted recall | Measures retained labeled sensitivity signal | Depends on labeler, perturbation, candidate coverage, and alignment quality. |
| Fractional coverage of annotated spans | Separates partial from exact survival | Token presence does not establish preserved meaning or scope. |
| Distractor token mass retained and non-distractor content retained | Measures actual GSM-IC filtering | Known boundaries are evaluation-only. Full-sentence absence is insufficient. |
| Query/options survival and quantity-unit/comparison group integrity | Diagnoses destruction of answer interfaces and relations | These are structural proxies, not certified semantic equivalence. |
| Critical retention versus correctness, stratified by task and budget | Tests whether the intrinsic score predicts the real objective | Associations remain confounded; include length/difficulty controls. |
| Warm compression latency p50/p95, peak memory, target input/output tokens, total pipeline time | Establishes whether token savings offset compressor overhead | Measure on a named device/configuration; short inputs may not save time. |
| Offline labeling/training cost and amortization estimate | Places the multi-model supervision cost in context | Clearly separate estimated from measured costs. |

**Do not use additional semantic-similarity metrics as a rescue strategy.** Embedding similarity can stay high when a negation or quantity is wrong. It can be supplemental but cannot justify improved reasoning performance.

### 7.3 Why existing PPL and ROUGE should be demoted

The current ROUGE implementation compares whitespace word sequences. For an exact order-preserving subsequence with `k` of `n` original words, LCS length is `k`, precision is 1, recall is `k/n`, and ROUGE-L F1 is:

`2 * (k/n) / (1 + k/n)`

At fixed retained word fraction, this is largely a length statistic. It does not tell us whether the surviving words preserve the arithmetic or logic. Real subword-based baselines may also change whitespace fragments, complicating the comparison.

DistilGPT2 perplexity measures compatibility with that model's text distribution. Telegraphed text can be awkward yet usable; fluent text can omit the only required fact. Here, the same model family supplies the surprisal signal, so treating its PPL as an independent quality judge is particularly weak. If retained, report cross-entropy/median or log-scale PPL with truncation handling and use it as a diagnostic, not the headline.

The intrinsic metrics are also not all "reference-free": critical retention uses intervention-derived annotations, and ROUGE uses the original question as a reference. Call them **no-new-answer-call diagnostics** instead.

### 7.4 Statistical discipline

- Resample base-question families for confidence intervals, keeping each family's variants and method outputs together. Apply the same clustering to span-level results. Consider within-subtask stratification for macro summaries.
- For repeated random seeds, average within question/group or use a hierarchical analysis; do not pretend each seed is a new independent question.
- State common-pair counts, missingness, original-correct counts, category supports, and independent group counts beside results.
- Use one declared strongest-baseline comparison and one neural-versus-heuristic comparison as primary tests. Treat other dataset/method analyses as secondary/exploratory, with an appropriate correction if claiming many significant effects.
- Do not infer equivalence or model independence from similar point estimates. A "near-perfect transfer" claim would require a predefined equivalence margin and adequate paired evidence, and still would not prove universal criticality.
- Do not interpret non-significance on a small subset as proof that methods are the same. Give the estimated difference and interval.

## 8. What the next coding turn should deliver

Keep working stage entry points where possible, but move shared logic into importable modules so importing utilities does not parse CLI flags, create output files, or initialize an API run. Do not perform a broad aesthetic rewrite while the data lineage is still uncertain.

### 8.1 Implementation map

Paths marked **new/proposed** do not exist yet. Final names may be adjusted during implementation; `RUNBOOK.md` must use the names that actually ship.

| Area | Existing code to repair | New/proposed pieces | Acceptance condition |
| --- | --- | --- | --- |
| Config/environment | `config.py`, `.env.example`, `requirements.txt` | Structured experiment configs; lightweight/GPU/optional-baseline dependency groups | Every run resolves configuration, validates paths/settings, records package/model revisions, and avoids printing secrets. |
| Legacy import and audit | `04_build_consensus.py` | `src/data/import_legacy.py`, `src/data/validate.py` | Explicit inputs; conflict report; canonical IDs; no overwrite of legacy sources; typed eligibility and failures. |
| Datasets and grouping | `01_load_data.py` | Dataset revision manifest; base-family reconstruction/splitting utility | Exact source IDs, clean GSM-IC pairing, distractor positions, deduplication, and zero group overlap across partitions. |
| Span extraction | `02_extract_spans.py`, compressor extractor | Shared `src/spans/extract.py` | Identical deployed span rules; stable offsets; category memberships; valid control diagnostics. |
| Answer parsing | `src/utils/answer_matching.py` | Task-specific parse records and regrading command | Unique final answers, explicit invalid statuses, documented numeric tolerance, adversarial edge cases covered. |
| Training | `src/scorer/train_scorer.py` | Feature baselines; calibration/labeler holdout reports | Real soft targets, group-safe splits, matched baselines, full checkpoint metadata, no test tuning. |
| Compression | `src/compressor/compress.py` | Shared reconstruction/budget logic; grouped selector; component toggles | Exact final counting; retained-range maps; no label/answer leakage; baseline-only execution works. |
| Baselines | `BaselineRunner` | Separate official adapters | Supported checkpoints, versioned settings, no silent substitute, per-example failure/status records. |
| Target generation | `src/utils/api_clients.py`, `evaluate.py` | `src/inference/local_hf.py`, hosted-compatible adapter | Same response schema; full raw outputs; provider-wide limits; target version and token usage captured. |
| Metrics/statistics | `metrics.py`, `evaluate.py`, notebook | Occurrence alignment; clustered comparisons; report generator | Common cohorts, robust joins, support/missingness counts, uncertainty, honest metric labels. |
| Orchestration | Current stage CLIs | Preflight/dry-run and manifest-based stage runner | Prints planned tasks/resources/cost envelope, can resume completed stages, isolates configuration changes. |
| Documentation/handover | `README.md`, reference findings | `RUNBOOK.md`; generated tables/figures and claims ledger | Teammates can follow concrete commands and return the expected artifacts. No reference PDFs/PPTX silently rewritten. |

### 8.2 Data and run contracts

Use separate normalized tables rather than repeating a full prompt and all model responses once per consensus annotation:

| Artifact | Essential fields |
| --- | --- |
| Problems | `problem_id`, source/subtask/revision/original source ID, `base_group_id`, exact question, question hash, gold answer/type; clean question and distractor offsets as evaluation metadata |
| Span occurrences | `span_id`, problem ID/hash, start/end, exact text, category memberships, extraction version |
| Model responses | query hash, original/mask/delete/compressed condition, model/provider/revision, prompt-template hash, decoding, full response, parsed answer, parse/finish/error status, usage, timestamps |
| Per-model interventions | span/model identity, original eligibility, perturbed correctness, valid criticality vote, status and provenance |
| Consensus | span ID, eligible/observed labeler counts, critical votes, soft fraction, majority label, missingness/protocol version |
| Splits | base group, role/fold, seed, source-manifest hash, independent group counts |
| Compression | run/problem/method/config/checkpoint hashes, requested budget, exact output, source retention mapping, actual tokens, structural flags, timing/status |
| Evaluation | generation/query identity, compression identity or full-input reference, target identity, full response, parsed answer and correctness, usage/error status |

Legacy CSV exports can remain for compatibility. Distinguish the model-response table from the per-category annotation table so overlapping annotations cannot manufacture extra votes.

Proposed generated artifact layout:

```text
data/legacy/<owner-or-export-id>/     immutable team inputs, checksummed
data/processed/<dataset-version>/    normalized problems, spans, votes, consensus
runs/<run-id>/manifest.json          source/config/data/split/model identity
runs/<run-id>/audit/                 import conflicts, missingness, alignment coverage
runs/<run-id>/scorer/                checkpoints, feature baselines, metrics
runs/<run-id>/compression/           outputs, retained ranges, timings, failures
runs/<run-id>/evaluation/            complete response records and grades
runs/<run-id>/reports/               tables, figures, statistics, claims ledger
```

Large data and secrets should remain excluded from Git. Provide a reproducible handover archive/manifest and a clear external artifact location rather than pretending ignored local files will travel with a clone. Use new run IDs when changing scientific configuration; resume only identical runs. Decide explicitly whether a lightweight example dataset and tiny fixtures belong in version control.

### 8.3 Meaningful lightweight verification

During implementation, run checks that exercise research failure modes without downloading model weights or making real API calls:

- Reject the empty-choice, numeric-list, fraction, incompatible-option, and yes/no ambiguity cases; accept documented legitimate formats.
- Merge duplicate worker rows without extra votes; fail on conflicting prompt hashes or criticality outcomes; preserve missing/failed statuses.
- Demonstrate that all variants of a fabricated clean base stay in one partition and that labeler-holdout targets exclude its votes.
- Show that intermediate consensus fractions survive training-table construction and that tie rules agree.
- Verify repeated `5` occurrences, partial multiword spans, subword-fragment outputs, ambiguous alignment, and distractor token-mass retention.
- Exercise exact final token budgets, protection infeasibility, original-order/newline preservation, and method-independent random seeding using a deterministic tokenizer fixture.
- Use fabricated target responses to test resume keys, interruption recovery, bounded retries, shared full-input references, and failure accounting.
- Check paired/clustered reporting on small hand-computable fixtures, including duplicate-key rejection and incomplete pairs.
- Run a full tiny fixture pipeline with a mocked scorer/target, then syntax/import/CLI preflight checks. Clearly label fixture outputs as synthetic and keep them out of research tables.

These tests are justified because a small error can invalidate the experiment. Passing them does not certify GPU compatibility or the scientific hypothesis. Server smoke tests remain a separate runbook gate.

## 9. Team handover: obtain these before replacing old results

This is an information checklist for your team; no messages have been sent to teammates.

1. All four workers' label shards, original-answer caches, error logs, and the exact candidate/problem CSV used for each collection. Retain even incomplete shards.
2. The teammate's merged per-model and consensus files, including the version reported as 924 questions/14,162 spans.
3. Exact training script/notebook, source commit if known, environment/package versions, group/fold assignments, scorer report, OOF predictions, tokenizer files, and every checkpoint used for evaluated compression.
4. Original hybrid/signal/scorer/random compressed prompts, external-baseline prompts, Kaggle notebook/configurations, actual baseline checkpoint names, and token-counting configuration.
5. Full target responses and uncompressed references for every evaluated method, plus any cached raw API JSON, served model IDs, generation settings, dates, and failure records.
6. The result-generation script or notebook that produced each Markdown table. `final_results_old.md` and `final_results_by_dataset.md` may represent different code/cohorts; do not merge their values by hand.
7. Tomorrow's additional API/server details: exact models, authorized available request/token budget, expiration/session limits, rate limits, supported endpoint, and GPU/VRAM availability. Record credential names/references, never actual secrets in manifests.

Recovering these artifacts may save more time than any new model experiment. If an artifact cannot be recovered, mark the corresponding old result as unverified or rerun only the affected frozen condition.

## 10. Nine-day execution schedule

This is a relative schedule, not a confirmed evaluation date. Teammates can divide tasks concurrently, but all branches must share versioned data/schema/configuration and converge before the main run.

| Day | Work and suggested ownership | Concrete exit gate |
| --- | --- | --- |
| 1 | You coordinate artifact handover; code owner implements schemas, answer parser, importer and dry-run. Another teammate inventories Kaggle/server/API access. | Immutable input manifest; conflicts/missingness identified; no heavy rerun yet. |
| 2 | Data owner regrades and validates full merged labels, reconstructs base families, freezes partitions. Runtime owner confirms additional quota; code owner fixes training/extraction/token counting. | Group overlap is zero; label counts explainable; target/backend and primary scope chosen. |
| 3 | Training owner runs cheap priors/feature baselines and one corrected DistilBERT fit. Baseline owner validates official compressors on tiny examples. | Soft labels confirmed; scorer outputs trace to checkpoints; budgets/answer formats pass smoke checks. |
| 4 | Compression owner implements/finishes grouping; team runs validation pilot, component ablations, runtime/target adequacy checks. | Freeze algorithm settings, seven main compressed methods, two budgets, cohort, and primary comparisons. Stop method expansion. |
| 5-6 | Runtime owner executes main compression and downstream sweep, with incremental exports. Separate teammate handles bounded E7 reliability check. | Paired coverage/status report, intact full responses, reproducible outputs; rerun only failed tasks. |
| 7 | Analysis owner generates clustered CIs, per-subtask tables, criticality/structure diagnostics, and measured runtime/token plots. | Every plotted value traces to run artifacts; no unsupported significance or transfer claims. |
| 8 | Team rewrites report/slides/README, chooses representative successes and failures, prepares limitations and demo. Stretch E8/second target only if core is already complete. | Consistent model/method/count descriptions, limitations, citations, and claims ledger. |
| 9 | Buffer for broken exports, targeted reruns, reproducibility smoke check, presentation rehearsal. | Final archive and runbook work from a clean server environment; no new algorithm hunting. |

Suggested four-person division: **data/provenance**, **training + grouped compression**, **official baselines + runtime**, and **statistics + report/integration**. These are proposed responsibilities, not assumptions about current ownership. You can coordinate code handoff without running the GPU work locally.

### 10.1 Scope cuts if access or time shrinks

Cut in this order:

1. Extra datasets, multilingual experiments, human studies beyond a small grading spot-check, full logical parsing, reinforcement learning, new encoders, or a larger counterfactual collection.
2. Adaptive compression policy, extra targets, 0.33 budget, broad hyperparameter searches, and extensive seed/fold sweeps.
3. Full 0.90/0.25 downstream evaluation; keep these as small diagnostics if useful.
4. Main cohort size, reduced uniformly across competitors before inspecting test outcomes; acknowledge wider uncertainty.

Keep grading, family-safe splits, official baseline honesty, exact budget accounting, full outputs, and a paired downstream comparison. If no usable target inference is possible at all, deliver a narrower annotation/measurement study and state that RQ5 remains unanswered; intrinsic similarity cannot replace it.

## 11. Replace the unsupported claims explicitly

| Existing claim or description | Problem | Safer wording/action |
| --- | --- | --- |
| "First direct evidence" / first task-aware or counterfactual compressor | Closely related prior work exists; current study is narrow | State the specific empirical setting and contribution; cite the literature above. |
| "Perfect generalization" / "criticality is model-agnostic" | Similar accuracies are not equivalence or label-transfer proof | "We test transfer across specified models; observed agreement/performance is conditional on this benchmark and protocol." |
| 43.3% versus 43.5% proves the neural scorer transfers | Those GSM8K entries are `ours_signal`, not the hybrid; that variant does not use neural criticality in its ranking | Correct attribution. Use paired, controlled scorer and held-out-labeler experiments for a learned-transfer claim. |
| "Near-zero accuracy loss" | Full prompt is around 96-98% in the summaries while compressed math scores are around 30-43% | Distinguish difference between readers from the large full-to-compressed accuracy loss. |
| "Hybrid beats baselines at every ratio" | The old table has scorer above hybrid at 0.33; external downstream baseline outputs are missing | Generate claims from exact paired tables; restrict any win to the measured comparator/cohort/budget. |
| "Hybrid is best on GSM-IC for DeepSeek" | The provided table ties hybrid and signal at 35.9% | Report the tie and uncertainty; no universal benefit of neural scoring follows. |
| "50% is the safe/optimal floor" | Only a few aggressive budgets were studied, with large accuracy losses | "These tested aggressive budgets caused substantial losses; moderate budgets require evaluation." |
| "100% of distractor tokens discarded" | Whole-sentence substring absence was used, and local extraction misses five annotations | Recompute actual distractor mass retention with coverage. |
| "DeBERTa-v3, 64-token local window, alpha=0.6 tuned" | Current source uses full-question DistilBERT span pooling, default alpha=0.5 | Describe the checkpoint/configuration that actually generated each result. No tuning claim without a validation record. |
| "TF-IDF + position + category weights" | Current signal uses causal-LM surprisal ranks and shared bonuses | Rewrite to match code; separately name any new feature-prior baseline. |
| "Soft consensus training" | Current loader misses the actual soft column | Retrain with repaired targets or label historical training as hard-target/unknown, according to recovered evidence. |
| "Neutral mask avoids grammatical artifacts" | A placeholder can introduce its own artifacts and different recoverability cues | "Masking is our chosen intervention; we measure agreement with deletion." |
| "BBH is multiple choice with roughly four options; random is chance" | Local tasks include numeric object counting, yes/no navigation, valid/invalid fallacies, three-option and six-option tasks | Report each answer format/subtask; no single BBH chance rate. Random deletion is not random guessing. |
| "Selective Context baseline" | Current implementation is a homegrown word-surprisal method | Rename legacy baseline; run official code separately. |
| "LLMLingua fails on short prompts" | Budget miss observed under one implementation/settings | "This recorded configuration missed the requested budgets; we investigate the adapter/settings before a broader conclusion." |
| "Uncompressed upper bound / random lower bound" | Neither is a formal bound | Use full-input reference and random sanity baseline. |
| "SOTA baselines assume universal importance" | This is not demonstrated by their objectives, and LLMLingua-2 already studies cross-model generalization | Describe each method's actual supervision/objective and the narrower question studied here. |

Correct the bibliography as well as the claims. The draft attributes Provence to Xu et al.; the [primary paper](https://arxiv.org/abs/2501.16214) is by Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and Stephane Clinchant, and describes sequence labeling with reranking rather than the draft's cross-attention account. The LLMLingua reference also needs its author metadata checked against the linked ACL record. Replace hand-entered references with verified BibTeX and ensure the text describes the cited method accurately.

Also reconcile the PPTX speaker notes: notes on later slides still say baseline runs are pending while visible slides contain intrinsic baseline comparisons. The final talk must distinguish completed intrinsic work from pending downstream work.

The reported **12,012 prompts** in the older summary equals `924 * (1 + 4 * 3)`: 924 questions, a full-input reference, four internal/random methods, and three ratios. This arithmetic is consistent with that experiment layout, but it is not 12,012 independent problems and does not imply external baselines were tested. Verify the actual response table before reusing the total.

## 12. How to explain success, failure, or mixed results

| Final observed outcome | Defensible interpretation | Avoid |
| --- | --- | --- |
| Grouped hybrid beats LLMLingua-2 at matched budgets with a clear paired difference | A limited improvement in the specified short-question setting; show budgets, intervals, hardware and exact cohort | Global superiority, benchmark-wide SOTA, or universal transfer. |
| Grouping helps, but non-neural grouping matches/beats neural grouping | Structural preservation explains the gain; supervision did not add a demonstrated benefit | Crediting the counterfactual-trained scorer for a heuristic gain. |
| Scorer predicts held-out criticality, but downstream accuracy stays poor | The single-span assay is predictive of its label yet insufficient for joint compression; test structure/interactions and consider mask mismatch | Calling label AUC a downstream performance improvement. |
| All methods degrade at 0.50, moderate budgets are useful | Dense question compression has a difficult trade-off in this setting; report the measured curve | Claiming a universal information-theoretic compression limit. |
| External baseline wins | Report the negative result, comparative costs, and which parts of the signal fail; preserve dataset/reliability contributions | Inventing a metric after the fact on which ours wins and calling it overall superiority. |
| Prior gains disappear after grading/split repairs | Retire the earlier claim and explain which protocol change affected it | Keeping old inflated headline numbers beside corrected tables. |
| Mask/deletion or cross-labeler agreement is weak | Criticality is intervention/model dependent under this protocol; qualify deployment claims | Describing these noisy labels as objective ground truth. |
| No statistically resolved difference on a small test set | Inconclusive at the available precision; show difference and interval | Proving equivalence or absence of an effect. |

A "good reason" for losing must be tested, not invented. Candidate reasons include structural loss, narrow candidate coverage, mask/deletion mismatch, source-model dependence, noisy grading, and short-input compressor overhead. The proposed ablations distinguish several of these. Others may remain limitations.

## 13. Final submission artifacts and definition of done

Required final outputs:

1. Versioned data/response manifest, clean normalized labels, group partitions, coverage/missingness/conflict audit, and documented legacy-to-current changes.
2. A working scorer and compression pipeline with official baseline adapters, local/hosted target options, safe resumability, and lightweight regression coverage.
3. One paired main downstream table at the declared budgets, full-input references, per-dataset/subtask results, actual token ratios, sample/group counts, and clustered intervals.
4. One controlled neural-versus-non-neural/structure ablation and a labeler-agreement analysis.
5. Occurrence-aware retention/distractor diagnostics; mask/deletion check if accessible, otherwise an explicit open limitation.
6. A small set of figures: accuracy versus actual tokens, category/labeler agreement, retention versus correctness, and measured overhead/cost. Save underlying numeric tables with each figure.
7. Qualitative examples selected using declared criteria, including failures and baseline wins, with intact originals, compressed text, retained ranges, and target answers. Do not present one favorable sample as evidence of a general property.
8. Corrected final report, slides and README, with a claims ledger linking every factual result to a run/table. Update reference documents only when explicitly included in the later writing scope; preserve the originals.
9. `RUNBOOK.md` and an exported artifact archive with checksums. No credentials or machine-specific secret paths.

The next runbook should give the **actual** command order, environment setup for Kaggle/server, resource preflight, how to import teammate data, exact expected output paths, stage-by-stage checks, resume/retry commands, how to export artifacts, and which result files to bring back for interpretation. It must include decision branches for weak target accuracy, missing labels, unavailable quota, budget misses, failed baselines, and findings that contradict the hypothesis. Do not publish speculative commands before their CLI implementation exists.

**Ready to write final claims when:** the pipeline can be traced from immutable inputs to tables; method/cohort/budget/model identities are explicit; grading and family splits pass checks; missingness and uncertainty are visible; and every conclusion remains true if the strongest baseline wins.
