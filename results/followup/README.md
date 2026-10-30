# Follow-up results (NEXT_STEPS steps 1–3)

All answers come from **`qwen-3.8-27b-fp8`** on Jarvislabs (temperature 0, output cap 8,192,
reasoning effort `low`). GPT-OSS was removed from Jarvislabs, so these numbers must **not** be
compared with the GPT-OSS rows in `RESULTS.md`. Confidence intervals are 95%, family-clustered
bootstrap (`base_group_id`), unadjusted.

`manifest.json` lists each file's source in the gitignored `outputs/followup/` tree and its SHA-256.
Commands to reproduce are in the main README ("Old-checkpoint versus v2 downstream reader
comparison (Qwen)" and "Scorer-only and signal-only ablations").

## `old_vs_v2_qwen/` — historical checkpoint versus v2 (step 2)

739 questions / 736 families (exposure-safe common cohort), `ours` and `grouped_hybrid` at
retention .50/.75, both checkpoints answered in one paired run. Originals: 99.46% correct.

| Method | Ratio | Historical | v2 | Δ v2 − historical [95% CI] |
| --- | --- | --- | --- | --- |
| `ours` | .50 | 41.54% | 41.00% | −0.54 pp [−2.72, +1.77] |
| `ours` | .75 | 71.31% | 71.85% | +0.54 pp [−0.94, +2.04] |
| `grouped_hybrid` | .50 | 42.14% | 41.19% | −0.95 pp [−3.12, +1.36] |
| `grouped_hybrid` | .75 | 71.45% | 71.99% | +0.54 pp [−0.95, +2.04] |

No detectable checkpoint difference. 64–82% of compressed prompts are identical across checkpoints;
`summary.csv` also reports the delta on differing-text pairs only. 6,649/6,651 answers `ok`
(2 truncated, excluded as one pair). Provider-reported cost ≈ ₹393; the cost ledger's higher figure
includes worst-case reservations for a local network outage, and its `limit_history` records the
user-approved cap raise from ₹700 to ₹850.

- `summary.csv`, `report.json`: per method × ratio results.
- `per_question.csv`: paired correctness per question (no prompt or response text).
- `common_cohort.csv` + manifest: the cohort; `compression_intrinsic_*.csv`: compression diagnostics.
- `pilot_check.json`, `response_manifest.json`, `paired_input_manifest.json`, `cost_ledger.json`: provenance.
- `kaggle_compression_log.txt`: log of `notebooks/04_followup_old_v2_compression_kaggle.ipynb`.

## `components_qwen/` — scorer-only and signal-only ablations (step 3)

Frozen 160 questions / 125 families. `ours`, `ours_scorer`, `ours_signal` at .50/.75, answered
in one matched run with originals (98.75% correct). The fresh `ours` arm reproduces the main
study's saved `ours` text in 320/320 conditions and its Qwen accuracy (46.25% / 71.25% versus
46.25% / 71.875%).

| Ratio | `ours` | `ours_scorer` | `ours_signal` | scorer − ours | signal − ours | scorer − signal |
| --- | --- | --- | --- | --- | --- | --- |
| .50 | 46.25% | 48.75% | 45.62% | +2.50 pp [−6.4, +11.9] | −0.62 pp [−9.6, +8.1] | +3.12 pp [−6.2, +14.1] |
| .75 | 71.25% | 70.62% | 74.38% | −0.62 pp [−10.5, +8.7] | +3.12 pp [−4.9, +11.1] | −3.75 pp [−13.4, +6.6] |

No component is detectably better or worse than the mixture. Per-dataset results
(`per_dataset.csv`, 15–16 questions each) are exploratory: e.g. `ours_scorer` scores 0/16 on
`bbh_object_counting` at both ratios. 1,120/1,120 answers `ok` after one retry pass;
provider-reported cost ≈ ₹138 (+ ₹15 pilot).

- `accuracy.csv`, `contrasts.csv`, `per_dataset.csv`, `report.json`: results.
- `per_question.csv`: outcome per question × method × ratio (no response text).
- `pilot_cohort.csv` + manifest, `pilot_check.json`: the disjoint 16-family pilot.
- `cohort_manifest.json`, `prepared_input_manifest.json`, `response_manifest.json`, `cost_ledger.json`: provenance.

## `summary/` — step 6: all follow-up paired results and failures

Built by `python -m final_eval.followup_summary` from the step 2–3 files above and Abhinav's step 4–5
files in `results/followup_v2/` (inputs and SHA-256 in `summary/manifest.json`). No API calls.

- `paired_results.csv`: every paired contrast from steps 2–5 in one table.
- **Step 4, bonuses on/off** (`bonuses_*.csv`). *Bonuses on* is our complete step-3 run;
  *bonuses off* is Abhinav's no-bonuses run. Abhinav's with-bonuses rerun asked exactly our step-3
  prompts (800/800 identical query hashes), and Qwen gave the same correctness to identical prompts
  in 99.3% of 731 cases (`reader_repeatability.csv`), so pairing the two runs adds little noise.

  | Arm | Ratio | Bonuses on | Bonuses off | off − on [95% CI] | Number words kept on → off |
  | --- | --- | --- | --- | --- | --- |
  | `ours_scorer` | .50 | 49.06% | 43.40% | −5.66 pp [−12.5, +0.7] | 97.8% → 95.3% |
  | `ours_scorer` | .75 | 70.25% | 68.99% | −1.27 pp [−6.3, +3.4] | 100% → 98.6% |
  | `ours_signal` | .50 | 45.62% | 31.25% | **−14.38 pp [−24.0, −4.7]** | 93.1% → 78.4% |
  | `ours_signal` | .75 | 74.38% | 76.88% | +2.50 pp [−3.6, +8.3] | 99.9% → 98.9% |

  Removing bonuses detectably hurts signal-only at .50, the only CI here that excludes zero, and it
  is also where bonuses protect the most numbers. The other three cells are compatible with no effect.
  Caveats: no pilot gate for step 4; the frozen 160 cohort was used instead of separate development
  families; bonuses off was run for `ours_scorer`/`ours_signal` only, not for the α=.5 `ours` mixture.
  Unadjusted CIs.
- **Step 5, mask vs deletion** (`mask_deletion_by_category.csv`): masking and deleting a span gave the
  same outcome for 112/116 eligible spans (96.6%, CI [92.2, 100]); deletion − mask accuracy −1.7 pp
  [−6.1, +1.8]. Every disagreement is in negation (3, all mask-right/delete-wrong) or entity (1). Each
  category has 22–24 spans, so per-category rates are exploratory.
- **Failures.** `failures_request_status.csv` lists every non-`ok` answer: steps 2 and 3 are fully `ok`,
  bonuses off has 3 non-`ok` (2 truncated, 1 budget-exhausted), and Abhinav's with-bonuses rerun has
  68 network errors (used only for the repeatability check). `failures_compression_by_dataset.csv` counts
  questions that the original answered correctly and compression broke, per arm × dataset. At .50,
  `ours` breaks 54% and signal-only without bonuses breaks 69%; GSM-IC is the dataset where bonuses
  matter most (signal-only 44% broken with bonuses, 78% without).

## `runtime/` — step 6: controlled CPU runtime and memory

`python -m final_eval.benchmark_runtime`: one process, Apple M2 CPU (8 GB), 4 torch threads, 5
warm-up questions, then 160 questions × 3 repeats. Each stage is timed on its own. The produced
compressions are identical to the evaluated step-3 texts (960/960), so this times the evaluated pipeline.

| Stage (per question) | Median ms | p90 ms |
| --- | ---: | ---: |
| spaCy span extraction (parser) | 12.3 | 21.3 |
| spaCy second pass (bonuses/groups) | 11.4 | 20.1 |
| DistilBERT span scorer | 19.8 | 26.7 |
| DistilGPT2 surprisal | 43.6 | 59.7 |
| Full preparation, neural (`ours`, `ours_scorer`) | 88.5 | 124.6 |
| Full preparation, signal-only | 68.4 | 98.9 |
| Budget selection (exact tokens) at .50 / .75 | 1.7 / 1.2 | 4.2 / 2.8 |
| **End to end, `ours` at .50** | **90.2** | **129.1** |
| **End to end, `ours_signal` at .50** | **69.9** | **102.9** |

Surprisal is the largest stage. Budget selection costs under 2% of the total. Loading takes about
4.5 s for DistilGPT2 (cold) and 1.2 s per scorer fold. Peak resident memory was 1.16 GB with all
five fold scorers loaded at once; the pipeline itself keeps one fold loaded at a time. LLMLingua
baselines are not installed on this machine, so they are not timed here; their GPU process times
stay in `RESULTS.md` §5 with that section's caveats.
