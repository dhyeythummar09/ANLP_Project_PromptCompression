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
