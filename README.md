# Symbiote: reasoning-critical prompt compression

[**Final model**](https://huggingface.co/Pranav-Sriv/symbiote-v3-prompt-compressor) | [Results](RESULTS.md) | [Reproduction artifacts](results/reproducibility.json) | [Code index](docs/CODE_MAP.csv)

Earlier models: [original scorer](https://huggingface.co/flashstep/reasoning-critical-span-scorer) | [corrected v2 scorer](https://huggingface.co/Pranav-Sriv/reasoning-critical-span-scorer-v2)

Symbiote studies whether counterfactual span criticality helps compress reasoning
questions. We mask spans, collect votes from readers that answered the original
question correctly, and train a DistilBERT scorer. The final v3 compressor
combines LLMLingua-2 retention supervision with an auxiliary criticality loss.
Compression runs locally without reader or API calls.

## Results

Final Qwen evaluation: 64 held-out families, one question per family. Full
questions scored 59/64. Ratios are ceilings on retained question tokens,
measured with `cl100k_base`.

| Method | 50% retention | 75% retention |
| --- | ---: | ---: |
| V3 combined student | 27/64 (42.19%) | 55/64 (85.94%) |
| Teacher-only control | 24/64 (37.50%) | 53/64 (82.81%) |
| V2 grouped hybrid | 17/64 (26.56%) | 49/64 (76.56%) |
| LLMLingua-2 | 31/64 (48.44%) | 53/64 (82.81%) |

At 50% retention, the primary combined-minus-control difference is +4.69 percentage points,
95% paired interval [-4.69, +15.63]. Span labels are predictable, but their added
compression benefit is not established. Actual retention differs between
methods; overall superiority to LLMLingua-2 is not established.
[RESULTS.md](RESULTS.md) records the studies, ablations, figures and limitations.

## Repository layout

```text
src/                   Original dataset collection, scorer and compressor
final_eval/
  preparation/         Label recovery, grading and family grouping
  scorer/              Corrected v2 span-scorer training
  spans/, compressor/  Extraction, token budgets and baseline compression
  inference/, utils/   Reader clients, answer grading and paired statistics
  analysis/            Analyses of saved labels, predictions and responses
  v3/                  Final student: data, model, training and evaluation
notebooks/             Original analysis and v2/v3 Kaggle workflows (01-05)
environments/          Recorded dependency versions
results/               Tables, plots, metadata and artifact checksums
docs/CODE_MAP.csv      File-by-file code index
RESULTS.md             Methods, results and interpretation
```

The original workflow remains in `src/`; v2 and v3 preserve the experiments
compared in the report. Scorer, signal and hybrid are v1/v2 compression
configurations. V3 teacher-only and combined are separate trained checkpoints.

## Use the model

The [Hugging Face model card](https://huggingface.co/Pranav-Sriv/symbiote-v3-prompt-compressor#use)
contains standalone installation and inference instructions. The custom loader
at the [pinned revision](https://huggingface.co/Pranav-Sriv/symbiote-v3-prompt-compressor/tree/6cd33f01c38910bbd1688a07f958d8bb801a3029)
reproduces 192/192 saved final-policy compressions.

With the completed run extracted locally, repository inference is:

```bash
python -m pip install -r final_eval/v3/requirements.txt
python -m final_eval.v3 infer --run-dir outputs/imports/v3_20261030_completed/run --ratio 0.75 --text "Alice has 8 apples and gives away 3. How many apples remain?"
```

## Reproduce the experiments

- **Score replay:** with the original return ZIP and status file, run
  `python -m final_eval.v3.audit_return --archive outputs/imports/v3_20261030_completed/symbiote_v3_results.zip --status outputs/imports/v3_20261030_completed/v3_status.json`.
  This checks saved artifacts, token budgets, grades and the primary interval
  without training or paid calls.
- **Training/evaluation:** use notebook 05 with its matching frozen Kaggle input.
  GPU dependencies are in [kaggle-requirements.txt](final_eval/v3/kaggle-requirements.txt).
  New hosted evaluation requires your own API access; score replay does not.
- **Earlier experiments:** notebooks 01-04 and the corresponding `final_eval/`
  modules cover label analysis, v2 and checkpoint/component follow-ups.

`data/`, `models/`, `outputs/`, `.env` and virtual environments are excluded from
Git. A clone contains code and compact results, not training data, weights or raw
responses. [The artifact index](results/reproducibility.json) records their local
locations and hashes. Keep the original experiment archives when transferring
machines. Models are linked above; exact score replay uses saved answers because
new hosted generations can differ.
