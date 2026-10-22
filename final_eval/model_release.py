"""Prepare a verified local Hugging Face release; never authenticates or uploads."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

from final_eval.utils.artifacts import atomic_json, claim_output, file_hash, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/span_scorer_v2_20261021"


def extracted_definition(path, name):
    source = Path(path).read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    return ast.get_source_segment(source, node)


def loader_source():
    # Preserve the actual architecture and scoring semantics, without importing
    # training/orchestration modules or downloading a base encoder on load.
    definitions = [extracted_definition(ROOT / "final_eval/scorer/train_scorer.py", name)
                   for name in ("SpanScorer", "span_token_mask")]
    definitions += [extracted_definition(ROOT / "final_eval/compressor/compress.py", "score_spans")]
    header = '''"""Custom scorer loader: local files only; not a Transformers AutoModel artifact."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 512

'''
    helpers = '''

def verify_bundle(directory=None):
    root = Path(directory) if directory is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Release checksum mismatch: {name}")
    return manifest


def fold_for_uid(uid, directory=None):
    root = Path(directory) if directory is not None else Path(__file__).resolve().parent
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    if uid not in split["folds"]:
        raise ValueError("Unknown study UID: choose and disclose a fold/ensemble policy for new inputs")
    return int(split["folds"][uid])


def load_fold(fold, directory=None, device="cpu"):
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("fold must be an integer from 0 to 4")
    root = Path(directory) if directory is not None else Path(__file__).resolve().parent
    folder = root / f"fold{fold}"
    config = DistilBertConfig.from_pretrained(str(folder), local_files_only=True)
    model = SpanScorer(encoder_config=config)
    model.load_state_dict(torch.load(folder / "scorer.pt", map_location="cpu", weights_only=True))
    model = model.to(torch.device(device)).eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    return model, tokenizer


if __name__ == "__main__":
    manifest = verify_bundle()
    print(f"PASS: {len(manifest['files'])} release files verified; no weights loaded or API calls made.")
'''
    return header + "\n\n".join(definitions) + helpers


def model_card():
    return '''---
language:
- en
base_model: distilbert-base-uncased
tags:
- pytorch
- prompt-compression
- span-classification
- counterfactual-analysis
- anlp-course-project
---

# Symbiote critical-span scorer v2 — recorded 21 October 2026 model

Five DistilBERT span-scoring checkpoints from the completed ANLP course study.
This is a custom PyTorch span classifier, not a chat model, token-classification
AutoModel, or standalone prompt compressor. Use the supplied `load_scorer.py`.
No new training was performed to prepare this release.

## Model and data

Each checkpoint includes the entire DistilBERT encoder plus a mean-pooled-span
linear head. The sigmoid predicts the fraction of eligible labelers whose answer
failed after masking a span, conditional on answering the original correctly.
This is not a certificate of semantic necessity or compressed-input sufficiency.

Training used recovered saved responses: 38,603 unique model/span votes,
37,979 eligible; 13,966 consensus positions over 912 questions/778 families.
The supplied team export was already consensus-filtered, so full raw worker
coverage is not established. Five family-disjoint folds, soft targets,
unweighted BCE, four epochs, batch 8, LR 3e-5, max length 512, seed 42.

On 13,855 shared held-out positions: ROC AUC .907385, average precision .713346,
critical F1 .634404, soft-target Brier .060593. The feature ridge AUC was .821711.
111 random-control spans lacked full tokenizer coverage and were excluded from
the shared comparison. See scorer_report.json and evidence/feature_report.json.

## Five folds are part of the result

These are cross-validation models, not five seeds of a full-data deployment model.
Reproduce a recorded question using its held-out assignment in split.json. Do not
choose the best-looking checkpoint on the test set or substitute an ensemble for
the reported held-out-fold predictions. For a genuinely new input, a declared
fold or ensemble is a deployment choice whose performance was not measured here.

## Contents and provenance

- fold0/ through fold4/: scorer.pt, encoder config, category prior and selection metadata.
- tokenizer/: the saved local fast tokenizer.
- split.json, training_metadata.json, scorer_report.json, scorer_oof_predictions.csv.
- load_scorer.py: the original architecture/span-mask/scoring definitions plus local loading helpers.
- source/: exact inference-source snapshot, including the original scorer implementation.
- evidence/: small verified comparison reports and scientific-audit metadata.
- release_manifest.json: every payload file's size/SHA256, training/inference fingerprints.

Model source directory: models/span_scorer_v2_20261021 in the project backup.
Project code, data scope, run history, baseline adaptations, API settings and full
reproduction guide: https://github.com/dhyeythummar09/ANLP_Project_PromptCompression
The inference snapshot includes later baseline/provider corrections; original
training metadata retains its distinct training-time fingerprint. Both are
recorded rather than presenting a Git commit with uncommitted changes as exact source.

## Local usage

Download this repository at a fixed full commit revision. From that downloaded
folder, install requirements-inference.txt in a suitable PyTorch environment.
The measured GPU environment used Python 3.13.15, Torch 2.11.0+cu128 and
Transformers 5.16.1; choose a Torch build appropriate for your actual CPU/CUDA.
No base encoder/tokenizer download is needed when loading a local fold.

```bash
python load_scorer.py
```

This verifies hashes without loading weights. Then, from the downloaded directory:

```python
import torch
from load_scorer import load_fold, score_spans, fold_for_uid

# Arbitrary fold 0 for a NEW-input demonstration, not the reported OOF evaluation.
model, tokenizer = load_fold(0, device="cpu")
text = "Mira has 3 apples."
spans = [{"span_start": 9, "span_end": 10, "span_text": "3", "span_category": "number"}]
scored = score_spans(text, spans, model, tokenizer, torch.device("cpu"))
print(scored[0]["criticality_score"])

# For a study question instead: load_fold(fold_for_uid("source_dataset|example_id"))
```

Offsets are Python character positions, end-exclusive. Call verify_bundle()
before use. Scoring follows the original conservative fallback: spans without
complete token coverage retain score 1.0; do not interpret those as neural predictions.

## Compression outcome and limitations

Good span-label prediction did not establish a neural downstream compression gain.
Grouped hybrid versus budgeted LLMLingua-2 on the full 160-question companion:

| Reader | Retained-token ceiling | Hybrid accuracy | LLMLingua-2 accuracy |
| --- | --- | --- | --- |
| GPT-OSS-120B | .50 | 36.875% | 60.000% |
| GPT-OSS-120B | .75 | 73.125% | 90.000% |
| Qwen-3.8-27B-FP8 | .50 | 46.250% | 69.375% |
| Qwen-3.8-27B-FP8 | .75 | 73.125% | 95.625% |

The full-cohort companion counts provider/parse failures as incorrect and was
added after failure review. Original common-success tables are retained in the
project: Qwen .50 uses 158 questions there. Qwen retains two empty outputs and
its all-responses-successful technical gate remains false. Hybrid/prior ablation
intervals include zero. No SOTA, causal sufficiency, or unseen-reader-family
transfer is claimed. Masking one span differs from joint deletion. Target LLMs,
LLMLingua-2, DistilGPT2 and spaCy are separate components, not included weights.

The optional seven-day follow-up is not part of these weights/results. Preserve
this dated release when publishing any later model. Team Symbiote, IIIT Hyderabad.
'''


def prepare(output_dir):
    target = Path(output_dir).resolve()
    target.relative_to(ROOT)
    claim_output(target)
    verification = json.loads((ROOT / "results/verification.json").read_text(encoding="utf-8"))
    if not verification["passed"] or len(verification["weights"]) != 5:
        raise ValueError("Need the independently verified five-fold model")
    if source_fingerprint() != verification["core_source_fingerprint"]:
        raise ValueError("Core source differs from the verified study; use its frozen release or audit a new version first")
    for name, info in verification["weights"].items():
        if file_hash(ROOT / name) != info["sha256"]:
            raise ValueError(f"Checkpoint differs from verified study: {name}")
    files = {p.relative_to(MODEL).as_posix(): p for p in MODEL.rglob("*") if p.is_file()}
    package = ROOT / "final_eval"
    sources = [package / "__init__.py", package / "settings.py", *package.glob("requirements*.txt")]
    for directory in ("compressor", "inference", "scorer", "spans", "utils", "preparation"):
        sources += sorted((package / directory).rglob("*.py"))
    sources += [ROOT / "requirements.txt", ROOT / "config.py"]
    files.update({"source/" + p.relative_to(ROOT).as_posix(): p for p in sources})
    for name in ("feature_report.json", "preparation.json"):
        files["evidence/" + name] = ROOT / "results/metadata" / name
    files["evidence/study_verification.json"] = ROOT / "results/verification.json"
    files["evidence/full_cohort_accuracy.csv"] = ROOT / "results/tables/full_cohort/accuracy.csv"
    generated = {"README.md": model_card(), "load_scorer.py": loader_source(),
                 "requirements-inference.txt": "# Select a CPU/CUDA Torch build appropriate for the machine.\ntorch>=2.6\ntransformers==5.16.1\nnumpy\n"}
    required = sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(ROOT).free < required + 100_000_000:
        raise ValueError("Insufficient free space for an independent release copy")
    target.mkdir(parents=True)
    for name, source in files.items():
        path = target / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        if file_hash(path) != file_hash(source):
            raise ValueError(f"Copy checksum failed: {name}")
    for name, content in generated.items():
        (target / name).write_text(content, encoding="utf-8", newline="\n")
    payload = {p.relative_to(target).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
               for p in sorted(target.rglob("*")) if p.is_file()}
    training = json.loads((MODEL / "training_metadata.json").read_text(encoding="utf-8"))
    atomic_json(target / "release_manifest.json", {"release": "symbiote-span-scorer-v2-20261021",
        "training_source_fingerprint": training["source"], "inference_source_fingerprint": source_fingerprint(),
        "source_model_directory": MODEL.relative_to(ROOT).as_posix(), "files": payload,
        "original_weights_unchanged": True, "new_training": False, "uploaded": False,
        "scope": "Byte-verified local packaging; real checkpoint forward pass not performed during packaging."})
    print(f"Prepared {len(payload)} verified payload files at {target}; {required / 1e9:.3f} GB. No model run, authentication or upload.")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/hf_release/span_scorer_v2_20261021")
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
