"""
Prepare the input CSV for Step 5 (mask vs deletion assay).

Reads original_prompt + ground_truth_answer from compressed.csv,
extracts candidate spans using the shared find_spans extractor,
and writes a spans CSV that deletion-check can consume.

Usage:
    cd /path/to/ANLP_Project_PromptCompression
    python prepare_step5_spans.py \
        --compressed compressed.csv \
        --output results/followup_v2/step5/spans_input.csv \
        [--dry-run]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compressed", default="compressed.csv")
    ap.add_argument("--output", default="results/followup_v2/step5/spans_input.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.compressed} ...")
    df = pd.read_csv(args.compressed, keep_default_na=False,
                     dtype={"ground_truth_answer": str})

    required = {"example_id", "source_dataset", "original_prompt", "ground_truth_answer"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: compressed.csv is missing columns: {missing}")

    # Deduplicate: one row per unique question (ignore method/ratio duplicates)
    questions = df[["example_id", "source_dataset", "original_prompt",
                    "ground_truth_answer"]].drop_duplicates("example_id").reset_index(drop=True)
    print(f"Unique questions found: {len(questions)}")
    print(f"Dataset breakdown:\n{questions['source_dataset'].value_counts().to_string()}")

    print("\nLoading spaCy model (en_core_web_sm) ...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        sys.exit("ERROR: Run: python -m spacy download en_core_web_sm")

    from final_eval.spans.extract import find_spans

    records = []
    skipped = 0

    for _, row in questions.iterrows():
        text = row["original_prompt"]
        if not text or not text.strip():
            skipped += 1
            continue
        try:
            spans = list(find_spans(text, nlp=nlp, include_controls=False))
        except Exception as e:
            print(f"  WARN: {row['example_id']}: {e}")
            skipped += 1
            continue

        for span_text, span_start, span_end, span_category, _ in spans:
            if text[span_start:span_end] != span_text:
                continue  # bad offset, skip
            records.append({
                "example_id": row["example_id"],
                "source_dataset": row["source_dataset"],
                "original_prompt": text,
                "ground_truth_answer": row["ground_truth_answer"],
                "span_text": span_text,
                "span_start": span_start,
                "span_end": span_end,
                "span_category": span_category,
            })

    spans_df = pd.DataFrame(records)
    print(f"\nExtracted {len(spans_df)} spans from {len(questions) - skipped} questions.")
    print(f"Category breakdown:\n{spans_df['span_category'].value_counts().to_string()}")

    if args.dry_run:
        print("\n[DRY RUN] No output written.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    spans_df.to_csv(out, index=False)
    print(f"\nWritten: {out}  ({len(spans_df)} rows)")
    print(f"\n--- Next command (dry-run first) ---")
    print(f"python -m final_eval.compressor.evaluate \\")
    print(f"  --mode deletion-check \\")
    print(f"  --input-csv {out} \\")
    print(f"  --output-csv results/followup_v2/step5/deletion_check_results.csv \\")
    print(f"  --model-config final_eval/configs/target.qwen.step5.json \\")
    print(f"  --n 120 --seed 42 --dry-run")


if __name__ == "__main__":
    main()
