"""
STEP 4: Aggregate per-model labels (one row per span per model) into a
single consensus criticality score per span.

One addition over the first draft: not every span gets labeled by all 4
models -- a model that already got the original question wrong is skipped
entirely for that problem (see counterfactual_labeling.py). So some spans
might end up with only 1 valid model label, and treating that the same as
a span backed by all 4 models would overstate how reliable it is. We flag
(and by default drop) spans below config.MIN_MODELS_FOR_CONSENSUS.

Run: python src/dataset_creation/04_build_consensus.py
Output: data/04_consensus_dataset.csv
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

GROUP_KEYS = [
    "example_id", "source_dataset", "span_text", "span_start", "span_end",
    "span_category", "span_granularity",
]


def main():
    df = pd.read_csv(config.LABELS_FILE)

    agg = df.groupby(GROUP_KEYS).agg(
        n_models_labeled=("model_specific_label", "count"),
        n_models_critical=("model_specific_label", lambda x: (x == "critical").sum()),
    ).reset_index()

    agg["agreement_score"] = agg["n_models_critical"] / agg["n_models_labeled"]
    # Tie-break rule, stated explicitly: exactly 0.5 agreement counts as
    # non_critical. This is a conservative choice -- since the compressor
    # will later "always keep" high-consensus spans regardless of budget,
    # we'd rather under-protect an ambiguous span than over-protect one,
    # keeping the "always keep" rule meaningful for genuinely high-agreement
    # spans only.
    agg["consensus_label"] = agg["agreement_score"].apply(lambda x: "critical" if x > 0.5 else "non_critical")

    n_before = len(agg)
    reliable = agg[agg["n_models_labeled"] >= config.MIN_MODELS_FOR_CONSENSUS].copy()
    n_dropped = n_before - len(reliable)
    if n_dropped:
        print(f"Dropping {n_dropped}/{n_before} spans labeled by fewer than "
              f"{config.MIN_MODELS_FOR_CONSENSUS} models (insufficient signal)")

    final = df.merge(
        reliable[GROUP_KEYS + ["consensus_label", "agreement_score", "n_models_labeled"]],
        on=GROUP_KEYS, how="inner",  # inner join drops the rows we just filtered out
    )
    final.to_csv(config.CONSENSUS_FILE, index=False)

    print(f"\nSaved {len(final)} rows ({reliable.shape[0]} unique reliable spans) to {config.CONSENSUS_FILE}")
    print("\nConsensus label distribution (unique spans):")
    print(reliable["consensus_label"].value_counts())
    print("\nAgreement score distribution (unique spans):")
    print(reliable["agreement_score"].describe())
    print("\nConsensus label by category (unique spans):")
    print(reliable.groupby("span_category")["consensus_label"].value_counts())


if __name__ == "__main__":
    main()
