"""
STEP 2: For each problem, find candidate spans in our target categories,
plus length-matched random control spans. Pure rule-based tagging with
spaCy -- no model API calls yet, so this step is fast and free to re-run.

Fixes a real bug from the first draft: multi-word phrases like "at least"
or "at most" were being matched with `token.text in a_set_of_strings`,
which can never succeed since spaCy tokenizes "at least" into two separate
tokens ("at", "least"), neither of which equals the two-word string. This
version uses spaCy's PhraseMatcher, which matches multi-token phrases
correctly.

Each span also gets a `span_granularity` ("token" / "phrase" / "clause"),
so later analysis can ask whether longer critical spans are harder for
existing compressors to preserve than single critical tokens.

First time only:
    pip install spacy && python -m spacy download en_core_web_sm

Run: python src/dataset_creation/02_extract_spans.py
Output: data/02_candidate_spans.csv
"""

import os
import sys
import random

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

nlp = spacy.load("en_core_web_sm")
random.seed(42)

# ---------------------------------------------------------------------------
# Category vocabularies
# ---------------------------------------------------------------------------
NEGATION_PHRASES = ["not", "no", "never", "n't", "none", "without", "neither", "nor", "no longer"]
COMPARISON_CONDITIONAL_PHRASES = [
    "if", "unless", "than", "more than", "less than", "at least", "at most",
    "greater than", "fewer than", "equal to", "before", "after", "until",
]
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT"}

_negation_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
_negation_matcher.add("NEGATION", [nlp.make_doc(p) for p in NEGATION_PHRASES])

_comparison_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
_comparison_matcher.add("COMPARISON_CONDITIONAL", [nlp.make_doc(p) for p in COMPARISON_CONDITIONAL_PHRASES])


def _granularity_for_length(n_tokens: int) -> str:
    """Token count -> granularity label. 1 token = token, 2-4 = phrase, 5+ = clause."""
    if n_tokens <= 1:
        return "token"
    if n_tokens <= 4:
        return "phrase"
    return "clause"


# def _matcher_spans(doc, matcher, category: str) -> list[tuple]:
#     """Run a PhraseMatcher over `doc` and return (text, start, end, category, granularity) tuples."""
#     results = []
#     for _, start, end in matcher(doc):
#         span = doc[start:end]
#         results.append((span.text, span.start_char, span.end_char, category, _granularity_for_length(end - start)))
#     return results

from spacy.util import filter_spans

def _matcher_spans(doc, matcher, category):
    spans = [doc[start:end] for _, start, end in matcher(doc)]
    spans = filter_spans(spans)  # drops "than" when "more than" already covers it
    return [(s.text, s.start_char, s.end_char, category, _granularity_for_length(len(s))) for s in spans]



def _number_spans(doc) -> list[tuple]:
    """
    Numeric tokens, extended to include an immediately following unit noun
    when there is one (e.g. "30 pens" rather than just "30") so the number
    category isn't artificially all single-token.
    """
    results = []
    for tok in doc:
        if tok.pos_ == "NUM" or tok.like_num:
            end_tok = tok
            # Extend to the next token if it's a noun directly attached to this number (e.g. "pens" in "30 pens").
            if tok.i + 1 < len(doc):
                nxt = doc[tok.i + 1]
                if nxt.pos_ in {"NOUN", "PROPN"} and (nxt.head == tok or tok.head == nxt):
                    end_tok = nxt
            span = doc[tok.i:end_tok.i + 1]
            results.append((span.text, span.start_char, span.end_char, "number", _granularity_for_length(len(span))))
    return results


def _entity_spans(doc) -> list[tuple]:
    """Named entities (people, orgs, places, products) via spaCy's NER."""
    results = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            n_tokens = ent.end - ent.start
            results.append((ent.text, ent.start_char, ent.end_char, "entity", _granularity_for_length(n_tokens)))
    return results


def _distractor_span(original_prompt: str, inserted_context: str) -> list[tuple]:
    """
    For GSM-IC problems, we already know exactly which sentence was
    inserted (computed in 01_load_data.py by diffing the original and
    GSM-IC versions of the question) -- no need to guess it with a parser.
    Tagged at clause granularity since it's a full inserted sentence.
    """
    if not isinstance(inserted_context, str) or not inserted_context.strip():
        return []
    start = original_prompt.find(inserted_context)
    if start == -1:
        return []  # couldn't locate it verbatim -- skip rather than guess
    end = start + len(inserted_context)
    return [(inserted_context, start, end, "distractor_context", "clause")]


def _random_control_spans(doc, used_ranges: set, n_controls: int = 3) -> list[tuple]:
    """
    Random-length (1-3 token) spans not overlapping any already-selected
    span, to serve as a control: if random spans turn out to be "critical"
    about as often as our curated categories, that's an important (if
    unflattering) finding about whether those categories are special.
    """
    candidates = []
    for start in range(len(doc)):
        for length in (1, 2, 3):
            end = start + length
            if end > len(doc):
                continue
            span = doc[start:end]
            if (span.start_char, span.end_char) in used_ranges:
                continue
            if not span.text.strip() or not any(t.is_alpha for t in span):
                continue
            candidates.append(span)

    if not candidates:
        return []
    chosen = random.sample(candidates, min(n_controls, len(candidates)))
    return [
        (s.text, s.start_char, s.end_char, "random_control", _granularity_for_length(len(s)))
        for s in chosen
    ]


def find_spans(text: str, inserted_context: str = "") -> list[tuple]:
    """Returns a list of (span_text, start_char, end_char, category, granularity)."""
    doc = nlp(text)

    spans = []
    spans += _matcher_spans(doc, _negation_matcher, "negation")
    spans += _matcher_spans(doc, _comparison_matcher, "comparison_conditional")
    spans += _number_spans(doc)
    spans += _entity_spans(doc)
    spans += _distractor_span(text, inserted_context)

    used_ranges = {(s, e) for _, s, e, _, _ in spans}
    spans += _random_control_spans(doc, used_ranges)

    return spans


def main():
    df = pd.read_csv(config.RAW_PROBLEMS_FILE).fillna("")
    rows = []

    for _, row in df.iterrows():
        inserted = row.get("inserted_context", "") or ""
        spans = find_spans(row["original_prompt"], inserted_context=inserted)

        for span_text, start, end, category, granularity in spans:
            # Neutral placeholder instead of straight deletion ==> deleting a span outright can leave grammatically broken text, and then a
            # wrong answer might just be the model getting confused by bad grammar rather than actually losing critical information.
            perturbed = row["original_prompt"][:start] + "[MASKED]" + row["original_prompt"][end:]
            rows.append({
                "example_id": row["example_id"],
                "source_dataset": row["source_dataset"],
                "original_prompt": row["original_prompt"],
                "ground_truth_answer": row["ground_truth_answer"],
                "span_text": span_text,
                "span_category": category,
                "span_granularity": granularity,
                "span_start": start,
                "span_end": end,
                "perturbed_prompt": perturbed,
            })

    out = pd.DataFrame(rows)
    out.to_csv(config.SPANS_FILE, index=False)
    print(f"Saved {len(out)} candidate spans across {out['example_id'].nunique()} problems")
    print("\nBy category:")
    print(out["span_category"].value_counts())
    print("\nBy granularity:")
    print(out["span_granularity"].value_counts())


if __name__ == "__main__":
    main()
