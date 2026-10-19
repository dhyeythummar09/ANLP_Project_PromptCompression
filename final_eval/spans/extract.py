"""Shared versioned extraction, with lazy spaCy loading and deterministic controls."""
import random
import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from final_eval.utils.artifacts import digest

EXTRACTOR_VERSION = "shared-v2"
NEGATION_PHRASES = ["not", "no", "never", "n't", "none", "without", "neither", "nor", "no longer"]
COMPARISON_CONDITIONAL_PHRASES = [
    "if", "unless", "than", "more than", "less than", "at least", "at most",
    "greater than", "fewer than", "equal to", "before", "after", "until",
]
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT"}

def _granularity_for_length(n_tokens: int) -> str:
    """Token count -> granularity label. 1 token = token, 2-4 = phrase, 5+ = clause."""
    if n_tokens <= 1:
        return "token"
    if n_tokens <= 4:
        return "phrase"
    return "clause"


# Runs a spaCy PhraseMatcher over a document and extracts matching spans, automatically filtering out redundant overlapping sub-spans
def _matcher_spans(doc, matcher, category):
    
    # Extract all raw span objects found by the matcher
    spans = [doc[start:end] for _, start, end in matcher(doc)]
    # Resolve overlaps by keeping only the longest, most complete span boundaries
    spans = filter_spans(spans)  # drops "than" when "more than" already covers it
    # Format the filtered spans into the required tuple structure for the dataset
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


def _action_verb_spans(doc) -> list[tuple]:
    """
    Capture main clausal verbs -- the operation a problem's numbers and
    entities get plugged into (e.g. "sold" vs "bought" changes the correct
    arithmetic direction even when every number/entity stays the same).

    We rely on pos_ == "VERB" to exclude auxiliaries/copulas: spaCy already
    tags auxiliary and copular "be" as pos_ == "AUX", not "VERB", so a
    separate lemma-based exclusion list isn't needed and would wrongly
    throw away genuine main-verb uses of "have" (e.g. "I have 3 apples"),
    which matter a lot for this dataset.
    """
    results = []
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"ROOT", "advcl", "conj", "xcomp", "relcl"}:
            results.append((tok.text, tok.idx, tok.idx + len(tok.text), "action_verb", "token"))
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


def _random_control_spans(doc, used_ranges: set, n_controls: int = 3, rng=None) -> list[tuple]:
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
            if any(span.start_char < e and span.end_char > s for s, e in used_ranges):
                continue
            if not span.text.strip() or not any(t.is_alpha for t in span):
                continue
            candidates.append(span)

    if not candidates:
        return []
    chosen = (rng or random.Random(42)).sample(candidates, min(n_controls, len(candidates)))
    return [
        (s.text, s.start_char, s.end_char, "random_control", _granularity_for_length(len(s)))
        for s in chosen
    ]


def find_spans(text: str, inserted_context: str = "", *, nlp=None, include_controls=True) -> list[tuple]:
    """Returns a list of (span_text, start_char, end_char, category, granularity)."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    _negation_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    _negation_matcher.add("NEGATION", [nlp.make_doc(p) for p in NEGATION_PHRASES])
    _comparison_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    _comparison_matcher.add("COMPARISON_CONDITIONAL", [nlp.make_doc(p) for p in COMPARISON_CONDITIONAL_PHRASES])
    spans = []
    spans += _matcher_spans(doc, _negation_matcher, "negation")
    spans += _matcher_spans(doc, _comparison_matcher, "comparison_conditional")
    spans += _number_spans(doc)
    spans += _entity_spans(doc)
    spans += _action_verb_spans(doc)
    spans += _distractor_span(text, inserted_context)

    used_ranges = {(s, e) for _, s, e, _, _ in spans}
    if include_controls:
        spans += _random_control_spans(doc, used_ranges, rng=random.Random(int(digest(text)[:16], 16)))

    return spans

