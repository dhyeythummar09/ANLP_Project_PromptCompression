"""
Shared logic for turning a raw model response into a clean answer, and for
deciding whether that answer matches the ground truth.

This lives in src/utils/ (not inline in the labeling script) because we will
need the exact same logic again later, when evaluating the final compressor
against LLMLingua/LLMLingua-2/Selective-Context -- keeping it in one place
means both uses stay consistent instead of silently drifting apart.

Why this matters: a substring check like `"3" in model_answer` is a common
first draft, but it produces false positives constantly (ground truth "3"
matches inside "13", "$3.50", "30", etc). We instead try a numeric-equality
comparison first, and only fall back to normalized exact-string matching for
answers that aren't numbers (BBH multiple-choice/yes-no answers, for
example).
"""

import re

# If the model was allowed to reason before answering, we look for this
# marker to find where the final answer starts. The prompt template in
# counterfactual_labeling.py explicitly asks the model to end with this.
FINAL_ANSWER_MARKER = "final answer:"


def extract_final_answer(raw_model_output: str) -> str:
    """
    Pull the final answer out of a (possibly multi-line, reasoning-included)
    model response.

    Looks for the "Final Answer:" marker first (case-insensitive) and takes
    everything after it. If the marker isn't present -- the model ignored
    the instruction, which happens -- falls back to the last non-empty line
    of the response, which is a reasonable heuristic for short-answer tasks.
    """
    if not raw_model_output:
        return ""

    lowered = raw_model_output.lower()
    marker_pos = lowered.rfind(FINAL_ANSWER_MARKER)
    if marker_pos != -1:
        return raw_model_output[marker_pos + len(FINAL_ANSWER_MARKER):].strip()

    # Fallback: last non-empty line
    lines = [line.strip() for line in raw_model_output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else raw_model_output.strip()


def _try_parse_number(text: str):
    """
    Try to parse `text` as a float after stripping common formatting:
    currency symbols, commas, surrounding punctuation, percent signs.
    Returns None if it doesn't look like a number at all.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"[,$%]", "", cleaned)
    cleaned = cleaned.strip(" .()")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_text(text: str) -> str:
    """Lowercase, strip whitespace/punctuation, collapse internal spaces."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)  # drop punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def answers_match(raw_model_output: str, ground_truth: str) -> bool:
    """
    Decide whether a model's response counts as correct.

    Numeric ground truths are compared as numbers (so "115" matches "115.0"
    or "$115"), not as substrings. Non-numeric ground truths (BBH-style
    multiple-choice or yes/no answers) are compared as normalized exact
    strings, not substrings, so "yes" doesn't spuriously match inside
    "yes and no" or similar.
    """
    predicted = extract_final_answer(raw_model_output)

    gt_num = _try_parse_number(str(ground_truth))
    pred_num = _try_parse_number(predicted)
    if gt_num is not None and pred_num is not None:
        return abs(gt_num - pred_num) < 1e-6

    # Non-numeric: exact match after normalization, not substring containment.
    return _normalize_text(predicted) == _normalize_text(str(ground_truth))
