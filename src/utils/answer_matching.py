"""
Shared logic for turning a raw model response into a clean answer, and for
deciding whether that answer matches the ground truth.

This lives in src/utils/ (not inline in the labeling script) because we will
need the exact same logic again later, when evaluating the final compressor
against LLMLingua/LLMLingua-2/Selective-Context -- keeping it in one place
means both uses stay consistent instead of silently drifting apart.
"""

import re

FINAL_ANSWER_MARKER = "final answer:"


def extract_final_answer(raw_model_output: str) -> str:
    """
    Pull the final answer out of a (possibly multi-line, reasoning-included)
    model response.
    """
    if not raw_model_output:
        return ""

    lowered = raw_model_output.lower()
    marker_pos = lowered.rfind(FINAL_ANSWER_MARKER)
    if marker_pos != -1:
        ans = raw_model_output[marker_pos + len(FINAL_ANSWER_MARKER):].strip()
    else:
        # Fallback: ignore </think> lines and take the last non-empty line
        lines = [line.strip() for line in raw_model_output.strip().splitlines() if line.strip() and line.strip() != "</think>"]
        ans = lines[-1] if lines else raw_model_output.strip()

    # Strip XML tags like <answer>80</answer>
    ans = re.sub(r"</?[a-zA-Z0-9_\-]+>", "", ans).strip()
    # Strip LaTeX boxed like \boxed{80}
    m_box = re.search(r"\\boxed\{([^}]+)\}", ans)
    if m_box:
        ans = m_box.group(1).strip()
    return ans


def _try_parse_number(text: str):
    """
    Try to parse `text` as a float after stripping common formatting:
    currency symbols, commas, surrounding punctuation, percent signs.
    Returns None if it doesn't look like a clean number.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"[,$%]", "", cleaned)
    cleaned = cleaned.strip(" .()")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all float-convertible numbers from text, handling commas and signs."""
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    nums = []
    for m in matches:
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return nums


def _normalize_text(text: str) -> str:
    """Lowercase, strip whitespace/punctuation, collapse internal spaces."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)  # drop punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_choice_letter(text: str):
    """Extract multiple-choice letter like '(A)' or 'A' if at start of string."""
    m = re.match(r"^\s*\(?([A-Za-z0-9])\)?(?:\s|$|[:.)])", text.strip())
    if m:
        return m.group(1).upper()
    return None


def answers_match(raw_model_output: str, ground_truth: str, question: str = "") -> bool:
    """
    Decide whether a model's response counts as correct.
    Supports:
    1. Direct numeric match or numeric with units (e.g. '60 miles' vs '60').
    2. Multiple-choice match (e.g. '(A) 07/27/2002' vs '(A)').
    3. Multiple-choice text match when model outputs option content (e.g. '07/27/2002' for (A)).
    4. Normalized exact string match.
    """
    predicted = extract_final_answer(raw_model_output)
    gt_str = str(ground_truth).strip()

    # 1. Numeric comparison: if ground truth is a number
    gt_num = _try_parse_number(gt_str)
    if gt_num is not None:
        # Check direct conversion first
        pred_num = _try_parse_number(predicted)
        if pred_num is not None and abs(gt_num - pred_num) < 1e-6:
            return True
        # Check if numbers inside predicted string match (e.g. "60 miles", "$60")
        pred_nums = _extract_numbers(predicted)
        if pred_nums and any(abs(gt_num - n) < 1e-6 for n in pred_nums):
            return True

    # 2. Multiple choice comparison (e.g. "(A)" vs "(A) 07/27/2002")
    gt_choice = _extract_choice_letter(gt_str)
    pred_choice = _extract_choice_letter(predicted)
    if gt_choice is not None and pred_choice is not None:
        if gt_choice == pred_choice:
            return True

    # If model answered with the option's text instead of its letter (e.g. "07/27/2002" instead of "(A)")
    if gt_choice is not None and question:
        opt_pattern = rf"\({gt_choice}\)\s*([^\n\r]+)"
        m_opt = re.search(opt_pattern, question)
        if m_opt:
            opt_text = m_opt.group(1).strip()
            norm_opt = _normalize_text(opt_text)
            norm_pred = _normalize_text(predicted)
            if norm_pred == norm_opt or norm_opt in norm_pred or norm_pred in norm_opt:
                return True

    # 3. Normalized string equality
    norm_pred = _normalize_text(predicted)
    norm_gt = _normalize_text(gt_str)
    if norm_pred == norm_gt:
        return True

    # 4. If ground truth is short (e.g. "valid", "invalid", "yes", "no") and matches start of prediction
    if norm_gt and (norm_pred.startswith(norm_gt + " ") or norm_pred.endswith(" " + norm_gt)):
        return True

    return False
