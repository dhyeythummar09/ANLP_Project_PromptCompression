"""Conservative, task-specific grading of a unique final answer.

No substring correctness and no searching reasoning for the gold number.
Parse failures are distinct from well-formed wrong answers.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

GRADER_VERSION = "unique-final-v2"
FINAL_ANSWER_MARKER = "final answer:"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
QUANTITY = rf"(?:{NUMBER})(?:\s*/\s*(?:{NUMBER}))?"

@dataclass(frozen=True)
class Grade:
    correct: bool
    status: str
    answer: str
    answer_type: str
    grader_version: str = GRADER_VERSION

    def to_dict(self):
        return asdict(self)

def extract_final_answer(raw_model_output: str) -> str:
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        return ""
    raw = re.sub(r"<think>.*?</think>", "", raw_model_output, flags=re.S | re.I)
    if "<think>" in raw.lower():
        return ""
    markers = list(re.finditer(r"final\s+answer\s*:", raw, flags=re.I))
    if markers:
        lines = [line.strip() for line in raw[markers[-1].end():].strip().splitlines() if line.strip()]
        if len(lines) != 1:
            return ""
        answer = lines[0]
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return ""
        answer = lines[-1]
    answer = re.sub(r"</?answer>", "", answer, flags=re.I).strip()
    boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}[.!]?", answer)
    return (boxed.group(1) if boxed else answer).strip().strip("*").strip()

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).strip(" .!\"'")

def _numeric_value(text):
    text = text.replace(",", "").replace(" ", "")
    try:
        if "/" in text:
            a, b = text.split("/")
            return Fraction(Decimal(a)) / Fraction(Decimal(b))
        value = Decimal(text)
        return Fraction(value) if value.is_finite() else None
    except (ValueError, InvalidOperation, ZeroDivisionError, OverflowError):
        return None

def _try_parse_number(text: str):
    cleaned = str(text).strip().lstrip("$").strip()
    return _numeric_value(cleaned) if re.fullmatch(QUANTITY, cleaned) else None

def _options(question):
    matches = list(re.finditer(r"(?:^|\s)\(([A-Z])\)\s*", question))
    return {m.group(1): question[m.end():matches[i + 1].start() if i + 1 < len(matches)
                              else len(question)].strip()
            for i, m in enumerate(matches)}

def grade_answer(raw_model_output, ground_truth, question="", source_dataset="") -> Grade:
    answer = extract_final_answer(raw_model_output)
    gold = str(ground_truth).strip()
    if not gold:
        raise ValueError("Ground truth is empty")
    numeric_gold = _try_parse_number(gold)
    choice = re.fullmatch(r"\(([A-Z])\)", gold)
    norm_gold = _normalize_text(gold)
    kind = ("numeric" if numeric_gold is not None else "choice" if choice else
            "binary" if norm_gold in {"yes", "no", "valid", "invalid"} else "text")
    if not answer:
        return Grade(False, "empty_or_unresolved_final", "", kind)
    if kind == "numeric":
        m = re.fullmatch(rf"\s*[$??]?\s*({QUANTITY})\s*(%|[A-Za-z][A-Za-z -]*)?[.!]?\s*", answer)
        if not m or (m.group(2) and re.search(r"\b(?:or|and|actually|equals|is)\b", m.group(2), re.I)):
            return Grade(False, "ambiguous_or_invalid_numeric", answer, kind)
        value = _numeric_value(m.group(1))
        if value is None:
            return Grade(False, "invalid_numeric", answer, kind)
        # Percent uses the benchmark's displayed numeric units, not automatic /100.
        return Grade(abs(value - numeric_gold) <= Fraction(1, 1_000_000), "ok", answer, kind)
    if kind == "choice":
        options = _options(question)
        m = re.fullmatch(r"\(?([A-Z])\)?[.!]?", answer.strip())
        if m:
            selected = m.group(1)
            if options and selected not in options:
                return Grade(False, "invalid_choice", answer, kind)
            return Grade(selected == choice.group(1), "ok", answer, kind)
        prefixed = re.fullmatch(r"\(?([A-Z])\)?[.):]?\s+(.+)", answer.strip())
        if prefixed and prefixed.group(1) in options:
            selected, content = prefixed.groups()
            if _normalize_text(content) == _normalize_text(options[selected]):
                return Grade(selected == choice.group(1), "ok", answer, kind)
        matching = [key for key, content in options.items()
                    if _normalize_text(answer) == _normalize_text(content)]
        if len(matching) == 1:
            return Grade(matching[0] == choice.group(1), "ok", answer, kind)
        return Grade(False, "ambiguous_or_invalid_choice", answer, kind)
    normalized = _normalize_text(answer)
    if kind == "binary" and normalized not in {"yes", "no", "valid", "invalid"}:
        return Grade(False, "ambiguous_or_invalid_binary", answer, kind)
    return Grade(normalized == norm_gold, "ok", answer, kind)

def answers_match(raw_model_output: str, ground_truth: str, question: str = "") -> bool:
    """Compatibility wrapper. New evaluators preserve the full Grade."""
    return grade_answer(raw_model_output, ground_truth, question).correct
