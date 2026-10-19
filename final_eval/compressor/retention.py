"""Occurrence-aware retention; uncertain black-box alignments expose bounds."""
import json
import re

def alignment_bounds(original, compressed):
    """Earliest/latest monotone TOKEN embeddings; no guessed exact alignment.

    Insertion/subword-fragment changes are unresolved. For a valid subsequence,
    lower ranges cover forced source tokens and upper ranges cover any possible
    token occurrence. Upper bounds may be loose for a whole multi-token span.
    """
    source = list(re.finditer(r"\w+|[^\w\s]", original))
    target = re.findall(r"\w+|[^\w\s]", compressed)
    values = [m.group() for m in source]
    earliest, latest, cursor = [], [], 0
    for token in target:
        while cursor < len(values) and values[cursor] != token:
            cursor += 1
        if cursor == len(values):
            return [], [], "unresolved"
        earliest.append(cursor); cursor += 1
    cursor = len(values) - 1
    for token in reversed(target):
        while cursor >= 0 and values[cursor] != token:
            cursor -= 1
        if cursor < 0:
            return [], [], "unresolved"
        latest.append(cursor); cursor -= 1
    latest.reverse()
    lower, upper = set(), set()
    for token, first, last in zip(target, earliest, latest):
        candidates = {i for i in range(first, last + 1) if values[i] == token}
        upper.update(candidates)
        if len(candidates) == 1:
            lower.update(candidates)
    ranges = lambda indices: [[source[i].start(), source[i].end()] for i in sorted(indices)]
    return ranges(lower), ranges(upper), "exact_token_alignment" if lower == upper else "ambiguous_token_alignment"

def retention(original, compressed, spans, kept_ranges=None):
    if kept_ranges is None:
        lower, upper, status = alignment_bounds(original, compressed)
    else:
        lower = upper = kept_ranges
        status = "source_ranges"
        for s, e in lower:
            if not 0 <= s < e <= len(original):
                raise ValueError("Invalid retained source range")
    result = {"alignment_status": status, "crit_total": 0, "crit_kept_lower": 0,
              "crit_kept_upper": 0, "cat_retention_json": "{}"}
    if status == "unresolved":
        result.update(crit_total=None, crit_kept_lower=None, crit_kept_upper=None,
                      crit_kept=None, all_critical_kept=None)
        return result
    cats = {}
    for r in spans.to_dict("records"):
        s, e = int(r["span_start"]), int(r["span_end"])
        pieces = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", original[s:e])]
        pieces = [(s + a, s + b) for a, b in pieces]
        if not pieces:
            continue
        covered = lambda ranges: sum(any(a >= x and b <= y for x, y in ranges) for a, b in pieces)
        lo, hi = covered(lower), covered(upper)
        critical = int(r.get("label", r.get("consensus_label") == "critical"))
        result["crit_total"] += critical
        result["crit_kept_lower"] += critical * int(lo == len(pieces))
        result["crit_kept_upper"] += critical * int(hi == len(pieces))
        categories = json.loads(r["span_categories"]) if isinstance(r.get("span_categories"), str) else [r["span_category"]]
        for category in categories:
            c = cats.setdefault(category, {"spans": 0, "full_lower": 0, "full_upper": 0,
                                           "source_tokens": 0, "tokens_lower": 0, "tokens_upper": 0})
            c["spans"] += 1; c["full_lower"] += int(lo == len(pieces)); c["full_upper"] += int(hi == len(pieces))
            c["source_tokens"] += len(pieces); c["tokens_lower"] += lo; c["tokens_upper"] += hi
    result["crit_kept"] = result["crit_kept_lower"]  # conservative compatibility column
    total = result["crit_total"]
    result["all_critical_kept"] = int(result["crit_kept_lower"] == total) if total else None
    result["cat_retention_json"] = json.dumps(cats, sort_keys=True)
    return result
