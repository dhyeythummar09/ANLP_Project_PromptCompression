"""Source-preserving selection with exact FINAL-string budget checks."""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

class TokenCounter:
    def __init__(self, name="cl100k_base"):
        self.name = name
        if name == "regex-test-only":
            self.encode = lambda s: re.findall(r"\w+|[^\w\s]", s)
        else:
            try:
                import tiktoken
            except ImportError as exc:
                raise RuntimeError("Scientific token counting requires tiktoken. Install requirements; no regex fallback.") from exc
            self.encode = tiktoken.get_encoding(name).encode

    def __call__(self, text):
        return len(self.encode(text))

def words(text):
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]

def render(text, units, selected):
    selected = sorted(selected)
    pieces = []
    previous = None
    for i in selected:
        start, end = units[i][:2]
        if previous is not None:
            prev_end = units[previous][1]
            gap = text[prev_end:start]
            pieces.append(gap if i == previous + 1 else "\n" if "\n" in gap else " ")
        pieces.append(text[start:end])
        previous = i
    return "".join(pieces)

@dataclass
class Selection:
    text: str
    kept_ranges: list
    budget: int
    tokens: int
    status: str
    groups: int

def make_groups(units, phrase_ranges=()):
    parent = list(range(len(units)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for s, e in phrase_ranges:
        indices = [i for i, (ws, we, _) in enumerate(units) if ws < e and we > s]
        if indices:
            for i in indices[1:]:
                parent[root(i)] = root(indices[0])
    groups = {}
    for i in range(len(units)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())

def select_to_budget(text, priority, ratio, counter, phrase_ranges=(), protected_ranges=()):
    if not 0 < ratio <= 1:
        raise ValueError("Retention ratio must be in (0, 1]")
    units = words(text)
    if len(priority) != len(units) or any(not math.isfinite(float(x)) for x in priority):
        raise ValueError("One finite priority per word required")
    budget = math.floor(ratio * counter(text))
    groups = make_groups(units, phrase_ranges)
    protected = {i for i, (s, e, _) in enumerate(units)
                 if any(s < pe and e > ps for ps, pe in protected_ranges)}
    protected_groups = {k for k, group in enumerate(groups) if protected.intersection(group)}
    minimum = {i for k in protected_groups for i in groups[k]}
    minimum_text = render(text, units, minimum)
    if counter(minimum_text) > budget:
        return Selection("", [], budget, 0, "infeasible_protection", len(groups))
    selected = set(range(len(units)))
    order = sorted((k for k in range(len(groups)) if k not in protected_groups),
                   key=lambda k: (max(float(priority[i]) for i in groups[k]), groups[k][0]))
    output = render(text, units, selected)
    for k in order:
        if counter(output) <= budget:
            break
        selected.difference_update(groups[k])
        output = render(text, units, selected)
    actual = counter(output)
    if actual > budget:
        raise AssertionError("Budget enforcement failed")
    # Greedy refill of whole removed groups, using actual reconstructed cost.
    for k in reversed(order):
        if set(groups[k]).issubset(selected):
            continue
        candidate = selected.union(groups[k])
        candidate_text = render(text, units, candidate)
        if counter(candidate_text) <= budget:
            selected, output = candidate, candidate_text
    return Selection(output, [[units[i][0], units[i][1]] for i in sorted(selected)], budget,
                     counter(output), "ok" if output else "empty_budget_output", len(groups))
