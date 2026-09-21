"""
fidelity_checks.py
-------------------
Fast, automated first-pass checks on an LLM-generated explanation of a
PARSE finding. These are a gate, not a substitute for reading real
outputs -- see Task 24's non-negotiables.

Three checks:
  - numerical_fidelity: every number that appears in the source finding
    (message text + attributes dict) must appear, unchanged, in the
    explanation.
  - causal_language_check: flags explanations that introduce causal
    phrasing ("causes", "leads to", "results in", ...) for findings
    whose kind is "interpretation" or whose message uses associational/
    inferential language. This is the highest-priority check per the
    training-plan review -- Cochrane fine-tuning data could plausibly
    make this worse, not just fail to fix it.
  - limitation_preserved: if the source finding has a limitations
    tuple, at least one limitation's key content must survive into the
    explanation in some recognizable form.
"""

import re
from dataclasses import dataclass


CAUSAL_PHRASES = (
    "causes", "caused by", "causing",
    "leads to", "led to", "leading to",
    "results in", "resulted in",
    "due to",  # ambiguous but frequently used causally; flagged for review, not auto-fail
    "because of",
)

ASSOCIATIONAL_MARKERS = (
    "associated", "association", "correlat", "may ", "might ", "appears to",
    "candidacy", "inferred", "heuristic", "suggests", "not a causal claim",
)

_NUMBER_RE = re.compile(r"-?\d+\.?\d*%?")


@dataclass
class FidelityResult:
    passed: bool
    detail: str


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens, normalized (trailing .0 / % stripped for comparison)."""
    numbers = set()
    for match in _NUMBER_RE.findall(text):
        cleaned = match.rstrip("%")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        # Normalize e.g. "4" and "4.0" to the same key; keep % as a separate fact.
        key = f"{value:g}" + ("%" if match.endswith("%") else "")
        numbers.add(key)
    return numbers


def numerical_fidelity(source_message: str, attributes: dict, explanation: str) -> FidelityResult:
    """Every number in the source message or attributes dict must survive, unchanged.

    Each source fact is represented as a group of acceptable forms (e.g. a
    0-1 fraction in `attributes` and its "NN%" rendering in `message` are
    the same fact) -- a fact is satisfied if *any* form in its group
    appears in the explanation, not all of them.
    """
    fact_groups: list[frozenset[str]] = [frozenset({n}) for n in _extract_numbers(source_message)]
    for value in attributes.values():
        if isinstance(value, (int, float)):
            forms = _extract_numbers(str(value))
            # PARSE often stores a raw 0-1 fraction in attributes but renders it as a
            # percentage in the message (e.g. numeric_parse_fraction=0.8 -> "80%") -
            # both forms represent the same fact, so either is acceptable.
            if 0 <= value <= 1:
                forms = forms | {f"{value * 100:g}%"}
            if forms:
                fact_groups.append(frozenset(forms))
        elif isinstance(value, str):
            fact_groups.extend(frozenset({n}) for n in _extract_numbers(value))

    if not fact_groups:
        return FidelityResult(True, "No numeric facts in source finding to check.")

    explanation_numbers = _extract_numbers(explanation)
    missing = [group for group in fact_groups if not (group & explanation_numbers)]

    if missing:
        return FidelityResult(
            False,
            f"Source number(s) not found unchanged in explanation: {[sorted(g) for g in missing]}",
        )
    return FidelityResult(True, f"All {len(fact_groups)} source number(s) preserved.")


def causal_language_check(source_message: str, kind: str, explanation: str) -> FidelityResult:
    """Flag explanations that introduce causal phrasing the source finding didn't support."""
    explanation_lower = explanation.lower()
    found_causal = [phrase for phrase in CAUSAL_PHRASES if phrase in explanation_lower]

    if not found_causal:
        return FidelityResult(True, "No causal phrasing detected.")

    source_lower = source_message.lower()
    source_is_associational = kind == "interpretation" or any(
        marker in source_lower for marker in ASSOCIATIONAL_MARKERS
    )
    source_uses_causal_language = any(phrase in source_lower for phrase in CAUSAL_PHRASES)

    if source_is_associational and not source_uses_causal_language:
        return FidelityResult(
            False,
            f"Explanation introduces causal phrasing {found_causal} for an associational/"
            f"inferential finding (kind={kind!r}). Source: {source_message!r}",
        )
    return FidelityResult(
        True,
        f"Causal phrasing {found_causal} present but source finding is not purely associational.",
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{4,}", text.lower())}


def limitation_preserved(limitations: tuple, explanation: str, min_overlap: int = 2) -> FidelityResult:
    """At least one limitation's key content should survive into the explanation.

    Uses a loose token-overlap heuristic (not exact-string matching, since
    the explanation is expected to reword) -- this is a first-pass gate,
    not a precise check; see Task 24's non-negotiables.
    """
    if not limitations:
        return FidelityResult(True, "Source finding has no limitations to preserve.")

    explanation_tokens = _tokenize(explanation)
    for limitation in limitations:
        limitation_tokens = _tokenize(limitation)
        overlap = limitation_tokens & explanation_tokens
        if len(overlap) >= min_overlap:
            return FidelityResult(
                True, f"Limitation content plausibly preserved (overlap: {sorted(overlap)})."
            )

    return FidelityResult(
        False,
        f"None of {len(limitations)} limitation(s) appear reflected in the explanation "
        f"(checked via token overlap, min_overlap={min_overlap}).",
    )


def run_all_checks(finding: dict, explanation: str) -> dict[str, FidelityResult]:
    """Run all three checks against a finding dict (as produced by
    build_parse_eval_set.py) and a model-generated explanation.
    """
    return {
        "numerical_fidelity": numerical_fidelity(
            finding["message"], finding.get("attributes", {}), explanation
        ),
        "causal_language": causal_language_check(
            finding["message"], finding.get("kind", ""), explanation
        ),
        "limitation_preserved": limitation_preserved(
            tuple(finding.get("limitations", ())), explanation
        ),
    }
