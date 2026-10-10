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
            # percentage in the message (e.g. numeric_parse_fraction=0.8 -> "80%" or
            # 0.786885 -> "79%") - both forms represent the same fact, so either is acceptable.
            if 0 <= value <= 1:
                forms = forms | {f"{value * 100:g}%", f"{round(value * 100):g}%"}
            if forms:
                # If any form of this attribute fact is already present in source_message,
                # expand the matching message fact group so either representation satisfies it.
                matched = False
                for idx, group in enumerate(fact_groups):
                    if group & forms:
                        fact_groups[idx] = group | forms
                        matched = True
                if not matched and not fact_groups:
                    fact_groups.append(frozenset(forms))
        elif isinstance(value, str):
            str_forms = frozenset(_extract_numbers(value))
            if str_forms and not fact_groups:
                fact_groups.append(str_forms)

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


# Words the model commonly uses that are benign elaborations, not novel claims.
_NOVELTY_STOPWORDS = {
    "this", "that", "these", "those", "which", "with", "from", "about",
    "have", "been", "were", "also", "such", "into", "than", "each",
    "some", "other", "more", "most", "very", "only", "does", "mean",
    "data", "value", "values", "column", "based", "analysis", "should",
    "could", "would", "might", "note", "however", "therefore",
    "overall", "particular", "important", "suggest", "suggests",
    "indicate", "indicates", "approximately", "around", "roughly",
}


def unsupported_novelty(
    source_message: str,
    attributes: dict,
    limitations: tuple,
    explanation: str,
    recommendation: str | None = None,
    max_novel_ratio: float = 0.45,
) -> FidelityResult:
    """Flag explanations that introduce substantial novel vocabulary.

    This is a soft advisory check -- it flags output for human review but
    should NOT be treated as proof of an error.  The model is allowed to
    rephrase; this check catches cases where it invents entire new concepts
    or terminology that have no basis in the source finding.

    The check computes the ratio of explanation-only tokens (4+ letter words
    not appearing anywhere in the source) to total explanation tokens.
    A high ratio suggests the model may be fabricating content.
    """
    # Combine all source text for comparison
    source_parts = [source_message]
    for v in attributes.values():
        source_parts.append(str(v))
    for lim in limitations:
        source_parts.append(str(lim))
    if recommendation:
        source_parts.append(str(recommendation))
    source_text = " ".join(source_parts)

    source_tokens = _tokenize(source_text) | _NOVELTY_STOPWORDS
    explanation_tokens = _tokenize(explanation)

    if not explanation_tokens:
        return FidelityResult(True, "Explanation has no scorable tokens.")

    novel = explanation_tokens - source_tokens
    ratio = len(novel) / len(explanation_tokens)

    if ratio > max_novel_ratio:
        return FidelityResult(
            False,
            f"High novel-vocabulary ratio ({ratio:.0%}): the explanation introduces "
            f"many terms not in the source finding. Novel tokens: {sorted(novel)[:10]}... "
            f"(flagged for review, not auto-fail).",
        )
    return FidelityResult(
        True,
        f"Novel-vocabulary ratio acceptable ({ratio:.0%}, "
        f"threshold {max_novel_ratio:.0%}).",
    )


_TENTATIVE_RECOMMENDATION_MARKERS = (
    "consider", "suggest", "potential", "optional", "evaluate",
    "investigate", "might want to", "may want to", "candidacy",
)
_MANDATORY_RECOMMENDATION_MARKERS = (
    "must", "mandatory", "required", "imperative", "obligation",
    "immediately remove", "immediately delete",
)
_UNSOLICITED_ACTION_PATTERNS = [
    re.compile(r"\b(?:you|we)\s+must\s+(?:remove|delete|drop|change|clean)\b", re.IGNORECASE),
    re.compile(r"\b(?:action\s+required|mandatory\s+action)\b", re.IGNORECASE),
    re.compile(r"\bmust\s+be\s+(?:removed|deleted|dropped)\b", re.IGNORECASE),
]
_IMPERATIVE_COMMAND_PATTERNS = [
    re.compile(r"\b(?:remove|delete|drop)\s+(?:the\s+)?(?:column|attribute|field|feature)\b", re.IGNORECASE),
]


def recommendation_fidelity(
    recommendation: str | None,
    explanation: str,
) -> FidelityResult:
    """Check that recommendation intent and strength are preserved accurately.

    - If a recommendation was provided:
        * Tentative recommendations ('consider removing') must NOT be strengthened
          into mandatory commands ('must remove', 'remove column').
        * The key recommendation intent/subject must appear in the explanation.
    - If no recommendation was provided:
        * The explanation must NOT invent unsolicited mandatory recommendations or
          unsupported imperative actions.
    """
    explanation_lower = explanation.lower()

    if recommendation is None or not recommendation.strip():
        for pat in _UNSOLICITED_ACTION_PATTERNS:
            if pat.search(explanation):
                return FidelityResult(
                    False,
                    "Explanation introduced unsolicited imperative action when no recommendation was supplied.",
                )
        return FidelityResult(True, "No recommendation was supplied and none was invented.")

    rec_clean = recommendation.strip()
    rec_lower = rec_clean.lower()

    # Check strength preservation
    is_tentative = any(m in rec_lower for m in _TENTATIVE_RECOMMENDATION_MARKERS)
    has_mandatory_source = any(m in rec_lower for m in _MANDATORY_RECOMMENDATION_MARKERS)

    if is_tentative and not has_mandatory_source:
        for m in _MANDATORY_RECOMMENDATION_MARKERS:
            if re.search(r"\b" + re.escape(m) + r"\b", explanation_lower):
                return FidelityResult(
                    False,
                    f"Recommendation strength violation: tentative recommendation {rec_clean!r} "
                    f"strengthened into mandatory language ('{m}') in explanation.",
                )
        for pat in _IMPERATIVE_COMMAND_PATTERNS:
            if pat.search(explanation) and not any(m in explanation_lower for m in _TENTATIVE_RECOMMENDATION_MARKERS):
                return FidelityResult(
                    False,
                    f"Recommendation strength violation: tentative recommendation {rec_clean!r} "
                    f"converted into imperative command without tentative framing.",
                )

    # Check intent preservation: key non-stopword tokens from recommendation should survive
    rec_tokens = (_tokenize(rec_clean) - _NOVELTY_STOPWORDS) - {
        "consider", "suggest", "potential", "optional", "investigate", "evaluate",
    }
    if rec_tokens:
        exp_tokens = _tokenize(explanation)
        overlap = rec_tokens & exp_tokens
        if not overlap:
            return FidelityResult(
                False,
                f"Recommendation intent not reflected in explanation: none of {sorted(rec_tokens)} found.",
            )

    return FidelityResult(True, "Recommendation intent and strength preserved.")


def metric_semantics_check(source_message: str, explanation: str) -> FidelityResult:
    """Verify that statistical metrics are not conflated or misrepresented.

    - IQR (Interquartile Range) must not be called 'the range' or spread between min and max.
    - Quartiles (Q1, Q3) must not be called 'the median'.
    """
    source_lower = source_message.lower()
    exp_lower = explanation.lower()

    # IQR vs Range
    has_iqr_source = "iqr" in source_lower or "interquartile" in source_lower
    has_range_source = "range" in source_lower

    if has_iqr_source and not has_range_source:
        if re.search(r"\b(?:the\s+range|full\s+range|range\s+of\s+values|overall\s+range)\b", exp_lower):
            return FidelityResult(
                False,
                "Metric semantics violation: Interquartile Range (IQR) was conflated with statistical range.",
            )

    # Quartiles vs Median
    has_quartile_source = "quartile" in source_lower or "q1" in source_lower or "q3" in source_lower
    has_median_source = "median" in source_lower

    if has_quartile_source and not has_median_source:
        if re.search(r"\b(?:the\s+median|median\s+value)\b", exp_lower):
            return FidelityResult(
                False,
                "Metric semantics violation: Quartile was conflated with median.",
            )

    return FidelityResult(True, "Metric semantics preserved.")


def run_all_checks(
    finding: dict,
    explanation: str,
    recommendation: str | None = None,
) -> dict[str, FidelityResult]:
    """Run all checks against a finding dict and explanation.

    Hard checks (numerical_fidelity, causal_language, limitation_preserved,
    recommendation_fidelity, metric_semantics) are gates that trigger fallback.
    unsupported_novelty is a soft advisory check for review.
    """
    rec = recommendation or finding.get("recommendation")
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
        "recommendation_fidelity": recommendation_fidelity(rec, explanation),
        "metric_semantics": metric_semantics_check(finding["message"], explanation),
        "unsupported_novelty": unsupported_novelty(
            finding["message"],
            finding.get("attributes", {}),
            tuple(finding.get("limitations", ())),
            explanation,
            recommendation=rec,
        ),
    }
