"""Versioned prompt builder for the PARSE Qwen Explainer v0.3.

Converts an Explainer record (as produced by explainer_adapter.py) into
the exact chat-message list expected by the v0.3 LoRA adapter.

Design rules
------------
- The prompt builder is the **only** place the prompt format is defined.
  Both production inference and the evaluation harness import from here.
- No prompt-building logic lives in run_comparison.py or explainer_llm.py.
- Recommendations are passed in separately and mapped onto their own prompt
  section; they are NEVER invented when absent.
- The system instruction enforces strict translation semantics so the model
  cannot invent statistics, causes, or consequences not present in the input.
"""

from __future__ import annotations

from typing import Any

# The schema version this builder targets.  Bump when the prompt contract changes.
PROMPT_VERSION = "v0.3"

_SYSTEM_INSTRUCTION = """\
You are a statistical translator for a data-quality analysis system called PARSE.

Your only job is to translate a structured statistical finding into a single clear, \
concise paragraph for a non-specialist reader.

STRICT RULES — follow them exactly:
1. Preserve every important number exactly as given (counts, percentages, \
correlation values, skewness values, spread values, etc.).
2. Preserve the recommendation exactly as supplied — do NOT weaken, strengthen, \
or omit it.  "Consider removing" must remain "consider removing", never "remove".
3. Preserve uncertainty and any supplied limitations or caveats.
4. Do NOT invent statistics, causes, mechanisms, or consequences that are not \
present in the supplied finding.
5. Do NOT reinterpret one statistic as another:
   - Q1 and Q3 are quartiles, NOT the median.
   - IQR is the interquartile range, NOT the range.
   - A correlation is NOT a cause.
6. Do NOT convert an associational finding into a causal claim.
7. Do NOT add recommendations when none is supplied.
8. Translate the supplied finding; do not extend, speculate, or editorialize.

Output: one plain-language paragraph only.  No bullet points, no headings.\
"""

_FINDING_TEMPLATE = """\
Finding:
  message: {message}
  category: {category}
  kind: {kind}\
"""

_ATTRIBUTES_SECTION = """\
  attributes: {attributes}\
"""

_LIMITATIONS_SECTION = """\
  limitations: {limitations}\
"""

_RECOMMENDATION_SECTION = """\
  recommendation: {recommendation}\
"""


def _format_attributes(attributes: dict[str, Any]) -> str:
    """Render attributes as a readable key=value list, skipping None."""
    if not attributes:
        return "none"
    parts = []
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts) if parts else "none"


def _format_limitations(limitations: list[str]) -> str:
    if not limitations:
        return "none"
    if len(limitations) == 1:
        return limitations[0]
    return "; ".join(limitations)


def build_user_message(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> str:
    """Build the user-turn message from an Explainer record dict.

    Parameters
    ----------
    record:
        A dict in the Explainer schema (schema_version, finding_id, source,
        category, kind, message, attributes, limitations).
    recommendation:
        Optional recommendation string from the analysis engine.  Pass it
        only when one genuinely exists; do NOT invent or default one.

    Returns
    -------
    str
        The formatted user-turn string ready for the chat template.
    """
    parts = [
        _FINDING_TEMPLATE.format(
            message=record.get("message", ""),
            category=record.get("category", ""),
            kind=record.get("kind", ""),
        )
    ]

    attributes = record.get("attributes", {})
    parts.append(_ATTRIBUTES_SECTION.format(
        attributes=_format_attributes(attributes)
    ))

    limitations = record.get("limitations", [])
    parts.append(_LIMITATIONS_SECTION.format(
        limitations=_format_limitations(limitations)
    ))

    if recommendation:
        parts.append(_RECOMMENDATION_SECTION.format(
            recommendation=recommendation.strip()
        ))

    return "\n".join(parts)


def build_chat_messages(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> list[dict[str, str]]:
    """Return the full chat-message list for the v0.3 adapter.

    Use this with tokenizer.apply_chat_template(messages, ...) to get the
    correctly formatted input tensor.

    Parameters
    ----------
    record:
        Explainer record dict (from explainer_adapter.py).
    recommendation:
        Optional recommendation string.  Pass only when one actually exists.

    Returns
    -------
    list[dict]
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_message(record, recommendation)},
    ]
