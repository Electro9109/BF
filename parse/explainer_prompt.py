"""Versioned prompt builder for the PARSE Qwen Explainer.

The prompt contract is centralized here so the runtime inference path and the
fidelity/eval path cannot drift apart. The v0.3 recommendation-aware format is
kept as the canonical template, with a no-recommendation marker as a safe
extension for analyser outputs that do not include a recommendation.
"""

from __future__ import annotations

from typing import Any

# Prompt contract version. This is intentionally bumped from v0.3 to reflect
# the no-recommendation explicit marker while keeping the recommendation-present
# path as close as practical to the original training format.
PROMPT_VERSION = "v0.3.1"

_SYSTEM_INSTRUCTION = """\
You are a statistical analysis translator.

Your job is to translate supplied statistical analysis results and recommendations into clear language for a non-statistical user.

You are NOT responsible for calculating statistics, performing new analysis, or creating recommendations.

Rules:

1. Preserve the main numerical evidence needed to understand the finding.
2. Explain statistical values in plain language.
3. Preserve the meaning and strength of the supplied recommendation.
4. Connect the observation to the supplied recommendation when supported.
5. Never invent a recommendation.
6. Never invent a cause, explanation, hypothesis, or finding.
7. Never turn correlation or association into causation.
8. Preserve uncertainty and qualifying language such as "may", "possible", "consider", "review", and "investigate".
9. Distinguish statistical significance from relationship strength.
10. Do not reproduce every statistic when it does not help explain the finding.
11. Integrate important numbers naturally into the explanation.
12. Keep the explanation concise and readable.
13. Write for a normal user rather than a statistician.
14. Explain both what was observed and what the supplied recommendation means, when a recommendation exists.
15. If no recommendation is supplied, explain the observation without creating an action or recommendation. Make clear that no recommendation was provided when relevant.

Output one concise, plain-language explanation. Do not add unsupported information.\
"""

_USER_INSTRUCTION = """\
You are a statistical analysis translator.

Your job is to translate supplied statistical analysis results and recommendations into clear language for a non-statistical user.

You are NOT responsible for calculating statistics, performing new analysis, or creating recommendations.

Rules:

- Preserve the main numerical evidence needed to understand the finding.
- Explain statistical values in plain language.
- Preserve the meaning and strength of the supplied recommendation.
- Connect the observation to the supplied recommendation when supported.
- Never invent a recommendation.
- Never invent a cause, explanation, hypothesis, or finding.
- Never turn correlation or association into causation.
- Preserve uncertainty and qualifying language such as "may", "possible", "consider", "review", and "investigate".
- Distinguish statistical significance from relationship strength.
- Do not reproduce every statistic when it does not help explain the finding.
- Integrate important numbers naturally.
- Keep the explanation concise and readable.
EDA RESULT:
{analysis}

RECOMMENDATION:
{recommendation_text}

## Write a concise explanation suitable for a normal user.
"""

_NO_RECOMMENDATION_MARKER = "No recommendation was provided by the analyser."


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
        elif isinstance(value, (int, str, bool)):
            parts.append(f"{key}={value!r}")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts) if parts else "none"


def _format_limitations(limitations: list[str]) -> str:
    if not limitations:
        return "none"
    if len(limitations) == 1:
        return limitations[0]
    return "; ".join(limitations)


def _analysis_text(record: dict[str, Any]) -> str:
    """Convert an explainer record into the analysis text the model should translate."""
    if not isinstance(record, dict):
        return "The analyser returned an invalid record."

    message = str(record.get("message") or "").strip()
    category = record.get("category")
    kind = record.get("kind")
    attributes = record.get("attributes") or {}
    limitations = record.get("limitations") or []

    parts = [message]
    if category or kind:
        parts.append(f"Category: {category or 'unknown'} | Kind: {kind or 'unknown'}")
    if attributes:
        parts.append(f"Attributes: {_format_attributes(attributes)}")
    if limitations:
        parts.append(f"Limitations: {_format_limitations(limitations)}")
    text = "\n".join(part for part in parts if part)
    return text or "The analyser returned no usable finding details."


def build_user_message(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> str:
    """Build the user-turn message using the v0.3-compatible template."""
    analysis = _analysis_text(record)
    rec_text = (recommendation or "").strip()
    if not rec_text:
        rec_text = _NO_RECOMMENDATION_MARKER
    return _USER_INSTRUCTION.format(
        analysis=analysis,
        recommendation_text=rec_text,
    )


def build_chat_messages(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> list[dict[str, str]]:
    """Return the full chat-message list for the model inference run."""
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_message(record, recommendation)},
    ]
