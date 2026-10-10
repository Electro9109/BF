"""Small, failure-safe bridge from the Data Explorer UI to the Qwen Explainer."""

from __future__ import annotations

import logging
from typing import Any

from parse.explainer_llm import ExplainerResult

logger = logging.getLogger(__name__)


def explain_record(
    record: dict[str, Any],
    recommendation: str | None = None,
) -> ExplainerResult:
    """Explain one adapted finding; preserve its source message on any integration error."""
    fallback_text = str(record.get("message", ""))
    finding_id = str(record.get("finding_id", ""))

    try:
        # Import the inference function at call time so the module itself stays lightweight.
        from parse.explainer_llm import explain

        return explain(record, recommendation=recommendation)
    except Exception as exc:  # The UI must remain usable if model loading/inference fails.
        logger.exception("Qwen Explainer integration failed for %s", finding_id)
        return ExplainerResult(
            text=fallback_text,
            fallback_used=True,
            validation_detail=f"Explainer integration error: {exc}",
            finding_id=finding_id,
        )
