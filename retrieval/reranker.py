"""
retrieval/reranker.py
─────────────────────────
Post-FAISS score adjustment based on section metadata. FAISS remains the
primary ranker; this is a small nudge, not an override. Returned scores are
adjusted ranking scores for explanatory queries, not raw cosine values or
calibrated confidence.
"""

from config.retrieval import (
    EXPLANATORY_QUERY_KEYWORDS,
    LOW_VALUE_SECTIONS,
    EXPLANATORY_BOOST,
    LOW_VALUE_PENALTY,
)
from config.retrieval_domain import EXPLANATORY_SECTIONS
from data.schemas import Chunk


def rerank_by_section(
    query: str,
    results: list[tuple[Chunk, float]],
    explanatory_sections=None,
    low_value_sections=None,
    explanatory_keywords=None,
    explanatory_boost: float = EXPLANATORY_BOOST,
    low_value_penalty: float = LOW_VALUE_PENALTY,
) -> list[tuple[Chunk, float]]:
    """
    For explanatory-style queries, boost chunks from high-value sections
    (overview, mechanism, ...) and penalise low-value sections
    (limitations, drawbacks, ...). Non-explanatory queries are returned
    unchanged.
    """
    q = query.lower()
    if explanatory_sections is None:
        explanatory_sections = EXPLANATORY_SECTIONS
    if low_value_sections is None:
        low_value_sections = LOW_VALUE_SECTIONS
    if explanatory_keywords is None:
        explanatory_keywords = EXPLANATORY_QUERY_KEYWORDS
    is_explanatory = any(kw in q for kw in explanatory_keywords)

    import re

    scen_match = re.search(r"\bscenario\s*(\d+)\b", q)
    target_scenario = f"scenario {scen_match.group(1)}" if scen_match else None
    has_scenario_intent = bool(
        re.search(
            r"\b(?:scenario\s*\d*|experiment\b|experimental|burden mix|blend case)\b",
            q,
        )
    )

    if not is_explanatory and not target_scenario and not has_scenario_intent:
        return results

    reranked = []
    for chunk, score in results:
        sec = chunk.section.lower()
        chunk_role = getattr(chunk, "role", "reference")

        if target_scenario:
            if target_scenario in sec:
                adjusted = score * 1.35 if score >= 0 else score * 0.65
            elif chunk_role == "experiment_scenario":
                adjusted = score * 0.95
            elif is_explanatory and any(s in sec for s in explanatory_sections):
                adjusted = score + abs(score) * (explanatory_boost - 1)
            elif is_explanatory and any(s in sec for s in low_value_sections):
                adjusted = score + abs(score) * (low_value_penalty - 1)
            else:
                adjusted = score
        elif has_scenario_intent and chunk_role == "experiment_scenario":
            adjusted = score * 1.15
        elif is_explanatory and any(s in sec for s in explanatory_sections):
            adjusted = score + abs(score) * (explanatory_boost - 1)
        elif is_explanatory and any(s in sec for s in low_value_sections):
            adjusted = score + abs(score) * (low_value_penalty - 1)
        else:
            adjusted = score
        reranked.append((chunk, adjusted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked