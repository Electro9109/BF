"""
retrieval/reranker.py
─────────────────────────
Post-FAISS score adjustment based on section metadata. FAISS remains the
primary ranker; this is a small nudge, not an override.
"""

from config.retrieval import (
    EXPLANATORY_QUERY_KEYWORDS,
    LOW_VALUE_SECTIONS,
    EXPLANATORY_BOOST,
    LOW_VALUE_PENALTY,
)
from config.retrieval_domain import EXPLANATORY_SECTIONS


def rerank_by_section(
    query: str,
    results: list[tuple[dict, float]],
    explanatory_sections=None,
    low_value_sections=None,
    explanatory_keywords=None,
    explanatory_boost: float = EXPLANATORY_BOOST,
    low_value_penalty: float = LOW_VALUE_PENALTY,
) -> list[tuple[dict, float]]:
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

    if not is_explanatory:
        return results

    reranked = []
    for chunk, score in results:
        section = chunk.section.lower()
        if any(s in section for s in explanatory_sections):
            adjusted = score * explanatory_boost
        elif any(s in section for s in low_value_sections):
            adjusted = score * low_value_penalty
        else:
            adjusted = score
        reranked.append((chunk, adjusted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked