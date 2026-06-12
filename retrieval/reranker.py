"""
retrieval/reranker.py
─────────────────────────
Post-FAISS score adjustment based on section metadata. FAISS remains the
primary ranker; this is a small nudge, not an override.
"""

from config.retrieval import (
    EXPLANATORY_QUERY_KEYWORDS,
    EXPLANATORY_SECTIONS,
    LOW_VALUE_SECTIONS,
    EXPLANATORY_BOOST,
    LOW_VALUE_PENALTY,
)


def rerank_by_section(query: str, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """
    For explanatory-style queries, boost chunks from high-value sections
    (overview, mechanism, ...) and penalise low-value sections
    (limitations, drawbacks, ...). Non-explanatory queries are returned
    unchanged.
    """
    q = query.lower()
    is_explanatory = any(kw in q for kw in EXPLANATORY_QUERY_KEYWORDS)

    if not is_explanatory:
        return results

    reranked = []
    for chunk, score in results:
        section = chunk.section.lower()
        if any(s in section for s in EXPLANATORY_SECTIONS):
            adjusted = score * EXPLANATORY_BOOST
        elif any(s in section for s in LOW_VALUE_SECTIONS):
            adjusted = score * LOW_VALUE_PENALTY
        else:
            adjusted = score
        reranked.append((chunk, adjusted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked