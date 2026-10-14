"""
retrieval/topic_filter.py
────────────────────────────
Pre-FAISS topic detection and candidate filtering.
"""

import re
from collections.abc import Mapping, Sequence

from config.retrieval import MIN_TOPIC_CANDIDATES
from config.retrieval_domain import TOPIC_KEYWORDS


def detect_topic(
    query_lower: str,
    topic_keywords: Mapping[str, Sequence[str]] | None = None,
) -> str | None:
    """
    Return a topic only when keywords identify exactly one canonical topic.

    Multiple topic matches are ambiguous and return None so candidate
    filtering does not exclude relevant evidence from a cross-topic query.
    """
    keywords_by_topic = TOPIC_KEYWORDS if topic_keywords is None else topic_keywords
    matched_topics = set()
    for topic, keywords in keywords_by_topic.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                matched_topics.add(topic)
                break
    if len(matched_topics) == 1:
        return next(iter(matched_topics))
    return None


def filter_by_topic(
    chunks: list,
    topic: str | None,
    min_candidates: int = MIN_TOPIC_CANDIDATES,
) -> tuple[list, str | None]:
    """
    Return (candidates, effective_topic).

    If `topic` is None, returns (all chunks, None).
    If `topic` is set but matches fewer than `min_candidates` chunks,
    falls back to the full corpus and returns effective_topic=None so the
    caller can log the fallback.
    """
    if topic is None:
        return chunks, None

    candidates = [c for c in chunks if c.topic == topic]
    if len(candidates) < min_candidates:
        return chunks, None

    return candidates, topic