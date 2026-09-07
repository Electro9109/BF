"""
retrieval/topic_filter.py
────────────────────────────
Pre-FAISS topic detection and candidate filtering.

NOTE on the original bug ("Broken Topic-Routing Filter"): the old
_detect_topic() returned a *list* of matched topics, but callers compared
it to a chunk's topic string with `chunk["topic"] == matched_topic`
(list == str is always False), so the topic filter never actually filtered
anything. detect_topic() below returns a single canonical topic string
(the first / most specific match) or None.
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
    Return a single canonical topic name if the query clearly targets one
    topic, otherwise None (-> search all chunks).

    Keywords are checked in TOPIC_KEYWORDS dict order; within a topic,
    longer/more specific phrases are listed first so e.g. "cohesive zone"
    is matched before the bare "cohesive".

    If multiple topics match, the first one encountered (dict order) wins.
    If this produces unexpected results for ambiguous queries, reorder
    TOPIC_KEYWORDS or tighten the keyword phrases.
    """
    keywords_by_topic = TOPIC_KEYWORDS if topic_keywords is None else topic_keywords
    for topic, keywords in keywords_by_topic.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                return topic
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