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

from config.retrieval import TOPIC_KEYWORDS, MIN_TOPIC_CANDIDATES


def detect_topic(query_lower: str) -> str | None:
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
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                return topic
    return None


def filter_by_topic(chunks: list[dict], topic: str | None) -> tuple[list[dict], str | None]:
    """
    Return (candidates, effective_topic).

    If `topic` is None, returns (all chunks, None).
    If `topic` is set but matches fewer than MIN_TOPIC_CANDIDATES chunks,
    falls back to the full corpus and returns effective_topic=None so the
    caller can log the fallback.
    """
    if topic is None:
        return chunks, None

    candidates = [c for c in chunks if c.topic == topic]
    if len(candidates) < MIN_TOPIC_CANDIDATES:
        return chunks, None

    return candidates, topic