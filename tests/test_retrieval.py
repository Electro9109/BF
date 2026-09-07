import numpy as np
import pytest

from data.schemas import Chunk
from retrieval.reranker import rerank_by_section
from retrieval.retriever import RetrievalEngine
from retrieval.topic_filter import detect_topic, filter_by_topic


def make_chunk(source: str, topic: str, section: str = "body") -> Chunk:
    return Chunk(
        content=f"Content from {source}",
        embed_text=f"TOPIC: {topic}\nSECTION: {section}\nContent",
        source=source,
        topic=topic,
        section=section,
    )


def test_topic_detection_and_filtering_are_explicit():
    chunks = [make_chunk("s1", "sinter"), make_chunk("p1", "pellet")]

    assert detect_topic("explain sinter reducibility") == "sinter"
    candidates, effective = filter_by_topic(chunks, "pellet", min_candidates=1)
    assert effective == "pellet"
    assert [chunk.source for chunk in candidates] == ["p1"]


def test_filter_falls_back_when_topic_evidence_is_insufficient():
    chunks = [make_chunk("s1", "sinter"), make_chunk("p1", "pellet")]

    candidates, effective = filter_by_topic(chunks, "pellet", min_candidates=2)

    assert effective is None
    assert candidates is chunks


def test_reranker_applies_injected_section_policy():
    preferred = make_chunk("preferred", "sinter", "mechanism")
    low_value = make_chunk("low", "sinter", "limitations")

    results = rerank_by_section(
        "explain the process",
        [(low_value, 1.0), (preferred, 0.8)],
        explanatory_sections={"mechanism"},
        low_value_sections={"limitations"},
        explanatory_keywords={"explain"},
        explanatory_boost=2.0,
        low_value_penalty=0.5,
    )

    assert [chunk.source for chunk, _ in results] == ["preferred", "low"]
    assert results[0][1] == 1.6
    assert results[1][1] == 0.5


class FakeEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones((1, 2), dtype=np.float32)


class FakeIndex:
    def __init__(self, size):
        self.size = size

    def search(self, query, k):
        return (
            np.array([[1.0 - index * 0.1 for index in range(k)]], dtype=np.float32),
            np.array([list(range(k))], dtype=np.int64),
        )


def test_retrieval_engine_injects_mechanisms_and_policy():
    chunks = [
        make_chunk("s1", "sinter"),
        make_chunk("s2", "sinter"),
        make_chunk("s3", "sinter"),
        make_chunk("p1", "pellet"),
    ]

    engine = RetrievalEngine(
        chunks,
        embedder=FakeEmbedder(),
        index_builder=lambda embeddings: FakeIndex(len(embeddings)),
        topic_detector=lambda query: "sinter",
    )

    results = engine.search("domain query", top_n=2)

    assert [chunk.source for chunk, _ in results] == ["s1", "s2"]


def test_faiss_wrapper_returns_nearest_vector():
    pytest.importorskip("faiss")
    from retrieval.faiss_index import build_index

    index = build_index(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    scores, indices = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        1,
    )

    assert indices[0, 0] == 0
    assert scores[0, 0] > 0.99