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

    assert detect_topic("explain sinter") == "sinter"
    candidates, effective = filter_by_topic(chunks, "pellet", min_candidates=1)
    assert effective == "pellet"
    assert [chunk.source for chunk in candidates] == ["p1"]


def test_topic_detection_uses_broad_retrieval_for_ambiguous_topics():
    assert detect_topic("compare sinter and pellet") is None
    assert detect_topic("basicity effects with hydrogen") is None
    assert detect_topic("what is reducibility") == "reducibility"
    assert detect_topic("hydrogenation effects") is None
    assert detect_topic("an unrelated question") is None


def test_topic_detection_does_not_filter_out_feo_for_softening_queries():
    assert detect_topic("residual feo and softening onset") is None
    assert detect_topic("wustite content") == "feo"


def test_topic_detection_broadens_overlapping_basicity_and_tmts_queries():
    assert detect_topic("what b2 range is typical for blast furnace sinter?") is None
    assert detect_topic("what tm-ts ranges are described as narrow or wide?") is None


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


def test_reranker_preserves_original_order_for_equal_adjusted_scores():
    first = make_chunk("first", "sinter", "mechanism")
    second = make_chunk("second", "sinter", "mechanism")

    results = rerank_by_section(
        "explain the process",
        [(first, 0.8), (second, 0.8)],
        explanatory_sections={"mechanism"},
        low_value_sections=set(),
        explanatory_keywords={"explain"},
    )

    assert [chunk.source for chunk, _score in results] == ["first", "second"]


def test_reranker_boosts_and_penalizes_negative_similarity_in_expected_direction():
    preferred = make_chunk("preferred", "sinter", "mechanism")
    neutral = make_chunk("neutral", "sinter", "body")
    low_value = make_chunk("low", "sinter", "limitations")

    results = rerank_by_section(
        "explain the process",
        [(preferred, -0.4), (neutral, -0.35), (low_value, -0.4)],
        explanatory_sections={"mechanism"},
        low_value_sections={"limitations"},
        explanatory_keywords={"explain"},
        explanatory_boost=1.15,
        low_value_penalty=0.75,
    )

    assert [chunk.source for chunk, _score in results] == [
        "preferred", "neutral", "low"
    ]
    assert results[0][1] == pytest.approx(-0.34)
    assert results[2][1] == pytest.approx(-0.5)


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


def test_reranker_can_promote_candidate_outside_initial_top_n_and_limits_results():
    chunks = [
        make_chunk("rank-1", "topic", "body"),
        make_chunk("rank-2", "topic", "body"),
        make_chunk("rank-3", "topic", "body"),
        make_chunk("promoted", "topic", "mechanism"),
    ]
    search_limits = []

    class OrderedIndex:
        def __init__(self, embeddings):
            self.size = len(embeddings)

        def search(self, query, k):
            search_limits.append(k)
            return (
                np.array([[1.0 - index * 0.1 for index in range(k)]], dtype=np.float32),
                np.array([list(range(k))], dtype=np.int64),
            )

    engine = RetrievalEngine(
        chunks,
        embedder=FakeEmbedder(),
        index_builder=OrderedIndex,
        topic_detector=lambda _query: None,
        reranker=lambda _query, results: sorted(
            results,
            key=lambda result: result[0].section == "mechanism",
            reverse=True,
        ),
        rerank_candidate_multiplier=2,
    )

    results = engine.search("query", top_n=2)

    assert search_limits == [len(chunks)]
    assert len(results) == 2
    assert [chunk.source for chunk, _score in results] == ["promoted", "rank-1"]


def test_rerank_candidate_pool_is_capped_to_filtered_corpus_size():
    chunks = [make_chunk(f"chunk-{index}", "topic") for index in range(3)]
    search_limits = []

    class RecordingIndex:
        def __init__(self, embeddings):
            self.size = len(embeddings)

        def search(self, query, k):
            search_limits.append(k)
            return (
                np.array([[1.0 - index * 0.1 for index in range(k)]], dtype=np.float32),
                np.array([list(range(k))], dtype=np.int64),
            )

    engine = RetrievalEngine(
        chunks,
        embedder=FakeEmbedder(),
        index_builder=RecordingIndex,
        topic_detector=lambda _query: "topic",
        candidate_filter=lambda all_chunks, topic: (all_chunks, topic),
        rerank_candidate_multiplier=4,
    )

    results = engine.search("query", top_n=5)

    assert search_limits == [len(chunks)]
    assert len(results) == len(chunks)


def test_filtered_search_reuses_precomputed_embedding_rows_and_maps_indices():
    chunks = [
        make_chunk(f"row-{index}", "topic" if index in {1, 3, 5} else "other")
        for index in range(6)
    ]
    for index, chunk in enumerate(chunks):
        chunk.embed_text = f"row-{index}"

    class RowEmbedder:
        def __init__(self):
            self.document_encode_calls = 0

        def encode(self, texts):
            self.document_encode_calls += 1
            return np.array(
                [[float(text.split("-")[-1]), 1.0] for text in texts],
                dtype=np.float32,
            )

        def encode_one(self, text):
            return np.array([[1.0, 1.0]], dtype=np.float32)

    captured_embeddings = []

    class ReversePairIndex:
        def __init__(self, embeddings):
            captured_embeddings.append(embeddings.copy())

        def search(self, query, k):
            return (
                np.array([[0.9, 0.8]], dtype=np.float32),
                np.array([[1, 0]], dtype=np.int64),
            )

    embedder = RowEmbedder()
    engine = RetrievalEngine(
        chunks,
        embedder=embedder,
        index_builder=ReversePairIndex,
        topic_detector=lambda query: "topic",
        candidate_filter=lambda all_chunks, topic: ([all_chunks[1], all_chunks[5]], topic),
        rerank_candidate_multiplier=1,
        reranker=lambda query, results: results,
    )

    results = engine.search("topic query", top_n=1)

    assert embedder.document_encode_calls == 1
    assert captured_embeddings[1].tolist() == [[1.0, 1.0], [5.0, 1.0]]
    assert [chunk.source for chunk, _ in results] == ["row-5"]


def test_filtered_faiss_ranking_matches_independent_candidate_dot_products():
    chunks = [make_chunk(f"doc-{index}", "topic" if index else "other") for index in range(4)]
    vectors = {
        "doc-0": [1.0, 0.0],
        "doc-1": [0.8, 0.6],
        "doc-2": [0.0, 1.0],
        "doc-3": [-1.0, 0.0],
        "query": [np.sqrt(0.5), np.sqrt(0.5)],
    }
    for chunk in chunks:
        chunk.embed_text = chunk.source

    class VectorEmbedder:
        def encode(self, texts):
            return np.asarray([vectors[text] for text in texts], dtype=np.float32)

        def encode_one(self, text):
            return np.asarray([vectors[text]], dtype=np.float32)

    from retrieval.faiss_index import build_index

    engine = RetrievalEngine(
        chunks,
        embedder=VectorEmbedder(),
        index_builder=build_index,
        topic_detector=lambda _query: "topic",
        candidate_filter=lambda all_chunks, topic: (
            [all_chunks[1], all_chunks[2], all_chunks[3]],
            topic,
        ),
        reranker=lambda _query, results: results,
    )

    actual = engine.search("query", top_n=3)
    candidates = [chunks[1], chunks[2], chunks[3]]
    reference_vectors = VectorEmbedder().encode([chunk.embed_text for chunk in candidates])
    reference_scores = reference_vectors @ VectorEmbedder().encode_one("query")[0]
    reference_order = np.argsort(reference_scores)[::-1]
    expected_sources = [candidates[index].source for index in reference_order]

    assert [chunk.source for chunk, _score in actual] == expected_sources
    assert [score for _chunk, score in actual] == pytest.approx(
        reference_scores[reference_order].tolist()
    )


def test_filtered_search_reembeds_and_keeps_noncorpus_candidate_compatibility():
    original = make_chunk("original", "topic")
    external = make_chunk("external", "topic")
    indexed_rows = []

    class CapturingIndex:
        def __init__(self, embeddings):
            indexed_rows.append(embeddings.copy())

        def search(self, query, k):
            return (
                np.array([[0.9]], dtype=np.float32),
                np.array([[0]], dtype=np.int64),
            )

    embedder = FakeEmbedder()
    engine = RetrievalEngine(
        [original],
        embedder=embedder,
        index_builder=CapturingIndex,
        topic_detector=lambda _query: "topic",
        candidate_filter=lambda _chunks, topic: ([external], topic),
        reranker=lambda _query, results: results,
    )

    results = engine.search("filtered query")

    assert len(indexed_rows) == 2
    assert len(results) == 1
    assert results[0][0] is external


def test_filtered_search_rejects_misaligned_external_candidate_embeddings():
    original = make_chunk("original", "topic")
    external = make_chunk("external", "topic")
    external.embed_text = "external candidate content"

    class MisalignedEmbedder(FakeEmbedder):
        def encode(self, texts):
            dimensions = 3 if texts == ["external candidate content"] else 2
            return np.ones((len(texts), dimensions), dtype=np.float32)

    engine = RetrievalEngine(
        [original],
        embedder=MisalignedEmbedder(),
        index_builder=lambda embeddings: FakeIndex(len(embeddings)),
        topic_detector=lambda _query: "topic",
        candidate_filter=lambda _chunks, topic: ([external], topic),
    )

    with pytest.raises(ValueError, match="Filtered candidate embeddings"):
        engine.search("filtered query")


def test_retrieval_engine_handles_empty_corpus_and_nonpositive_top_n():
    embedder = FakeEmbedder()
    engine = RetrievalEngine(
        [],
        embedder=embedder,
        index_builder=lambda embeddings: pytest.fail("empty corpus must not build an index"),
    )

    assert engine.search("query") == []

    chunks = [make_chunk("s1", "sinter")]
    engine = RetrievalEngine(chunks, embedder=FakeEmbedder())
    assert engine.search("query", top_n=0) == []
    assert engine.search("query", top_n=-1) == []


@pytest.mark.parametrize("multiplier", [0, -1, 1.5, True])
def test_retrieval_engine_rejects_invalid_rerank_candidate_multiplier(multiplier):
    with pytest.raises(ValueError, match="positive integer"):
        RetrievalEngine(
            [make_chunk("chunk", "topic")],
            embedder=FakeEmbedder(),
            rerank_candidate_multiplier=multiplier,
        )


def test_retrieval_engine_handles_empty_filtered_candidates_without_building_index():
    index_sizes = []

    def build_index(embeddings):
        index_sizes.append(len(embeddings))
        return FakeIndex(len(embeddings))

    engine = RetrievalEngine(
        [make_chunk("available", "topic")],
        embedder=FakeEmbedder(),
        index_builder=build_index,
        topic_detector=lambda _query: "topic",
        candidate_filter=lambda _chunks, _topic: ([], None),
    )

    assert engine.search("query") == []
    assert index_sizes == [1]


def test_retrieval_engine_ignores_invalid_search_indices():
    chunk = make_chunk("valid", "topic")

    class SentinelIndex:
        def search(self, query, k):
            return (
                np.array([[0.9, 0.8]], dtype=np.float32),
                np.array([[0, -1]], dtype=np.int64),
            )

    engine = RetrievalEngine(
        [chunk],
        embedder=FakeEmbedder(),
        index_builder=lambda embeddings: SentinelIndex(),
        reranker=lambda query, results: results,
    )

    assert [item[0].source for item in engine.search("query", top_n=5)] == ["valid"]


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


def test_faiss_wrapper_caps_k_to_number_of_indexed_rows():
    pytest.importorskip("faiss")
    from retrieval.faiss_index import build_index

    index = build_index(np.array([[1.0, 0.0]], dtype=np.float32))
    scores, indices = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        5,
    )

    assert scores.shape == (1, 1)
    assert indices.tolist() == [[0]]


@pytest.mark.parametrize(
    "embeddings",
    [
        np.empty((0, 2), dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, np.nan]], dtype=np.float32),
    ],
)
def test_faiss_wrapper_rejects_invalid_embedding_matrices(embeddings):
    pytest.importorskip("faiss")
    from retrieval.faiss_index import build_index

    with pytest.raises(ValueError):
        build_index(embeddings)


def test_faiss_wrapper_rejects_wrong_query_dimensions():
    pytest.importorskip("faiss")
    from retrieval.faiss_index import build_index

    index = build_index(np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        index.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), 1)