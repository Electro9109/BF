"""
retrieval/retriever.py - Orchestrates topic filtering, FAISS search, and
section reranking. This is the only retrieval class other layers import.

Filtered candidate searches use rows selected from the original embedding
matrix, avoiding repeated document inference while keeping the mapping from
local search indices to original chunks explicit.
"""

from collections import defaultdict
import logging
import numpy as np

from config.retrieval import RERANK_CANDIDATE_MULTIPLIER, TOP_K
from data.schemas import Chunk
from retrieval.embeddings import Embedder
from retrieval.faiss_index import build_index
from retrieval.reranker import rerank_by_section
from retrieval.topic_filter import detect_topic, filter_by_topic

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Semantic retrieval using sentence-transformers + FAISS.

    Search scores are raw inner-product similarities for normalised vectors
    unless an explanatory-query section boost/penalty has adjusted them.
    They are ranking signals, not calibrated confidence.

    Pipeline per query
    ──────────────────
    1. detect_topic()      -> restrict candidates to the matched topic (or all)
    2. FAISS search         -> cosine similarity over embed_text vectors of
                                the candidate subset
    3. rerank_by_section()  -> small multiplier nudge based on section type
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder=None,
        index_builder=build_index,
        topic_detector=detect_topic,
        candidate_filter=filter_by_topic,
        reranker=rerank_by_section,
        rerank_candidate_multiplier: int = RERANK_CANDIDATE_MULTIPLIER,
    ):
        if (
            not isinstance(rerank_candidate_multiplier, int)
            or isinstance(rerank_candidate_multiplier, bool)
            or rerank_candidate_multiplier < 1
        ):
            raise ValueError("rerank_candidate_multiplier must be a positive integer.")
        self.chunks = chunks
        self._embedder = Embedder() if embedder is None else embedder
        self._index_builder = index_builder
        self._topic_detector = topic_detector
        self._candidate_filter = candidate_filter
        self._reranker = reranker
        self._rerank_candidate_multiplier = rerank_candidate_multiplier

        texts = [c.embed_text if c.embed_text else c.content for c in chunks]
        self.embeddings = (
            self._embedder.encode(texts)
            if texts
            else np.empty((0, 0), dtype=np.float32)
        )
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(chunks):
            raise ValueError(
                "Document embeddings must be a 2D matrix with one row per chunk."
            )
        try:
            self._index = self._index_builder(self.embeddings) if chunks else None
        except Exception as exc:
            raise RuntimeError(
                f"Could not build retrieval index for {len(chunks)} chunks "
                f"with embedding shape {self.embeddings.shape}."
            ) from exc

    def _indices_for_candidates(self, candidates: list[Chunk]) -> list[int] | None:
        indices_by_chunk: dict[int, list[int]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            indices_by_chunk[id(chunk)].append(index)

        offsets: dict[int, int] = defaultdict(int)
        candidate_indices = []
        for candidate in candidates:
            key = id(candidate)
            offset = offsets[key]
            if key not in indices_by_chunk or offset >= len(indices_by_chunk[key]):
                return None
            candidate_indices.append(indices_by_chunk[key][offset])
            offsets[key] += 1
        return candidate_indices

    def search(self, query: str, top_n: int = TOP_K) -> list[tuple[Chunk, float]]:
        """
        Topic-aware semantic search.

        Returns [(chunk, ranking_score), ...] sorted descending. Explanatory
        queries may have section-adjusted scores; scores are not confidence.
        """
        if top_n <= 0 or not self.chunks:
            return []

        q_lower = query.lower()
        matched_topic = self._topic_detector(q_lower)
        candidates, effective_topic = self._candidate_filter(self.chunks, matched_topic)
        if matched_topic and effective_topic is None:
            logger.info(
                "Topic filter for %r had insufficient candidates; searching full corpus.",
                matched_topic,
            )

        if not candidates:
            return []

        if candidates is self.chunks:
            cand_embeddings = self.embeddings
            tmp_index = self._index
        else:
            candidate_indices = self._indices_for_candidates(candidates)
            if candidate_indices is None:
                cand_texts = [
                    c.embed_text if c.embed_text else c.content
                    for c in candidates
                ]
                cand_embeddings = self._embedder.encode(cand_texts)
            else:
                cand_embeddings = self.embeddings[candidate_indices]
            if (
                cand_embeddings.ndim != 2
                or len(cand_embeddings) != len(candidates)
                or cand_embeddings.shape[1] != self.embeddings.shape[1]
            ):
                raise ValueError(
                    "Filtered candidate embeddings must align with candidate "
                    "chunks and the corpus embedding dimension."
                )
            try:
                tmp_index = self._index_builder(cand_embeddings)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not build retrieval index for {len(candidates)} "
                    f"filtered candidate chunks."
                ) from exc

        if tmp_index is None:
            return []

        q_vec = self._embedder.encode_one(query)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        if q_vec.ndim != 2 or q_vec.shape[1] != cand_embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimensions do not match indexed document embeddings."
            )

        candidate_pool_size = min(
            len(candidates),
            top_n * self._rerank_candidate_multiplier,
        )
        k = candidate_pool_size
        scores, indices = tmp_index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(candidates):
                results.append((candidates[idx], float(score)))

        return self._reranker(query, results)[:top_n]