"""
retrieval/retriever.py - Orchestrates topic filtering, FAISS search, and
section reranking. This is the only retrieval class other layers import.

REPAIR NOTES
─────────────
The previous version computed `cand_texts` from the filtered
`candidates` list, but then built/searched the FAISS index using
`self.embeddings` (the FULL corpus embeddings) - so even if topic
filtering correctly narrowed `candidates`, the search still ran over
every chunk and `indices` returned positions into the full corpus that
were then used to index into `candidates` (wrong list -> wrong chunks,
or IndexError once corpus > candidates in size).

This version re-encodes the candidate subset and builds a temporary
index over exactly that subset, restoring the v5 behaviour. Re-encoding
per query is fine at corpus sizes < ~1000 chunks (see docstring below);
if that becomes a bottleneck, switch to slicing self.embeddings by
precomputed candidate indices instead of re-encoding.
"""

import numpy as np

from config.retrieval import MIN_TOPIC_CANDIDATES, TOP_K
from data.schemas import Chunk
from retrieval.embeddings import Embedder
from retrieval.faiss_index import build_index
from retrieval.reranker import rerank_by_section
from retrieval.topic_filter import detect_topic


class RetrievalEngine:
    """
    Semantic retrieval using sentence-transformers + FAISS.

    Scores returned are cosine similarities (0.0 - 1.0).
    Typical values:
        Strong match   : 0.65 - 0.90
        Moderate match : 0.40 - 0.65
        Weak / noise   : < 0.35

    If scores cluster below 0.35 for all results, the query vocabulary
    does not match the corpus - either rephrase the query or restructure
    the relevant .txt file.

    Pipeline per query
    ──────────────────
    1. detect_topic()      -> restrict candidates to the matched topic (or all)
    2. FAISS search         -> cosine similarity over embed_text vectors of
                                the candidate subset
    3. rerank_by_section()  -> small multiplier nudge based on section type
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._embedder = Embedder()

        texts = [c.embed_text if c.embed_text else c.content for c in chunks]
        self.embeddings = self._embedder.encode(texts)
        self._index = build_index(self.embeddings)

    def search(self, query: str, top_n: int = TOP_K) -> list[tuple[Chunk, float]]:
        """
        Topic-aware semantic search.

        Returns [(chunk_dict, cosine_score), ...] sorted descending.
        """
        q_lower = query.lower()
        matched_topic = detect_topic(q_lower)

        if matched_topic:
            candidates = [c for c in self.chunks if c.topic == matched_topic]
            if len(candidates) < MIN_TOPIC_CANDIDATES:
                print(
                    f"[RAG] WARN: topic filter '{matched_topic}' returned "
                    f"{len(candidates)} chunk(s) - falling back to full corpus."
                )
                candidates = self.chunks
        else:
            candidates = self.chunks

        if not candidates:
            return []

        # Build a temporary index over exactly the candidate subset, so
        # FAISS indices map back to `candidates` correctly.
        if candidates is self.chunks:
            cand_embeddings = self.embeddings
        else:
            cand_texts = [c.embed_text if c.embed_text else c.content for c in candidates]
            cand_embeddings = self._embedder.encode(cand_texts)

        tmp_index = build_index(cand_embeddings)

        q_vec = self._embedder.encode_one(query)

        k = min(top_n, len(candidates))
        scores, indices = tmp_index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((candidates[idx], float(score)))

        results = rerank_by_section(query, results)
        return results