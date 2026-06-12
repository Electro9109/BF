"""
retrieval/faiss_index.py - Thin wrapper around a FAISS IndexFlatIP.

Scores returned are cosine similarities (0.0 - 1.0) because all vectors
going in are normalised by retrieval/embeddings.py.
"""

import numpy as np


class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices), each shape (1, k)."""
        return self._index.search(query_vec, k)


def build_index(embeddings: np.ndarray) -> FaissIndex:
    return FaissIndex(embeddings)
