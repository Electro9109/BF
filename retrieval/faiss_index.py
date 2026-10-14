"""
retrieval/faiss_index.py - Thin wrapper around a FAISS IndexFlatIP.

The configured metric is inner product. Embedder normalizes vectors, so the
raw inner products are cosine similarities (not calibrated confidence).
"""

import numpy as np


class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or not len(embeddings) or not embeddings.shape[1]:
            raise ValueError("FAISS embeddings must be a non-empty 2D matrix.")
        if not np.isfinite(embeddings).all():
            raise ValueError("FAISS embeddings must contain only finite values.")

        self._size = len(embeddings)
        self._dimension = embeddings.shape[1]
        dim = self._dimension
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return raw inner-product scores and indices, capped to indexed rows."""
        if k <= 0:
            return (
                np.empty((1, 0), dtype=np.float32),
                np.empty((1, 0), dtype=np.int64),
            )
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.ndim != 2 or query_vec.shape != (1, self._dimension):
            raise ValueError(
                f"Query vector must have shape (1, {self._dimension})."
            )
        if not np.isfinite(query_vec).all():
            raise ValueError("Query vector must contain only finite values.")
        return self._index.search(query_vec, min(k, self._size))


def build_index(embeddings: np.ndarray) -> FaissIndex:
    return FaissIndex(embeddings)
