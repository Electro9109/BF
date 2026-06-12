"""
retrieval/embeddings.py - Thin wrapper around SentenceTransformer.

Isolating this means swapping all-MiniLM-L6-v2 for BAAI/bge-small-en-v1.5
(the A/B test discussed earlier) touches only this file and
config/models.py.
"""

from pathlib import Path

import numpy as np

from config.models import EMBED_MODEL_NAME
from config.paths import EMBED_MODEL_DIR


class Embedder:
    """Loads once, encodes many times. normalize_embeddings=True throughout
    so that inner product == cosine similarity downstream in FAISS."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        embed_path = Path(EMBED_MODEL_DIR)
        model_source = str(embed_path) if embed_path.exists() else EMBED_MODEL_NAME
        self._model = SentenceTransformer(model_source)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])
