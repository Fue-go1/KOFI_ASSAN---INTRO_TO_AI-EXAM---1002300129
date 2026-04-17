# Student: Kofi Assan | Index: 1002300129 | CS4241-Introduction to Artificial Intelligence
"""Local embedding pipeline using sentence-transformers (no LangChain)."""
from __future__ import annotations

import os

import numpy as np

_MODEL = None
_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
