"""Create local Ollama embeddings and a cosine-similarity FAISS index."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def embed_text(text: str, model_name: str = "nomic-embed-text") -> np.ndarray:
    """Request one embedding from a locally running Ollama server."""
    import ollama

    response = ollama.embeddings(model=model_name, prompt=str(text))
    return np.asarray(response["embedding"], dtype="float32")


def embed_items(
    texts: Iterable[str],
    model_name: str = "nomic-embed-text",
) -> np.ndarray:
    """Embed all item texts as one float32 matrix."""
    return np.vstack([embed_text(text, model_name) for text in texts])


def build_faiss_index(embeddings: np.ndarray):
    """Normalize vectors and index them; inner product then equals cosine similarity."""
    import faiss

    matrix = np.asarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index

