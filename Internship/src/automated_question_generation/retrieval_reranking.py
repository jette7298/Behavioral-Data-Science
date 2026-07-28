"""Retrieve survey items with FAISS and rerank them with a cross-encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .embedding_index import embed_text
except ImportError:  # Allows direct execution from this directory.
    from embedding_index import embed_text


def retrieve_items(
    query: str,
    items: pd.DataFrame,
    index,
    n_candidates: int = 25,
    embedding_model: str = "nomic-embed-text",
) -> pd.DataFrame:
    """Return the highest-cosine-similarity candidates from the FAISS index."""
    import faiss

    query_vector = embed_text(query, embedding_model).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    scores, positions = index.search(query_vector, min(n_candidates, len(items)))

    retrieved = items.iloc[positions[0]].copy()
    retrieved["retrieval_score"] = scores[0]
    return retrieved.reset_index(drop=True)


def retrieve_and_rerank(
    query: str,
    items: pd.DataFrame,
    index,
    cross_encoder,
    n_candidates: int = 25,
    n_final: int = 8,
) -> pd.DataFrame:
    """Rerank dense-retrieval candidates using query-item relevance scores."""
    candidates = retrieve_items(query, items, index, n_candidates)
    pairs = [(query, text) for text in candidates["item_text"].tolist()]
    candidates["reranking_score"] = np.asarray(cross_encoder.predict(pairs))
    return (
        candidates.sort_values("reranking_score", ascending=False)
        .head(n_final)
        [["item_text", "retrieval_score", "reranking_score"]]
        .reset_index(drop=True)
    )

