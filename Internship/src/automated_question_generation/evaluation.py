"""Readability and semantic-similarity evaluation of generated survey items."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_groups(
    data: pd.DataFrame,
    value_column: str,
    group_column: str = "method",
) -> tuple[object, pd.DataFrame]:
    """Run Kruskal-Wallis followed by pairwise Dunn tests with Holm correction."""
    import scikit_posthocs as sp
    from scipy.stats import kruskal

    groups = [
        group[value_column].dropna().to_numpy()
        for _, group in data.groupby(group_column, sort=False)
    ]
    if len(groups) < 2 or any(len(group) == 0 for group in groups):
        raise ValueError("At least two non-empty groups are required.")

    omnibus = kruskal(*groups)
    posthoc = sp.posthoc_dunn(
        data,
        val_col=value_column,
        group_col=group_column,
        p_adjust="holm",
    )
    return omnibus, posthoc


def evaluate_readability(
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """Calculate Flesch Reading Ease and compare item-generation methods."""
    import textstat

    scored = items[["method", "item_text"]].dropna().copy()
    scored["flesch_reading_ease"] = scored["item_text"].map(
        textstat.flesch_reading_ease
    )
    kruskal_result, dunn_result = compare_groups(
        scored,
        value_column="flesch_reading_ease",
    )
    return scored, kruskal_result, dunn_result


def evaluate_semantic_similarity(
    items: pd.DataFrame,
    target_prompt: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """Compare each item's cosine similarity to the same target prompt."""
    from sentence_transformers import SentenceTransformer

    scored = items[["method", "item_text"]].dropna().copy()
    model = SentenceTransformer(model_name)
    item_embeddings = model.encode(
        scored["item_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    prompt_embedding = model.encode(
        [target_prompt],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scored["semantic_similarity"] = np.asarray(item_embeddings) @ prompt_embedding
    kruskal_result, dunn_result = compare_groups(
        scored,
        value_column="semantic_similarity",
    )
    return scored, kruskal_result, dunn_result
