"""Minimal lexical and semantic deduplication of a survey item bank."""

from __future__ import annotations

import pandas as pd


def normalize_text(text: object) -> str:
    """Normalize text for exact and fuzzy comparison."""
    return " ".join(str(text).lower().split())


def fuzzy_deduplicate(
    items: pd.DataFrame,
    text_column: str = "item_text",
    threshold: int = 95,
) -> pd.DataFrame:
    """Keep the first item when two normalized strings are very similar."""
    from thefuzz import fuzz

    kept_indices: list[object] = []
    seen: list[str] = []

    for index, text in items[text_column].items():
        normalized = normalize_text(text)
        if not any(fuzz.token_sort_ratio(normalized, previous) >= threshold for previous in seen):
            kept_indices.append(index)
            seen.append(normalized)

    return items.loc[kept_indices].reset_index(drop=True)


def semantic_deduplicate(
    items: pd.DataFrame,
    text_column: str = "item_text",
    threshold: float = 0.90,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Keep one representative from each high-similarity embedding cluster."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        items[text_column].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    clusters = util.community_detection(
        embeddings,
        min_community_size=2,
        threshold=threshold,
    )
    duplicate_positions = {position for cluster in clusters for position in cluster[1:]}
    keep_mask = [position not in duplicate_positions for position in range(len(items))]
    return items.loc[keep_mask].reset_index(drop=True)


def clean_item_bank(
    items: pd.DataFrame,
    text_column: str = "item_text",
    fuzzy_threshold: int = 95,
    semantic_threshold: float | None = 0.90,
) -> pd.DataFrame:
    """Apply missing-value, exact, fuzzy, and optional semantic cleaning."""
    cleaned = items.dropna(subset=[text_column]).copy()
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[text_column].ne("")]
    cleaned["_normalized"] = cleaned[text_column].map(normalize_text)
    cleaned = cleaned.drop_duplicates("_normalized").drop(columns="_normalized")
    cleaned = fuzzy_deduplicate(cleaned, text_column, fuzzy_threshold)

    # Set semantic_threshold=None for a lightweight lexical-only pass.
    if semantic_threshold is not None and len(cleaned) > 1:
        cleaned = semantic_deduplicate(
            cleaned,
            text_column=text_column,
            threshold=semantic_threshold,
        )
    return cleaned


if __name__ == "__main__":
    bank = pd.read_csv("data/item_bank.csv")
    clean_item_bank(bank).to_csv("data/clean_item_bank.csv", index=False)

