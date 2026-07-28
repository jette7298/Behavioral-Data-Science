"""Core lexical, semantic, and sentiment measures for open-ended responses."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd


def tokenize(text: str) -> list[str]:
    """Apply a transparent lowercase word tokenizer."""
    return re.findall(r"\b[\w']+\b", str(text).lower(), flags=re.UNICODE)


def mattr(tokens: list[str], window_size: int = 50) -> float:
    """Calculate the mean type-token ratio across moving windows."""
    if not tokens:
        return math.nan
    if len(tokens) <= window_size:
        return len(set(tokens)) / len(tokens)

    scores = [
        len(set(tokens[start : start + window_size])) / window_size
        for start in range(len(tokens) - window_size + 1)
    ]
    return float(np.mean(scores))


def shannon_entropy(tokens: list[str]) -> float:
    """Calculate word-level Shannon entropy in bits."""
    if not tokens:
        return math.nan
    counts = np.asarray(list(Counter(tokens).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def encode_responses(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """Generate normalized multilingual sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def semantic_similarity(
    responses: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every cross-source pair and each synthetic response's nearest human."""
    human_mask = responses["source"].eq("human").to_numpy()
    synthetic_mask = responses["source"].eq("synthetic").to_numpy()
    human = responses.loc[human_mask].reset_index(drop=True)
    synthetic = responses.loc[synthetic_mask].reset_index(drop=True)

    similarities = embeddings[synthetic_mask] @ embeddings[human_mask].T
    pair_rows = [
        {
            "synthetic_id": synthetic.loc[i, "respondent_id"],
            "human_id": human.loc[j, "respondent_id"],
            "cosine_similarity": float(similarities[i, j]),
        }
        for i, j in np.ndindex(similarities.shape)
    ]
    pairwise = pd.DataFrame(pair_rows)

    nearest_positions = similarities.argmax(axis=1)
    nearest = synthetic[["respondent_id", "response_text"]].copy()
    nearest = nearest.rename(
        columns={
            "respondent_id": "synthetic_id",
            "response_text": "synthetic_response",
        }
    )
    nearest["nearest_human_id"] = human.loc[
        nearest_positions, "respondent_id"
    ].to_numpy()
    nearest["nearest_human_response"] = human.loc[
        nearest_positions, "response_text"
    ].to_numpy()
    nearest["cosine_similarity"] = similarities.max(axis=1)
    return pairwise, nearest


def analyze_sentiment(
    texts: list[str],
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
) -> pd.DataFrame:
    """Classify multilingual sentiment and provide a signed confidence score."""
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model=model_name)
    outputs = classifier(texts, truncation=True, max_length=512)
    rows = []
    for text, output in zip(texts, outputs):
        label = str(output["label"])
        confidence = float(output["score"])
        normalized_label = label.lower()
        signed_score = (
            confidence
            if "pos" in normalized_label or normalized_label == "label_2"
            else -confidence
            if "neg" in normalized_label or normalized_label == "label_0"
            else 0.0
        )
        rows.append(
            {
                "response_text": text,
                "sentiment_label": label,
                "model_confidence": confidence,
                "sentiment_score": signed_score,
            }
        )
    return pd.DataFrame(rows)
