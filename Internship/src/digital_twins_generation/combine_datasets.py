"""Align anonymized human and synthetic response tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_source(
    data: pd.DataFrame,
    source: str,
    id_column: str,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    """Rename selected columns and add common respondent metadata."""
    selected = data[[id_column, *column_mapping.keys()]].rename(columns=column_mapping)
    selected.insert(0, "source", source)
    selected.insert(
        0,
        "respondent_id",
        source + "_" + data[id_column].astype(str),
    )
    return selected.drop(columns=id_column)


def combine_datasets(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    human_mapping: dict[str, str],
    synthetic_mapping: dict[str, str],
    human_id: str = "participant_id",
    synthetic_id: str = "persona_id",
) -> pd.DataFrame:
    """Map both sources to a shared schema and stack their rows."""
    human_common = prepare_source(human, "human", human_id, human_mapping)
    synthetic_common = prepare_source(
        synthetic,
        "synthetic",
        synthetic_id,
        synthetic_mapping,
    )
    shared_columns = list(
        dict.fromkeys([*human_common.columns, *synthetic_common.columns])
    )
    return pd.concat(
        [
            human_common.reindex(columns=shared_columns),
            synthetic_common.reindex(columns=shared_columns),
        ],
        ignore_index=True,
    )


def save_combined(data: pd.DataFrame, output_path: str | Path) -> None:
    """Store the aligned table without writing the source datasets."""
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

