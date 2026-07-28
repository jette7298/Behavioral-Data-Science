"""Categorical comparisons between human and synthetic respondents."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def bootstrap_percentage_gap_ci(
    human: Iterable[bool | int],
    synthetic: Iterable[bool | int],
    iterations: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap the human-minus-synthetic percentage-point difference."""
    human_values = np.asarray(list(human), dtype=float)
    synthetic_values = np.asarray(list(synthetic), dtype=float)
    rng = np.random.default_rng(random_state)

    human_draws = rng.choice(
        human_values,
        size=(iterations, len(human_values)),
        replace=True,
    ).mean(axis=1)
    synthetic_draws = rng.choice(
        synthetic_values,
        size=(iterations, len(synthetic_values)),
        replace=True,
    ).mean(axis=1)
    gaps = 100 * (human_draws - synthetic_draws)
    alpha = (1 - confidence) / 2
    return tuple(np.quantile(gaps, [alpha, 1 - alpha]))


def chi_square_test(
    data: pd.DataFrame,
    variable: str,
    source_column: str = "source",
) -> dict[str, float | int | str]:
    """Test whether one categorical distribution differs by data source."""
    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(data[source_column], data[variable])
    statistic, p_value, degrees_of_freedom, _ = chi2_contingency(contingency)
    return {
        "variable": variable,
        "chi_square": float(statistic),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
    }


def compare_categorical_variables(
    data: pd.DataFrame,
    variables: list[str],
    source_column: str = "source",
) -> pd.DataFrame:
    """Run chi-square tests and control FDR with Benjamini-Hochberg."""
    from statsmodels.stats.multitest import multipletests

    results = pd.DataFrame(
        [chi_square_test(data, variable, source_column) for variable in variables]
    )
    results["p_value_adjusted"] = multipletests(
        results["p_value"],
        method="fdr_bh",
    )[1]
    return results
