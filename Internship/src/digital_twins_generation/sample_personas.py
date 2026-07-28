"""Reproducible random or stratified sampling from a persona table."""

from __future__ import annotations

import pandas as pd


def sample_personas(
    personas: pd.DataFrame,
    n: int,
    stratify_by: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample personas, optionally distributing the sample across strata."""
    if n <= 0 or n > len(personas):
        raise ValueError("n must be between 1 and the number of available personas.")

    if stratify_by is None:
        return personas.sample(n=n, random_state=random_state).reset_index(drop=True)

    # Allocate approximately equal sample fractions to every stratum.
    sampled = (
        personas.groupby(stratify_by, group_keys=False)
        .sample(frac=n / len(personas), random_state=random_state)
    )

    # Rounding per group can miss the exact target; fill from unused rows.
    if len(sampled) < n:
        remaining = personas.drop(index=sampled.index)
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(n=n - len(sampled), random_state=random_state),
            ]
        )
    elif len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)

    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

