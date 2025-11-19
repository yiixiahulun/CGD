# tests/helpers.py
"""
Core test helper functions for the cgd test suite.
"""
from typing import List, Dict, Literal
import numpy as np
import pandas as pd

from cgd.core.source import GravitationalSource
from cgd.core.universe import Universe
from cgd.geometry.transformations import log_map_from_origin

def create_test_source(
    p: np.ndarray,
    name: str,
    v_type: Literal['absolute', 'substitution'],
    labels: List[str]
) -> GravitationalSource:
    """
    Safely creates a GravitationalSource object for testing from a probability vector.

    This ensures that the effect vector v_eigen is derived from a valid probability
    distribution on the simplex using the correct geometric transformation.

    Args:
        p: The probability vector (must sum to 1).
        name: The name of the source.
        v_type: The theoretical type of the source ('absolute' or 'substitution').
        labels: The coordinate system labels.

    Returns:
        A GravitationalSource instance.
    """
    # Arrange: Ensure p is a valid probability vector
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Input probability vector p must sum to 1.")

    # Act: Calculate v_eigen using the log map
    v_eigen = log_map_from_origin(p)

    # Assert (implicit): Create and return the source object
    return GravitationalSource(
        name=name,
        v_type=v_type,
        labels=tuple(labels),
        v_eigen=v_eigen
    )

def create_test_universe(
    K: int,
    alpha: float,
    backend: Literal['numpy', 'jax'] = 'numpy'
) -> Universe:
    """
    Factory function to quickly create a Universe instance for testing.

    Args:
        K: The dimensionality of the universe.
        alpha: The alpha parameter of the universe.
        backend: The computation backend ('numpy' or 'jax').

    Returns:
        A Universe instance.
    """
    return Universe(K=K, alpha=alpha, backend=backend)

def generate_mock_data(
    groups: Dict[str, Dict[str, int]],
    event_labels: List[str],
    group_col: str = 'group',
    event_col: str = 'event'
) -> pd.DataFrame:
    """
    Generates a mock pandas.DataFrame mimicking experimental count data.

    Args:
        groups: A dictionary where keys are group names (e.g., 'Stage', 'Stim_A')
                and values are dictionaries mapping event labels to their counts.
        event_labels: The full list of possible event labels.
        group_col: The name for the group column in the output DataFrame.
        event_col: The name for the event column in the output DataFrame.

    Returns:
        A pandas.DataFrame in the format expected by workflow functions.
    """
    # Arrange
    records = []

    # Act
    for group_name, counts in groups.items():
        for event, count in counts.items():
            if event not in event_labels:
                raise ValueError(f"Event '{event}' in group '{group_name}' is not in event_labels.")
            # Add 'count' number of rows for this event
            for _ in range(count):
                records.append({group_col: group_name, event_col: event})

    # Assert (implicit): Return the created DataFrame
    return pd.DataFrame(records)
