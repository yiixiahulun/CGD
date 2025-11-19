# cgd/geometry/simplex.py
"""Functions and classes related to the geometry of the probability simplex.

This module provides standalone functions for calculating key geometric
properties of the (K-1)-dimensional probability simplex, such as its origin,
radius, and the Fisher-Rao distance metric. It also provides a convenience
`Simplex` class to encapsulate these properties for a given dimensionality K.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------
# Standalone geometric functions
# ----------------------------------------------------


@lru_cache(maxsize=32)
def get_chaos_origin(K: int) -> NDArray[np.float64]:
    """Returns the chaos origin O (uniform distribution) for dimension K.

    The chaos origin is the barycenter of the simplex, representing a state
    of maximum uncertainty or uniform probability.

    Args:
        K (int): The dimensionality of the simplex (number of atomic events).

    Returns:
        NDArray[np.float64]: A (K,)-dimensional NumPy array where each element is 1/K.

    Raises:
        ValueError: If K is not a positive integer.
    """
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"Dimensionality K must be a positive integer, but got {K}.")
    return np.full(K, 1.0 / K)


@lru_cache(maxsize=32)
def get_radius(K: int) -> float:
    """Returns the maximum radius D_max(K) of a universe of dimension K.

    The radius is defined as the geodesic distance (Fisher-Rao distance) from
    the chaos origin to any of the simplex's vertices (pure states).

    Args:
        K (int): The dimensionality of the simplex.

    Returns:
        float: The geodesic radius of the simplex. Returns 0.0 for K=1.

    Raises:
        ValueError: If K is not a positive integer.
    """
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"Dimensionality K must be a positive integer, but got {K}.")
    if K == 1:
        return 0.0
    return 2 * cast(float, np.arccos(np.sqrt(1.0 / K)))


def distance_FR(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Calculates the Fisher-Rao distance between two probability distributions.

    The Fisher-Rao distance is the geodesic distance on the manifold of the
    probability simplex. It measures the shortest path between two points (p, q)
    on the surface of the hypersphere onto which the simplex is projected.

    Args:
        p (NDArray[np.float64]): The first probability vector.
        q (NDArray[np.float64]): The second probability vector.

    Returns:
        float: The geodesic distance between p and q.

    Raises:
        ValueError: If the shapes of p and q do not match.
    """
    p_arr: NDArray[np.float64] = np.asarray(p, dtype=float)
    q_arr: NDArray[np.float64] = np.asarray(q, dtype=float)

    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"Input vector shapes do not match: {p_arr.shape} vs {q_arr.shape}"
        )

    # Ensure vectors are normalized and non-negative to avoid domain errors
    p_sum: float = np.sum(p_arr)
    q_sum: float = np.sum(q_arr)
    if not (np.isclose(p_sum, 1.0) and np.isclose(q_sum, 1.0)):
        p_arr = p_arr / p_sum
        q_arr = q_arr / q_sum

    p_arr[p_arr < 0] = 0
    q_arr[q_arr < 0] = 0

    # Calculate the dot product of the square roots of the vectors
    dot_product: float = np.sum(np.sqrt(p_arr * q_arr))

    # Clip the dot product to handle potential floating-point inaccuracies
    dot_product = np.clip(dot_product, -1.0, 1.0)

    distance: float = 2 * cast(float, np.arccos(dot_product))
    return distance


# ----------------------------------------------------
# Simplex class for convenience
# ----------------------------------------------------


@dataclass(frozen=True)
class Simplex:
    """A convenience dataclass for storing geometric constants of a simplex.

    This class serves as an attribute of the `Universe` class, providing quick
    access to dimension-dependent geometric constants and leveraging caching
    to avoid redundant calculations.

    Attributes:
        K (int): The dimensionality of the simplex (number of atomic events).
    """

    K: int

    @property
    def origin(self) -> NDArray[np.float64]:
        """The coordinates of the chaos origin O."""
        return get_chaos_origin(self.K)

    @property
    def radius(self) -> float:
        """The maximum radius D_max(K) of the simplex."""
        return get_radius(self.K)

    def __repr__(self) -> str:
        """Provides a concise string representation of the Simplex."""
        return f"Simplex(K={self.K}, radius={self.radius:.4f})"
