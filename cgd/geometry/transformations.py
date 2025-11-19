# cgd/geometry/transformations.py
"""Core geometric mappings between the probability simplex and tangent space.

This module provides the logarithmic and exponential maps, which are fundamental
operations for navigating the geometry of the probability simplex. These functions
allow for the conversion between points on the simplex (probability distributions)
and vectors in the tangent space at the simplex's origin (effect vectors).

Note:
    This module only implements mappings relative to the chaos origin O, as
    this is the canonical reference point in CGD theory.
"""
import numpy as np
from numpy.typing import NDArray

from .simplex import get_chaos_origin

# A small epsilon for numerical stability
EPSILON = 1e-12


def log_map_from_origin(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the logarithmic map from the chaos origin O to a point p.

    This function calculates the "effect vector" v that corresponds to the
    geodesic path starting at the origin O and ending at the probability
    distribution p. This vector v lies in the tangent space T_O at the origin.
    In essence, it linearizes the position p relative to the origin.

    v = Log_O(p)

    Args:
        p (NDArray[np.float64]): The target probability distribution on the simplex.

    Returns:
        NDArray[np.float64]: The corresponding effect vector v in the tangent space T_O.
            This vector is guaranteed to be zero-sum.
    """
    p_arr: NDArray[np.float64] = np.asarray(p, dtype=float)
    K: int = len(p_arr)
    origin: NDArray[np.float64] = get_chaos_origin(K)

    # Cosine of the angle is related to the dot product on the hypersphere
    dot_product: float = np.sum(np.sqrt(origin * p_arr))
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # The angle theta is half the Fisher-Rao distance
    theta: float = np.arccos(dot_product)

    # If p is at the origin, the tangent vector is a zero vector
    if theta < EPSILON:
        return np.zeros(K)

    # Project sqrt(p) onto the tangent space at sqrt(origin)
    u_raw: NDArray[np.float64] = np.sqrt(p_arr) - dot_product * np.sqrt(origin)
    norm_u: float = np.linalg.norm(u_raw)

    if norm_u < EPSILON:
        return np.zeros(K)

    # The final tangent vector v, scaled and transformed back to simplex coordinates
    v: NDArray[np.float64] = (2 * theta / norm_u) * u_raw / np.sqrt(origin)

    # Enforce the zero-sum constraint to correct for floating-point errors
    return v - np.mean(v)


def exp_map_from_origin(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the exponential map from the origin O along an effect vector v.

    This function "follows" the geodesic path starting from the origin O in
    the direction and length specified by the effect vector v to find the
    resulting point p on the probability simplex. It is the inverse operation
    of the logarithmic map.

    p = Exp_O(v)

    Args:
        v (NDArray[np.float64]): An effect vector in the tangent space T_O at the origin.

    Returns:
        NDArray[np.float64]: The resulting probability distribution p on the simplex.
    """
    v_arr: NDArray[np.float64] = np.asarray(v, dtype=float)
    K: int = len(v_arr)
    origin: NDArray[np.float64] = get_chaos_origin(K)

    # First, transform the effect vector v from simplex coordinates to sphere coordinates
    u: NDArray[np.float64] = v_arr * np.sqrt(origin)
    norm_u: float = np.linalg.norm(u)

    # If the effect vector has no magnitude, the destination is the origin
    if norm_u < EPSILON:
        return origin

    # The geodesic equation in terms of square-root representation
    # The length of the path on the sphere is ||u||/2
    sqrt_p: NDArray[np.float64] = np.cos(norm_u / 2.0) * np.sqrt(origin) + np.sin(
        norm_u / 2.0
    ) * (u / norm_u)

    # Recover the probability by squaring and ensuring normalization
    p_final: NDArray[np.float64] = sqrt_p**2
    return p_final / np.sum(p_final)
