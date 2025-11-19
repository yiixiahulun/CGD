# cgd/geometry/_jax_impl.py
"""JAX-native implementations of core geometric functions.

This module provides JIT-compiled, JAX-native versions of the geometric
functions defined in `simplex.py` and `transformations.py`. These functions
operate on JAX arrays and are designed to be composed and differentiated
within the JAX ecosystem.
"""
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import jit, partial

EPSILON = 1e-12


@partial(jit, static_argnames=("K",))  # type: ignore[misc]
def get_chaos_origin_jax(K: int) -> jax.Array:
    """JAX version of get_chaos_origin, JIT-compiled.

    Args:
        K (int): The dimensionality of the simplex (static argument).

    Returns:
        jax.Array: A JAX array of shape (K,) representing the uniform distribution.
    """
    return jnp.full(K, 1.0 / K)


@jit  # type: ignore[misc]
def distance_FR_jax(p: jax.Array, q: jax.Array) -> float:
    """JAX version of Fisher-Rao distance, JIT-compiled.

    Args:
        p (jax.Array): The first probability vector.
        q (jax.Array): The second probability vector.

    Returns:
        float: The scalar geodesic distance between p and q.
    """
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    p_safe: jax.Array = jnp.maximum(p, 0)
    q_safe: jax.Array = jnp.maximum(q, 0)
    dot_product: float = jnp.clip(jnp.sum(jnp.sqrt(p_safe * q_safe)), -1.0, 1.0)
    return 2 * cast(float, jnp.arccos(dot_product))


@jit  # type: ignore[misc]
def log_map_from_origin_jax(p: jax.Array) -> jax.Array:
    """JAX version of the logarithmic map from the origin, JIT-compiled.

    This version infers the dimensionality K from the input array `p` and uses
    `jax.lax.cond` to handle the edge case where p is at the origin, ensuring
    the function is traceable by the JIT compiler.

    Args:
        p (jax.Array): The target probability distribution on the simplex.

    Returns:
        jax.Array: The corresponding effect vector v in the tangent space.
    """
    K: int = p.shape[0]
    origin: jax.Array = get_chaos_origin_jax(K)
    p_normalized: jax.Array = p / jnp.sum(p)
    dot_product: float = jnp.clip(jnp.sum(jnp.sqrt(origin * p_normalized)), -1.0, 1.0)
    theta: float = jnp.arccos(dot_product)

    def compute_v(_: Any) -> jax.Array:
        """Computes v when p is not at the origin."""
        u_raw: jax.Array = jnp.sqrt(p_normalized) - dot_product * jnp.sqrt(origin)
        norm_u: float = jnp.linalg.norm(u_raw)

        def u_is_nonzero(_: Any) -> jax.Array:
            v_raw: jax.Array = (2 * theta / norm_u) * u_raw / jnp.sqrt(origin)
            return v_raw - jnp.mean(v_raw)

        def u_is_zero(_: Any) -> jax.Array:
            return jnp.zeros_like(p)

        return jax.lax.cond(norm_u < EPSILON, u_is_zero, u_is_nonzero, operand=None)

    def return_zero_vector(_: Any) -> jax.Array:
        """Returns a zero vector when p is at the origin."""
        return jnp.zeros_like(p)

    return jax.lax.cond(theta < EPSILON, return_zero_vector, compute_v, operand=None)


@jit  # type: ignore[misc]
def exp_map_from_origin_jax(v: jax.Array) -> jax.Array:
    """JAX version of the exponential map from the origin, JIT-compiled.

    This version infers dimensionality K from the input vector `v` and uses
    `jax.lax.cond` to handle the edge case of a zero-length effect vector,
    making it fully JIT-compatible.

    Args:
        v (jax.Array): An effect vector in the tangent space at the origin.

    Returns:
        jax.Array: The resulting probability distribution p on the simplex.
    """
    K: int = v.shape[0]
    origin: jax.Array = get_chaos_origin_jax(K)
    v_centered: jax.Array = v - jnp.mean(v)
    u: jax.Array = v_centered * jnp.sqrt(origin)
    norm_u: float = jnp.linalg.norm(u)

    def v_is_nonzero(_: Any) -> jax.Array:
        """Computes p when v has a non-zero magnitude."""
        sqrt_p: jax.Array = jnp.cos(norm_u / 2.0) * jnp.sqrt(origin) + jnp.sin(
            norm_u / 2.0
        ) * (u / norm_u)
        p_final: jax.Array = sqrt_p**2
        return p_final / jnp.sum(p_final)

    def v_is_zero(_: Any) -> jax.Array:
        """Returns the origin when v is a zero vector."""
        return origin

    return jax.lax.cond(norm_u < EPSILON, v_is_zero, v_is_nonzero, operand=None)
