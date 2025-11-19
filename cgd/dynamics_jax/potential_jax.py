# cgd/dynamics_jax/potential_jax.py
"""JAX implementation of the Gaussian potential for CGD physics.

This module provides a JAX-native, JIT-compiled function to calculate the
total potential energy at a point in the probability simplex. The inputs are
JAX arrays, making the function suitable for integration into larger JAX
computational graphs and enabling features like automatic differentiation.
"""
from functools import partial
from typing import Callable, cast

import jax
import jax.numpy as jnp
from jax import jit

from ..geometry._jax_impl import distance_FR_jax


@partial(jit, static_argnames=("K",))
def potential_gaussian_jax(
    p: jax.Array,
    sources_p_stim: jax.Array,
    sources_strength: jax.Array,
    alpha: float,
    radius: float,
    K: int,
) -> float:
    """Calculates the total Gaussian potential energy at a point p (JAX version).

    This function is JIT-compiled for performance. It computes the sum of
    potentials from all sources, where each source's contribution is a
    Gaussian function of the geodesic Fisher-Rao distance. The calculation
    is performed iteratively over the sources using `jax.lax.fori_loop` for
    efficiency within a compiled context.

    Args:
        p (jax.Array): The point in the simplex (a probability distribution)
            at which to calculate the potential.
        sources_p_stim (jax.Array): A JAX array of shape (n_sources, K)
            containing the stimulus locations (p_stim) for all sources.
        sources_strength (jax.Array): A JAX array of shape (n_sources,)
            containing the strengths for all sources.
        alpha (float): The universe's alpha parameter, controlling the range
            of the potential.
        radius (float): The radius of the K-dimensional simplex.
        K (int): The dimensionality of the universe. This is a static argument
            for the JIT compiler.

    Returns:
        float: The total scalar potential energy at point p.
    """
    # Ensure radius is non-zero to prevent division by zero
    radius = jnp.maximum(radius, 1e-12)

    def body_fun(i: int, current_potential: float) -> float:
        """Calculates the potential for one source and adds it to the total."""
        p_stim: jax.Array = sources_p_stim[i]
        strength: float = sources_strength[i]

        relative_dist: float = distance_FR_jax(p, p_stim) / radius
        potential_s: float = -strength * jnp.exp(-((alpha * relative_dist) ** 2))
        return current_potential + potential_s

    # Efficiently loop over all sources within the JIT-compiled function
    total_potential: float = cast(
        float, jax.lax.fori_loop(0, len(sources_strength), body_fun, 0.0)
    )
    return total_potential
