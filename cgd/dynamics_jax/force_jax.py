# cgd/dynamics_jax/force_jax.py
"""JAX implementation of the net force calculation using automatic differentiation.

This module leverages JAX's `grad` transformation to compute the net force
vector F_net precisely and efficiently. By defining the force as the negative
gradient of the potential, this approach avoids the numerical inaccuracies and
computational overhead of manual finite difference methods.
"""
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, jit

from cgd.dynamics_jax.potential_jax import potential_gaussian_jax

# --- 1. Automatically derive the gradient function using jax.grad ---
# Here, we create a new function, `grad_potential_fn`, which is the gradient
# of `potential_gaussian_jax` with respect to its first argument (`p`).
# JAX handles the entire differentiation process under the hood.
grad_potential_fn: Callable[..., jax.Array] = grad(potential_gaussian_jax, argnums=0)

# --- 2. Compile the final force calculation function ---


@partial(jit, static_argnames=("K",))
def calculate_F_net_jax(
    p: jax.Array,
    sources_p_stim: jax.Array,
    sources_strength: jax.Array,
    alpha: float,
    radius: float,
    K: int,
) -> jax.Array:
    """Calculates the net force F_net at a point p using automatic differentiation.

    This function computes the force F = -∇U by directly calling the
    auto-generated gradient function `grad_potential_fn`. The resulting gradient
    vector is then projected onto the tangent space of the simplex to ensure it
    is a valid (zero-sum) force vector. The entire function is JIT-compiled
    for optimal performance.

    Args:
        p (jax.Array): The point in the simplex at which to calculate the force.
        sources_p_stim (jax.Array): A JAX array of shape (n_sources, K)
            containing the stimulus locations (p_stim) for all sources.
        sources_strength (jax.Array): A JAX array of shape (n_sources,)
            containing the strengths for all sources.
        alpha (float): The universe's alpha parameter.
        radius (float): The radius of the K-dimensional simplex.
        K (int): The dimensionality of the universe (a static argument for JIT).

    Returns:
        jax.Array: The net force vector F_net at point p. This is a zero-sum
            vector in the tangent space of the simplex.
    """
    # Directly call the function JAX generated for us
    grad_vec: jax.Array = grad_potential_fn(
        p, sources_p_stim, sources_strength, alpha, radius, K
    )

    # Force is the negative gradient, projected onto the zero-sum tangent space
    force_net: jax.Array = -(grad_vec - jnp.mean(grad_vec))

    return force_net
