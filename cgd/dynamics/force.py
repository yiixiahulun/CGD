# cgd/dynamics/force.py
"""NumPy implementation of the net force calculation.

This module provides the function to calculate the net force vector at a point
in the simplex by taking the numerical gradient of the potential function.
"""
from typing import List

import numpy as np
from numpy.typing import NDArray

from ..core.source import GravitationalSource
from ..core.universe import Universe
from .potential import potential_gaussian


def calculate_F_net(
    p: NDArray[np.float64],
    sources: List[GravitationalSource],
    universe: Universe,
    h: float = 1e-6,
) -> NDArray[np.float64]:
    """Calculates the net force F_net at a point p using numerical gradient.

    The force is defined as the negative gradient of the potential energy
    (F = -∇U). This function computes the gradient using the central difference
    method for each dimension. The resulting gradient vector is then projected
    onto the tangent space of the simplex to ensure it is a valid (zero-sum)
    force vector.

    Args:
        p (NDArray[np.float64]): The point in the simplex (a probability distribution)
            at which to calculate the force.
        sources (List[GravitationalSource]): A list of all sources contributing
            to the potential.
        universe (Universe): The universe object, providing physical constants.
        h (float): The small step size used for the finite difference
            calculation. Defaults to 1e-6.

    Returns:
        NDArray[np.float64]: The net force vector F_net at point p. This is a zero-sum
            vector in the tangent space of the simplex.
    """
    K: int = universe.K
    grad: NDArray[np.float64] = np.zeros(K)

    # 1. Calculate the partial derivative for each dimension
    for i in range(K):
        p_fwd: NDArray[np.float64] = p.copy()
        p_bwd: NDArray[np.float64] = p.copy()

        p_fwd[i] += h
        p_bwd[i] -= h

        # Renormalization is crucial to stay on the simplex, though the
        # effect is small for a small h.
        p_fwd /= np.sum(p_fwd)
        p_bwd /= np.sum(p_bwd)

        potential_fwd: float = potential_gaussian(p_fwd, sources, universe)
        potential_bwd: float = potential_gaussian(p_bwd, sources, universe)

        grad[i] = (potential_fwd - potential_bwd) / (2 * h)

    # 2. Project the gradient vector onto the zero-sum tangent space
    grad_projected: NDArray[np.float64] = grad - np.mean(grad)

    # 3. Force is the negative of the gradient
    force_net: NDArray[np.float64] = -grad_projected

    return force_net
