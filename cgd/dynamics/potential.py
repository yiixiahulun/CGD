# cgd/dynamics/potential.py
"""NumPy implementation of the Gaussian potential for CGD physics.

This module provides the function to calculate the total potential energy at a
point in the probability simplex, based on the standard CGD Gaussian model.
"""
from typing import List

import numpy as np
from numpy.typing import NDArray

from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..geometry.simplex import distance_FR
from ..geometry.transformations import exp_map_from_origin


def potential_gaussian(
    p: NDArray[np.float64], sources: List[GravitationalSource], universe: Universe
) -> float:
    """Calculates the total Gaussian potential energy at a point p (NumPy).

    The total potential is the sum of the potentials from all individual
    sources. Each source generates a potential field that follows a Gaussian
    profile, where the distance is the geodesic Fisher-Rao distance on the
    simplex, normalized by the simplex's radius.

    Potential_s = -strength * exp(-(alpha * (distance / radius))^2)

    Args:
        p (NDArray[np.float64]): The point in the simplex (a probability distribution)
            at which to calculate the potential.
        sources (List[GravitationalSource]): A list of all sources contributing
            to the potential.
        universe (Universe): The universe object, which provides the physical
            constants K, alpha, and the simplex geometry.

    Returns:
        float: The total scalar potential energy at point p.
    """
    total_potential: float = 0.0

    alpha: float = universe.alpha
    radius: float = universe.simplex.radius

    if radius == 0:
        return 0.0

    for source in sources:
        # p_stim is the location of the potential's minimum for this source
        p_stim: NDArray[np.float64] = exp_map_from_origin(source.v_eigen)

        dist: float = distance_FR(p, p_stim)
        relative_dist: float = dist / radius

        potential_s: float = -source.strength * np.exp(-((alpha * relative_dist) ** 2))
        total_potential += potential_s

    return total_potential
