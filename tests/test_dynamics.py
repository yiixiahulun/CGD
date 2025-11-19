# tests/test_dynamics.py
"""Functional tests for the cgd.dynamics and cgd.dynamics_jax modules."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from cgd.core import Universe, GravitationalSource
from cgd.geometry import get_chaos_origin

# === Test EquilibriumFinder for both NumPy and JAX backends ===

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_equilibrium_finder_symmetry(backend):
    """Test that a symmetric universe has an equilibrium at the center."""
    if backend == "jax":
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed, skipping JAX backend test.")

    # Create a symmetric universe
    universe = Universe(K=3, alpha=1.0, backend=backend, labels=("A", "B", "C"))
    sources = [
        GravitationalSource(
            name="s1",
            v_eigen=np.array([0.1, -0.1, 0.0]),
            v_type="absolute",
            labels=("A", "B", "C"),
        ),
        GravitationalSource(
            name="s2",
            v_eigen=np.array([-0.1, 0.1, 0.0]),
            v_type="absolute",
            labels=("A", "B", "C"),
        ),
    ]

    # Find the equilibrium
    equilibria = universe.find_equilibria(sources, validate_stability=True)

    # Assert that the equilibrium is at the center
    assert len(equilibria) == 1
    assert_allclose(equilibria[0], get_chaos_origin(3), atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_equilibrium_finder_known_result(backend):
    """Test that the equilibrium finder can find a known equilibrium."""
    if backend == "jax":
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed, skipping JAX backend test.")

    # Create a simple K=2 universe with a known equilibrium
    universe = Universe(K=2, alpha=1.0, backend=backend, labels=("A", "B"))
    sources = [
        GravitationalSource(
            name="s1",
            v_eigen=np.array([0.5, -0.5]),
            v_type="absolute",
            labels=("A", "B"),
        )
    ]

    # The equilibrium for this system is at the stimulus location
    from cgd.geometry import exp_map_from_origin
    known_equilibrium = exp_map_from_origin(sources[0].v_eigen)

    # Find the equilibrium
    equilibria = universe.find_equilibria(sources, validate_stability=True)

    # Assert that the equilibrium is at the known location
    assert len(equilibria) == 1
    assert_allclose(equilibria[0], known_equilibrium, atol=1e-7)
