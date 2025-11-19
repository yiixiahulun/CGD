# tests/geometry/test_consistency.py
"""
Backend consistency tests for the geometry module.

These tests ensure that the JAX implementations of geometric functions
are numerically equivalent to their NumPy counterparts.
"""
import pytest
import numpy as np

# Import NumPy versions
from cgd.geometry import simplex as simplex_np
from cgd.geometry import transformations as trans_np

# Import JAX versions
# Use importorskip to allow tests to be skipped if JAX is not installed
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from cgd.geometry import _jax_impl as jax_impl

# === Test Data ===
# A comprehensive set of p-vectors in various dimensions
P_VECTORS = [
    np.array([1.0]),                      # K=1
    np.array([0.5, 0.5]),                 # K=2 Center
    np.array([1.0, 0.0]),                 # K=2 Vertex
    np.array([1/3, 1/3, 1/3]),            # K=3 Center
    np.array([1.0, 0.0, 0.0]),            # K=3 Vertex
    np.array([0.5, 0.5, 0.0]),            # K=3 Edge
    np.array([0.8, 0.1, 0.1]),            # K=3 Asymmetric
    np.array([0.25, 0.25, 0.25, 0.25]),   # K=4 Center
    np.array([0.4, 0.3, 0.2, 0.1]),       # K=4 Asymmetric
]

# A comprehensive set of v-vectors (sum-zero)
V_VECTORS = [
    np.array([0.0]),                            # K=1
    np.array([0.0, 0.0]),                       # K=2 Center
    np.array([0.5, -0.5]),                      # K=2 Non-zero
    np.array([0.0, 0.0, 0.0]),                  # K=3 Center
    np.array([0.6, -0.3, -0.3]),                # K=3 Non-zero
    np.array([0.0, 0.0, 0.0, 0.0]),             # K=4 Center
    np.array([0.3, -0.1, -0.1, -0.1]),          # K=4 Non-zero
]


@pytest.mark.consistency
@pytest.mark.parametrize("p1_np", P_VECTORS)
@pytest.mark.parametrize("p2_np", P_VECTORS)
def test_distance_fr_consistency(p1_np, p2_np):
    """
    Compares the output of NumPy distance_FR with JAX distance_FR_jax.
    """
    # Arrange
    # Skip if dimensions don't match
    if p1_np.shape != p2_np.shape:
        pytest.skip("Skipping test for mismatched shapes.")

    p1_jax = jnp.array(p1_np)
    p2_jax = jnp.array(p2_np)

    # Act
    dist_np = simplex_np.distance_FR(p1_np, p2_np)
    dist_jax = jax_impl.distance_FR_jax(p1_jax, p2_jax)

    # Assert
    assert np.isclose(dist_np, dist_jax, atol=1e-6)


@pytest.mark.consistency
@pytest.mark.parametrize("p_np", P_VECTORS)
def test_log_map_from_origin_consistency(p_np):
    """
    Compares the output of NumPy log_map_from_origin with JAX log_map_from_origin_jax.
    """
    # Arrange
    p_jax = jnp.array(p_np)

    # Act
    v_np = trans_np.log_map_from_origin(p_np)
    v_jax = jax_impl.log_map_from_origin_jax(p_jax)

    # Assert
    np.testing.assert_allclose(v_np, v_jax, atol=1e-6)


@pytest.mark.consistency
@pytest.mark.parametrize("v_np", V_VECTORS)
def test_exp_map_from_origin_consistency(v_np):
    """
    Compares the output of NumPy exp_map_from_origin with JAX exp_map_from_origin_jax.
    """
    # Arrange
    v_jax = jnp.array(v_np)

    # Act
    p_np = trans_np.exp_map_from_origin(v_np)
    p_jax = jax_impl.exp_map_from_origin_jax(v_jax)

    # Assert
    np.testing.assert_allclose(p_np, p_jax, atol=1e-6)
