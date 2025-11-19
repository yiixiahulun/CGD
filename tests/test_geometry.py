# tests/test_geometry.py
"""Unit tests for the cgd.geometry module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_raises

from cgd.geometry import (
    get_chaos_origin,
    get_radius,
    distance_FR,
    log_map_from_origin,
    exp_map_from_origin,
)
from cgd.geometry import (
    get_chaos_origin_jax,
    distance_FR_jax,
    log_map_from_origin_jax,
    exp_map_from_origin_jax,
    JAX_AVAILABLE,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp


# === Test NumPy Implementations ===

def test_get_chaos_origin_numpy():
    """Test the NumPy get_chaos_origin function."""
    assert_allclose(get_chaos_origin(3), np.array([1/3, 1/3, 1/3]))
    assert_allclose(get_chaos_origin(1), np.array([1.0]))
    with assert_raises(ValueError):
        get_chaos_origin(0)
    with assert_raises(ValueError):
        get_chaos_origin(-1)

def test_get_radius_numpy():
    """Test the NumPy get_radius function."""
    assert get_radius(1) == 0.0
    assert_allclose(get_radius(2), np.pi / 2)
    assert_allclose(get_radius(3), 2 * np.arccos(np.sqrt(1/3)))
    with assert_raises(ValueError):
        get_radius(0)

def test_distance_fr_numpy():
    """Test the NumPy distance_FR function."""
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert_allclose(distance_FR(p, q), np.pi / 2)

    # Test with zero values
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    assert_allclose(distance_FR(p, q), np.pi)

    # Test mismatched dimensions
    with assert_raises(ValueError):
        distance_FR(np.array([0.5, 0.5]), np.array([1/3, 1/3, 1/3]))

def test_log_exp_map_numpy_identity():
    """Test that Exp(Log(p)) == p for the NumPy implementations."""
    p = np.array([0.1, 0.7, 0.2])
    v = log_map_from_origin(p)
    p_reconstructed = exp_map_from_origin(v)
    assert_allclose(p, p_reconstructed, atol=1e-9)

    # Edge case: origin
    origin = get_chaos_origin(4)
    v_zero = log_map_from_origin(origin)
    assert_allclose(v_zero, np.zeros(4))
    p_reconstructed_origin = exp_map_from_origin(v_zero)
    assert_allclose(origin, p_reconstructed_origin)

# === Test JAX Implementations (if available) ===

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_get_chaos_origin_jax():
    """Test the JAX get_chaos_origin function."""
    assert_allclose(np.asarray(get_chaos_origin_jax(3)), np.array([1/3, 1/3, 1/3]))
    assert_allclose(np.asarray(get_chaos_origin_jax(1)), np.array([1.0]))

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_distance_fr_jax():
    """Test the JAX distance_FR function."""
    p = jnp.array([0.5, 0.5], dtype=jnp.float64)
    q = jnp.array([1.0, 0.0], dtype=jnp.float64)
    assert_allclose(distance_FR_jax(p, q), np.pi / 2)

    # Test with zero values
    p = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
    q = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
    assert_allclose(distance_FR_jax(p, q), np.pi)

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_log_exp_map_jax_identity():
    """Test that Exp(Log(p)) == p for the JAX implementations."""
    p = jnp.array([0.1, 0.7, 0.2])
    v = log_map_from_origin_jax(p)
    p_reconstructed = exp_map_from_origin_jax(v)
    assert_allclose(np.asarray(p), np.asarray(p_reconstructed), atol=1e-7)

    # Edge case: origin
    origin = get_chaos_origin_jax(4)
    v_zero = log_map_from_origin_jax(origin)
    assert_allclose(np.asarray(v_zero), np.zeros(4))
    p_reconstructed_origin = exp_map_from_origin_jax(v_zero)
    assert_allclose(np.asarray(origin), np.asarray(p_reconstructed_origin))
