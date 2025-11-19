# tests/geometry/test_simplex.py
"""
Unit tests for cgd.geometry.simplex.
"""
import pytest
import numpy as np
from cgd.geometry import simplex

# === Test Data ===
P_3D_CENTER = np.array([1/3, 1/3, 1/3])
P_3D_VERTEX_A = np.array([1.0, 0.0, 0.0])
P_3D_VERTEX_B = np.array([0.0, 1.0, 0.0])
P_3D_EDGE_MIDPOINT = np.array([0.5, 0.5, 0.0])
P_3D_ASYMMETRIC = np.array([0.6, 0.3, 0.1])


@pytest.mark.core
@pytest.mark.parametrize("K, expected_origin", [
    (1, np.array([1.0])),
    (3, np.array([1/3, 1/3, 1/3])),
    (4, np.array([0.25, 0.25, 0.25, 0.25])),
])
def test_get_chaos_origin_happy_path(K, expected_origin):
    """
    Tests that get_chaos_origin returns the correct uniform distribution.
    """
    # Arrange & Act
    origin = simplex.get_chaos_origin(K)
    # Assert
    np.testing.assert_allclose(origin, expected_origin)


@pytest.mark.validation
@pytest.mark.parametrize("K", [0, -1, 1.5, "a"])
def test_get_chaos_origin_validation(K):
    """
    Tests that get_chaos_origin raises ValueError for invalid K.
    """
    with pytest.raises(ValueError, match="K 必须为正整数"):
        simplex.get_chaos_origin(K)


@pytest.mark.core
@pytest.mark.parametrize("K, expected_radius", [
    (1, 0.0),  # K=1 is a single point, radius is 0
    (2, np.pi / 2), # Theoretical value: 2 * arccos(sqrt(1/2))
    (3, 2 * np.arccos(np.sqrt(1/3))),
    (4, 2 * np.arccos(np.sqrt(1/4))), # This is 2*pi/3, not pi.
])
def test_get_radius_happy_path(K, expected_radius):
    """
    Tests that get_radius calculates the correct simplex radius.
    """
    # Arrange & Act
    radius = simplex.get_radius(K)
    # Assert
    assert np.isclose(radius, expected_radius)


@pytest.mark.validation
@pytest.mark.parametrize("K", [0, -1, 2.5, "b"])
def test_get_radius_validation(K):
    """
    Tests that get_radius raises ValueError for invalid K.
    """
    with pytest.raises(ValueError, match="K 必须为正整数"):
        simplex.get_radius(K)


@pytest.mark.core
class TestDistanceFR:
    """Tests for the Fisher-Rao distance function."""

    @pytest.mark.parametrize("p, q, expected_distance", [
        # Distance to self is zero
        (P_3D_CENTER, P_3D_CENTER, 0.0),
        (P_3D_VERTEX_A, P_3D_VERTEX_A, 0.0),
        # Distance from center to a vertex is the radius
        (P_3D_CENTER, P_3D_VERTEX_A, simplex.get_radius(3)),
        # Distance between two vertices
        (P_3D_VERTEX_A, P_3D_VERTEX_B, 2 * np.arccos(0)), # PI
        # A known, non-trivial distance
        (P_3D_VERTEX_A, P_3D_EDGE_MIDPOINT, 2 * np.arccos(np.sqrt(0.5))), # PI/2
    ])
    def test_distance_fr_happy_path(self, p, q, expected_distance):
        """
        Tests Fisher-Rao distance for various standard cases.
        """
        # Arrange & Act
        dist = simplex.distance_FR(p, q)
        # Assert
        assert np.isclose(dist, expected_distance)

    @pytest.mark.consistency
    def test_distance_fr_is_symmetric(self):
        """
        Tests that distance_FR(p, q) == distance_FR(q, p).
        """
        p = P_3D_ASYMMETRIC
        q = P_3D_EDGE_MIDPOINT
        dist_pq = simplex.distance_FR(p, q)
        dist_qp = simplex.distance_FR(q, p)
        assert np.isclose(dist_pq, dist_qp)

    @pytest.mark.validation
    def test_distance_fr_mismatched_shapes(self):
        """
        Tests that distance_FR raises an error for inputs with different shapes.
        """
        p_3d = P_3D_CENTER
        q_4d = np.array([0.25, 0.25, 0.25, 0.25])
        with pytest.raises(ValueError, match="输入向量的形状不匹配"):
            simplex.distance_FR(p_3d, q_4d)

    def test_distance_fr_handles_non_normalized_input(self):
        """
        Tests that the function correctly normalizes non-unit-sum inputs.
        """
        p_unnorm = np.array([1, 1, 1])
        q_unnorm = np.array([2, 0, 0])
        p_norm = p_unnorm / 3
        q_norm = q_unnorm / 2

        expected_dist = simplex.distance_FR(p_norm, q_norm)
        actual_dist = simplex.distance_FR(p_unnorm, q_unnorm)

        assert np.isclose(expected_dist, actual_dist)
