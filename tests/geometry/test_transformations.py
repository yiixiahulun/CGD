# tests/geometry/test_transformations.py
"""
Unit tests for cgd.geometry.transformations.
"""
import pytest
import numpy as np
from cgd.geometry import transformations as trans

# === Test Data (p-space) ===
P_3D_CENTER = np.array([1/3, 1/3, 1/3])
P_3D_VERTEX = np.array([1.0, 0.0, 0.0])
P_3D_EDGE = np.array([0.5, 0.5, 0.0])
P_3D_ASYMMETRIC = np.array([0.8, 0.1, 0.1])
P_5D_ASYMMETRIC = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

# === Test Data (v-space) ===
# A v-vector must be sum-zero.
V_3D_ZERO = np.array([0.0, 0.0, 0.0])
V_3D_NON_ZERO = np.array([0.5, -0.2, -0.3])


@pytest.mark.core
class TestLogMapFromOrigin:
    @pytest.mark.parametrize("p", [
        P_3D_CENTER, P_3D_VERTEX, P_3D_EDGE, P_3D_ASYMMETRIC
    ])
    def test_log_map_output_sums_to_zero(self, p):
        """
        Tests that the output v_eigen vector is always on the tangent plane (sums to zero).
        """
        # Arrange & Act
        v = trans.log_map_from_origin(p)
        # Assert
        assert np.isclose(np.sum(v), 0.0)

    @pytest.mark.validation
    def test_log_map_on_invalid_input(self):
        """
        Tests that log_map raises an error for a vector that isn't a probability distribution.
        """
        p_invalid = np.array([0.5, 0.6, 0.1]) # sums to > 1
        with pytest.raises(ValueError, match="输入向量 p 必须是一个有效的概率分布"):
            trans.log_map_from_origin(p_invalid)


@pytest.mark.core
class TestExpMapFromOrigin:
    @pytest.mark.parametrize("v", [
        V_3D_ZERO, V_3D_NON_ZERO
    ])
    def test_exp_map_output_is_probability_vector(self, v):
        """
        Tests that the output p vector is always on the simplex (sums to one).
        """
        # Arrange & Act
        p = trans.exp_map_from_origin(v)
        # Assert
        assert np.isclose(np.sum(p), 1.0)
        assert np.all(p >= 0)

    @pytest.mark.validation
    def test_exp_map_on_invalid_input(self):
        """
        Tests that exp_map raises an error for a vector that is not on the tangent plane.
        """
        v_invalid = np.array([0.5, 0.1, 0.1]) # does not sum to zero
        with pytest.raises(ValueError, match="输入向量 v 必须在切空间中"):
            trans.exp_map_from_origin(v_invalid)


@pytest.mark.consistency
@pytest.mark.parametrize("p_orig", [
    P_3D_CENTER,
    P_3D_VERTEX,
    P_3D_EDGE,
    P_3D_ASYMMETRIC,
    P_5D_ASYMMETRIC
])
def test_log_exp_map_reversibility(p_orig):
    """
    Tests the critical mathematical property that exp_map(log_map(p)) == p.
    This ensures that the coordinate transformation is self-consistent.
    """
    # Arrange: log_map(p) -> v
    v = trans.log_map_from_origin(p_orig)

    # Act: exp_map(v) -> p_reconstructed
    p_reconstructed = trans.exp_map_from_origin(v)

    # Assert: p_reconstructed should be identical to p_orig
    np.testing.assert_allclose(p_reconstructed, p_orig, atol=1e-9)


@pytest.mark.consistency
@pytest.mark.parametrize("v_orig", [
    V_3D_ZERO,
    V_3D_NON_ZERO,
    np.array([0.5, 0.2, 0.1, -0.4, -0.4]) # 5D vector
])
def test_exp_log_map_reversibility(v_orig):
    """
    Tests the critical mathematical property that log_map(exp_map(v)) == v.
    This ensures that the coordinate transformation is self-consistent.
    """
    # Arrange: exp_map(v) -> p
    p = trans.exp_map_from_origin(v_orig)

    # Act: log_map(p) -> v_reconstructed
    v_reconstructed = trans.log_map_from_origin(p)

    # Assert: v_reconstructed should be identical to v_orig
    np.testing.assert_allclose(v_reconstructed, v_orig, atol=1e-9)
