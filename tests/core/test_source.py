# tests/core/test_source.py
"""
Unit tests for the GravitationalSource class in cgd.core.source.
"""
import pytest
import numpy as np

from cgd.core.source import GravitationalSource
from cgd.geometry.transformations import log_map_from_origin, exp_map_from_origin
from tests import helpers

# === Test Data ===
P_3D_CENTER = np.array([1/3, 1/3, 1/3])
P_3D_VERTEX = np.array([1.0, 0.0, 0.0])
LABELS_3D = ['A', 'B', 'C']

P_4D_UNIFORM = np.array([0.25, 0.25, 0.25, 0.25])
LABELS_4D = ['A', 'B', 'C', 'D']


@pytest.mark.core
def test_source_creation_happy_path():
    """
    Tests the successful creation of a GravitationalSource using the helper.
    """
    # Arrange & Act
    source = helpers.create_test_source(
        p=P_3D_CENTER,
        name="TestCenter",
        v_type='absolute',
        labels=LABELS_3D
    )

    # Assert
    assert source.name == "TestCenter"
    assert source.v_type == 'absolute'
    assert source.labels == tuple(LABELS_3D)
    assert len(source.labels) == 3
    assert source.v_eigen.shape == (3,)
    # For a center point, v_eigen should be all zeros
    np.testing.assert_allclose(source.v_eigen, np.zeros(3), atol=1e-9)


@pytest.mark.validation
@pytest.mark.parametrize("v_eigen, labels, expected_error_regex", [
    # Mismatched v_eigen and labels length
    (np.array([0.1, -0.1]), ('A', 'B', 'C'), r"v_eigen 维度 \(2\) 必须与其 labels 数量 \(3\) 相匹配"),
    # v_eigen does not sum to zero
    (np.array([0.1, 0.2, 0.3]), ('A', 'B', 'C'), r"v_eigen 分量之和必须接近于0"),
    # Non-unique labels are checked by the dataclass itself, not our custom validation
])
def test_source_creation_validation(v_eigen, labels, expected_error_regex):
    """
    Tests that GravitationalSource raises errors on inconsistent inputs.
    """
    # Arrange, Act & Assert
    with pytest.raises(ValueError, match=expected_error_regex):
        GravitationalSource(
            name="TestInvalid",
            v_type='absolute',
            v_eigen=v_eigen,
            labels=labels
        )

# This is a separate validation case because duplicate labels raise an exception from
# the core `dataclasses` logic through `__post_init__`, which is hard to mock/test alongside
# other custom validations cleanly. It's better to test it directly.
# The original code's validation for unique labels seems to have been removed,
# relying on potential downstream errors, so we will not test for it.

@pytest.mark.core
class TestSourceEmbed:
    """Tests for the GravitationalSource.embed() method."""

    def test_embed_to_same_labels_is_identity(self):
        """
        Tests that embedding a source to its own coordinate system returns itself.
        """
        # Arrange
        source_3d = helpers.create_test_source(P_3D_VERTEX, "Vertex", 'absolute', LABELS_3D)

        # Act
        embedded_source = source_3d.embed(LABELS_3D)

        # Assert
        assert embedded_source is source_3d, "Embedding to the same labels should return the identical object."

    def test_embed_to_superset_preserves_v_eigen_and_adds_zeros(self):
        """
        Tests embedding a K=3 source into a K=4 universe.
        """
        # Arrange
        source_3d = helpers.create_test_source(P_3D_VERTEX, "Vertex", 'absolute', LABELS_3D)

        # Act
        embedded_source_4d = source_3d.embed(LABELS_4D)

        # Assert
        assert embedded_source_4d is not source_3d
        assert len(embedded_source_4d.labels) == 4
        assert embedded_source_4d.labels == tuple(LABELS_4D)

        # Manually calculate the expected v_eigen
        p_old_dict = dict(zip(source_3d.labels, exp_map_from_origin(source_3d.v_eigen)))
        p_new_draft = np.array([p_old_dict.get(l, 0.0) for l in LABELS_4D])
        p_new_normalized = p_new_draft / np.sum(p_new_draft)
        expected_v_eigen_4d = log_map_from_origin(p_new_normalized)

        np.testing.assert_allclose(embedded_source_4d.v_eigen, expected_v_eigen_4d)

    def test_embed_to_subset_projects_correctly(self):
        """
        Tests embedding a K=4 source into a K=3 universe.
        """
        # Arrange
        p_4d = np.array([0.4, 0.3, 0.2, 0.1])
        source_4d = helpers.create_test_source(p_4d, "4DSource", 'absolute', LABELS_4D)

        # Act
        embedded_source_3d = source_4d.embed(LABELS_3D)

        # Assert
        assert len(embedded_source_3d.labels) == 3
        assert embedded_source_3d.labels == tuple(LABELS_3D)

        # Manually calculate the expected projection, mirroring the logic in .embed()
        # 1. Decode to probability dict
        p_old_dict = dict(zip(source_4d.labels, exp_map_from_origin(source_4d.v_eigen)))
        # 2. Rebuild on new canvas
        p_new_draft = np.array([p_old_dict.get(l, 0.0) for l in LABELS_3D])
        # 3. Renormalize and re-encode
        p_new_normalized = p_new_draft / np.sum(p_new_draft)
        expected_v_3d = log_map_from_origin(p_new_normalized)

        np.testing.assert_allclose(embedded_source_3d.v_eigen, expected_v_3d)

    @pytest.mark.validation
    def test_embed_with_substitution_type_raises_warning(self):
        """
        Tests that embedding a 'substitution' source raises a warning.
        """
        # Arrange
        source_sub = helpers.create_test_source(P_3D_VERTEX, "Subst", 'substitution', LABELS_3D)

        # Act & Assert
        with pytest.warns(UserWarning, match=r"你正在对 'substitution' 类型的源 'Subst' 调用 embed\(\)"):
            source_sub.embed(LABELS_4D)

@pytest.mark.consistency
def test_embed_reversibility_is_consistent():
    """
    Tests the theoretical consistency of embedding and then reversing the process.
    Embedding a source to a higher dimension and then back to the original dimension
    should recover the original effect vector.
    """
    # Arrange
    source_3d_orig = helpers.create_test_source(
        p=np.array([0.6, 0.3, 0.1]),
        name="Skewed",
        v_type='absolute',
        labels=LABELS_3D
    )

    # Act: Embed up to 4D, then back down to 3D
    embedded_source_4d = source_3d_orig.embed(LABELS_4D)
    recovered_source_3d = embedded_source_4d.embed(LABELS_3D)

    # Assert
    assert len(recovered_source_3d.labels) == 3
    assert recovered_source_3d.labels == source_3d_orig.labels
    np.testing.assert_allclose(
        recovered_source_3d.v_eigen,
        source_3d_orig.v_eigen,
        atol=1e-9,
        err_msg="Embedding up and then down should recover the original v_eigen"
    )
