# tests/core/test_universe.py
"""
Unit tests for the Universe class in cgd.core.universe.
"""
import pytest
from unittest.mock import patch
import numpy as np

from cgd.core.universe import Universe
from cgd.dynamics.solvers import EquilibriumFinder, TrajectorySimulator
from cgd.dynamics_jax.solvers_jax import EquilibriumFinderJax
from tests import helpers

# === Test Data ===
LABELS_3D = ('A', 'B', 'C')
LABELS_4D = ('A', 'B', 'C', 'D')


@pytest.mark.core
@pytest.mark.parametrize("backend", ['numpy', 'jax'])
def test_universe_creation_happy_path(backend):
    """
    Tests the successful creation of a Universe instance with both backends.
    """
    # Arrange & Act
    # JAX check
    if backend == 'jax':
        pytest.importorskip("jax")

    universe = Universe(K=3, alpha=1.0, labels=LABELS_3D, backend=backend)

    # Assert
    assert universe.K == 3
    assert universe.alpha == 1.0
    assert universe.labels == LABELS_3D
    assert universe.backend == backend
    assert universe.simplex.K == 3


@pytest.mark.core
def test_universe_default_labels():
    """
    Tests that a Universe creates default numerical labels if none are provided.
    """
    # Arrange & Act
    universe = Universe(K=4, alpha=0.5)

    # Assert
    assert universe.labels == ('0', '1', '2', '3')


@pytest.mark.validation
@pytest.mark.parametrize("K, alpha, labels, expected_error", [
    (3, -0.1, LABELS_3D, "alpha 值必须为非负数"),
    (3, 1.0, LABELS_4D, "提供的 labels 数量 .4. 必须与宇宙维度 K .3. 匹配"),
    (4, 1.0, LABELS_3D, "提供的 labels 数量 .3. 必须与宇宙维度 K .4. 匹配"),
])
def test_universe_creation_validation(K, alpha, labels, expected_error):
    """
    Tests that Universe initialization raises errors on invalid inputs.
    """
    # Arrange, Act & Assert
    with pytest.raises(ValueError, match=expected_error):
        Universe(K=K, alpha=alpha, labels=labels)


@pytest.mark.core
class TestUniverseBackendSelection:
    """Tests the dynamic backend selection logic."""

    def test_backend_numpy_selection(self):
        """
        Tests that the 'numpy' backend correctly binds NumPy solver classes.
        """
        # Arrange & Act
        universe = Universe(K=3, alpha=1.0, backend='numpy')

        # Assert
        assert universe._finder_class is EquilibriumFinder
        assert universe._simulator_class is TrajectorySimulator

    def test_backend_jax_selection(self):
        """
        Tests that the 'jax' backend correctly binds JAX solver classes.
        """
        # Arrange
        pytest.importorskip("jax") # Skip if jax is not installed

        # Act
        universe = Universe(K=3, alpha=1.0, backend='jax')

        # Assert
        assert universe._finder_class is EquilibriumFinderJax
        # Currently, simulator falls back to NumPy
        assert universe._simulator_class is TrajectorySimulator

    @pytest.mark.validation
    def test_backend_invalid_selection_raises_error(self):
        """
        Tests that providing an unknown backend name raises a ValueError.
        """
        with pytest.raises(ValueError, match="不支持的后端: 'invalid_backend'"):
            Universe(K=3, alpha=1.0, backend='invalid_backend')


@pytest.mark.core
class TestUniverseDispatcher:
    """Tests the Universe's role as a dispatcher to its backend."""

    @patch('cgd.dynamics.solvers.EquilibriumFinder.find')
    def test_find_equilibria_dispatches_to_numpy_backend(self, mock_find):
        """
        Verifies that find_equilibria calls the numpy backend's find method.
        """
        # Arrange
        universe = Universe(K=3, alpha=1.0, backend='numpy')
        source = helpers.create_test_source(np.array([.5,.5,0]), "S1", 'absolute', ['A','B','C'])

        # Act
        universe.find_equilibria([source], num_random_seeds=10)

        # Assert
        mock_find.assert_called_once_with(num_random_seeds=10)

    def test_find_equilibria_with_mismatched_source_raises_error(self):
        """
        Tests the validation layer that prevents mismatched sources from being used.
        """
        # Arrange
        universe = Universe(K=3, alpha=1.0, labels=LABELS_3D)
        mismatched_source = helpers.create_test_source(
            p=np.array([.25,.25,.25,.25]),
            name="4D_Source",
            v_type='absolute',
            labels=LABELS_4D
        )

        # Act & Assert
        with pytest.raises(ValueError, match="与宇宙维度 .K=3. 不匹配"):
            universe.find_equilibria([mismatched_source])
