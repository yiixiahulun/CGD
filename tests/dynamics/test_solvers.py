# tests/dynamics/test_solvers.py
import pytest
import numpy as np
from unittest.mock import patch
from cgd.dynamics.solvers import EquilibriumFinder
from cgd.geometry.simplex import distance_FR
from tests import helpers

SOURCE_A = helpers.create_test_source(p=np.array([0.9, 0.05, 0.05]), name="A", v_type='absolute', labels=['X','Y','Z'])

@pytest.mark.core
def test_equilibrium_finder_initialization():
    universe = helpers.create_test_universe(K=3, alpha=1.0)
    sources = [SOURCE_A]
    finder = EquilibriumFinder(universe, sources)
    assert finder.universe is universe
    assert finder.sources is sources
    assert finder.K == 3

@pytest.mark.core
@patch('cgd.dynamics.solvers.Parallel')
def test_find_returns_validated_and_unique_points(MockParallel):
    p1 = np.array([0.8, 0.1, 0.1])
    p2 = np.array([0.1, 0.8, 0.1])
    p_duplicate = np.array([0.8, 0.1, 0.1000001])
    p_unstable = np.array([1/3, 1/3, 1/3])
    universe = helpers.create_test_universe(K=3, alpha=1.0)
    finder = EquilibriumFinder(universe, [SOURCE_A])
    mock_parallel_instance = MockParallel.return_value
    mock_parallel_instance.return_value = [p1, p2, p_duplicate, p_unstable]
    def mock_validate_stability(p, *args, **kwargs):
        if np.allclose(p, p_unstable):
            return "Unstable"
        return "Stable"
    with patch.object(finder, '_validate_point_stability', side_effect=mock_validate_stability):
        equilibria = finder.find(validate_stability=True, num_random_seeds=4)
    assert len(equilibria) == 2
    assert not any(np.allclose(eq, p_unstable) for eq in equilibria)
    assert any(np.allclose(eq, p1) for eq in equilibria)
    assert any(np.allclose(eq, p2) for eq in equilibria)

@pytest.mark.consistency
def test_numpy_uniqueness_logic_fails_on_close_points():
    p1 = np.array([0.5, 0.5, 1e-8])
    p1 /= np.sum(p1)
    p2 = np.array([0.5, 0.5, 1e-7])
    p2 /= np.sum(p2)
    assert np.allclose(p1, p2, atol=1e-6)
    uniqueness_tolerance = 1e-4
    assert distance_FR(p1, p2) > uniqueness_tolerance
    universe = helpers.create_test_universe(K=3, alpha=1.0)
    finder = EquilibriumFinder(universe, [SOURCE_A])
    with patch('cgd.dynamics.solvers.Parallel') as MockParallel:
        mock_parallel_instance = MockParallel.return_value
        mock_parallel_instance.return_value = [p1, p2]
        with patch.object(finder, '_validate_point_stability', return_value="Stable"):
            equilibria = finder.find(validate_stability=True, num_random_seeds=2, uniqueness_tolerance=uniqueness_tolerance)
    assert len(equilibria) == 2, "NumPy uniqueness logic incorrectly merged two distinct solutions"
