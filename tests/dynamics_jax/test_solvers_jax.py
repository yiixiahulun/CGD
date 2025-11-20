# tests/dynamics_jax/test_solvers_jax.py
import pytest
import numpy as np
from unittest.mock import patch
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from cgd.dynamics_jax.solvers_jax import EquilibriumFinderJax
from cgd.core.source import GravitationalSource
from cgd.geometry.transformations import exp_map_from_origin
from tests import helpers

SOURCE_A_NP = helpers.create_test_source(p=np.array([0.9, 0.05, 0.05]), name="A", v_type='absolute', labels=['X','Y','Z'])

@pytest.mark.core
def test_equilibrium_finder_jax_initialization():
    universe = helpers.create_test_universe(K=3, alpha=1.0, backend='jax')
    sources = [SOURCE_A_NP]
    finder = EquilibriumFinderJax(universe, sources)
    assert finder.universe is universe
    assert finder.K == 3
    expected_p_stim = jnp.array([exp_map_from_origin(SOURCE_A_NP.v_eigen)])
    expected_strength = jnp.array([SOURCE_A_NP.strength])
    assert jnp.allclose(finder.sources_p_stim_jax, expected_p_stim)
    assert jnp.allclose(finder.sources_strength_jax, expected_strength)

@pytest.mark.core
@patch('cgd.dynamics_jax.solvers_jax.Parallel')
def test_find_returns_validated_and_unique_points(MockParallel):
    p1 = np.array([0.8, 0.1, 0.1])
    p2 = np.array([0.1, 0.8, 0.1])
    p_unstable = np.array([1/3, 1/3, 1/3])
    universe = helpers.create_test_universe(K=3, alpha=1.0, backend='jax')
    finder = EquilibriumFinderJax(universe, [SOURCE_A_NP])
    mock_parallel_instance = MockParallel.return_value
    mock_parallel_instance.return_value = [p1, p2, p_unstable]
    def mock_is_stable(p, *args, **kwargs):
        return not np.allclose(p, p_unstable)
    with patch.object(finder, '_validate_point_stability_jax', side_effect=mock_is_stable):
        equilibria = finder.find(validate_stability=True, num_random_seeds=3)
    assert len(equilibria) == 2
    assert not any(np.allclose(eq, p_unstable) for eq in equilibria)
    assert any(np.allclose(eq, p1) for eq in equilibria)
    assert any(np.allclose(eq, p2) for eq in equilibria)

@pytest.mark.consistency
def test_jax_stability_check_incorrectly_validates_saddle_point():
    """
    This test is EXPECTED TO FAIL.
    It verifies that the current JAX stability check, which only uses the gradient,
    incorrectly validates a known saddle point as stable.
    """
    labels = ['A', 'B', 'C']
    source1 = helpers.create_test_source(np.array([0.9, 0.05, 0.05]), "S1", 'absolute', labels)
    source2 = helpers.create_test_source(np.array([0.05, 0.9, 0.05]), "S2", 'absolute', labels)
    source3 = helpers.create_test_source(np.array([0.05, 0.05, 0.9]), "S3", 'absolute', labels)
    source4 = helpers.create_test_source(np.array([1/3, 1/3, 1/3]), "S4", 'absolute', labels)
    sources = [source1, source2, source3, source4]
    p_saddle = jnp.array([1/3, 1/3, 1/3])
    universe = helpers.create_test_universe(K=3, alpha=2.0, backend='jax', labels=labels)
    finder = EquilibriumFinderJax(universe, sources)
    is_stable = finder._validate_point_stability_jax(p_saddle)
    assert not is_stable, "JAX stability check incorrectly validated a saddle point as stable"
