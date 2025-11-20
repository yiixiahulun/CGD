# tests/dynamics/test_solvers_consistency.py
import pytest
import numpy as np
jax = pytest.importorskip("jax")
from cgd.core.universe import Universe
from cgd.geometry.simplex import distance_FR
from tests import helpers

SOURCE_A = helpers.create_test_source(p=np.array([0.9, 0.05, 0.05]), name="A", v_type='absolute', labels=['X','Y','Z'])
SOURCE_B = helpers.create_test_source(p=np.array([0.05, 0.9, 0.05]), name="B", v_type='absolute', labels=['X','Y','Z'])
SOURCE_C = helpers.create_test_source(p=np.array([0.05, 0.05, 0.9]), name="C", v_type='absolute', labels=['X','Y','Z'])

SCENARIO_SINGLE_POINT = {
    "sources": [SOURCE_A],
    "alpha": 1.0,
    "K": 3
}
SCENARIO_MULTI_POINT = {
    "sources": [SOURCE_A, SOURCE_B, SOURCE_C],
    "alpha": 0.5,
    "K": 3
}

def find_closest_point(p, points):
    if not points:
        return None
    distances = [distance_FR(p, pt) for pt in points]
    return points[np.argmin(distances)]

@pytest.mark.consistency
@pytest.mark.parametrize("scenario", [SCENARIO_SINGLE_POINT, SCENARIO_MULTI_POINT])
def test_solver_showdown(scenario):
    sources = scenario["sources"]
    alpha = scenario["alpha"]
    K = scenario["K"]
    solver_options = {
        "num_random_seeds": 20,
        "validate_stability": False,
    }
    universe_np = helpers.create_test_universe(K, alpha, labels=['X','Y','Z'], backend='numpy')
    universe_jax = helpers.create_test_universe(K, alpha, labels=['X','Y','Z'], backend='jax')
    equilibria_np = universe_np.find_equilibria(sources, **solver_options)
    equilibria_jax = universe_jax.find_equilibria(sources, **solver_options)
    assert len(equilibria_np) == len(equilibria_jax), f"Solvers found a different number of equilibria: NumPy ({len(equilibria_np)}) vs JAX ({len(equilibria_jax)})"
    remaining_jax_points = list(equilibria_jax)
    for p_np in equilibria_np:
        closest_p_jax = find_closest_point(p_np, remaining_jax_points)
        assert closest_p_jax is not None, f"No corresponding JAX point found for NumPy point {p_np}"
        dist = distance_FR(p_np, closest_p_jax)
        assert dist < 1e-5, f"NumPy point {p_np} is too far from closest JAX point {closest_p_jax} (distance: {dist})"
        remaining_jax_points = [p for p in remaining_jax_points if not np.allclose(p, closest_p_jax)]
