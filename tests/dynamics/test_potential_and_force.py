# tests/dynamics/test_potential_and_force.py
import pytest
import numpy as np
from cgd.dynamics import potential as pot_np
from cgd.dynamics import force as force_np
from cgd.core.universe import Universe
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from cgd.dynamics_jax import potential_jax as pot_jax
from cgd.dynamics_jax import force_jax as force_jax
from tests import helpers

P_3D_CENTER = np.array([1/3, 1/3, 1/3])
P_3D_VERTEX = np.array([1.0, 0.0, 0.0])
P_3D_ASYMMETRIC = np.array([0.8, 0.1, 0.1])
SOURCE_A = helpers.create_test_source(p=np.array([0.6, 0.2, 0.2]), name="A", v_type='absolute', labels=['X','Y','Z'])
SOURCE_B = helpers.create_test_source(p=np.array([0.2, 0.6, 0.2]), name="B", v_type='absolute', labels=['X','Y','Z'])

@pytest.mark.core
@pytest.mark.parametrize("p_test", [P_3D_CENTER, P_3D_VERTEX, P_3D_ASYMMETRIC])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_potential_gaussian_returns_scalar(p_test, alpha):
    universe = helpers.create_test_universe(K=3, alpha=alpha)
    sources = [SOURCE_A, SOURCE_B]
    potential = pot_np.potential_gaussian(p_test, sources, universe)
    assert isinstance(potential, float)
    assert np.isfinite(potential)

@pytest.mark.core
def test_potential_gaussian_zero_sources_is_zero():
    universe = helpers.create_test_universe(K=3, alpha=1.0)
    potential = pot_np.potential_gaussian(P_3D_ASYMMETRIC, [], universe)
    assert np.isclose(potential, 0.0)

@pytest.mark.core
@pytest.mark.parametrize("p_test", [P_3D_CENTER, P_3D_VERTEX, P_3D_ASYMMETRIC])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_calculate_F_net_returns_correct_shape(p_test, alpha):
    universe = helpers.create_test_universe(K=3, alpha=alpha)
    sources = [SOURCE_A, SOURCE_B]
    F_net = force_np.calculate_F_net(p_test, sources, universe)
    assert F_net.shape == (3,)
    assert np.isclose(np.sum(F_net), 0.0)

@pytest.mark.consistency
@pytest.mark.parametrize("p_test", [P_3D_CENTER, P_3D_VERTEX, P_3D_ASYMMETRIC])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.5])
def test_numpy_numerical_vs_jax_analytical_gradient(p_test, alpha):
    K = len(p_test)
    universe_np = helpers.create_test_universe(K=K, alpha=alpha, backend='numpy')
    sources = [SOURCE_A, SOURCE_B]
    p_jax = jnp.array(p_test)
    from cgd.geometry.transformations import exp_map_from_origin
    sources_p_stim_jax = jnp.array([exp_map_from_origin(s.v_eigen) for s in sources])
    sources_strength_jax = jnp.array([s.strength for s in sources])
    F_net_numpy = force_np.calculate_F_net(p_test, sources, universe_np)
    F_net_jax = force_jax.calculate_F_net_jax(
        p_jax,
        sources_p_stim_jax,
        sources_strength_jax,
        alpha,
        universe_np.simplex.radius,
        K
    )
    np.testing.assert_allclose(
        F_net_numpy,
        F_net_jax,
        atol=1e-5,
        err_msg="Numerical (NumPy) and analytical (JAX) gradients do not match."
    )
