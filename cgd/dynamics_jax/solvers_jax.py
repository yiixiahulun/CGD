# cgd/dynamics_jax/solvers_jax.py
"""High-performance JAX-based solver for finding equilibrium points.

This module provides `EquilibriumFinderJax`, a JAX-native implementation for
finding stable equilibrium points in a CGD universe. It uses a JIT-compiled
coordinate descent algorithm for local minimization, offering significant
performance advantages over the NumPy-based solver.

Note:
    Trajectory simulation is currently handled by the NumPy-based
    `TrajectorySimulator` and is not yet implemented in JAX.
"""
from __future__ import annotations

import os
import warnings
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.stats import qmc

if TYPE_CHECKING:
    from ..core import Universe, GravitationalSource

from ..geometry import distance_FR, exp_map_from_origin
from ..geometry._jax_impl import distance_FR_jax
from .force_jax import calculate_F_net_jax
from .potential_jax import potential_gaussian_jax


# ============================================================================
# Core JAX-native Solver Function
# ============================================================================
@partial(jit, static_argnames=("K", "max_rounds", "cd_steps", "learning_rate"))
def _solve_cd_single_jax(
    p_initial: jax.Array,
    sources_p_stim: jax.Array,
    sources_strength: jax.Array,
    alpha: float,
    radius: float,
    K: int,
    max_rounds: int,
    cd_steps: int,
    learning_rate: float,
) -> jax.Array:
    """A fully JAX-native coordinate descent solver, designed for JIT compilation.

    This function performs local minimization of the potential energy landscape
    using a coordinate descent approach where each 1D minimization is achieved
    through gradient descent.

    Args:
        p_initial (jax.Array): The starting point for the minimization.
        sources_p_stim (jax.Array): JAX array of source stimulus locations.
        sources_strength (jax.Array): JAX array of source strengths.
        alpha (float): The universe's alpha parameter.
        radius (float): The radius of the simplex.
        K (int): The dimensionality of the universe (static).
        max_rounds (int): Number of full coordinate descent cycles (static).
        cd_steps (int): Number of gradient descent steps for each 1D slice (static).
        learning_rate (float): Step size for the gradient descent (static).

    Returns:
        jax.Array: The position of the found local minimum.
    """

    def find_min_on_slice_jax(p_start: jax.Array, d_opt: int) -> jax.Array:
        """Helper to find the minimum on a 1D slice using gradient descent."""
        d_comp = K - 1
        s_val: float = (
            1.0 - cast(float, jnp.sum(p_start)) + p_start[d_opt] + p_start[d_comp]
        )
        s_val = jnp.maximum(s_val, 1e-9)

        def gd_step(_: int, x_current: float) -> float:
            def objective_1d(x: float) -> float:
                p_vec = p_start.at[d_opt].set(x).at[d_comp].set(s_val - x)
                return cast(
                    float,
                    potential_gaussian_jax(
                        p_vec, sources_p_stim, sources_strength, alpha, radius, K
                    ),
                )
            g: float = grad(objective_1d)(x_current)
            x_next = x_current - learning_rate * g
            return cast(float, jnp.clip(x_next, 0, s_val))

        final_x: float = jax.lax.fori_loop(0, cd_steps, gd_step, s_val / 2.0)
        p_res = p_start.at[d_opt].set(final_x).at[d_comp].set(s_val - final_x)
        return p_res

    def main_loop_body(_: int, p_current: jax.Array) -> jax.Array:
        """One full round of coordinate descent."""

        def cd_loop_body(i: int, p_inner: jax.Array) -> jax.Array:
            return find_min_on_slice_jax(p_inner, i)

        p_after_round = jax.lax.fori_loop(0, K - 1, cd_loop_body, p_current)
        return p_after_round

    p_final = jax.lax.fori_loop(0, max_rounds, main_loop_body, p_initial)
    return p_final / cast(float, jnp.sum(p_final))


# ============================================================================
# EquilibriumFinderJax Class
# ============================================================================


class EquilibriumFinderJax:
    """A high-performance JAX version of the EquilibriumFinder.

    This class provides the same functionality as its NumPy counterpart but
    leverages JAX for the core minimization routine, which is JIT-compiled
    for significant speedup. It manages the conversion of data into
    JAX-compatible formats and wraps the JAX-native solver.

    Args:
        universe (Universe): The universe defining the simulation space.
        sources (List[GravitationalSource]): The sources generating the potential.
    """

    def __init__(self, universe: "Universe", sources: List["GravitationalSource"]):
        self.universe: "Universe" = universe
        self.sources: List["GravitationalSource"] = sources
        self.K: int = universe.K
        self.sources_p_stim_jax: jax.Array
        self.sources_strength_jax: jax.Array
        self._prepare_jax_inputs()

    def _prepare_jax_inputs(self) -> None:
        """Flattens the list of source objects into JAX-compatible arrays."""
        p_stims_np: List[NDArray[np.float64]] = [
            exp_map_from_origin(s.v_eigen) for s in self.sources
        ]
        strengths_np: List[float] = [s.strength for s in self.sources]

        self.sources_p_stim_jax = jnp.array(p_stims_np)
        self.sources_strength_jax = jnp.array(strengths_np)

    def _generate_intelligent_guesses(
        self, num_random_seeds: int
    ) -> List[NDArray[np.float64]]:
        """Generates a diverse set of high-quality initial guesses.

        (This method is identical to the one in the NumPy version).
        """
        k: int = self.K
        guesses: List[NDArray[np.float64]] = [self.universe.simplex.origin]

        # This will not be None, established in Universe.__post_init__
        universe_labels: List[str] = list(cast(Tuple[str, ...], self.universe.labels))

        try:
            v_net_lesa: NDArray[np.float64] = sum(
                s.embed(universe_labels).v_eigen for s in self.sources
            )
            p_lesa_guess = exp_map_from_origin(v_net_lesa)
            guesses.append(p_lesa_guess)
        except Exception as e:
            warnings.warn(f"LESA initial guess failed: {e}")

        for i in range(k):
            p: NDArray[np.float64] = np.full(k, 0.01 / (k - 1) if k > 1 else 0)
            p[i] = 0.99
            guesses.append(p)
        if k > 2:
            for i, j in combinations(range(k), 2):
                p = np.zeros(k)
                p[i], p[j] = 0.5, 0.5
                guesses.append(p)

        if num_random_seeds > 0:
            n_power_of_2 = 1 << (num_random_seeds - 1).bit_length()
            try:
                sampler = qmc.Sobol(d=k, scramble=True)
                samples: NDArray[np.float64] = sampler.random(n=n_power_of_2)
                samples /= np.sum(samples, axis=1, keepdims=True)
                guesses.extend(samples)
            except Exception:
                dirichlet_samples: NDArray[np.float64] = np.random.dirichlet(
                    np.ones(k), size=num_random_seeds
                )
                guesses.extend(dirichlet_samples)

        return guesses

    def _solve_from_single_point_wrapper(
        self, p_initial_np: NDArray[np.float64], solver_params: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """A Python wrapper to call the JIT-compiled JAX core solver.

        This function handles the data conversion from NumPy to JAX and back,
        allowing the JIT-compiled solver to be called from a standard Python
        loop (e.g., in `joblib`).
        """
        p_initial_jax: jax.Array = jnp.array(p_initial_np)

        p_final_jax: jax.Array = _solve_cd_single_jax(
            p_initial=p_initial_jax,
            sources_p_stim=self.sources_p_stim_jax,
            sources_strength=self.sources_strength_jax,
            alpha=self.universe.alpha,
            radius=self.universe.simplex.radius,
            K=self.K,
            **solver_params,
        )
        return np.array(p_final_jax)

    def _validate_point_stability_jax(
        self, p_star: NDArray[np.float64], grad_tolerance: float = 1e-5
    ) -> bool:
        """Validates stability using the JAX backend (gradient check only)."""
        p_star_jax: jax.Array = jnp.array(p_star)
        force_jax: jax.Array = calculate_F_net_jax(
            p_star_jax,
            self.sources_p_stim_jax,
            self.sources_strength_jax,
            self.universe.alpha,
            self.universe.simplex.radius,
            self.K,
        )
        return cast(bool, np.linalg.norm(force_jax) < grad_tolerance)

    def find(
        self,
        num_random_seeds: int = 20,
        uniqueness_tolerance: float = 1e-4,
        validate_stability: bool = False,
        n_jobs: int = -1,
        max_rounds: int = 50,
        cd_steps: int = 15,
        learning_rate: float = 0.05
    ) -> List[NDArray[np.float64]]:
        """Finds all unique, stable equilibrium points using the JAX backend.

        Args:
            num_random_seeds (int): Number of quasi-random starting points.
            uniqueness_tolerance (float): Fisher-Rao distance for considering
                two points as identical.
            validate_stability (bool): If True, filters results to include only
                points with a near-zero gradient. (Note: Hessian check is not
                implemented in the JAX version).
            n_jobs (int): Number of CPU cores for parallel execution.
            max_rounds (int): Number of coordinate descent cycles in the solver.
            cd_steps (int): Number of gradient descent steps per 1D slice.
            learning_rate (float): Step size for the gradient descent.

        Returns:
            List[NDArray[np.float64]]: A list of unique equilibrium points.
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        initial_guesses: List[NDArray[np.float64]] = self._generate_intelligent_guesses(
            num_random_seeds
        )

        solver_params: Dict[str, Any] = {
            "max_rounds": max_rounds,
            "cd_steps": cd_steps,
            "learning_rate": learning_rate,
        }

        candidate_points_raw = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_from_single_point_wrapper)(guess, solver_params)
            for guess in initial_guesses
        )
        candidate_points: List[NDArray[np.float64]] = [
            p for p in candidate_points_raw if p is not None
        ]

        unique_candidates: List[NDArray[np.float64]] = []
        if candidate_points:
            for p in candidate_points:
                if p is not None and np.isfinite(p).all():
                    is_close_to_existing = any(
                        distance_FR(p, up) < uniqueness_tolerance
                        for up in unique_candidates
                    )
                    if not is_close_to_existing:
                        unique_candidates.append(p)

        if not validate_stability:
            return sorted(unique_candidates, key=lambda p: -p.max())
        else:
            final_solutions: List[NDArray[np.float64]] = [
                p for p in unique_candidates if self._validate_point_stability_jax(p)
            ]
            return sorted(final_solutions, key=lambda p: -p.max())
