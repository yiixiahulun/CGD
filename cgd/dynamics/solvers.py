# cgd/dynamics/solvers.py
"""Core solvers for CGD dynamics.

This module provides the primary classes for simulating a CGD system:
- EquilibriumFinder: A high-performance, global solver for finding all stable
  attractors (equilibrium points) in a potential field.
- TrajectorySimulator: An ordinary differential equation (ODE) solver for
  simulating the time evolution of a system's trajectory.
"""
from __future__ import annotations

import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any, List, Tuple, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.stats import qmc

if TYPE_CHECKING:
    from ..core.universe import Universe
    from ..core.source import GravitationalSource

from ..dynamics import calculate_F_net, potential_gaussian
from ..geometry import distance_FR, exp_map_from_origin


class EquilibriumFinder:
    """Finds stable equilibrium points (attractors) in a CGD universe.

    This class uses a multi-stage, global optimization strategy to locate all
    points in the simplex where the net force is zero and the location is a
    stable minimum of the potential. It combines heuristic guesses, quasi-random
    sampling, and parallelized local minimization.

    Args:
        universe (Universe): The universe defining the simulation space.
        sources (List[GravitationalSource]): The sources generating the potential.
    """

    def __init__(self, universe: "Universe", sources: List["GravitationalSource"]):
        self.universe: "Universe" = universe
        self.sources: List["GravitationalSource"] = sources
        self.K: int = universe.K

    def _find_min_on_slice(
        self, p_start: NDArray[np.float64], dim_to_optimize: int, compensating_dim: int
    ) -> NDArray[np.float64]:
        """Helper for coordinate descent: minimizes potential along a 1D slice."""
        k: int = self.K
        fixed_indices: List[int] = [
            i for i in range(k) if i != dim_to_optimize and i != compensating_dim
        ]
        fixed_sum: float = np.sum(p_start[fixed_indices])
        sum_val: float = 1.0 - fixed_sum
        if sum_val < 0:
            return p_start

        def objective_1d(x: float) -> float:
            p_vec: NDArray[np.float64] = np.zeros(k)
            p_vec[dim_to_optimize] = x
            p_vec[compensating_dim] = sum_val - x
            if fixed_indices:
                p_vec[fixed_indices] = p_start[fixed_indices]
            return potential_gaussian(p_vec, self.sources, self.universe)

        res: OptimizeResult = minimize_scalar(
            objective_1d,
            bounds=(0, sum_val),
            method="bounded",
            options={"xatol": 1e-10},
        )
        p_res: NDArray[np.float64] = np.zeros(k)
        p_res[dim_to_optimize] = res.x
        p_res[compensating_dim] = sum_val - res.x
        if fixed_indices:
            p_res[fixed_indices] = p_start[fixed_indices]
        return p_res

    def _solve_cd_from_single_point(
        self,
        p_initial: NDArray[np.float64],
        tolerance: float = 1e-8,
        max_rounds: int = 50,
    ) -> NDArray[np.float64]:
        """Performs local minimization from a single starting point.

        Uses a coordinate descent algorithm to find a local minimum of the
        potential energy landscape.

        Args:
            p_initial (NDArray[np.float64]): The starting point for the minimization.
            tolerance (float): The convergence tolerance, based on the Fisher-Rao
                distance between iterations.
            max_rounds (int): The maximum number of full coordinate descent
                cycles to perform.

        Returns:
            NDArray[np.float64]: The position of the found local minimum.
        """
        p_current: NDArray[np.float64] = np.array(p_initial, dtype=float)
        if np.abs(np.sum(p_current) - 1.0) > 1e-9:
            p_current /= np.sum(p_current)

        compensating_dim: int = self.K - 1
        dims_to_optimize: List[int] = list(range(self.K - 1))

        for _ in range(max_rounds):
            p_before_round: NDArray[np.float64] = p_current.copy()
            for i in dims_to_optimize:
                p_current = self._find_min_on_slice(p_current, i, compensating_dim)
            if np.abs(np.sum(p_current) - 1.0) > 1e-9:
                p_current /= np.sum(p_current)

            if distance_FR(p_current, p_before_round) < tolerance:
                break
        return p_current

    def _generate_intelligent_guesses(
        self, num_random_seeds: int
    ) -> List[NDArray[np.float64]]:
        """Generates a diverse set of high-quality initial guesses.

        This method creates a list of starting points for the local minimizer,
        combining heuristic, deterministic, and quasi-random strategies to
        ensure good coverage of the search space.

        Args:
            num_random_seeds (int): The number of quasi-random (Sobol sequence)
                points to generate.

        Returns:
            List[NDArray[np.float64]]: A list of initial guess points.
        """
        k: int = self.K
        guesses: List[NDArray[np.float64]] = []

        # --- LESA Prediction (Heuristic Guess) ---
        try:
            v_net_lesa: NDArray[np.float64] = np.zeros(k)
            # This will not be None, established in Universe.__post_init__
            universe_labels: List[str] = list(
                cast(Tuple[str, ...], self.universe.labels)
            )
            for s in self.sources:
                embedded_source = s.embed(new_labels=universe_labels)
                v_net_lesa += embedded_source.v_eigen
            p_lesa_guess = exp_map_from_origin(v_net_lesa)
            guesses.extend([self.universe.simplex.origin, p_lesa_guess])
        except Exception as e:
            warnings.warn(
                f"LESA initial guess failed ({type(e).__name__}: {e}). This may"
                " be due to incompatible source labels. Falling back to using "
                "only the universe origin as the heuristic starting point."
            )
            guesses.append(self.universe.simplex.origin)

        # --- Deterministic Guesses (Vertices and Edges) ---
        for i in range(k):
            p: NDArray[np.float64] = np.full(k, 0.01 / (k - 1) if k > 1 else 0)
            p[i] = 0.99
            guesses.append(p)

        if k > 2:
            for i, j in combinations(range(k), 2):
                p = np.zeros(k)
                p[i], p[j] = 0.5, 0.5
                guesses.append(p)

        # --- Quasi-Random Sampling (Sobol Sequence) ---
        if num_random_seeds > 0:
            n_power_of_2 = 1 << (num_random_seeds - 1).bit_length()
            try:
                sampler = qmc.Sobol(d=k, scramble=True)
                samples: NDArray[np.float64] = sampler.random(n=n_power_of_2)
                samples /= np.sum(samples, axis=1, keepdims=True)
                guesses.extend(samples)
            except Exception:
                # Fallback to Dirichlet if Sobol fails
                samples_dirichlet: NDArray[np.float64] = np.random.dirichlet(
                    np.ones(k), size=num_random_seeds
                )
                guesses.extend(samples_dirichlet)

        return guesses

    def _calculate_hessian(
        self, p_star: NDArray[np.float64], h: float = 1e-5
    ) -> NDArray[np.float64]:
        """Numerically computes the Hessian matrix of the potential at a point."""
        hessian: NDArray[np.float64] = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                # Central difference for second partial derivatives
                p_ii, p_io = p_star.copy(), p_star.copy()
                p_oi, p_oo = p_star.copy(), p_star.copy()
                p_ii[i] += h
                p_ii[j] += h
                p_io[i] += h
                p_io[j] -= h
                p_oi[i] -= h
                p_oi[j] += h
                p_oo[i] -= h
                p_oo[j] -= h
                for p_vec in [p_ii, p_io, p_oi, p_oo]:
                    p_vec /= np.sum(p_vec)
                U_ii = potential_gaussian(p_ii, self.sources, self.universe)
                U_io = potential_gaussian(p_io, self.sources, self.universe)
                U_oi = potential_gaussian(p_oi, self.sources, self.universe)
                U_oo = potential_gaussian(p_oo, self.sources, self.universe)
                hessian[i, j] = hessian[j, i] = (U_ii - U_io - U_oi + U_oo) / (4 * h**2)
        return hessian

    def _validate_point_stability(
        self, p_star: NDArray[np.float64], grad_tolerance: float = 1e-4
    ) -> str:
        """Checks if a point is a stable minimum."""
        force: NDArray[np.float64] = calculate_F_net(
            p_star, self.sources, self.universe
        )
        if np.linalg.norm(force) > grad_tolerance:
            return f"Unstable (High Gradient: {np.linalg.norm(force):.2e})"

        hessian = self._calculate_hessian(p_star)
        eigenvalues: NDArray[np.float64] = eigh(hessian, eigvals_only=True)
        # Filter out the zero eigenvalue corresponding to movement off the simplex
        relevant_eigenvalues: NDArray[np.float64] = eigenvalues[
            np.abs(eigenvalues) > 1e-6
        ]

        if np.any(relevant_eigenvalues < 0):
            num_negative: int = np.sum(relevant_eigenvalues < 0)
            return f"Unstable (Saddle Point, {num_negative} negative eigval(s))"
        return "Stable"

    def find(
        self,
        num_random_seeds: int = 20,
        uniqueness_tolerance: float = 1e-4,
        validate_stability: bool = False,
        n_jobs: int = -1,
    ) -> List[NDArray[np.float64]]:
        """Finds all unique, stable equilibrium points.

        Args:
            num_random_seeds (int): Number of quasi-random starting points to use.
            uniqueness_tolerance (float): The tolerance for considering two
                points as identical.
            validate_stability (bool): If True, performs a Hessian analysis to
                filter out unstable saddle points, returning only true minima.
            n_jobs (int): The number of CPU cores to use for parallel execution.
                -1 means using all available cores.

        Returns:
            List[NDArray[np.float64]]: A list of unique equilibrium points, sorted.
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        initial_guesses = self._generate_intelligent_guesses(num_random_seeds)

        candidate_points_raw = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_cd_from_single_point)(guess)
            for guess in initial_guesses
        )
        candidate_points: List[NDArray[np.float64]] = [
            p for p in candidate_points_raw if p is not None
        ]

        unique_candidates: List[NDArray[np.float64]] = []
        if candidate_points:
            for p in candidate_points:
                if p is not None and np.isfinite(p).all():
                    is_unique = not any(
                        np.allclose(p, up, atol=uniqueness_tolerance)
                        for up in unique_candidates
                    )
                    if is_unique:
                        unique_candidates.append(p)

        if not validate_stability:
            return sorted(unique_candidates, key=lambda p: p[0], reverse=True)
        else:
            final_solutions: List[NDArray[np.float64]] = []
            for p_candidate in unique_candidates:
                verdict = self._validate_point_stability(p_candidate)
                if verdict == "Stable":
                    final_solutions.append(p_candidate)
            return sorted(final_solutions, key=lambda p: p[0], reverse=True)


class TrajectorySimulator:
    """Simulates the trajectory of a particle in a CGD potential field.

    This class wraps an ODE solver (`scipy.integrate.solve_ivp`) to trace the
    path of a particle over time, governed by the net force from the potential.

    Args:
        universe (Universe): The universe defining the simulation space.
        sources (List[GravitationalSource]): The sources generating the potential.
        p_initial (NDArray[np.float64]): The starting position of the particle.
        T (float): The total simulation time.
        num_points (int): The number of points at which to evaluate the trajectory.
    """

    def __init__(
        self,
        universe: Universe,
        sources: List[GravitationalSource],
        p_initial: NDArray[np.float64],
        T: float = 1.0,
        num_points: int = 100,
        **kwargs: Any,
    ):
        self.universe: Universe = universe
        self.sources: List[GravitationalSource] = sources
        self.p_initial: NDArray[np.float64] = p_initial
        self.T: float = T
        self.num_points: int = num_points

    def _dynamics(self, t: float, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """The system of ODEs: dp/dt = F(p)."""
        # Ensure the point stays on the simplex for the force calculation
        p_normalized: NDArray[np.float64] = np.maximum(p, 0)
        p_normalized /= np.sum(p_normalized)
        return calculate_F_net(p_normalized, self.sources, self.universe)

    def simulate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Runs the trajectory simulation.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: A tuple containing:
            - t_points (NDArray[np.float64]): The time points of the simulation.
            - trajectory (NDArray[np.float64]): The positions of the particle at each
              time point.
        """
        t_eval: NDArray[np.float64] = np.linspace(0, self.T, self.num_points)
        solution = solve_ivp(
            self._dynamics,
            [0, self.T],
            self.p_initial,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
        )
        trajectory: NDArray[np.float64] = solution.y.T
        # Normalize each step to ensure it remains a valid probability distribution
        trajectory_normalized: NDArray[np.float64] = np.array(
            [row / np.sum(row) for row in trajectory]
        )
        return solution.t, trajectory_normalized
