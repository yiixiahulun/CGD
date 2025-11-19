# cgd/core/universe.py
"""The Universe class, a container for CGD simulation settings.

This module defines the Universe class, which encapsulates the physical
properties (dimensionality and force law) and the computational backend
(NumPy or JAX) for a specific CGD simulation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from ..geometry.simplex import Simplex

if TYPE_CHECKING:
    from ..dynamics.solvers import EquilibriumFinder, TrajectorySimulator
    from ..dynamics_jax.solvers_jax import EquilibriumFinderJax
    from .source import GravitationalSource


@dataclass
class Universe:
    """A class that encapsulates the physics and backend of a CGD universe.

    The Universe is the primary container for a CGD simulation. It defines the
    dimensionality of the probability simplex, the force law through the `alpha`
    parameter, and selects the computational backend. It also provides high-level
    methods for running simulations like finding equilibria and tracing trajectories.

    Attributes:
        K (int): The dimensionality of the universe, representing the number of
            distinct atomic events.
        alpha (float): A non-negative float that determines the "sharpness" or
            range of the Gaussian potential. Higher values lead to shorter-range
            forces.
        labels (Optional[Tuple[str, ...]]): An optional tuple of strings to label
            the K dimensions. If None, integer labels ('0', '1', ...) are
            automatically generated. Defaults to None.
        backend (Literal['numpy', 'jax']): The computational backend to use for
            simulations. 'numpy' is the default, while 'jax' offers higher

            performance through JIT compilation and automatic differentiation.
            Defaults to 'numpy'.
        simplex (Simplex): An object holding pre-calculated geometric properties
            of the K-dimensional simplex, such as its origin and radius.
            Automatically initialized.
    """

    K: int
    alpha: float
    labels: Optional[Tuple[str, ...]] = None
    backend: Literal["numpy", "jax"] = "numpy"

    simplex: Simplex = field(init=False)

    _finder_class: Type[EquilibriumFinder] | Type[EquilibriumFinderJax] = field(
        init=False, repr=False
    )
    _simulator_class: Type[TrajectorySimulator] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initializes the simplex and dynamically binds the backend."""
        if self.alpha < 0:
            raise ValueError(
                f"alpha value must be non-negative, but received {self.alpha}."
            )
        self.simplex = Simplex(self.K)

        labels: Tuple[str, ...]
        if self.labels is None:
            labels = tuple(str(i) for i in range(self.K))
        elif len(self.labels) != self.K:
            raise ValueError(
                f"The number of provided labels ({len(self.labels)}) must match "
                f"the universe dimensionality K ({self.K})."
            )
        else:
            labels = tuple(self.labels)
        # mypy requires this weird assignment to convince it that self.labels is not Optional
        self.labels = labels

        # --- Dynamically bind the computational backend ---
        if self.backend == "jax":
            try:
                from ..dynamics_jax.solvers_jax import EquilibriumFinderJax

                self._finder_class = EquilibriumFinderJax
                from ..dynamics.solvers import TrajectorySimulator

                self._simulator_class = TrajectorySimulator
            except ImportError as e:
                raise ImportError(
                    "Could not load the JAX backend. Please ensure JAX is "
                    "installed: `pip install 'jax[cpu]'` or `pip install "
                    "'jax[cuda]'` (for GPU)."
                ) from e
        elif self.backend == "numpy":
            from ..dynamics.solvers import EquilibriumFinder, TrajectorySimulator

            self._finder_class = EquilibriumFinder
            self._simulator_class = TrajectorySimulator
        else:
            raise ValueError(
                f"Unsupported backend: '{self.backend}'. Please choose "
                "'numpy' or 'jax'."
            )

    def _validate_sources(self, sources: List["GravitationalSource"]) -> None:
        """Validates that all sources have the same dimension as the universe.

        Args:
            sources (List['GravitationalSource']): A list of GravitationalSource
                objects to be validated.

        Raises:
            ValueError: If any source's dimension (len(source.v_eigen)) does
                not match the universe's dimension (self.K).
        """
        for source in sources:
            if len(source.v_eigen) != self.K:
                raise ValueError(
                    f"GravitationalSource '{source.name}' (K="
                    f"{len(source.v_eigen)}) is incompatible with the universe "
                    f"dimension (K={self.K}).\nPlease ensure all sources are "
                    f"correctly projected to the target universe '{self.labels}'"
                    " via the .embed() method before calling a Universe method."
                )

    def find_equilibria(
        self, sources: List["GravitationalSource"], **kwargs: Any
    ) -> List[NDArray[np.float64]]:
        """Finds equilibrium points in the universe using the configured backend.

        This method initializes an equilibrium finder with the given sources
        and uses it to locate points in the simplex where the net force is zero.

        Args:
            sources (List['GravitationalSource']): A list of sources whose combined
                potential will be used to find equilibria.
            **kwargs: Additional keyword arguments to be passed directly to the
                backend's `find` method (e.g., `n_samples`, `tol`).

        Returns:
            List[NDArray[np.float64]]: A list of found equilibrium points. Each point is
                a K-dimensional NumPy array representing a probability
                distribution.
        """
        self._validate_sources(sources)
        finder = self._finder_class(universe=self, sources=sources)
        return finder.find(**kwargs)

    def simulate_trajectory(
        self,
        p_initial: NDArray[np.float64],
        sources: List["GravitationalSource"],
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulates the trajectory of a particle from an initial position.

        This method simulates the path of a probability particle through the
        simplex under the influence of the provided gravitational sources.

        Args:
            p_initial (NDArray[np.float64]): The starting probability distribution (a
                K-dimensional NumPy array) for the simulation.
            sources (List['GravitationalSource']): The list of sources generating
                the potential field for the simulation.
            **kwargs: Additional keyword arguments to be passed directly to the
                backend's `simulate` method (e.g., `dt`, `n_steps`).

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: A tuple containing two arrays:
            - The time points of the simulation.
            - The positions (probability distributions) at each time point.
        """
        self._validate_sources(sources)
        simulator = self._simulator_class(
            universe=self, sources=sources, p_initial=p_initial, **kwargs
        )
        return simulator.simulate()

    def __repr__(self) -> str:
        """Provides a concise string representation of the Universe."""
        if not self.labels:
            # This should not be reachable due to __post_init__, but mypy needs it
            return f"Universe(K={self.K}, alpha={self.alpha:.4f}, backend='{self.backend}')"

        labels_repr: List[str] = list(self.labels[:2])
        if len(self.labels) > 2:
            labels_repr = list(labels_repr) + ["..."]
        return (
            f"Universe(K={self.K}, alpha={self.alpha:.4f}, "
            f"backend='{self.backend}', labels={labels_repr})"
        )
