# cgd/core/source.py
"""The GravitationalSource class, representing matter in a CGD universe.

This module defines the GravitationalSource, an immutable dataclass that
encapsulates the intrinsic properties of a source of "gravitational" potential.
Crucially, it is "aware" of its own theoretical level ('absolute' vs.
'substitution') and its native coordinate system ('labels') to ensure
correct scientific usage and unambiguous dimensional transformations.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from ..geometry.transformations import exp_map_from_origin, log_map_from_origin


@dataclass(frozen=True)
class GravitationalSource:
    """An immutable representation of a source of potential in a CGD universe.

    This class is the digital representation of "matter" in a CGD simulation.
    Each source is fully defined by its "physical DNA" (v_eigen), its
    theoretical type (v_type), and the coordinate system in which it was
    defined (labels).

    Attributes:
        name (str): A unique identifier for the source.
        v_eigen (NDArray[np.float64]): The source's "eigen-effect" vector. This is a
            zero-sum vector in the tangent space of the probability simplex,
            representing the direction and magnitude of the source's influence.
        v_type (Literal['absolute', 'substitution']): The theoretical type of
            the v_eigen vector.
            - 'absolute': Represents a complete physical state or law. These
              sources have a standalone meaning and can be meaningfully
              projected into different dimensional spaces using the `embed`
              method.
            - 'substitution': Represents the difference between two 'absolute'
              sources. These sources are only meaningful for comparisons within
              their original dimensional space. They lack the theoretical basis
              for cross-dimensional projection.
        labels (Tuple[str, ...]): A tuple of strings defining the coordinate
            axes for v_eigen. This labeling is fundamental for performing
            unambiguous dimensional transformations.
        strength (float): The Euclidean norm (||v_eigen||) of the eigen-effect
            vector, representing the source's intrinsic strength. It is
            computed automatically after initialization.
    """

    name: str
    v_eigen: NDArray[np.float64]
    v_type: Literal["absolute", "substitution"]
    labels: Tuple[str, ...]
    strength: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Computes the strength attribute and validates the source's data."""
        # Use object.__setattr__ because the dataclass is frozen
        strength_val = np.linalg.norm(self.v_eigen)
        object.__setattr__(self, "strength", strength_val)

        if self.v_type not in ["absolute", "substitution"]:
            raise ValueError(
                f"v_type must be 'absolute' or 'substitution', but got "
                f"'{self.v_type}'."
            )

        if len(self.v_eigen) != len(self.labels):
            raise ValueError(
                f"Source '{self.name}': v_eigen dimension ({len(self.v_eigen)})"
                f" must match the number of labels ({len(self.labels)})."
            )

        if not np.isclose(np.sum(self.v_eigen), 0):
            raise ValueError(
                f"Source '{self.name}': v_eigen components must sum to zero "
                f"(current sum is {np.sum(self.v_eigen):.2e})."
            )

    def embed(self, new_labels: List[str]) -> GravitationalSource:
        """Projects the source into a new dimensional space via label matching.

        This method implements an embedding based on the principle of "conservation
        of deviation pattern". It decodes the source's effect vector into a
        probability distribution, reconstructs this pattern onto the new set of
        labels, and then re-encodes the result into a new effect vector. This
        approach is theoretically pure and general, handling up-projection,
        down-projection, and reordering of dimensions simultaneously.

        Warning:
            This method is theoretically designed only for 'absolute' type
            sources. Applying it to 'substitution' type sources is
            mathematically ill-defined and will produce unreliable results.

        Args:
            new_labels (List[str]): The list of atomic event labels defining
                the target dimensional space.

        Returns:
            GravitationalSource: A new source object projected into the target
                dimensional space.
        """
        old_labels = list(self.labels)

        if new_labels == old_labels:
            return self

        if self.v_type == "substitution":
            warnings.warn(
                f"Warning: Calling .embed() on a 'substitution' type source "
                f"'{self.name}'. This operation is not theoretically sound "
                "and the results may be unreliable.",
                UserWarning,
            )

        # 1. Decode the old pattern and associate with labels
        p_old_dict = dict(zip(old_labels, exp_map_from_origin(self.v_eigen)))

        # 2. Reconstruct the probability distribution on the new canvas
        p_new_draft: NDArray[np.float64] = np.zeros(len(new_labels))
        for i, label in enumerate(new_labels):
            if label in p_old_dict:
                p_new_draft[i] = p_old_dict[label]

        # 3. Renormalize and encode the new effect
        sum_draft = np.sum(p_new_draft)
        v_embedded: NDArray[np.float64]
        if sum_draft < 1e-12:
            warnings.warn(
                f"During embed for source '{self.name}', no overlap was found "
                f"between old labels {old_labels} and new labels "
                f"{new_labels}. Returning a zero-effect source."
            )
            v_embedded = np.zeros(len(new_labels))
        else:
            p_new_normalized: NDArray[np.float64] = p_new_draft / sum_draft
            v_embedded = log_map_from_origin(p_new_normalized)

        return GravitationalSource(
            name=self.name,
            v_eigen=v_embedded,
            v_type=self.v_type,
            labels=tuple(new_labels),
        )

    def to_substitution(
        self, reference_source: GravitationalSource
    ) -> GravitationalSource:
        """Converts an 'absolute' source to a 'substitution' source.

        This is done by subtracting a 'reference' absolute source. The resulting
        source represents the effect of substituting the reference with this
        source.

        Args:
            reference_source (GravitationalSource): Another 'absolute' source
                that serves as the baseline for comparison. It must share the
                same labels as the current source.

        Returns:
            GravitationalSource: A new 'substitution' type source representing
                the difference between the two absolute sources.

        Raises:
            TypeError: If either this source or the reference source is not
                of type 'absolute'.
            ValueError: If the labels of the two sources do not match.
        """
        if self.v_type != "absolute" or reference_source.v_type != "absolute":
            raise TypeError(
                "The .to_substitution() method can only be used to subtract "
                "two 'absolute' type sources."
            )
        if self.labels != reference_source.labels:
            raise ValueError("Source and reference must have the exact same labels.")

        v_sub = self.v_eigen - reference_source.v_eigen
        new_name = f"{self.name}_vs_{reference_source.name}"

        return GravitationalSource(
            name=new_name, v_eigen=v_sub, v_type="substitution", labels=self.labels
        )

    def __repr__(self) -> str:
        """Provides a concise string representation of the source."""
        # Truncate label list for cleaner output
        labels_repr: List[str] = list(self.labels[:3])
        if len(self.labels) > 3:
            labels_repr = labels_repr + ["..."]

        return (
            f"GravitationalSource(name='{self.name}', type='{self.v_type}', "
            f"K={len(self.v_eigen)}, strength={self.strength:.4f}, "
            f"labels={labels_repr})"
        )

    def __add__(self, other: GravitationalSource) -> GravitationalSource:
        """Overloads the addition operator for linear superposition of effects."""
        if self.labels != other.labels:
            raise ValueError("Sources can only be added if they share the same labels.")

        # The sum is a substitution if either operand is one
        new_v_type: Literal["absolute", "substitution"] = (
            "substitution"
            if "substitution" in (self.v_type, other.v_type)
            else "absolute"
        )
        new_v_eigen = self.v_eigen + other.v_eigen
        new_name = f"({self.name})+({other.name})"

        return GravitationalSource(
            name=new_name, v_eigen=new_v_eigen, v_type=new_v_type, labels=self.labels
        )

    def __sub__(self, other: GravitationalSource) -> GravitationalSource:
        """Overloads the subtraction operator, primarily for substitution."""
        if self.labels != other.labels:
            raise ValueError(
                "Sources can only be subtracted if they share the same labels."
            )

        if self.v_type == "absolute" and other.v_type == "absolute":
            # This is the primary use case: creating a substitution effect
            return self.to_substitution(other)

        if self.v_type == other.v_type:
            # Subtraction of two substitution sources is also allowed
            new_v_eigen = self.v_eigen - other.v_eigen
            new_name = f"({self.name})-({other.name})"
            return GravitationalSource(
                name=new_name,
                v_eigen=new_v_eigen,
                v_type=self.v_type,
                labels=self.labels,
            )
        else:
            raise TypeError(
                f"Direct subtraction between different v_types ('{self.v_type}'"
                f" and '{other.v_type}') is not permitted. If the intent is to "
                "calculate a substitution effect, ensure both sources are of "
                "'absolute' type."
            )

    def __mul__(self, scalar: float) -> GravitationalSource:
        """Overloads the multiplication operator for scaling the effect."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("A GravitationalSource can only be multiplied by a scalar.")

        new_v_eigen: NDArray[np.float64] = self.v_eigen * scalar
        new_name = f"{self.name}*{scalar:.2f}"

        return GravitationalSource(
            name=new_name, v_eigen=new_v_eigen, v_type=self.v_type, labels=self.labels
        )
