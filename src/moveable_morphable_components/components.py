from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from moveable_morphable_components.domain import Point2D

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ComponentGroup:
    """A collection of components that share the same topology description function and design variables."""

    topology_description_function: Callable[[NamedTuple], np.float64 | jnp.ndarray]
    variable_initial: NDArray[np.float64]
    variable_mins: NDArray[np.float64]
    variable_maxes: NDArray[np.float64]
    frozen_variables: list[int] | None = None

    @property
    def tdf(self) -> Callable[[NamedTuple], np.float64 | jnp.ndarray]:
        """Allow abbreviated access to the topology description function."""
        return self.topology_description_function

    @property
    def free_variable_col_indexes(self) -> NDArray:
        """Indexes of non-frozen design variables."""
        return np.setdiff1d(
            np.arange(self.variable_initial.shape[1]),
            self.frozen_variables,
        )

    @property
    def num_components(self) -> int:
        """Number of components in the group."""
        return self.variable_initial.shape[0]

    @property
    def num_design_variables(self) -> int:
        """Number of free (non-frozen) design variables for the whole group."""
        return self.variable_initial[:, self.free_variable_col_indexes].size

    @property
    def variable_initials_flattened(self) -> NDArray:
        """Starting values of the design variables for each component.

        Stacked into a flat array.
        """
        return self.variable_initial[:, self.free_variable_col_indexes].flatten()

    @property
    def bounds_flattened(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a tuple containing the bounds for each design variable in the group.

        E.g. a group of two circles defined by [Point2D(x,y), radius] could have bounds
        as follows:

        ([Point2D(0.0, 0.0), 0.0, Point2D(0.0, 0.0), 0.0],
            [Point2D(2.0, 1.0), 2.0, Point2D(2.0, 1.0), 2.0])

        The first array is the min for each variable, and the second array is the max.
        """
        mins: NDArray[np.float64] = np.tile(
            self.variable_mins[self.free_variable_col_indexes],
            self.num_components,
        )
        maxes: NDArray[np.float64] = np.tile(
            self.variable_maxes[self.free_variable_col_indexes],
            self.num_components,
        )
        return mins, maxes


class CircleSpec(NamedTuple):
    """Design variables for a circle."""

    center: Point2D
    radius: float | jnp.ndarray


def circle(
    point: Point2D,
) -> Callable[[CircleSpec], np.float64 | jnp.ndarray]:
    """Create a topological description function for a circle."""

    def tdf(spec: CircleSpec) -> np.float64 | jnp.ndarray:
        """Topological Description Function for a circle."""
        center = Point2D(*spec[:2])
        radius = spec[2]
        return radius**2 - (point.x - center.x) ** 2 - (point.y - center.y) ** 2

    return tdf


class BeamSpec(NamedTuple):
    """Design Variables for a beam."""

    center: Point2D
    angle: float | jnp.ndarray
    length: float | jnp.ndarray
    thickness: float | jnp.ndarray


def uniform_beam(
    point: Point2D,
) -> Callable[[BeamSpec], np.float64 | jnp.ndarray]:
    """Create Topological Description Function for a beam with a uniform width."""

    def tdf(spec: BeamSpec) -> np.float64 | jnp.ndarray:
        """Topological Description Function for a beam with a uniform width."""
        center = Point2D(*spec[:2])
        angle, length, thickness = spec[2:]
        # because the matrix gets flipped top to bottom
        rotation_matrix: NDArray = jnp.array(
            [
                [jnp.cos(angle), jnp.sin(angle)],
                [-jnp.sin(angle), jnp.cos(angle)],
            ],
        )
        # Local coordinates
        _x, _y = rotation_matrix @ jnp.stack(
            [
                (point.x - center.x),
                (point.y - center.y),
            ],
        )

        return -jnp.maximum(jnp.abs(_x) - length / 2, jnp.abs(_y) - thickness / 2)

    return tdf
