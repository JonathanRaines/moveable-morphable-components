from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class Point2D(NamedTuple):
    x: float | jnp.ndarray
    y: float | jnp.ndarray


class CircleSpec(NamedTuple):
    center: Point2D
    radius: float | jnp.ndarray


def circle(
    point: Point2D,
) -> tuple[
    Callable[[CircleSpec], float | jnp.ndarray],
    Callable[[CircleSpec], float | jnp.ndarray],
]:
    """Create a topological description function for a circle."""

    def φ(spec: CircleSpec) -> float | jnp.ndarray:
        """Topological Description Function for a circle."""
        return (
            spec.radius**2
            - (point.x - spec.center.x) ** 2
            - (point.y - spec.center.y) ** 2
        )

    J = jax.jacobian(φ)

    return φ, J


class BeamSpec(NamedTuple):
    center: Point2D
    angle: float | jnp.ndarray
    length: float | jnp.ndarray
    thickness: float | jnp.ndarray


def uniform_beam(point: Point2D) -> tuple[Callable, Callable]:
    def φ(spec: BeamSpec) -> float | jnp.ndarray:
        """Topological Description Function for a circle."""
        center, angle, length, thickness = spec
        # because the matrix gets flipped top to bottom
        rotation_matrix: NDArray = jnp.array(
            [
                [jnp.cos(angle), jnp.sin(angle)],
                [-jnp.sin(angle), jnp.cos(angle)],
            ]
        )
        # Local coordinates
        _x, _y = rotation_matrix @ jnp.stack(
            [
                (point.x - center.x),
                (point.y - center.y),
            ]
        )

        return -jnp.maximum(jnp.abs(_x) - length / 2, jnp.abs(_y) - thickness / 2)

    J = jax.jacobian(φ)

    return φ, J


class Component:
    def __call__(self, points: tuple[NDArray[float] | float]) -> NDArray:
        return self.φ(points[0], points[1])

    @property
    def design_variables(self) -> list[float]:
        return [float(v) for v in vars(self).values()]

    @property
    def num_design_variables(self) -> int:
        return len(self.design_variables)

    def φ(self, x: NDArray[float] | float, y: NDArray[float] | float) -> float:
        return jnp.vectorize(self._φ)(*self.design_variables, x, y)

    def _φ(x: float, y: float) -> float:
        raise NotImplementedError

    def φ_grad(self, point: tuple[float] | NDArray[float]) -> float | NDArray[float]:
        design_var_args = np.arange(self.num_design_variables)
        f = partial(
            jax.grad(self._φ, argnums=design_var_args), *list(self.design_variables)
        )

        return jnp.vectorize(f)(point[0], point[1])


class Circle(Component):
    def __init__(self, center: Point2D, radius: float):
        self.x = center.x
        self.y = center.y
        self.radius = radius

    def __call__(self, x: NDArray, y: NDArray) -> NDArray:
        return self.φ(x, y)

    def φ(self, x: NDArray, y: NDArray) -> float:
        return self._φ(*self.design_variables, x, y)

    def _φ(
        self, center_x: float, center_y: float, radius: float, x: NDArray, y: NDArray
    ) -> NDArray:
        φ: float = radius**2 - (x - center_x) ** 2 - (y - center_y) ** 2
        return φ


class UniformBeam(Component):
    def __init__(self, center: Point2D, angle: float, length: float, thickness: float):
        """A beam of uniform thickness.

        Args:
            center (Point2D): Center of the beam
            angle (float): rotation relative to x-axis in radians
            length (float): length of the beam
            thickness (float): thickness of the beam
        """
        self.x = center.x
        self.y = center.y
        self.angle = angle
        self.length = length
        self.thickness = thickness

    def _φ(
        self,
        center_x: float,
        center_y: float,
        angle: float,
        length: float,
        thickness: float,
        x: float,
        y: float,
    ) -> float:
        # -angle because the matrix gets flipped top to bottom
        rotation_matrix: NDArray = jnp.array(
            [
                [jnp.cos(angle), jnp.sin(angle)],
                [-jnp.sin(angle), jnp.cos(angle)],
            ]
        )
        # Local coordinates
        _x, _y = rotation_matrix @ jnp.stack(
            [
                (x - center_x),
                (y - center_y),
            ]
        )

        φ: NDArray = -jnp.maximum(jnp.abs(_x) - length / 2, jnp.abs(_y) - thickness / 2)
        return φ

    def __repr__(self) -> str:
        return f"UniformBeam(center=({self.x}, {self.y}), angle={float(np.rad2deg(self.angle)):2f}°, length={self.length:2f}, thickness={self.thickness:2f})"

    def __str__(self) -> str:
        return self.__repr__()


class UniformBeamFixedThickness(Component):
    def __init__(self, center: Point2D, angle: float, length: float, thickness: float):
        """A beam of uniform thickness.

        Args:
            center (Point2D): Center of the beam
            angle (float): rotation relative to x-axis in radians
            length (float): length of the beam
            thickness (float): thickness of the beam
        """
        self.x = center.x
        self.y = center.y
        self.angle = angle
        self.length = length
        self.thickness = thickness

    # Override the design variables to exclude the thickness
    @Component.design_variables.getter
    def design_variables(self) -> list[float]:
        return [float(v) for v in [self.x, self.y, self.angle, self.length]]

    def _φ(
        self,
        center_x: float,
        center_y: float,
        angle: float,
        length: float,
        x: float,
        y: float,
    ) -> float:
        # -angle because the matrix gets flipped top to bottom
        rotation_matrix: NDArray = jnp.array(
            [
                [jnp.cos(-angle), -jnp.sin(-angle)],
                [jnp.sin(-angle), jnp.cos(-angle)],
            ]
        )
        # Local coordinates
        _x, _y = rotation_matrix @ jnp.stack(
            [
                (x - center_x),
                (y - center_y),
            ]
        )

        φ: NDArray = -jnp.maximum(
            jnp.abs(_x) - length / 2, jnp.abs(_y) - self.thickness / 2
        )
        return φ

    def __repr__(self) -> str:
        return f"UniformBeamFixedThickness(center=({self.x}, {self.y}), angle={float(np.rad2deg(self.angle)):2f}°, length={self.length:2f}, thickness={self.thickness:2f})"

    def __str__(self) -> str:
        return self.__repr__()
