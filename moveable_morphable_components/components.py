from dataclasses import dataclass
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import plotly.express as px


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


def circle(
    x: float, y: float, center_x: float, center_y: float, radius: float
) -> tuple[Callable, Callable]:
    assert all([isinstance(i, float) for i in [x, y, center_x, center_y, radius]])

    def phi(center_x, center_y, radius, x, y):
        return radius**2 - (x - center_x) ** 2 - (y - center_y) ** 2

    phi_grad = jax.grad(phi, argnums=(0, 1, 2))

    return (
        partial(phi, center_x, center_y, radius),
        partial(
            phi_grad,
            center_x,
            center_y,
            radius,
        ),
    )


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

        φ: NDArray = 1 - jnp.maximum(
            jnp.abs(_x) - length / 2, jnp.abs(_y) - thickness / 2
        )
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
        @property
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

        φ: NDArray = 1 - jnp.maximum(
            jnp.abs(_x) - length / 2, jnp.abs(_y) - self.thickness / 2
        )
        return φ

    def __repr__(self) -> str:
        return f"UniformBeamFixedThickness(center=({self.x}, {self.y}), angle={float(np.rad2deg(self.angle)):2f}°, length={self.length:2f}, thickness={self.thickness:2f})"

    def __str__(self) -> str:
        return self.__repr__()
