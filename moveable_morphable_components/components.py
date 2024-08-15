from dataclasses import dataclass
import itertools
from typing import Callable
from functools import partial

import jax
import jax.numpy as np
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
    def __call__(self, points: tuple | NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def design_variables(self) -> list[float]:
        raise NotImplementedError

    def φ(x: float, y: float) -> float:
        raise NotImplementedError

    def _φ(x: float, y: float) -> float:
        raise NotImplementedError

    def φ_grad(self, x: float, y: float) -> float:
        raise NotImplementedError

    def φ_grad(self, x: float, y: float) -> float:
        design_var_args = np.arange(self.num_design_variables)
        return np.array(
            jax.vmap(
                lambda x, y: jax.grad(self._φ, argnums=design_var_args)(
                    *self.design_variables, x, y
                )
            )(x, y)
        )


class Circle(Component):
    def __init__(self, center: Point2D, radius: float):
        self.x = center.x
        self.y = center.y
        self.radius = radius
        self.num_design_variables = 3

    @property
    def design_variables(self) -> list[float]:
        """Returns the design variables as a list of floats"""
        return map(float, [self.center.x, self.center.y, self.radius])

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
        self.num_design_variables = 5

    @property
    def design_variables(self) -> list[float]:
        return map(
            float,
            [self.x, self.y, self.angle, self.length, self.thickness],
        )

    def __call__(self, points: tuple | NDArray) -> NDArray:
        if isinstance(points, tuple):
            points = np.array(points).reshape(1, 2)
        if isinstance(points, itertools.product):
            points = np.array(list(points))
        return self.φ(x=points[:, 0], y=points[:, 1])

    def φ(self, x: float, y: float) -> float:
        return self._φ(*self.design_variables, x, y)

    def _φ(
        self,
        center_x: float,
        center_y: float,
        angle: float,
        length: float,
        thickness: float,
        x: NDArray,
        y: NDArray,
    ) -> float:
        # -angle because the matrix gets flipped top to bottom
        rotation_matrix: NDArray = np.array(
            [
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)],
            ]
        )
        # Local coordinates
        _x, _y = rotation_matrix @ np.array(
            [
                x - center_x,
                y - center_y,
            ]
        )

        φ: NDArray = 1 - np.linalg.norm(
            np.array([(_x / length) * 2, (_y / thickness) * 2]),
            ord=6,
            axis=0,
        )
        return φ

    def __repr__(self) -> str:
        return f"UniformBeam(center={self.center}, angle={float(np.rad2deg(self.angle)):2f}°, length={self.length:2f}, thickness={self.thickness:2f})"

    def __str__(self) -> str:
        return self.__repr__()


def plot_φ(φ: NDArray, resolution: tuple[int]) -> None:
    """Convinience function to plot the level set function φ."""

    # Must flip the array top to bottom as plotly has the origin at the top left
    # and MMC uses a bottom left origin
    px.imshow(
        φ.reshape(resolution[0], resolution[1], order="F").T,
        template="simple_white",
        origin="lower",
    ).show()


# def plot_boundary(φ: NDArray, resolution: tuple[int]) -> None:
#     plot_φ(heaviside(φ, minimum_value=0.5, transition_width=0.1), resolution)


if __name__ == "__main__":
    c: Circle = Circle(radius=0.2, center=Point2D(1, 0.0))
    r: UniformBeam = UniformBeam(
        center=Point2D(1.5, 0.5), angle=np.pi / 4, length=1, thickness=0.1
    )
    lower = [0.0, 0.0]
    upper = [2.0, 1.0]
    resolution: int = [80, 40]
    coords: NDArray = np.array(
        list(
            itertools.product(
                np.linspace(lower[1], upper[1], resolution[1]),
                np.linspace(lower[0], upper[0], resolution[0]),
            )
        )
    )
    # phi = c(coords[:, 0], coords[:, 1])
    # phi_grad = c.φ_grad(coords[:, 0], coords[:, 1])
    # --
    coords = np.fliplr(coords)
    phi = r(coords)
    plot_φ(phi, resolution)
    phi_grads = r.φ_grad(coords[:, 0], coords[:, 1])
    for phi_grad in phi_grads:
        plot_φ(phi_grad, resolution)

    plot_φ(r(coords), resolution)
    pass
