from dataclasses import dataclass
import itertools
from typing import Generator

import numpy as np
from numpy.typing import NDArray
import plotly.express as px


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class Component:
    def __call__(self, points: tuple | NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def design_variables(self) -> list[float]:
        raise NotImplementedError


class Circle(Component):
    def __init__(self, center: Point2D, r: float):
        self.center = center
        self.r = r

    def __call__(self, points: tuple | NDArray) -> NDArray:
        if isinstance(points, tuple):
            points = np.array(points).reshape(1, 2)

        x: NDArray = points[:, 0]
        y: NDArray = points[:, 1]
        φ: NDArray = self.r**2 - (x - self.center.x) ** 2 - (y - self.center.y) ** 2
        return φ

    @property
    def design_variables(self) -> list[float]:
        return [self.center.x, self.center.y, self.r]


class UniformBeam(Component):
    def __init__(self, center: Point2D, angle: float, length: float, thickness: float):
        """A beam of uniform thickness.

        Args:
            center (Point2D): Center of the beam
            angle (float): rotation relative to x-axis in radians
            length (float): length of the beam
            thickness (float): thickness of the beam
        """
        self.center = center
        self.angle = angle
        self.length = length
        self.thickness = thickness

    def __call__(self, points: tuple | NDArray) -> NDArray:
        if isinstance(points, tuple):
            points = np.array(points).reshape(1, 2)
        if isinstance(points, itertools.product):
            points = np.array(list(points))

        x: NDArray = points[:, 0]
        y: NDArray = points[:, 1]

        rotation_matrix: NDArray = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )
        # Local coordinates
        _x, _y = rotation_matrix @ np.array(
            [
                x - self.center.x,
                y - self.center.y,
            ]
        )

        φ: NDArray = 1 - np.linalg.norm(
            [_x / self.thickness * 2, _y / self.length * 2],
            ord=6,
            axis=0,
        )
        return φ

    @property
    def design_variables(self) -> list[float]:
        return [self.center.x, self.center.y, self.angle, self.length, self.thickness]

    def __repr__(self) -> str:
        return f"UniformBeam(center={self.center}, angle={float(np.rad2deg(self.angle)):2f}°, length={self.length:2f}, thickness={self.thickness:2f})"


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
    c: Circle = Circle(r=0.2, center=Point2D(1, 0))
    r: UniformBeam = UniformBeam(
        center=Point2D(0.25, 0.75), angle=np.pi / 4, length=0.2, thickness=0.1
    )
    lower = 0
    upper = 1
    resolution: int = 100
    coords: NDArray = np.array(
        list(
            itertools.product(
                np.linspace(lower, upper, resolution),
                np.linspace(lower, upper, resolution),
            )
        )
    )
    plot_φ(r(coords), (resolution, resolution))
    # plot_boundary(c(coords), (resolution, resolution))
