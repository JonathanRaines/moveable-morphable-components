import itertools

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from moveable_morphable_components import components


def grid_of_uniform_beams(
    n_x: int,
    n_y: int,
    dimensions: tuple[float, float],
    thickness: float,
) -> jnp.ndarray:
    """Initialise a grid of crossed Uniform Beams in the domain.

    Args:
        n_x: int - The number of component pairs (crosses) in the x direction
        n_y: int - The number of component pairs (crosses) in the y direction
        dimensions: tuple[float, float] - The dimensions to fill with the grid
        thickness: float - The thickness of the beam components

    Returns:
        list[Component] - The list of components

    """
    # Work out the size of the regions each crossed pair will occupy
    region_size: NDArray = dimensions / (n_x, n_y)

    # Generate the center coordinates of regions
    x_coords: NDArray = np.linspace(
        region_size[0] / 2,
        dimensions[0] - region_size[0] / 2,
        n_x,
    )
    y_coords: NDArray = np.linspace(
        region_size[1] / 2,
        dimensions[1] - region_size[1] / 2,
        n_y,
    )

    # Angle and length to put the components across the region diagonals
    angle: float = np.arctan2(region_size[1], region_size[0])
    length: float = np.linalg.norm(region_size)

    design_variables = [
        [x, y, sign * angle, length, thickness]
        for sign, y, x in itertools.product([-1, 1], y_coords, x_coords)
    ]

    return jnp.array(design_variables)


def grid_of_uniform_beams_of_fixed_thickness(
    n_x: int,
    n_y: int,
    dimensions: tuple[float, float],
    thickness: list[float] | float,
) -> jnp.ndarray:
    """Initialise a grid of crossed Uniform Beams in the domain.

    Parameters;
        n_x: int - The number of component pairs (crosses) in the x direction
        n_y: int - The number of component pairs (crosses) in the y direction
        dimensions: tuple[float, float] - The dimensions to fill with the grid
        thickness: float - The thickness of the beam components

    Returns:
        list[Component] - The list of components

    """
    if isinstance(thickness, float):
        thickness = [thickness] * n_x * n_y * 2

    if len(thickness) != n_x * n_y * 2:
        msg = f"""Thickness list must be the correct length (nx * ny * 2),
          got {len(thickness)}"""
        raise ValueError(
            msg,
        )

    # Work out the size of the regions each crossed pair will occupy
    region_size: NDArray = dimensions / (n_x, n_y)

    # Generate the center coordinates of regions
    x_coords: NDArray = np.linspace(
        region_size[0] / 2,
        dimensions[0] - region_size[0] / 2,
        n_x,
    )
    y_coords: NDArray = np.linspace(
        region_size[1] / 2,
        dimensions[1] - region_size[1] / 2,
        n_y,
    )

    # Angle and length to put the components across the region diagonals
    angle: float = np.arctan2(region_size[1], region_size[0])
    length: float = np.linalg.norm(region_size)

    design_variables = [
        [x, y, sign * angle, length, thickness[i]]
        for i, (sign, y, x) in enumerate(itertools.product([-1, 1], y_coords, x_coords))
    ]

    return jnp.array(design_variables)


def random(
    low: NDArray,
    high: NDArray,
    n: int,
    np_random: np.random.Generator | None,
) -> jnp.ndarray:
    """Randomly place n components in the domain."""
    if np_random is None:
        np_random = np.random.default_rng()

    design_vars = np_random.uniform(low=low, high=high, size=(n, low.size))

    return jnp.array(design_vars)


def fixed_thickness_beam_from_df(df: pd.DataFrame) -> jnp.ndarray:
    component_list = []
    for i, row in df.iterrows():
        component_list.append(
            components.UniformBeamFixedThickness(
                center=components.Point2D(row.x, row.y),
                angle=row.angle,
                length=row.length,
                thickness=row.width,
            ),
        )
    return component_list
