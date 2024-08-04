from dataclasses import dataclass
import itertools
from typing import Generator

import numpy as np  # Might be worth going to JAX for GPU acceleration
from numpy.typing import NDArray

E: float = 1.0  # Young's modulus
α: float = 1e-9  # Young's modulus of void
ε: float = 0.2  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness

MAX_ITERATIONS: int = 100


@dataclass
class Component:
    # Placehoder for now. Want to make this general.
    # Class or function? Use the SDF library?
    x: float
    y: float
    θ: float
    t_1: float
    t_2: float
    t_3: float


class Domain2D:
    """
    Represents the 2D domain for the problem. The domain is a rectangle with dimensions [x, y].
    Quadrilateral bi-linear elements are used to discretise the domain.

    Attributes:
        dimensions: tuple[float, float] - The dimensions of the domain [x, y]
        num_elements: tuple[int, int] - The number of elements in each direction [x, y]
        num_nodes: tuple[int, int] - The number of nodes in each direction [x, y]
        ν: float - Poisson's ratio - used to calculate the stiffness matrix
        num_dofs: int - The number of degrees of freedom
        element_size: NDArray - The size of each element [x, y]
    """

    def __init__(
        self,
        dimensions: tuple[float, float],
        num_elements: tuple[int, int],
    ) -> None:
        self.dimensions: NDArray = np.array(dimensions, dtype=np.float32)
        self.num_elements: NDArray = np.array(num_elements, dtype=np.int32)
        self.num_nodes: NDArray = self.num_elements + 1

        self.num_dofs: int = 2 * np.prod(self.num_nodes)

        self.element_size: NDArray = dimensions / self.num_elements

        # self.stiffness_matrix = make_stiffness_matrix(E, ν, self.element_size, h=1)

    @property
    def node_coordinates(self) -> Generator[tuple[float, float], None, None]:
        return itertools.product(
            np.linspace(0, self.dimensions[0], self.num_nodes[0]),
            np.linspace(0, self.dimensions[1], self.num_nodes[1]),
        )


def main() -> None:
    define_design_space()
    define_objective()
    define_constraints()
    components: list[Component] = initialise_components()

    for i in range(MAX_ITERATIONS):
        calculate_φ(components)
        finite_element()
        sensitivity_analysis()
        update_design_variables()

        if is_converged():
            break


def define_design_space(
    width,
    height,
) -> None:
    raise NotImplementedError


def define_objective() -> None:
    raise NotImplementedError


def define_constraints() -> None:
    raise NotImplementedError


def initialise_components() -> None:
    raise NotImplementedError


def calculate_φ() -> None:
    """Calculates the level set function φ"""
    raise NotImplementedError


def finite_element() -> None:
    raise NotImplementedError


def sensitivity_analysis() -> None:
    raise NotImplementedError


def update_design_variables() -> None:
    raise NotImplementedError


def is_converged() -> bool:
    raise NotImplementedError


def heaviside(
    x: NDArray, transition_width: float, minimum_value: float = 0.0
) -> NDArray:
    """
    Smoothed Heaviside function
    https://en.wikipedia.org/wiki/Heaviside_step_function

    An element-wise function that 1 when x > 0, and minimum_value when x < 0.
    The step is smoothed over the transition_width.

    Parameters:
        x: NDArray - The input array
        minimum_value: float - The Young's modulus of void (outside the components)
        transition_width: float - The size of the transition region

    Returns:
        NDArray - The smoothed Heaviside of the input array

    Example:
        >>> heaviside(np.array([-1.0, 1.0]), 0.1, 0.1)
        array([0.1 , 1. ])
    """
    h_x = (
        3 * (1 - x) / 4 * (x / transition_width - x**3 / (3 * transition_width**3))
        + (1 + minimum_value) / 2
    )
    h_x = np.where(x < -transition_width, minimum_value, h_x)
    h_x = np.where(x > transition_width, 1, h_x)
    return h_x


def make_stiffness_matrix(
    E: float, ν: float, element_size: tuple[float, float], h: int
) -> NDArray:
    """
    Create the stiffness matrix for a single element

    Parameters:
        E: float - Young's modulus
        ν: float - Poisson's ratio
        element_size: tuple[float, float] - The size of the element
        h: int - The thickness of the element

    Returns:
        NDArray - The stiffness matrix for a single element
    """
    # TODO: Mostly based off the original 218 line MMC-2D code.
    # I would prefer to use the shape functions and domain to generate the matrix.
    # High likelihood of errors in this function.

    a, b = element_size
    k_1_1: float = -1 / (6 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_2: float = (ν + 1) / 8
    k_1_3: float = -1 / (12 * a * b) * (a**2 * (ν - 1) + 4 * b**2)
    k_1_4: float = (3 * ν - 1) / 8
    k_1_5: float = 1 / (12 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_6: float = -k_1_2
    k_1_7: float = 1 / (6 * a * b) * (a**2 * (ν - 1) + b**2)
    k_1_8: float = -k_1_4
    k_2_1: float = -1 / (6 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_2: float = k_1_2
    k_2_3: float = -1 / (12 * a * b) * (b**2 * (ν - 1) + 4 * a**2)
    k_2_4: float = k_1_4
    k_2_5: float = 1 / (12 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_6: float = -k_1_2
    k_2_7: float = 1 / (6 * a * b) * (b**2 * (ν - 1) + a**2)
    k_2_8: float = -k_1_4

    K_e_triu: NDArray = (
        E
        * h
        / (1 - ν**2)
        * np.array(
            [
                [k_1_1, k_1_2, k_1_3, k_1_4, k_1_5, k_1_6, k_1_7, k_1_8],
                [0.0, k_2_1, k_2_8, k_2_7, k_2_6, k_2_5, k_2_4, k_2_3],
                [0.0, 0.0, k_1_1, k_1_6, k_1_7, k_1_4, k_1_5, k_1_2],
                [0.0, 0.0, 0.0, k_2_1, k_2_8, k_2_3, k_2_2, k_2_5],
                [0.0, 0.0, 0.0, 0.0, k_1_1, k_1_2, k_1_3, k_1_4],
                [0.0, 0.0, 0.0, 0.0, 0.0, k_2_1, k_2_8, k_2_7],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_1_1, k_1_6],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_2_1],
            ]
        )
    )

    K_e: NDArray = K_e_triu + K_e_triu.T - np.diag(np.diag(K_e))
    return K_e


if __name__ == "__main__":
    domain = Domain2D(dimensions=(1.0, 1.0), num_elements=(10, 10))
    main()
