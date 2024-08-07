from dataclasses import dataclass
import itertools
from typing import Generator


import numpy as np  # Might be worth going to JAX for GPU acceleration
from numpy.typing import NDArray
import plotly.express as px
from scipy.sparse.linalg import spsolve

import components
import method_moving_asymptotes as mma

E: float = 1.0  # Young's modulus
α: float = 1e-9  # Young's modulus of void
ε: float = 0.3  # Size of transition region for the smoothed Heaviside function
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

    # TODO currently (y, x) as the way product generates it.
    @property
    def node_coordinates(self) -> Generator[tuple[float, float], None, None]:
        return itertools.product(
            np.linspace(0, self.dimensions[1], self.num_nodes[1]),
            np.linspace(0, self.dimensions[0], self.num_nodes[0]),
        )


def main() -> None:
    domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), num_elements=(80, 40))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), num_elements=(3, 2))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), num_elements=(1, 1))

    # define_objective() # TODO

    # Fix the left hand side in place
    fixed_node_ids: NDArray = np.arange(
        np.prod(domain.num_nodes), step=domain.num_nodes[0]
    )
    fixed_dof_ids: NDArray = np.concatenate(
        [2 * fixed_node_ids, 2 * fixed_node_ids + 1]
    )
    fixed_dofs = np.zeros(domain.num_dofs, dtype=bool)
    fixed_dofs[[fixed_dof_ids]] = True

    # Load the beam on the RHS half way up
    loaded_node_index: NDArray = np.array(
        [domain.num_nodes[0] - 1, domain.num_nodes[1] // 2], dtype=np.int32
    )
    loaded_node_ids: int = np.ravel_multi_index(
        loaded_node_index, domain.num_nodes, order="F"
    )
    loaded_dof_ids = [2 * loaded_node_ids, 2 * loaded_node_ids + 1]

    loaded_dofs = np.zeros(domain.num_dofs, dtype=bool)
    loaded_dofs[[loaded_dof_ids]] = True

    F = np.zeros(domain.num_dofs)
    F[2 * loaded_node_ids + 1] = -1.0  # Force in negative y direction

    # Generate the iniiial components
    component_list: list[Component] = initialise_components(n_x=4, n_y=2, domain=domain)

    free_dofs = np.logical_not(fixed_dofs)

    for i in range(MAX_ITERATIONS):
        φ: NDArray = calculate_φ(component_list, domain.node_coordinates)
        components.plot_φ(φ, domain.num_nodes)  # TODO for debug

        # finite_element()
        node_densities: NDArray = heaviside(φ, transition_width=ε, minimum_value=α)
        node_densities = node_densities.reshape(domain.num_nodes)

        element_densities: NDArray = np.mean(
            [
                node_densities[:-1, :-1],
                node_densities[1:, :-1],
                node_densities[:-1, 1:],
                node_densities[1:, 1:],
            ],
            axis=0,
        ).flatten(order="F")

        U: NDArray = np.zeros(domain.num_dofs)
        K_e: NDArray = make_stiffness_matrix(E, ν, domain.element_size, t)
        element_ids = np.arange(np.prod(domain.num_elements))
        element_multi_index = np.array(
            np.unravel_index(element_ids, domain.num_elements, order="F")
        ).T
        node_global_multi_index = np.array(
            [
                element_multi_index,
                element_multi_index + [1, 0],
                element_multi_index + [1, 1],
                element_multi_index + [0, 1],
            ]
        ).reshape((-1, 2), order="F")

        # The global node ids for each element
        # Start in top left, sweep along x direction, then down a row.
        # Nodes are numbered starting in the bottom left corner and moving anti-clockwise
        # 4 nodes per element. Adjacent elements share nodes.
        """
            8   9  10  11
            *---*---*---*
            | 3 | 4 | 5 |
          4 *---*---*---* 7
            | 0 | 1 | 2 |
            *---*---*---*
            0   1   2   3
        """
        node_global_indices = np.ravel_multi_index(
            (node_global_multi_index[:, 0], node_global_multi_index[:, 1]),
            domain.num_nodes,
            order="F",
        )
        dof_indices = np.array(
            [
                2 * node_global_indices,
                2 * node_global_indices + 1,
            ]
        ).flatten(order="F")

        # TODO Seems to be a bug here. K is not being assembled correctly.
        K = np.zeros((domain.num_dofs, domain.num_dofs))
        for element in element_ids:
            global_dofs = dof_indices[8 * element : 8 * element + 8]
            for i, j in itertools.product(range(8), range(8)):
                K[global_dofs[i], global_dofs[j]] += (
                    K_e[i, j] * element_densities[element]
                )
        K_free = K[free_dofs, :][:, free_dofs]

        U[free_dofs] = spsolve(K_free, F[free_dofs])

        # Visualise the result TODO debug
        # X displacements
        px.imshow(U[::2].reshape(domain.num_nodes, order="F").T, origin="lower").show()
        # Y displacements
        px.imshow(U[1::2].reshape(domain.num_nodes, order="F").T, origin="lower").show()

        sensitivity_analysis()
        update_design_variables()

        if is_converged():
            break


def define_objective() -> None:
    raise NotImplementedError


def define_constraints() -> None:
    raise NotImplementedError


def initialise_components(n_x, n_y, domain: Domain2D) -> list[Component]:
    """Initialises a grid of crossed Uniform Beams in the domain

    Parameters:
        n_x: int - The number of component pairs (crosses) in the x direction
        n_y: int - The number of component pairs (crosses) in the y direction
        domain: Domain2D - The domain in which the components are placed

    Returns:
        list[Component] - The list of components
    """

    region_size: NDArray = domain.dimensions / (n_x, n_y)
    x_coords: NDArray = np.linspace(
        region_size[0] / 2, domain.dimensions[0] - region_size[0] / 2, n_x
    )
    y_coords: NDArray = np.linspace(
        region_size[1] / 2, domain.dimensions[1] - region_size[1] / 2, n_y
    )

    angle: float = np.atan2(region_size[1], region_size[0])
    length: float = np.linalg.norm(region_size)

    component_list: list[Component] = []
    for y, x in itertools.product(y_coords, x_coords):
        for sign in [-1, 1]:
            component_list.append(
                components.UniformBeam(
                    center=components.Point2D(x, y),
                    angle=sign * angle,
                    length=length,
                    thickness=0.1,
                )
            )

    return component_list


def calculate_φ(
    component_list: list[Component], coordinates, ks_aggregation_power: int = 100
) -> NDArray:
    """Calculates the level set function φ"""
    # TODO: domain returns coordinate generator. When/if to convert to NDArray?
    coords = np.array(list(coordinates))
    coords = np.fliplr(
        coords
    )  # TODO: currently (y, x) as the way product generates it.
    φs = np.array([component(coords) for component in component_list])
    ## Simple max aggregation
    # φ: NDArray = np.max(φs, axis=0)

    # Kolmogorov-Smirnov (KS) aggregation as per the original MMC-2D code
    temp: NDArray = np.exp(φs * ks_aggregation_power)
    φ: NDArray = np.maximum(
        np.full(temp.shape[1], -1e3),
        np.log(np.sum(temp, axis=0)) / ks_aggregation_power,
    )

    return φ


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
    E: float, ν: float, element_size: tuple[float, float], t: float
) -> NDArray:
    """
    Create the stiffness matrix for a single element

    Parameters:
        E: float - Young's modulus
        ν: float - Poisson's ratio
        element_size: tuple[float, float] - The size of the element
        h: float - The thickness of the element

    Returns:
        NDArray - The stiffness matrix for a single element
    """
    # TODO: Mostly based off the original 218 line MMC-2D code.
    # I would prefer to use the shape functions and domain to generate the matrix.
    # High likelihood of errors in this function.

    # My calculation of K_e matches that in the 218 line code.  k_1_1 here is equivalent to k1(1)
    # It is the first row of the element stiffness matrix.
    # I have adjusted indices for k2 from the 218 line code. There K_e is described as a 1D matrix
    # so the k2 indices are in a strange order to allow for the process of turning those values into a
    # symmetric 8 x 8 matrix. All values of k2 match my derivation, I have just changed their indices them for clarity.

    # Note: the indices in the variable names are 1-indexed to match the 218 line code and mathematical matrix notation.
    # They are never indexed on their names or used outside this function.

    a, b = element_size
    k_1_1: float = -1 / (6 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_2: float = (ν + 1) / 8
    k_1_3: float = -1 / (12 * a * b) * (a**2 * (ν - 1) + 4 * b**2)
    k_1_4: float = (3 * ν - 1) / 8
    k_1_5: float = 1 / (12 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_7: float = 1 / (6 * a * b) * (a**2 * (ν - 1) + b**2)
    k_2_2: float = -1 / (6 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_4: float = 1 / (6 * a * b) * (b**2 * (ν - 1) + a**2)
    k_2_6: float = 1 / (12 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_8: float = -1 / (12 * a * b) * (b**2 * (ν - 1) + 4 * a**2)

    K_e_triu: NDArray = (
        E
        * t
        / (1 - ν**2)
        * np.array(
            [
                [k_1_1, k_1_2, k_1_3, k_1_4, k_1_5, -k_1_2, k_1_7, -k_1_4],
                [0.0, k_2_2, -k_1_4, k_2_4, -k_1_2, k_2_6, k_1_4, k_2_8],
                [0.0, 0.0, k_1_1, -k_1_2, k_1_7, k_1_4, k_1_5, k_1_2],
                [0.0, 0.0, 0.0, k_2_2, -k_1_4, k_2_8, k_1_2, k_2_6],
                [0.0, 0.0, 0.0, 0.0, k_1_1, k_1_2, k_1_3, k_1_4],
                [0.0, 0.0, 0.0, 0.0, 0.0, k_2_2, -k_1_4, k_2_4],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_1_1, -k_1_2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_2_2],
            ]
        )
    )

    K_e: NDArray = K_e_triu + K_e_triu.T - np.diag(np.diag(K_e_triu))
    return K_e


if __name__ == "__main__":
    main()
