import itertools
from typing import Generator


import numpy as np  # Might be worth going to JAX for GPU acceleration
from numpy.typing import NDArray

import plotly.express as px
from scipy.sparse.linalg import spsolve
import tqdm

import components
import method_moving_asymptotes as mma

E: float = 1  # 000.0  # Young's modulus
α: float = 1e-9  # Young's modulus of void
ε: float = 0.3  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness

MAX_ITERATIONS: int = 100
VOLUME_FRACTION: float = 0.4


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
    free_dofs = np.logical_not(fixed_dofs)

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
    component_list: list[components.Component] = initialise_components(
        n_x=2,  # 4,
        n_y=1,  # 2,
        domain=domain,
    )

    # inialise the starting values for mma
    xmin = np.vstack((0, 0, -np.pi / 2, 0, 0))
    xmin = np.matlib.repmat(xmin, len(component_list), 1)
    xmax = np.vstack(
        (
            domain.dimensions[0],
            domain.dimensions[1],
            np.pi / 2,
            np.linalg.norm(domain.dimensions) / 2,
            np.min(domain.dimensions) / 2,
        )
    )
    xmax = np.matlib.repmat(xmax, len(component_list), 1)
    m = 1
    low = xmin
    upp = xmax
    a0 = 1
    a = np.zeros((m, 1))
    c = np.full((m, 1), 1000)
    d = np.zeros((m, 1))

    for iteration in tqdm.trange(MAX_ITERATIONS):
        φ, dφ_dφs = calculate_φ(component_list, domain.node_coordinates)
        # plot_values(φ, domain.num_nodes).show()
        # plot_values(dφ_dφs[0], domain.num_nodes).show()
        # H is Heaviside(φ)
        H: NDArray = heaviside(φ, transition_width=ε, minimum_value=α)
        # plot_values(H, domain.num_nodes).show()
        # Calculate the derivative of H with respect to φ using the analytical form
        dH_dφ: NDArray = 3 * (1 - α) / (4 * ε) * (1 - φ**2 / ε**2)
        dH_dφ = np.where(abs(φ) > ε, 0, dH_dφ)
        # plot_values(dH_dφ, domain.num_nodes).show()
        coords = np.fliplr(np.array(list(domain.node_coordinates)))
        # Calculate the derivative of φ with respect to the design variables
        dφ_component_d_design_vars = np.concat(
            [comp.φ_grad(coords[:, 0], coords[:, 1]) for comp in component_list]
        )
        dφ_d_design_vars = np.repeat(dφ_dφs, 5, axis=0) * dφ_component_d_design_vars
        # for design_var in dφ_d_design_vars:
        #     plot_values(design_var, domain.num_nodes).show()

        # finite_element()
        node_densities: NDArray = H.copy()
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

        K = np.zeros((domain.num_dofs, domain.num_dofs))

        for element, element_dof_global_indices in enumerate(
            dof_indices.reshape((-1, 8))
        ):
            for i, j in itertools.product(range(8), range(8)):
                K[element_dof_global_indices[i], element_dof_global_indices[j]] += (
                    K_e[i, j] * element_densities[element]
                )
        K_free = K[free_dofs, :][:, free_dofs]

        U[free_dofs] = np.linalg.solve(K_free, F[free_dofs])

        # Visualise the result TODO debug
        # X displacements
        # plot_values(U[::2], domain.num_nodes).show()
        # # Y displacements
        # plot_values(U[1::2], domain.num_nodes).show()

        # Calculate the Energy of the Elements
        U_by_element = U[dof_indices.reshape((-1, 8))]
        element_energy = np.sum(
            (U_by_element @ K_e) * U_by_element,
            axis=1,
        ).reshape(domain.num_elements, order="F")
        # plot_values(element_energy, domain.num_elements).show()

        node_energy = np.zeros(domain.num_nodes)
        node_energy[:-1, :-1] += element_energy / 4
        node_energy[1:, :-1] += element_energy / 4
        node_energy[1:, 1:] += element_energy / 4
        node_energy[:-1, 1:] += element_energy / 4
        node_energy = node_energy.flatten(order="F")
        # plot_values(node_energy, domain.num_nodes).show()

        node_volumes = np.zeros(domain.num_nodes)
        node_volumes[:-1, :-1] += 1 / 4
        node_volumes[1:, :-1] += 1 / 4
        node_volumes[1:, 1:] += 1 / 4
        node_volumes[:-1, 1:] += 1 / 4
        node_volumes = node_volumes.flatten(order="F")
        # plot_values(node_volumes, domain.num_nodes).show()

        # s_energy = np.expand_dims(energy, axis=1) @ np.ones((1, 4) / 4
        # node_energy[node_global_indices] = sum(s_energy[node_global_indices]

        comp = F.T @ U

        # sensitivity_analysis()
        design_variables = np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten()
        # Define the initial values for the design variables
        initial_design_variables = np.expand_dims(
            design_variables.flatten(order="F"), axis=1
        )
        design_variables_prev = initial_design_variables.copy()
        design_variables_prev_2 = initial_design_variables.copy()

        # components x design variables x nodes
        d_objective_d_design_vars = np.nansum(
            -node_energy * dH_dφ * dφ_d_design_vars, axis=1
        )
        # for d_obj_d_dv in d_objective_d_design_var:
        #     plot_values(
        #         d_obj_d_dv,
        #         domain.num_nodes,
        #     ).show()
        d_volume_fraction_d_design_vars = np.nansum(
            node_volumes * dH_dφ * dφ_d_design_vars,
            axis=1,
        )

        f0val = comp
        d_objective_d_design_vars = (-d_objective_d_design_vars) / np.max(
            abs(d_objective_d_design_vars)
        )
        fval = (
            np.sum(node_densities)
            * np.prod(domain.element_size)
            / np.prod(domain.dimensions)
            - VOLUME_FRACTION
        )
        d_volume_fraction_d_design_vars = d_volume_fraction_d_design_vars / np.max(
            abs(d_volume_fraction_d_design_vars)
        )

        # update_design_variables()
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, ss, low, upp = mma.mmasub(
            m=1,
            n=5 * len(component_list),
            iter=iteration + 1,
            # [x, y, angle, length, thickness]
            xval=np.expand_dims(design_variables, 1),
            xmin=xmin,
            xmax=xmax,
            xold1=design_variables_prev,
            xold2=design_variables_prev_2,
            f0val=np.expand_dims(np.array([f0val]), 1),
            df0dx=np.expand_dims(d_objective_d_design_vars, 1),
            fval=fval,
            dfdx=np.expand_dims(d_volume_fraction_d_design_vars, 0),
            low=low,
            upp=upp,
            a0=a0,
            a=a,
            c=c,
            d=d,
            move=1.0,
        )

        # Update the components
        design_variables_prev_2 = design_variables_prev.copy()
        design_variables_prev = design_variables.copy()
        design_variables = xmma.copy()
        # change = np.max(abs(design_variables - design_variables_prev))
        component_list = [
            components.UniformBeam(
                center=components.Point2D(x, y),
                angle=angle,
                length=length,
                thickness=thickness,
            )
            for x, y, angle, length, thickness in design_variables.reshape(-1, 5)
        ]
        if iteration % 5 == 0:
            new_phi, _ = calculate_φ(component_list, domain.node_coordinates)
            components.plot_φ(new_phi, domain.num_nodes)
            pass

        # if is_converged():
        #     break


def define_objective() -> None:
    raise NotImplementedError


def define_constraints() -> None:
    raise NotImplementedError


def initialise_components(n_x, n_y, domain: Domain2D) -> list[components.Component]:
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

    component_list: list[components.Component] = []
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
    component_list: list[components.Component],
    coordinates,
    ks_aggregation_power: int = 10,  # Was 100 but getting NAN
) -> tuple[NDArray, NDArray]:
    """Calculates the level set function φ"""
    # TODO: domain returns coordinate generator. When/if to convert to NDArray?
    coords = np.array(list(coordinates))
    coords = np.fliplr(
        coords
    )  # TODO: currently (y, x) as the way product generates it.
    φ_components = np.array([component(coords) for component in component_list])
    ## Simple max aggregation
    # φ_simple: NDArray = np.max(φs, axis=0)

    # Kolmogorov-Smirnov (KS) aggregation as per the original MMC-2D code
    temp: NDArray = np.exp(φ_components * ks_aggregation_power)
    φ_global: NDArray = np.maximum(
        np.full(temp.shape[1], -1e3),
        np.log(np.sum(temp, axis=0)) / ks_aggregation_power,
    )

    dφ_global_dφ_components = temp / np.sum(temp, axis=0)

    return φ_global, dφ_global_dφ_components


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


def plot_values(values: NDArray, domain_shape: tuple[int, int]) -> None:
    return px.imshow(values.reshape(domain_shape, order="F").T, origin="lower")


if __name__ == "__main__":
    main()
