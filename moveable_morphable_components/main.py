import itertools
from typing import Callable

import jax.numpy as jnp
import networkx as nx
import numpy as np  # TODO: use jax.numpy fully
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.sparse
import tqdm
import scipy

import components
from domain import Domain2D
import finite_element
import method_moving_asymptotes as mma

import warnings

warnings.filterwarnings("error")

E: float = 1e1  # 1e7  # Young's modulus N/mm^2
α: float = 1e-9  # 1e-3  # Heaviside minimum value
ε: float = 0.01  # 10  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness
FORCE_MAGNITUDE: float = 1  #  N

SCALE: float = 1.0
NUM_CONSTRAINTS: int = 1  # Volume fraction constraint
A0: float = 1.0
# A0: float = 0.01
A: NDArray = np.full((NUM_CONSTRAINTS, 1), 0)
C: NDArray = np.full((NUM_CONSTRAINTS, 1), 1000)
D: NDArray = np.full((NUM_CONSTRAINTS, 1), 1)
MOVE = 1.0  # Proportion of the design variable range that can be moved in a single iteration
OBJECTIVE_TOLERANCE: float = 1e-2 * SCALE  # within a 1% change

MAX_ITERATIONS: int = 100
VOLUME_FRACTION: float = 0.4


def main() -> None:
    domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(80, 40))

    # Fix the left hand side in place
    fixed_dof_ids: NDArray[np.uint] = domain.left_boundary_dof_ids()

    # Make a mask for the free dofs
    free_dofs: NDArray[np.uint] = np.setdiff1d(
        np.arange(domain.num_dofs), fixed_dof_ids
    )

    # Load the beam on the RHS half way up
    loaded_dof_ids: NDArray[np.uint] = domain.coords_to_nearest_dof_ids(
        point=(domain.dimensions[0], domain.dimensions[1] / 2)
    )

    # sparse force vector [x1, y1, x2, y2, ...]
    F = scipy.sparse.csc_array(
        ([-FORCE_MAGNITUDE], ([loaded_dof_ids[1]], [0])), shape=(domain.num_dofs, 1)
    )

    # Define the element stiffness matrix
    K_e: NDArray[float] = finite_element.element_stiffness_matrix(
        E, ν, domain.element_size, t
    )

    # Generate the initial components
    component_list: list[components.Component] = layout_grid_of_uniform_beams(
        n_x=4, n_y=2, dimensions=domain.dimensions, thickness=0.1
    )

    # Define the initial values for the design variables
    initial_design_variables: NDArray[float] = np.expand_dims(
        np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten(),
        axis=1,
    )

    # Set the bounds for the design variables
    design_variables_min: NDArray[float] = np.array((0.0, 0.0, -np.pi / 2, 0.0, -4 * ε))
    design_variables_min: NDArray[float] = np.expand_dims(
        np.tile(design_variables_min, len(component_list)), axis=1
    )
    design_variables_max: NDArray[float] = np.array(
        (
            domain.dimensions[0],
            domain.dimensions[1],
            np.pi / 2,
            np.linalg.norm(domain.dimensions) / 2,
            np.min(domain.dimensions) / 2,
        )
    )
    design_variables_max: NDArray[float] = np.expand_dims(
        np.tile(design_variables_max, len(component_list)), axis=1
    )

    # Initialise the starting values for mma optimization
    design_variables: NDArray[float] = initial_design_variables.copy()
    design_variables_prev: NDArray[float] = initial_design_variables.copy()
    design_variables_prev_2: NDArray[float] = initial_design_variables.copy()
    low: NDArray[float] = design_variables_min
    upp: NDArray[float] = design_variables_max

    H_history: NDArray = np.zeros((*domain.node_shape, MAX_ITERATIONS))
    design_variable_history = np.zeros((design_variables.shape[0], MAX_ITERATIONS))
    objective_history: list[float] = []
    constraint_history: list[float] = []

    # Optimisation loop
    for iteration in tqdm.trange(MAX_ITERATIONS):
        # Combine the level set functions from the components to form a global one
        component_φs: NDArray[float] = evaluate_signed_distance_functions(
            component_list=component_list, coordinates=domain.node_coordinates
        )

        # component_connectivity: NDArray[bool] = connected_components(component_φs)

        # load_coordinates: NDArray = domain.dof_ids_to_coords(loaded_dof_ids)
        # ground_coordinates: NDArray = domain.dof_ids_to_coords(fixed_dof_ids)
        # components_touching_load: list[int] = point_is_in_component(
        #     signed_distance_functions=component_list, points=load_coordinates
        # )
        # components_touching_ground: list[int] = point_is_in_component(
        #     signed_distance_functions=component_list, points=ground_coordinates
        # )
        # component_graph = nx.from_numpy_array(component_connectivity)
        # component_islands = nx.connected_components(component_graph)

        # islands_in_load_path: list[set[int]] = [
        #     isle
        #     for isle in component_islands
        #     if any(np.intersect1d(list(isle), components_touching_load))
        #     and any(np.intersect1d(list(isle), components_touching_ground))
        # ]

        load_path_exists: bool = False
        # if len(islands_in_load_path) != 0:
        #     load_path_exists = True

        #     components_in_load_path: set[int] = set.union(*islands_in_load_path)500
        #             list(components_in_load_path)
        #         ].reshape(-1, 1)
        #         design_variables_prev_2 = design_variables_prev_2.reshape(-1, 5)[
        #             list(components_in_load_path)
        #         ].reshape(-1, 1)
        #         design_variables_min = design_variables_min.reshape(-1, 5)[
        #             list(components_in_load_path)
        #         ].reshape(-1, 1)
        #         design_variables_max = design_variables_max.reshape(-1, 5)[
        #             list(components_in_load_path)
        #         ].reshape(-1, 1)
        #         low = low.reshape(-1, 5)[list(components_in_load_path)].reshape(-1, 1)
        #         upp = upp.reshape(-1, 5)[list(components_in_load_path)].reshape(-1, 1)

        φ, dφ_dφs = combine_φs(signed_distance_functions=component_φs)

        # dφ_dφs is the derivative of the global level set function with respect to the component level set functions
        # It is 1 where the component is the maximum and 0 elsewhere
        # plot_values(φ, domain.node_shape).show()

        # H is Heaviside(φ), it is used to modify the Young's modulus (E) of the elements
        H: NDArray[float] = heaviside(φ, transition_width=ε, minimum_value=α)
        H_history[:, :, iteration] = H.reshape(domain.node_shape, order="F")

        # Calculate the derivative of H with respect to φ using the analytical form
        # TODO: Replace with automatic differentiation?
        dH_dφ: NDArray[float] = 3 * (1 - α) / (4 * ε) * (1 - φ**2 / ε**2)
        dH_dφ = np.where(abs(φ) > ε, 0.0, dH_dφ)
        # plot_values(dH_dφ, domain.node_shape).show()

        # Calculate the derivative of φ with respect to the design variables
        dφ_component_d_design_vars: NDArray[float] = np.concatenate(
            [comp.φ_grad(domain.node_coordinates) for comp in component_list]
        )
        # TODO hard coded value 5
        dφ_d_design_vars: NDArray[float] = (
            np.repeat(dφ_dφs, 5, axis=0) * dφ_component_d_design_vars
            # .reshape(dφ_component_d_design_vars.shape[0], -1)
            # .sum(axis=1)
        )

        # Calculate the density of the elements
        element_densities: NDArray[float] = domain.average_node_values_to_element(H)

        # Stiffness Matrix
        K: scipy.sparse.csc_matrix = finite_element.assemble_stiffness_matrix(
            element_dof_ids=domain.element_dof_ids,
            element_densities=element_densities,
            element_stiffness_matrix=K_e,
        )

        # If the load path exists, downselect the free dofs to only those
        # that related to elements that intersect the structure
        if load_path_exists:
            unused_dof_xys = np.argwhere(element_densities <= α)
            unused_dof_ids = np.ravel_multi_index(
                unused_dof_xys.T, domain.node_shape, order="F"
            )
            free_dofs = np.setdiff1d(free_dofs, unused_dof_ids)
            pass

        # Reduce the stiffness matrix to the free dofs
        K_free: scipy.sparse.csc_matrix = K[free_dofs, :][:, free_dofs]

        # Solve the system
        U: NDArray[float] = np.zeros(domain.num_dofs)
        U[free_dofs] = scipy.sparse.linalg.spsolve(K_free, F[free_dofs])

        # Calculate the Energy of the Elements
        U_by_element: NDArray[float] = U[domain.element_dof_ids]
        element_energy: NDArray[float] = np.sum(
            (U_by_element @ K_e) * U_by_element,
            axis=1,
        ).reshape(domain.element_shape, order="F")

        node_energy: NDArray[float] = domain.element_value_to_nodes(
            element_energy
        )  # .flatten(order="F")

        # Sensitivity_analysis()
        design_variables: NDArray[float] = np.expand_dims(
            np.array(
                [list(component.design_variables) for component in component_list]
            ).flatten(),
            1,
        )

        # Objective and derivative
        objective: NDArray[float] = F.T @ U * SCALE
        objective_history.append(objective[0])
        d_objective_d_design_vars = -node_energy * dH_dφ * dφ_d_design_vars
        d_objective_d_design_vars = np.nansum(
            d_objective_d_design_vars.reshape(d_objective_d_design_vars.shape[0], -1),
            axis=1,
        )

        # Volume fraction constraint and derivative
        volume_fraction_constraint: float = (
            np.sum(
                heaviside(φ, transition_width=ε, minimum_value=0) * domain.node_volumes
            )
            / np.sum(domain.node_volumes)
            - VOLUME_FRACTION
        )
        d_volume_fraction_d_design_vars: NDArray[float] = (
            domain.node_volumes * dH_dφ * dφ_d_design_vars
        )
        ## Enable to make the volume fraction a target rather than upper limit
        # d_volume_fraction_d_design_vars *= volume_fraction_constraint // np.abs(
        #     volume_fraction_constraint
        # )  # flip the sign of the derivative
        # volume_fraction_constraint = (
        #     np.abs(volume_fraction_constraint) - 0.1
        # )  # make within 10% of target
        constraint_history.append(volume_fraction_constraint)

        d_volume_fraction_d_design_vars = np.nansum(
            d_volume_fraction_d_design_vars.reshape(
                d_volume_fraction_d_design_vars.shape[0], -1, order="F"
            ),
            axis=1,
        )

        # Normalise the derivatives
        # scale_factor: float = np.max(
        #     np.abs(
        #         np.concatenate(
        #             [d_objective_d_design_vars, d_volume_fraction_d_design_vars]
        #         )
        #     )
        # )
        d_objective_d_design_vars *= SCALE
        # d_volume_fraction_d_design_vars /= 200_000

        # Update design variables
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, ss, low, upp = mma.mmasub(
            m=1,
            n=5 * len(component_list),
            iter=iteration + 1,
            # [x, y, angle, length, thickness]
            xval=design_variables,
            xmin=design_variables_min,
            xmax=design_variables_max,
            xold1=design_variables_prev,
            xold2=design_variables_prev_2,
            f0val=np.expand_dims(objective, 1),
            df0dx=np.expand_dims(d_objective_d_design_vars, 1),
            fval=volume_fraction_constraint,
            dfdx=np.expand_dims(d_volume_fraction_d_design_vars, 0),
            low=low,
            upp=upp,
            a0=A0,
            a=A,
            c=C,
            d=D,
            move=MOVE,
        )

        # Update the components
        new_design_variables: NDArray = xmma.copy().reshape(-1, 5)
        keep_mask = np.ones_like(new_design_variables, dtype=bool)
        keep_mask[new_design_variables[:, -1] < -ε, :] = False
        keep_mask = keep_mask.reshape(-1)
        design_variables_prev_2 = design_variables_prev.copy()[keep_mask]
        design_variables_prev = design_variables.copy()[keep_mask]
        design_variables = xmma.copy()[keep_mask]
        design_variables_min = design_variables_min[keep_mask]
        design_variables_max = design_variables_max[keep_mask]
        # design_variable_history[:, iteration][keep_mask] = design_variables.flatten()
        low = low[keep_mask]
        upp = upp[keep_mask]

        component_list: list[components.Component] = [
            components.UniformBeam(
                center=components.Point2D(x, y),
                angle=angle,
                length=length,
                thickness=thickness,
            )
            for x, y, angle, length, thickness in design_variables.reshape(-1, 5)
        ]

        if is_converged(
            iteration=iteration,
            objective_tolerance=OBJECTIVE_TOLERANCE,
            objective_history=objective_history,
            constraint_value=volume_fraction_constraint,
            window_size=5,
        ):
            print("Converged")
            break

    return (
        H_history[:, :, :iteration],
        objective_history[:iteration],
        constraint_history[:iteration],
    )


def layout_grid_of_uniform_beams(
    n_x: int, n_y: int, dimensions: tuple[float, float], thickness: float
) -> list[components.Component]:
    """Initialises a grid of crossed Uniform Beams in the domain

    Parameters:
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
        region_size[0] / 2, dimensions[0] - region_size[0] / 2, n_x
    )
    y_coords: NDArray = np.linspace(
        region_size[1] / 2, dimensions[1] - region_size[1] / 2, n_y
    )

    # Angle and length to put the components across the region diagonals
    angle: float = np.arctan2(region_size[1], region_size[0])
    length: float = np.linalg.norm(region_size)

    component_list: list[components.Component] = []
    for y, x in itertools.product(y_coords, x_coords):
        for sign in [-1, 1]:
            component_list.append(
                components.UniformBeam(
                    center=components.Point2D(x, y),
                    angle=sign * angle,
                    length=length,
                    thickness=thickness,
                )
            )

    return component_list


def evaluate_signed_distance_functions(
    component_list: list[components.Component],
    coordinates: tuple[NDArray[float], NDArray[float]],
) -> NDArray[float]:
    return np.array([component(coordinates) for component in component_list])


def combine_φs(
    signed_distance_functions: NDArray[float],
    ks_aggregation_power: int = 10,  # Was 100 but getting NAN
) -> tuple[NDArray, NDArray]:
    """Calculates the level set function φ"""

    # # Kolmogorov-Smirnov (KS) aggregation as per the original MMC-2D code
    # temp: NDArray = jnp.exp(signed_distance_functions * ks_aggregation_power)
    # φ_global: NDArray = jnp.maximum(
    #     jnp.full(temp.shape[1:], -1e3),
    #     jnp.log(np.sum(temp, axis=0)) / ks_aggregation_power,
    # )

    # dφ_global_dφ_components = temp / jnp.sum(temp, axis=0)

    φ_global: NDArray = np.max(signed_distance_functions, axis=0)
    φ_global = np.where(φ_global < -1e3, -1e3, φ_global)
    dφ_global_dφ_components = np.where(
        signed_distance_functions == φ_global, 1, 0
    ).astype(float)

    return φ_global, dφ_global_dφ_components


def is_converged(
    iteration,
    objective_tolerance,
    objective_history,
    constraint_value,
    window_size,
) -> bool:
    if iteration > window_size and constraint_value < 0:
        smoothed_objective_change: NDArray = moving_average(
            objective_history, window_size
        )
        smoothed_objective_deltas: NDArray = np.diff(smoothed_objective_change)
        if np.all(np.abs(smoothed_objective_deltas) < objective_tolerance):
            return True
        return False


def moving_average(values, n):
    ret: NDArray = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


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
    x = np.array(x)
    h_x = (
        3
        * (1 - minimum_value)
        / 4
        * (x / transition_width - x**3 / (3 * transition_width**3))
        + (1 + minimum_value) / 2
    )
    h_x = np.where(x < -transition_width, minimum_value, h_x)
    h_x = np.where(x > transition_width, 1, h_x)
    return h_x


def plot_values(values: NDArray, domain_shape: tuple[int, int]) -> None:
    # return px.imshow(values.reshape(domain_shape, order="F").T, origin="lower")
    return go.Figure(
        data=go.Contour(
            z=values.reshape(domain_shape, order="F").T,
        )
    )


def connected_components(singed_distance_functions: NDArray[float]) -> NDArray[bool]:
    """Finds connected components in the list of components
    parameters:
        singed_distance_functions: NDArray[float] - The stack of signed distance functions for each component
    returns:
        NDArray[bool] - The connectivity matrix
    """
    num_components = singed_distance_functions.shape[0]
    connectivity_matrix = np.zeros((num_components, num_components), dtype=bool)
    for component_1, component_2 in itertools.combinations(range(num_components), 2):
        sdf_1 = singed_distance_functions[component_1, :, :]
        sdf_2 = singed_distance_functions[component_2, :, :]
        if np.any(np.logical_and(sdf_1 > 0, sdf_2 > 0)):
            connectivity_matrix[component_1, component_2] = True
            connectivity_matrix[component_2, component_1] = True
    return connectivity_matrix


def point_is_in_component(
    signed_distance_functions: list[components.Component], points: NDArray[float]
) -> list[int]:
    """Returns a list of indices indicating if a point is in a component
    parameters:
        singed_distance_functions: NDArray[float] - The stack of signed distance functions for each component
    returns:
        NDArray[int] - the indexes of components that the point is within
    """
    components_touching_point = set()
    for point in points:
        signed_distances: list[float] = np.array(
            [sdf(point) for sdf in signed_distance_functions]
        )
        components_touching_point.update(np.argwhere(signed_distances > 0).flatten())
    return list(components_touching_point)


if __name__ == "__main__":
    steps, objective, constraint = main()
    steps_downsampled = steps[:, :, ::1]
    objective_downsampled = objective[::1]
    duration = 5_000  # ms
    frame_duration = duration // steps_downsampled.shape[2]
    fig = go.Figure(
        data=[go.Contour(z=steps_downsampled[:, :, 0].T)],
        layout=go.Layout(
            title="MMC",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": frame_duration}}],
                        )
                    ],
                )
            ],
            width=1_600,
            height=800,
        ),
        frames=[
            go.Frame(data=[go.Contour(z=steps_downsampled[:, :, i].T)])
            for i in range(steps_downsampled.shape[2])
        ],
    )
    fig.show()

    obj_fig = make_subplots(specs=[[{"secondary_y": True}]])
    obj_fig.add_trace(
        go.Scatter(
            x=np.arange(len(constraint)), y=objective, mode="lines", name="Objective"
        ),
        secondary_y=False,
    )
    obj_fig.add_trace(
        go.Scatter(
            x=np.arange(len(constraint)),
            y=constraint,
            mode="lines",
            name="Volume Fraction Error",
        ),
        secondary_y=True,
    )
    obj_fig.update_layout(title="Objective", template="simple_white")
    obj_fig.show()
