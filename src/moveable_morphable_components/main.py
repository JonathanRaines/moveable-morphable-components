"""The main loop of the MMC method."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np  # TODO(JonathanRaines): use jax.numpy fully
import plotly.graph_objects as go
import scipy
import scipy.sparse
import tqdm

from moveable_morphable_components import finite_element
from moveable_morphable_components import method_moving_asymptotes as mma

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array
    from jax.typing import ArrayLike
    from numpy.typing import NDArray

    from moveable_morphable_components.components import ComponentGroup
    from moveable_morphable_components.domain import Domain2D

HEAVISIDE_MIN_VALUE: float = 1e-6  # 1e-9  # Heaviside minimum value
# 10  # Size of transition region for the smoothed Heaviside function
HEAVISIDE_TRANSITION_WIDTH: float = 0.01
POISSON_RATIO: float = 0.3  # Poisson's ratio
THICKNESS: float = 1.0  # Thickness

YOUNGS_MODULUS: float = 1e1  # 1e7  # Young's modulus N/mm^2

SCALE: float = 1.0
NUM_CONSTRAINTS: int = 1  # Volume fraction constraint
A0: float = 1.0
A: NDArray = np.full((NUM_CONSTRAINTS, 1), 0)
C: NDArray = np.full((NUM_CONSTRAINTS, 1), 1000)
D: NDArray = np.full((NUM_CONSTRAINTS, 1), 1)
MOVE = 1.0  # Proportion of the design variable range that can be moved in a single step
OBJECTIVE_TOLERANCE: float = 1e-2 * SCALE  # within a 1% change


def main(
    max_iterations: int,
    domain: Domain2D,
    boundary_conditions: dict,
    volume_fraction_limit: float,
    component_list: list[ComponentGroup],
) -> tuple[NDArray, NDArray, NDArray]:
    """Do MMC method."""
    # Make a mask for the free dofs
    free_dofs: NDArray[np.uint] = np.setdiff1d(
        np.arange(domain.num_dofs),
        boundary_conditions["fixed_dof_ids"],
    )

    # Load the beam on the RHS half way up
    loaded_dof_ids: NDArray[np.uint] = boundary_conditions["loaded_dof_ids"]
    load_magnitudes = boundary_conditions["load_magnitudes"]

    # sparse force vector [x1, y1, x2, y2, ...]
    forces = scipy.sparse.csc_array(
        (load_magnitudes, (loaded_dof_ids, np.zeros_like(loaded_dof_ids))),
        shape=(domain.num_dofs, 1),
    )

    # Define the element stiffness matrix
    element_stiffness: NDArray[np.float64] = finite_element.element_stiffness_matrix(
        YOUNGS_MODULUS,
        POISSON_RATIO,
        domain.element_size,
        THICKNESS,
    )

    # Combine the initials, mins, and maxes for the design variables from each group
    design_variables: NDArray[np.float64] = np.hstack(
        [cg.variable_initials_flattened for cg in component_list],
    )
    design_variables_min: NDArray[np.float64] = np.hstack(
        [cg.bounds_flattened[0] for cg in component_list],
    )
    design_variables_max: NDArray[np.float64] = np.hstack(
        [cg.bounds_flattened[1] for cg in component_list],
    )
    num_design_variables: int = design_variables.size

    # Initialise the starting values for mma optimization
    low: NDArray[np.float64] = np.expand_dims(design_variables_min, axis=1)
    upp: NDArray[np.float64] = np.expand_dims(design_variables_max, axis=1)

    design_variables_history: NDArray[np.float64] = np.zeros(
        (max_iterations, design_variables.size),
    )
    objective_history: list[float] = []
    constraint_history: list[float] = []

    # Combine the level set functions from the components to form a global one
    group_tdfs: list[Callable] = [compose_tdfs(tdf=cg.tdf) for cg in component_list]

    structure_tdf: Callable = compose_structure_tdf(group_tdfs, component_list)

    # Used to modify the Young's modulus (E) of the elements
    heaviside_structure: Callable = make_heaviside(
        tdf=structure_tdf,
        transition_width=HEAVISIDE_TRANSITION_WIDTH,
        minimum_value=HEAVISIDE_MIN_VALUE,
    )
    # heaviside_structure: Callable = make_leaky_relu(tdf=structure_tdf)

    # Optimisation loop
    for iteration in tqdm.trange(max_iterations):
        # Save the design variables
        design_variables_history[iteration] = design_variables
        # Calculate the density of the elements
        node_densities: NDArray[np.float64] = heaviside_structure(design_variables)
        element_densities: NDArray[np.float64] = domain.average_node_values_to_element(
            node_densities,
        )

        # Stiffness Matrix
        stiffness: scipy.sparse.csc_matrix = finite_element.assemble_stiffness_matrix(
            element_dof_ids=domain.element_dof_ids,
            element_densities=element_densities,
            element_stiffness_matrix=element_stiffness,
        )

        # Reduce the stiffness matrix to the free dofs
        stiffness_free: scipy.sparse.csc_matrix = stiffness[free_dofs, :][:, free_dofs]

        # Solve the system
        displacements: NDArray[np.float64] = np.zeros(domain.num_dofs)
        displacements[free_dofs] = scipy.sparse.linalg.spsolve(
            stiffness_free,
            forces[free_dofs],
        )

        # Calculate the Energy of the Elements
        element_displacements: NDArray[np.float64] = displacements[
            domain.element_dof_ids
        ]
        element_energy: NDArray[np.float64] = np.sum(
            (element_displacements @ element_stiffness) * element_displacements,
            axis=1,
        ).reshape(domain.element_shape, order="F")

        node_energy: NDArray[np.float64] = domain.element_value_to_nodes(
            element_energy,
        ).reshape((-1, 1), order="F")
        log_node_energy = np.log(node_energy + 1 + 1e-9)
        # plot_values(log_node_energy, (81, 41), title="Log Node Energy").show()

        # Sensitivity Analysis

        # Objective and derivative
        objective: NDArray[np.float64] = forces.T @ displacements * SCALE
        objective_history.append(objective[0])
        grads = jax.jacobian(heaviside_structure)(design_variables)
        d_objective_d_design_vars = jnp.nansum(
            -log_node_energy * grads,
            axis=0,
        )

        # Volume fraction constraint and derivative
        volume_fraction_constraint: float = (
            jnp.mean(node_densities) - volume_fraction_limit
        )

        # TODO(JonathanRaines): The solution seems independent
        # of the constraint gradient sign. Flipping it does nothing...
        constraint_grads: NDArray[np.float64] = jnp.nansum(
            domain.node_volumes.reshape((-1, 1), order="F") * grads,
            axis=0,
        )

        constraint_history.append(volume_fraction_constraint)

        # Update design variables
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, ss, low, upp = mma.mmasub(
            m=1,
            n=num_design_variables,
            iter=iteration + 1,
            xval=np.expand_dims(design_variables, axis=1),
            xmin=np.expand_dims(design_variables_min, axis=1),
            xmax=np.expand_dims(design_variables_max, axis=1),
            xold1=np.expand_dims(
                design_variables_history[max(0, iteration - 1)],
                axis=1,
            ),
            xold2=np.expand_dims(
                design_variables_history[max(0, iteration - 2)],
                axis=1,
            ),
            f0val=objective,
            df0dx=np.expand_dims(d_objective_d_design_vars, 1),
            fval=np.expand_dims(volume_fraction_constraint, axis=0),
            dfdx=np.expand_dims(constraint_grads, 0),
            low=low,
            upp=upp,
            a0=A0,
            a=A,
            c=C,
            d=D,
            move=MOVE,
        )

        # Update the components
        new_design_variables: NDArray = xmma.copy().flatten()
        keep_mask = np.ones_like(new_design_variables, dtype=bool)
        # keep_mask[new_design_variables[:, -1] < -ε, :] = False
        # keep_mask = keep_mask.reshape(-1)
        design_variables = new_design_variables[keep_mask]
        design_variables_min = design_variables_min[keep_mask]
        design_variables_max = design_variables_max[keep_mask]
        # design_variable_history[:, iteration][keep_mask] = design_variables.flatten()
        # low = low[keep_mask]
        # upp = upp[keep_mask]

        # if is_converged(
        #     iteration=iteration,
        #     objective_tolerance=OBJECTIVE_TOLERANCE,
        #     objective_history=objective_history,
        #     constraint_value=volume_fraction_constraint,
        #     window_size=5,
        # ):
        #     print("Converged")
        # break

    return (
        design_variables_history[: iteration + 1],
        objective_history[: iteration + 1],
        constraint_history[: iteration + 1],
    )


def compose_tdfs(
    tdf: Callable[[NDArray[np.float64]], ArrayLike],
) -> Callable[[NDArray[np.float64]], Array]:
    """Compose a group-level topology description function.

    Group tdf takes in an num_components by num_design_variables array
    """

    def group_tdf(design_variables: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the value of the combined Topology Description Function for a group of components.

        Args:
            design_variables: NDArray[np.float64] - The design variables for the group of components. dims: num_components x num_design_variables

        Returns:
            NDArray[np.float64] - The combined Topology Description Function value at each point in the domain
                (the domain is baked into the TDF of the group).

        """
        return jnp.max(jax.vmap(tdf)(design_variables), axis=0)
        # K-S method as per original paper
        return (
            jnp.log(jnp.sum(jnp.exp(jax.vmap(tdf)(design_variables) * 100), axis=0))
            / 100
        )

    return group_tdf


def compose_structure_tdf(
    group_tdfs: list[Callable],
    component_list: list[ComponentGroup],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Compose component Topology Description Functions into a structure TDF.

    Return a function that unravels the design variables and calculates the combined TDF
        for the structure.
    """
    group_design_variable_counts: Array = jnp.asarray(
        [cg.num_design_variables for cg in component_list],
    )
    group_design_variables_per_component: list[int] = [
        cg.free_variable_col_indexes.size for cg in component_list
    ]

    def structure_tdf(design_variables: ArrayLike) -> Array:
        """Evaluate the Topology Description Function for the structure.

        Given some design variable, calculate the value of the topology description
        function at each point in the domain. Points are set when the component
        group TDF is created.
        """
        group_design_variables_flat: list[NDArray] = jnp.split(
            design_variables,
            jnp.cumsum(group_design_variable_counts)[:-1],
        )
        group_design_variables = [
            design_variables.reshape(-1, n)
            for design_variables, n in zip(
                group_design_variables_flat,
                group_design_variables_per_component,
                strict=True,
            )
        ]

        group_tdf_values = [
            tdf(dv) for tdf, dv in zip(group_tdfs, group_design_variables, strict=True)
        ]

        return jnp.max(jnp.asarray(group_tdf_values), axis=0)

        # K-S method as per original paper
        return (
            jnp.log(jnp.sum(jnp.exp(jnp.asarray(group_tdf_values) * 100), axis=0)) / 100
        )

    return structure_tdf


def is_converged(
    iteration,
    objective_tolerance,
    objective_history,
    constraint_value,
    window_size,
) -> bool:
    """Check if the optimisation has converged."""
    if iteration > window_size and constraint_value < 0:
        smoothed_objective_change: NDArray = moving_average(
            objective_history,
            window_size,
        )
        smoothed_objective_deltas: NDArray = np.diff(smoothed_objective_change)
        return bool(np.all(np.abs(smoothed_objective_deltas) < objective_tolerance))
    return False


def moving_average(values, n):
    """Calculate a rolling average for values with window size n."""
    ret: NDArray = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def make_heaviside(
    tdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    transition_width: float,
    minimum_value: float = 0.0,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Smoothed Heaviside function.

    https://en.wikipedia.org/wiki/Heaviside_step_function.

    An element-wise function that 1 when x > 0, and minimum_value when x < 0.
    The step is smoothed over the transition_width.

    Args:
        tdf: Callable - The topology description function to make crisp
        minimum_value: float - The lower value for areas where x<0
        transition_width: float - The size of the transition region

    Returns:
        NDArray - The smoothed Heaviside of the input array

    """

    @jax.jit
    def smooth_heaviside(design_variables: NDArray[np.float64]) -> NDArray[np.float64]:
        x = tdf(design_variables)
        h_x = (
            3
            * (1 - minimum_value)
            / 4
            * (x / transition_width - x**3 / (3 * transition_width**3))
            + (1 + minimum_value) / 2
        )
        h_x = jnp.where(x < -transition_width, minimum_value, h_x)
        return jnp.where(x > transition_width, 1, h_x)

    return smooth_heaviside


def make_leaky_relu(
    tdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def leaky_relu(design_variables: NDArray[np.float64]) -> NDArray[np.float64]:
        x = tdf(design_variables)
        negative_slope = 1
        return jnp.where(x > 0, 1 + x * negative_slope, x * negative_slope)

    return leaky_relu


def plot_values(
    values: NDArray,
    domain_shape: tuple[int, int],
    title: str = "",
) -> go.Figure:
    """Plot values for debugging."""
    return go.Figure(
        data=go.Contour(
            z=values.reshape(domain_shape, order="F").T,
        ),
        layout=go.Layout(
            title=title,
        ),
    )


def connected_components(
    singed_distance_functions: NDArray[np.float64],
) -> NDArray[np.bool]:
    """Find connected components in the list of components.

    Args:
        singed_distance_functions: NDArray[np.float64] - The stack of TDFs
    Returns:
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


# def point_is_in_component(
#     signed_distance_functions: list[components.Component], points: NDArray[np.float64]
# ) -> list[int]:
#     """Returns a list of indices indicating if a point is in a component
#     parameters:
#         singed_distance_functions: NDArray[np.float64] - The stack of signed distance functions for each component
#     returns:
#         NDArray[int] - the indexes of components that the point is within
#     """
#     components_touching_point = set()
#     for point in points:
#         signed_distances: list[float] = np.array(
#             [sdf(point) for sdf in signed_distance_functions]
#         )
#         components_touching_point.update(np.argwhere(signed_distances > 0).flatten())
#     return list(components_touching_point)
