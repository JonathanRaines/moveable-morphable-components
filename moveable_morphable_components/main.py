import itertools

import numpy as np  # TODO: use jax.numpy fully
import jax.numpy as jnp
from numpy.typing import NDArray

import plotly.graph_objects as go
import scipy.sparse
import tqdm
import scipy

import components
from domain import Domain2D
import finite_element
import method_moving_asymptotes as mma

E: float = 1e5  # 1e7  # Young's modulus
α: float = 1e-3  # Heaviside minimum value
ε: float = 0.3  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness

NUM_CONSTRAINTS: int = 1  # Volume fraction constraint
A0: int = 1
A: NDArray = np.zeros((NUM_CONSTRAINTS, 1))
C: NDArray = np.full((NUM_CONSTRAINTS, 1), 1000)
D: NDArray = np.zeros((NUM_CONSTRAINTS, 1))
MOVE: float = 0.02
OBJECTIVE_TOLERANCE: float = 1.0

MAX_ITERATIONS: int = 500
VOLUME_FRACTION: float = 0.4


def main() -> None:
    domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(80, 40))

    # Fix the left hand side in place
    fixed_dof_ids: NDArray[np.uint] = domain.select_dofs_on_left_boundary()

    # Make a mask for the free dofs
    free_dofs: NDArray[np.uint] = np.setdiff1d(
        np.arange(domain.num_dofs), fixed_dof_ids
    )

    # Load the beam on the RHS half way up
    loaded_dof_ids: NDArray[np.uint] = domain.select_dofs_with_point(
        point=(domain.dimensions[0], domain.dimensions[1] / 2)
    )

    # sparse force vector [x1, y1, x2, y2, ...]
    F = scipy.sparse.csc_array(
        ([-100], ([loaded_dof_ids[1]], [0])), shape=(domain.num_dofs, 1)
    )

    # Define the element stiffness matrix
    K_e: NDArray[float] = finite_element.element_stiffness_matrix(
        E, ν, domain.element_size, t
    )

    # Generate the initial components
    component_list: list[components.Component] = layout_grid_of_uniform_beams(
        n_x=4,
        n_y=2,
        dimensions=domain.dimensions,
    )

    # Define the initial values for the design variables
    initial_design_variables: NDArray[float] = np.expand_dims(
        np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten(),
        axis=1,
    )

    # Set the bounds for the design variables
    design_variables_min: NDArray[float] = np.array((0, 0, -np.pi / 2, 0.1, 0.01))
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

    objective_history: list[float] = []
    H_history: NDArray = np.zeros((*domain.node_shape, MAX_ITERATIONS))
    # Optimisation loop
    for iteration in tqdm.trange(MAX_ITERATIONS):
        # Combine the level set functions from the components to form a global one
        # dφ_dφs is the derivative of the global level set function with respect to the component level set functions
        # It is 1 where the component is the maximum and 0 elsewhere
        φ, dφ_dφs = calculate_φ(component_list, domain.node_coordinates)
        # plot_values(φ, domain.node_shape).show()
        # plot_values(dφ_dφs[0], domain.node_shape).show()

        # H is Heaviside(φ), it is used to modify the Young's modulus (E) of the elements
        H: NDArray[float] = heaviside(φ, transition_width=ε, minimum_value=α)
        H_history[:, :, iteration] = H.reshape(domain.node_shape, order="F")

        # Calculate the derivative of H with respect to φ using the analytical form
        # TODO: Replace with automatic differentiation?
        dH_dφ: NDArray[float] = 3 * (1 - α) / (4 * ε) * (1 - φ**2 / ε**2)
        dH_dφ = np.where(abs(φ) > ε, 0.0, dH_dφ)

        coords: NDArray[np.uint] = np.fliplr(np.array(list(domain.node_coordinates)))
        # Calculate the derivative of φ with respect to the design variables
        dφ_component_d_design_vars: NDArray[float] = np.concatenate(
            [comp.φ_grad(coords[:, 0], coords[:, 1]) for comp in component_list]
        )
        dφ_d_design_vars: NDArray[float] = (
            np.repeat(dφ_dφs, 5, axis=0) * dφ_component_d_design_vars
        )

        # Calculate the density of the elements
        element_densities: NDArray[float] = domain.average_node_values_to_element(H)

        # Stiffness Matrix
        K: scipy.sparse.csc_matrix = finite_element.assemble_stiffness_matrix(
            element_dof_ids=domain.element_dof_ids,
            element_densities=element_densities,
            element_stiffness_matrix=K_e,
        )

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
        ).flatten(order="F")

        # Sensitivity_analysis()
        design_variables: NDArray[float] = np.expand_dims(
            np.array(
                [list(component.design_variables) for component in component_list]
            ).flatten(),
            1,
        )

        # Objective and derivative
        objective: NDArray[float] = F.T @ U
        objective_history.append(objective)
        d_objective_d_design_vars = np.nansum(
            -node_energy * dH_dφ * dφ_d_design_vars, axis=1
        )

        # Volume fraction constraint and derivative
        volume_fraction_constraint: float = (
            np.sum(H * domain.node_volumes) / np.sum(domain.node_volumes)
            - VOLUME_FRACTION
        )
        d_volume_fraction_d_design_vars: NDArray[float] = np.nansum(
            domain.node_volumes * dH_dφ * dφ_d_design_vars,
            axis=1,
        )

        # Normalise the derivatives
        scale_factor: float = np.max(
            np.abs(
                np.concatenate(
                    [d_objective_d_design_vars, d_volume_fraction_d_design_vars]
                )
            )
        )
        d_objective_d_design_vars /= scale_factor
        d_volume_fraction_d_design_vars /= scale_factor

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
            f0val=np.expand_dims(np.array([objective]), 1),
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
        design_variables_prev_2 = design_variables_prev.copy()
        design_variables_prev = design_variables.copy()
        design_variables = xmma.copy()

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
            constraint_tolerance=1e-4,
            constraint_error=volume_fraction_constraint / VOLUME_FRACTION,
            window_size=5,
        ):
            print("Converged")
            break

    return H_history


def layout_grid_of_uniform_beams(
    n_x: int, n_y: int, dimensions: tuple[float, float], thickness: float = 0.1
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
    # Simple max aggregation
    # φ_global: NDArray = np.max(φ_components, axis=0)
    # dφ_global_dφ_components = np.zeros_like(φ_components)
    # dφ_global_dφ_components[np.argmax(φ_components, axis=0)] = 1

    # # Kolmogorov-Smirnov (KS) aggregation as per the original MMC-2D code
    temp: NDArray = jnp.exp(φ_components * ks_aggregation_power)
    φ_global: NDArray = jnp.maximum(
        jnp.full(temp.shape[1], -1e3),
        jnp.log(np.sum(temp, axis=0)) / ks_aggregation_power,
    )

    dφ_global_dφ_components = temp / jnp.sum(temp, axis=0)

    return φ_global, dφ_global_dφ_components


# def sensitivity_analysis() -> None:
#     raise NotImplementedError


def is_converged(
    iteration,
    objective_tolerance,
    objective_history,
    constraint_tolerance,
    constraint_error,
    window_size,
) -> bool:
    if iteration > window_size and constraint_error < constraint_tolerance:
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


if __name__ == "__main__":
    steps = main()
    steps_downsampled = steps[:, :, ::1]
    fig = go.Figure(
        data=[go.Contour(z=steps_downsampled[:, :, 0].T)],
        layout=go.Layout(
            title="MMC",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[dict(label="Play", method="animate", args=[None])],
                )
            ],
        ),
        frames=[
            go.Frame(data=[go.Contour(z=steps_downsampled[:, :, i].T)])
            for i in range(steps_downsampled.shape[2])
        ],
    )
    fig.show()
