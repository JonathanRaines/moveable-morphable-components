import itertools


import numpy as np  # TODO: use jax.numpy fully
from jax.experimental import sparse
import jax.numpy as jnp
from numpy.typing import NDArray

import plotly.express as px
import plotly.graph_objects as go
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

NUM_CONSTRAINTS = 1  # Volume fraction constraint
A0 = 1
A = np.zeros((NUM_CONSTRAINTS, 1))
C = np.full((NUM_CONSTRAINTS, 1), 1000)
D = np.zeros((NUM_CONSTRAINTS, 1))
MOVE = 1.0

MAX_ITERATIONS: int = 500
VOLUME_FRACTION: float = 0.4


def main() -> None:
    domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(80, 40))
    # domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(20, 10))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), element_shape=(3, 2))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 0.2), element_shape=(5, 1))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), element_shape=(2, 2))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), element_shape=(1, 1))

    # define_objective() # TODO

    # Fix the left hand side in place
    fixed_dof_ids: NDArray = domain.select_dofs_on_left_boundary()

    # Make a mask for the free dofs
    fixed_dofs = np.zeros(domain.num_dofs, dtype=bool)
    fixed_dofs[[fixed_dof_ids]] = True
    free_dofs = np.logical_not(fixed_dofs)

    # Load the beam on the RHS half way up
    # loaded_dof_ids = domain.select_dofs_with_point(point=(2.0, 0.5))
    loaded_dof_ids = domain.select_dofs_with_point(
        point=(domain.dimensions[0], domain.dimensions[1] / 2)
    )

    # force vector [x1, y1, x2, y2, ...]
    # F = jnp.zeros(domain.num_dofs, dtype=jnp.float32)
    # F = F.at[loaded_dof_ids[1]].set(-1_000.0)  # Force in negative y direction (N)
    # F = F.at[loaded_dof_ids[0]].set(1_000.0)  # Tensile Force in positive x direction (N)
    F = scipy.sparse.csc_array(
        ([-100], ([loaded_dof_ids[1]], [0])), shape=(domain.num_dofs, 1)
    )

    # Define the element stiffness matrix
    # This is independent of the components or domain
    # It's a function of the material properties and the element size
    # Assumes a bi-linear quadrilateral element
    K_e: NDArray = finite_element.element_stiffness_matrix(E, ν, domain.element_size, t)

    # Generate the initial components
    component_list: list[components.Component] = initialise_components(
        n_x=4,
        n_y=2,
        domain=domain,
    )
    # component_list: list[components.Component] = [
    #     components.UniformBeam(
    #         center=components.Point2D(*domain.dimensions / 2),
    #         angle=np.pi / 2,
    #         length=domain.dimensions[1],
    #         thickness=0.4,
    #     )
    # ]

    # Define the initial values for the design variables
    initial_design_variables = np.expand_dims(
        np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten(),
        axis=1,
    )

    design_variables_min = np.array((0, 0, -np.pi / 2, 0.1, 0.01))
    design_variables_min = np.expand_dims(
        np.tile(design_variables_min, len(component_list)), axis=1
    )
    design_variables_max = np.array(
        (
            domain.dimensions[0],
            domain.dimensions[1],
            np.pi / 2,
            np.linalg.norm(domain.dimensions) / 2,
            np.min(domain.dimensions) / 2,
        )
    )
    design_variables_max = np.expand_dims(
        np.tile(design_variables_max, len(component_list)), axis=1
    )

    # initialise the starting values for mma optimization
    design_variables = initial_design_variables.copy()
    design_variables_prev = initial_design_variables.copy()
    design_variables_prev_2 = initial_design_variables.copy()
    low = design_variables_min
    upp = design_variables_max

    objective_change = 1.0
    objective_history = []
    # Optimisation loop
    for iteration in tqdm.trange(MAX_ITERATIONS):
        if objective_change < 1e-4:
            break

        # Combine the level set functions from the components to form a global one
        # dφ_dφs is the derivative of the global level set function with respect to the component level set functions
        # It is 1 where the component is the maximum and 0 elsewhere
        φ, dφ_dφs = calculate_φ(component_list, domain.node_coordinates)
        # plot_values(φ, domain.node_shape).show()
        # plot_values(dφ_dφs[0], domain.node_shape).show()

        # H is Heaviside(φ), it is used to modify the Young's modulus (E) of the elements
        H: NDArray = heaviside(φ, transition_width=ε, minimum_value=α)

        # Calculate the derivative of H with respect to φ using the analytical form
        # TODO: Replace with automatic differentiation?
        dH_dφ: NDArray = 3 * (1 - α) / (4 * ε) * (1 - φ**2 / ε**2)
        dH_dφ = np.where(abs(φ) > ε, 0, dH_dφ)
        # plot_values(dH_dφ, domain.node_shape).show()

        # Plot every 10 iterations
        if iteration % 10 == 0:
            # go.Figure(
            #     data=go.Contour(
            #         z=φ.reshape(domain.node_shape, order="F").T,
            #         contours=dict(
            #             start=0,
            #             end=1,
            #             size=0.1,
            #         ),
            #     )
            # ).show()
            go.Figure(
                data=go.Contour(
                    z=H.reshape(domain.node_shape, order="F").T,
                    contours=dict(
                        start=0,
                        end=1,
                        size=0.1,
                    ),
                )
            ).show()
            pass

        coords = np.fliplr(np.array(list(domain.node_coordinates)))
        # Calculate the derivative of φ with respect to the design variables
        dφ_component_d_design_vars = np.concatenate(
            [comp.φ_grad(coords[:, 0], coords[:, 1]) for comp in component_list]
        )
        dφ_d_design_vars = np.repeat(dφ_dφs, 5, axis=0) * dφ_component_d_design_vars
        # for design_var in dφ_d_design_vars:
        #     plot_values(design_var, domain.node_shape).show()

        # Calculate the density of the elements
        element_densities: NDArray = domain.average_node_values_to_element(H)
        # plot_values(element_densities, domain.element_shape).show()

        # Stiffness Matrix
        K = finite_element.assemble_stiffness_matrix(
            element_dof_ids=domain.element_dof_ids,
            element_densities=element_densities,
            element_stiffness_matrix=K_e,
        )

        # k_display = K.toarray()
        # k_diag_x = np.diag(k_display)[::2]
        # plot_values(k_diag_x, domain.node_shape).show()
        # # k_diag_y = np.diag(k_display)[1::2]
        # # plot_values(k_diag_y, domain.node_shape).show()

        # K_free: NDArray = K[np.ix_(free_dofs, free_dofs)]
        # K_free = jnp.array(K_free)

        # For sparse:
        K_free = K[free_dofs, :][:, free_dofs]

        # Solve the system
        # U_free = finite_element.solve_displacements(K_free, F[free_dofs])
        # U: NDArray = jnp.zeros(domain.num_dofs, dtype=jnp.float32)
        # U = U.at[free_dofs].set(U_free)
        U: NDArray = np.zeros(domain.num_dofs)
        U[free_dofs] = scipy.sparse.linalg.spsolve(K_free, F[free_dofs])

        # if iteration % 10 == 0:
        #     # # Visualise the result TODO debug
        #     # # X displacements
        #     plot_values(U[::2], domain.node_shape).show()
        #     # # Y displacements
        #     plot_values(U[1::2], domain.node_shape).show()
        #     # Total displacements
        #     plot_values(
        #         np.linalg.norm([U[::2], U[1::2]], axis=0), domain.node_shape
        #     ).show()

        # Calculate the Energy of the Elements
        U_by_element = U[domain.element_dof_ids]
        element_energy = np.sum(
            (U_by_element @ K_e) * U_by_element,
            axis=1,
        ).reshape(domain.element_shape, order="F")
        # plot_values(element_energy, domain.element_shape).show()

        node_energy = domain.element_value_to_nodes(element_energy).flatten(order="F")
        # plot_values(node_energy, domain.node_shape).show()

        # plot_values(node_volumes, domain.node_shape).show()

        # sensitivity_analysis()
        design_variables = np.expand_dims(
            np.array(
                [list(component.design_variables) for component in component_list]
            ).flatten(),
            1,
        )

        # Objective and derivative
        objective = F.T @ U
        objective_history.append(objective)
        d_objective_d_design_vars = np.nansum(
            -node_energy * dH_dφ * dφ_d_design_vars, axis=1
        )
        # if iteration % 10 == 0:
        #     plot_values(-node_energy, domain.node_shape).show()
        #     plot_values(-dH_dφ, domain.node_shape).show()

        # Volume fraction constraint and derivative
        volume_fraction_constraint = (
            np.sum(H * domain.node_volumes) / np.sum(domain.node_volumes)
            - VOLUME_FRACTION
        )
        d_volume_fraction_d_design_vars = np.nansum(
            domain.node_volumes * dH_dφ * dφ_d_design_vars,
            axis=1,
        )

        # digit = (5 - np.floor(np.log10(np.max(abs(d_objective_d_design_vars))))).astype(
        #     np.int32
        # )
        # d_objective_d_design_vars = np.round(d_objective_d_design_vars, digit).astype(
        #     np.float64
        # )
        # digit = 5 - np.floor(
        #     np.log10(np.max(abs(d_volume_fraction_d_design_vars)))
        # ).astype(np.int32)
        # d_volume_fraction_d_design_vars = np.round(
        #     d_volume_fraction_d_design_vars, digit
        # ).astype(np.float64)

        scale_factor = np.max(
            np.abs(
                np.concatenate(
                    [d_objective_d_design_vars, d_volume_fraction_d_design_vars]
                )
            )
        )
        d_objective_d_design_vars /= scale_factor
        d_volume_fraction_d_design_vars /= scale_factor

        # update_design_variables()
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

        # xmma, ymma, zmma, lam, xsi, eta, mu, zet, ss, low, upp = mma.gmmasub(
        #     m=1,
        #     n=5 * len(component_list),
        #     iter=iteration + 1,
        #     # [x, y, angle, length, thickness]
        #     epsimin=1e-7,
        #     xval=design_variables,
        #     xmin=design_variables_min,
        #     xmax=design_variables_max,
        #     xold1=design_variables_prev,
        #     xold2=design_variables_prev_2,
        #     f0val=np.expand_dims(np.array([objective]), 1),
        #     df0dx=np.expand_dims(d_objective_d_design_vars, 1),
        #     fval=volume_fraction_constraint,
        #     dfdx=np.expand_dims(d_volume_fraction_d_design_vars, 0),
        #     low=low,
        #     upp=upp,
        #     a0=a0,
        #     a=a,
        #     c=c,
        #     d=d,
        #     move=1.0,
        # )

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

        # if is_converged():
        #     break
        if iteration > 5 and volume_fraction_constraint / VOLUME_FRACTION < 1e-4:
            objective_change = np.abs(
                np.max(np.abs(objective_history[-5:]) - np.mean(objective_history[-5:]))
            ) / np.mean(objective_history[-5:])


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
    main()
