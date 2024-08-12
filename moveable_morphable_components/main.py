import itertools


import numpy as np  # TODO: use jax.numpy fully
import jax.numpy as jnp
from numpy.typing import NDArray

import plotly.express as px
import tqdm


import components
from domain import Domain2D
import finite_element
import method_moving_asymptotes as mma

E: float = 1e7  # Young's modulus
α: float = 1e-9  # Young's modulus of void
ε: float = 0.1  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness

MAX_ITERATIONS: int = 500
VOLUME_FRACTION: float = 0.4


def main() -> None:
    # domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(20, 10))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), element_shape=(3, 2))
    domain: Domain2D = Domain2D(dimensions=(1.0, 0.2), element_shape=(5, 1))
    # domain: Domain2D = Domain2D(dimensions=(1.0, 1.0), element_shape=(2, 1))
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
    F = jnp.zeros(domain.num_dofs, dtype=jnp.float32)
    # F = F.at[loaded_dof_ids[1]].set(-1_000.0)  # Force in negative y direction (N)
    F = F.at[loaded_dof_ids[0]].set(1_000.0)  # Force in negative y direction (N)

    # Define the element stiffness matrix
    # This is independent of the components or domain
    # It's a function of the material properties and the element size
    # Assumes a bi-linear quadrilateral element
    K_e: NDArray = finite_element.make_stiffness_matrix(E, ν, domain.element_size, t)

    # Generate the initial components
    # component_list: list[components.Component] = initialise_components(
    #     n_x=4,
    #     n_y=2,
    #     domain=domain,
    # )
    component_list: list[components.Component] = [
        components.UniformBeam(
            center=components.Point2D(*domain.dimensions / 2),
            angle=0.0,
            length=domain.dimensions[0],
            thickness=0.2,
        )
    ]

    # Define the initial values for the design variables
    initial_design_variables = np.expand_dims(
        np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten(),
        axis=1,
    )

    design_variables_min = np.array((0, 0, -np.pi / 2, 0.3, 1))
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
    m = 1
    design_variables = initial_design_variables.copy()
    design_variables_prev = initial_design_variables.copy()
    design_variables_prev_2 = initial_design_variables.copy()
    low = design_variables_min
    upp = design_variables_max
    a0 = 1
    a = np.zeros((m, 1))
    c = np.full((m, 1), 1000)
    d = np.zeros((m, 1))

    # Optimisation loop
    for iteration in tqdm.trange(MAX_ITERATIONS):
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
            components.plot_φ(H, domain.node_shape)
            pass

        coords = np.fliplr(np.array(list(domain.node_coordinates)))
        # Calculate the derivative of φ with respect to the design variables
        dφ_component_d_design_vars = np.concat(
            [comp.φ_grad(coords[:, 0], coords[:, 1]) for comp in component_list]
        )
        dφ_d_design_vars = np.repeat(dφ_dφs, 5, axis=0) * dφ_component_d_design_vars
        # for design_var in dφ_d_design_vars:
        #     plot_values(design_var, domain.node_shape).show()

        # Calculate the density of the elements
        element_densities: NDArray = domain.average_node_values_to_element(H)

        K = finite_element.stiffness_matrix(
            domain.element_dof_ids,
            # element_densities * E,
            np.ones(domain.element_shape) * E,
            K_e,
        )

        K_free = K[free_dofs, :][:, free_dofs]
        K_free = jnp.array(K[free_dofs, :][:, free_dofs])

        # Solve the system
        U_free = finite_element.solve_displacements(K_free, F[free_dofs])
        U: NDArray = jnp.zeros(domain.num_dofs, dtype=jnp.float32)
        U = U.at[free_dofs].set(U_free)

        # # Visualise the result TODO debug
        # # X displacements
        # plot_values(U[::2], domain.node_shape).show()
        # # Y displacements
        # plot_values(U[1::2], domain.node_shape).show()
        # Total displacements
        plot_values(np.linalg.norm([U[::2], U[1::2]], axis=0), domain.node_shape).show()

        # Calculate the Energy of the Elements
        U_by_element = U[domain.element_dof_ids]
        element_energy = np.sum(
            (U_by_element @ K_e) * U_by_element,
            axis=1,
        ).reshape(domain.element_shape, order="F")
        # plot_values(element_energy, domain.element_shape).show()

        node_energy = domain.element_value_to_nodes(element_energy).flatten(order="F")
        # plot_values(node_energy, domain.node_shape).show()

        node_volumes = domain.element_value_to_nodes(
            np.ones(domain.element_shape)
        ).flatten(order="F")
        # plot_values(node_volumes, domain.node_shape).show()

        # sensitivity_analysis()
        design_variables = np.array(
            [list(component.design_variables) for component in component_list]
        ).flatten()

        # Objective and derivative
        objective = F.T @ U
        d_objective_d_design_vars = jnp.nansum(
            -node_energy * dH_dφ * dφ_d_design_vars, axis=1
        )

        # Volume fraction constraint and derivative
        volume_fraction_constraint = (
            np.sum(node_densities)
            * np.prod(domain.element_size)
            / np.prod(domain.dimensions)
            - VOLUME_FRACTION
        )
        d_volume_fraction_d_design_vars = np.nansum(
            node_volumes
            * dH_dφ
            * dφ_d_design_vars
            * np.prod(domain.element_size)
            / np.prod(domain.dimensions),
            axis=1,
        )

        # digit = 5 - np.floor(np.log10(np.max(abs(d_objective_d_design_vars))))
        # d_objective_d_design_vars = np.round(d_objective_d_design_vars, digit)
        # digit = 5 - np.floor(np.log10(np.max(abs(d_volume_fraction_d_design_vars))))
        # d_volume_fraction_d_design_vars = np.round(
        #     d_volume_fraction_d_design_vars, digit
        # )

        d_objective_d_design_vars = d_objective_d_design_vars / np.max(
            abs(d_objective_d_design_vars)
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

        # if is_converged():
        #     break


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
    x = jnp.array(x)
    h_x = (
        3 * (1 - x) / 4 * (x / transition_width - x**3 / (3 * transition_width**3))
        + (1 + minimum_value) / 2
    )
    h_x = jnp.where(x < -transition_width, minimum_value, h_x)
    h_x = jnp.where(x > transition_width, 1, h_x)
    return h_x


def plot_values(values: NDArray, domain_shape: tuple[int, int]) -> None:
    return px.imshow(values.reshape(domain_shape, order="F").T, origin="lower")


if __name__ == "__main__":
    main()
