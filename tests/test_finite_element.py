import pytest

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

from moveable_morphable_components.domain import Domain2D
from moveable_morphable_components import finite_element


@pytest.fixture
def d():
    return Domain2D(dimensions=(1.0, 0.2), element_shape=(10, 2))


def test_deflection(d):
    # Fix the left hand side in place
    fixed_dof_ids: NDArray = d.select_dofs_on_left_boundary()

    # Make a mask for the free dofs
    fixed_dofs = np.zeros(d.num_dofs, dtype=bool)
    fixed_dofs[[fixed_dof_ids]] = True
    free_dofs = np.logical_not(fixed_dofs)

    # Load the beam on the RHS half way up
    loaded_dof_ids = d.select_dofs_with_point(
        point=(d.dimensions[0], d.dimensions[1] / 2)
    )

    # force vector [x1, y1, x2, y2, ...]
    F = jnp.zeros(d.num_dofs, dtype=jnp.float32)
    F = F.at[loaded_dof_ids[1]].set(-1_000.0)  # Force in negative y direction (N)

    E = 1e7
    t = 1.0
    K_e: NDArray = finite_element.make_stiffness_matrix(
        E, ν=0.3, element_size=d.element_size, t=t
    )

    element_densities = jnp.ones(d.num_elements, dtype=jnp.float32)

    K = finite_element.stiffness_matrix(
        d.element_dof_ids,
        element_densities * E,
        K_e,
    )

    K_free = K[free_dofs][:, free_dofs]
    # Solve for the displacements
    # U_free = finite_element.solve_displacements(K_free, F[free_dofs])
    U_free = np.linalg.solve(K_free, F[free_dofs])
    U_max = jnp.max(U_free)

    # Analytical solution
    I = t * d.dimensions[1] ** 3 / 12
    U_analytical = 1_000.0 * d.dimensions[0] / (3 * E * I)

    assert assert_allclose(U_max, U_analytical, rtol=1e-2)


def test_extension(d):
    # Fix the left hand side in place
    fixed_dof_ids: NDArray = d.select_dofs_on_left_boundary()

    # Make a mask for the free dofs
    fixed_dofs = np.zeros(d.num_dofs, dtype=bool)
    fixed_dofs[[fixed_dof_ids]] = True
    free_dofs = np.logical_not(fixed_dofs)

    # Load the beam on the RHS half way up
    loaded_dof_ids = d.select_dofs_with_point(
        point=(d.dimensions[0], d.dimensions[1] / 2)
    )

    # force vector [x1, y1, x2, y2, ...]
    F = jnp.zeros(d.num_dofs, dtype=jnp.float32)
    F = F.at[loaded_dof_ids[0]].set(1_000.0)  # Tension in x direction (N)

    E = 1e7
    t = 1.0
    K_e: NDArray = finite_element.make_stiffness_matrix(
        E, ν=0.3, element_size=d.element_size, t=t
    )

    element_densities = jnp.ones(d.num_elements, dtype=jnp.float32)

    K = finite_element.stiffness_matrix(
        d.element_dof_ids,
        element_densities * E,
        K_e,
    )

    K_free = K[free_dofs][:, free_dofs]
    # Solve for the displacements
    # U_free = finite_element.solve_displacements(K_free, F[free_dofs])
    U_free = np.linalg.solve(K_free, F[free_dofs])
    U_max = jnp.max(U_free)

    # Analytical solution
    stress = 1_000.0 / (d.dimensions[1] * t)
    strain = stress / E
    U_analytical = d.dimensions[0] * strain

    assert assert_allclose(U_max, U_analytical, rtol=1e-2)
