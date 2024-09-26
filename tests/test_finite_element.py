
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from moveable_morphable_components import finite_element
from moveable_morphable_components.domain import Domain2D

YOUNGS_MODULUS: np.float64 = 1e7
POISSON_RATIO: np.float64 = 0.3
THICKNESS: np.float64 = 1.0
FORCE_MAGNITUDE: np.float64 = 1_000.0

ABSOLUTE_TOLERANCE: np.float64 = 0.005


@pytest.fixture
def d() -> Domain2D:
    """Create a 2D domain fixture."""
    return Domain2D(dimensions=(1.0, 0.2), element_shape=(10, 2))


def test_deflection(d: Domain2D):
    """Compare the deflection calculated by FEM to an analytical solution.

    A cantileverd beam with point load at the tip.
    """
    # Fix the left hand side in place
    fixed_dof_ids: NDArray = d.left_boundary_dof_ids()

    # Make a mask for the free dofs
    free_dofs: NDArray[np.uint] = np.setdiff1d(
        np.arange(d.num_dofs), fixed_dof_ids,
    )

    # Load the beam on the RHS half way up
    loaded_dof_ids: NDArray[np.uint] = [d.coords_to_nearest_dof_ids(
        point=(d.dimensions[0], d.dimensions[1] / 2),
    )[1]]

    load_magnitudes = [-FORCE_MAGNITUDE]
    # force vector [x1, y1, x2, y2, ...]
    forces = scipy.sparse.csc_array(
        (load_magnitudes, (loaded_dof_ids, np.zeros_like(loaded_dof_ids))),
        shape=(d.num_dofs, 1),
    )

    element_stiffness: NDArray = finite_element.element_stiffness_matrix(
        YOUNGS_MODULUS, POISSON_RATIO, d.element_size, THICKNESS,
    )

    element_densities = np.ones(d.num_elements, dtype=np.float32)

    stiffness = finite_element.assemble_stiffness_matrix(
        d.element_dof_ids,
        element_densities,
        element_stiffness,
    )

    stiffness_free: scipy.sparse.csc_matrix = stiffness[free_dofs,
                                                        :][:, free_dofs]
    # Solve for the displacements
    displacements: NDArray[np.float64] = np.zeros(d.num_dofs)
    displacements[free_dofs] = scipy.sparse.linalg.spsolve(
        stiffness_free, forces[free_dofs])
    x_eval = np.linspace(0, d.dimensions[0], d.node_shape[0])
    midline_dof_ids = [d.coords_to_nearest_dof_ids(
        (x, d.dimensions[1] / 2))[1] for x in x_eval]
    displacements_y_midline = displacements[midline_dof_ids]

    # Analytical solution
    expected_displacements = cantilver_analytical_displacement(
        length=d.dimensions[0],
        height=d.dimensions[1],
        thickness=THICKNESS,
        force=-FORCE_MAGNITUDE,
        youngs_modulus=YOUNGS_MODULUS,
        x_eval=x_eval,
    )

    # import plotly.express as px
    # fig = px.line(x=x_eval, y=expected_displacements)
    # fig.add_scatter(x=x_eval, y=displacements_y_midline, mode="lines")
    # fig.show()

    assert_allclose(
        displacements_y_midline,
        expected_displacements,
        verbose=True,
        atol=ABSOLUTE_TOLERANCE,
    )


def cantilver_analytical_displacement(
        length: float,
        height: float,
        thickness: float,
        force: float,
        youngs_modulus: float,
        x_eval: NDArray,
) -> NDArray:
    """Calculate the deflection of a cantilevered beam with a point load at the tip.

    Args:
        force: The magnitude of the force applied to the beam.
        length: The length of the beam.
        height: The height of the beam.
        thickness: The thickness of the beam.
        x_eval: The x coordinates to evaluate the deflection at.

    Returns:
        The deflection in y at the x coordinates.

    """
    second_moment_of_area = thickness * height ** 3 / 12
    return (
        force
        * x_eval ** 2
        / (6 * youngs_modulus * second_moment_of_area)
        * (3 * length - x_eval)
    )


def test_extension(d: Domain2D):
    """Compare the deflection calculated by FEM to an analytical solution.

    A vertical beam with point load at the bottom.
    """
    # Fix the left hand side in place
    fixed_dof_ids: NDArray = d.left_boundary_dof_ids()

    # Make a mask for the free dofs
    free_dofs: NDArray[np.uint] = np.setdiff1d(
        np.arange(d.num_dofs), fixed_dof_ids,
    )

    # Load the beam on the RHS half way up
    loaded_dof_ids: NDArray[np.uint] = [d.coords_to_nearest_dof_ids(
        point=(d.dimensions[0], d.dimensions[1] / 2),
    )[0]]

    load_magnitudes = [-FORCE_MAGNITUDE]
    # force vector [x1, y1, x2, y2, ...]
    forces = scipy.sparse.csc_array(
        (load_magnitudes, (loaded_dof_ids, np.zeros_like(loaded_dof_ids))),
        shape=(d.num_dofs, 1),
    )

    element_stiffness: NDArray = finite_element.element_stiffness_matrix(
        YOUNGS_MODULUS, POISSON_RATIO, d.element_size, THICKNESS,
    )

    element_densities = np.ones(d.num_elements, dtype=np.float32)

    stiffness = finite_element.assemble_stiffness_matrix(
        d.element_dof_ids,
        element_densities,
        element_stiffness,
    )

    stiffness_free: scipy.sparse.csc_matrix = stiffness[free_dofs,
                                                        :][:, free_dofs]
    # Solve for the displacements
    displacements: NDArray[np.float64] = np.zeros(d.num_dofs)
    displacements[free_dofs] = scipy.sparse.linalg.spsolve(
        stiffness_free, forces[free_dofs])
    max_displacement = np.max(np.absolute(displacements))

    # Analytical solution
    stress = FORCE_MAGNITUDE / (d.dimensions[1] * THICKNESS)
    strain = stress / YOUNGS_MODULUS
    expected_dispalcement = d.dimensions[0] * strain

    assert_allclose(max_displacement, expected_dispalcement,
                    atol=ABSOLUTE_TOLERANCE)
