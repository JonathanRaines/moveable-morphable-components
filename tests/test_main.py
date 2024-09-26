from typing import TYPE_CHECKING

import numpy as np

from moveable_morphable_components import components, domain, finite_element, main
from shape_func_and_stiffness import derive_stiffness_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_heaviside():
    """Test the heaviside function."""
    min_val = 0.1

    dom: domain.Domain2D = domain.Domain2D(
        dimensions=(2.0, 1.0), element_shape=(3, 2))

    design_variables: NDArray = np.array([1.0, 0.5, 0.5])
    tdf: Callable = components.circle(
        components.Point2D(*dom.node_coordinates))

    crisp_tdf: Callable = main.make_heaviside(
        tdf=tdf, minimum_value=min_val, transition_width=0.1)
    expected_values: NDArray = tdf(design_variables)
    expected_crisp_values: NDArray = np.where(
        expected_values > 0, 1.0, min_val)

    assert np.allclose(crisp_tdf(design_variables), expected_crisp_values)


def test_make_stiffness_matrix():
    """Confirms the hard-coded stiffness matrix is the same as the one derived using sympy."""
    element_stiffness = finite_element.element_stiffness_matrix(
        youngs_modulus=1.0,
        poissons_ratio=0.3,
        element_size=(1.0, 1.0),
        element_thickness=1.0,
    )
    element_stiffness_symbolic, thickness, youngs_modulus, poisson_ratio, element_width, element_height = derive_stiffness_matrix()
    expected_element_stiffness = np.array(
        element_stiffness_symbolic.subs(
            {
                youngs_modulus: 1.0,
                poisson_ratio: 0.3,
                element_width: 1.0,
                element_height: 1.0,
                thickness: 1.0,
            },
        ),
    ).astype(np.float64)
    np.testing.assert_allclose(element_stiffness, expected_element_stiffness)
