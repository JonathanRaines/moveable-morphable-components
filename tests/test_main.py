import numpy as np
from numpy.typing import NDArray

from moveable_morphable_components import main
from moveable_morphable_components import finite_element
from shape_func_and_stiffness import derive_stiffness_matrix


def test_heaviside():
    α = 0.1

    φ: NDArray = np.array([[-1.0, -0.5, 0, 0.5, 1.0], [-1.0, -0.5, 0, 0.5, 1.0]])

    h_φ: NDArray = main.heaviside(x=φ, minimum_value=α, transition_width=0.1)
    expected_h_φ: NDArray = np.array([[α, α, 0.55, 1.0, 1.0], [α, α, 0.55, 1.0, 1.0]])

    assert np.allclose(h_φ, expected_h_φ)


def test_make_stiffness_matrix():
    """Confirms the hard-coded stiffness matrix is the same as the one derived using sympy."""
    k_e_hardcoded = finite_element.make_stiffness_matrix(
        E=1.0, ν=0.3, element_size=(1.0, 1.0), t=1.0
    )
    k_e, t, E, ν, a, b = derive_stiffness_matrix()
    k_e_numeric = np.array(k_e.subs({E: 1.0, ν: 0.3, a: 1.0, b: 1.0, t: 1.0})).astype(
        np.float64
    )
    np.testing.assert_allclose(k_e_hardcoded, k_e_numeric)
