import numpy as np
from numpy.typing import NDArray

from moveable_morphable_components.main import heaviside, Domain2D


def test_heaviside():
    α = 0.1

    φ: NDArray = np.array([[-1.0, -0.5, 0, 0.5, 1.0], [-1.0, -0.5, 0, 0.5, 1.0]])

    h_φ: NDArray = heaviside(x=φ, minimum_value=α, transition_width=0.1)
    expected_h_φ: NDArray = np.array([[α, α, 0.55, 1.0, 1.0], [α, α, 0.55, 1.0, 1.0]])

    assert np.allclose(h_φ, expected_h_φ)


def test_domain():
    domain = Domain2D(dimensions=(1.0, 2.0), num_elements=(10, 10), ν=0.3)

    assert all(domain.dimensions == (1.0, 2.0))
    assert all(domain.num_elements == (10, 10))
    assert all(domain.num_nodes == (11, 11))
    assert domain.num_dofs == 242  # 11 * 11 * 2D
    assert all(domain.element_size == (0.1, 0.2))
    coordinates = list(domain.node_coordinates)
    assert len(coordinates) == 121
    assert coordinates[0] == (0.0, 0.0)
    assert coordinates[-1] == (1.0, 2.0)
