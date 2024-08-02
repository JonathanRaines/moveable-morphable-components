import numpy as np
from numpy.typing import NDArray

from moveable_morphable_components.main import heaviside


def test_heaviside():
    α = 0.1

    φ: NDArray = np.array([[-1.0, -0.5, 0, 0.5, 1.0], [-1.0, -0.5, 0, 0.5, 1.0]])

    h_φ: NDArray = heaviside(x=φ, α=α, ε=0.1)
    expected_h_φ: NDArray = np.array([[α, α, 0.55, 1.0, 1.0], [α, α, 0.55, 1.0, 1.0]])

    assert np.allclose(h_φ, expected_h_φ)
