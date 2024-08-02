from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

E: float = 1.0  # Young's modulus
α: float = 1e-9  # Young's modulus of void
ε: float = 0.2  # Size of transition region for the smoothed Heaviside function
ν: float = 0.3  # Poisson's ratio
t: float = 1.0  # Thickness

MAX_ITERATIONS: int = 100


@dataclass
class Component:
    x: float
    y: float
    θ: float
    t_1: float
    t_2: float
    t_3: float


def main() -> None:
    define_design_space()
    define_objective()
    define_constraints()
    components: list[Component] = initialise_components()

    for i in range(MAX_ITERATIONS):
        calculate_φ(components)
        finite_element()
        sensitivity_analysis()
        update_design_variables()

        if is_converged():
            break


def define_design_space(
    width,
    height,
) -> None:
    raise NotImplementedError


def define_objective() -> None:
    raise NotImplementedError


def define_constraints() -> None:
    raise NotImplementedError


def initialise_components() -> None:
    raise NotImplementedError


def calculate_φ() -> None:
    """Calculates the level set function φ"""
    raise NotImplementedError


def finite_element() -> None:
    raise NotImplementedError


def sensitivity_analysis() -> None:
    raise NotImplementedError


def update_design_variables() -> None:
    raise NotImplementedError


def is_converged() -> bool:
    raise NotImplementedError


def heaviside(x: NDArray, α: float, ε: float) -> NDArray:
    """
    Smoothed Heaviside function
    https://en.wikipedia.org/wiki/Heaviside_step_function

    Parameters:
        x: NDArray - The input array
        α: float - The Young's modulus of void (outside the components)
        ε: float - The size of the transition region

    Returns:
        NDArray - The smoothed Heaviside of the input array
    """
    h_x = 3 * (1 - x) / 4 * (x / ε - x**3 / (3 * ε**3)) + (1 + α) / 2
    h_x = np.where(x < -ε, α, h_x)
    h_x = np.where(x > ε, 1, h_x)
    return h_x


if __name__ == "__main__":
    main()
