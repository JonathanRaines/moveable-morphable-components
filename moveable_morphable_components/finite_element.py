import itertools

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def make_stiffness_matrix(
    E: float, ν: float, element_size: tuple[float, float], t: float
) -> NDArray:
    """
    Create the stiffness matrix for a single element

    Parameters:
        E: float - Young's modulus
        ν: float - Poisson's ratio
        element_size: tuple[float, float] - The size of the element
        h: float - The thickness of the element

    Returns:
        NDArray - The stiffness matrix for a single element
    """
    # TODO: Mostly based off the original 218 line MMC-2D code.
    # I would prefer to use the shape functions and domain to generate the matrix.
    # High likelihood of errors in this function.

    # My calculation of K_e matches that in the 218 line code.  k_1_1 here is equivalent to k1(1)
    # It is the first row of the element stiffness matrix.
    # I have adjusted indices for k2 from the 218 line code. There K_e is described as a 1D matrix
    # so the k2 indices are in a strange order to allow for the process of turning those values into a
    # symmetric 8 x 8 matrix. All values of k2 match my derivation, I have just changed their indices them for clarity.

    # Note: the indices in the variable names are 1-indexed to match the 218 line code and mathematical matrix notation.
    # They are never indexed on their names or used outside this function.

    a, b = element_size
    k_1_1: float = -1 / (6 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_2: float = (ν + 1) / 8
    k_1_3: float = -1 / (12 * a * b) * (a**2 * (ν - 1) + 4 * b**2)
    k_1_4: float = (3 * ν - 1) / 8
    k_1_5: float = 1 / (12 * a * b) * (a**2 * (ν - 1) - 2 * b**2)
    k_1_7: float = 1 / (6 * a * b) * (a**2 * (ν - 1) + b**2)
    k_2_2: float = -1 / (6 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_4: float = 1 / (6 * a * b) * (b**2 * (ν - 1) + a**2)
    k_2_6: float = 1 / (12 * a * b) * (b**2 * (ν - 1) - 2 * a**2)
    k_2_8: float = -1 / (12 * a * b) * (b**2 * (ν - 1) + 4 * a**2)

    K_e_triu: NDArray = (
        E
        * t
        / (1 - ν**2)
        * np.array(
            [
                [k_1_1, k_1_2, k_1_3, k_1_4, k_1_5, -k_1_2, k_1_7, -k_1_4],
                [0.0, k_2_2, -k_1_4, k_2_4, -k_1_2, k_2_6, k_1_4, k_2_8],
                [0.0, 0.0, k_1_1, -k_1_2, k_1_7, k_1_4, k_1_5, k_1_2],
                [0.0, 0.0, 0.0, k_2_2, -k_1_4, k_2_8, k_1_2, k_2_6],
                [0.0, 0.0, 0.0, 0.0, k_1_1, k_1_2, k_1_3, k_1_4],
                [0.0, 0.0, 0.0, 0.0, 0.0, k_2_2, -k_1_4, k_2_4],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_1_1, -k_1_2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_2_2],
            ]
        )
    )

    K_e: NDArray = K_e_triu + K_e_triu.T - np.diag(np.diag(K_e_triu))
    return K_e


def stiffness_matrix(
    element_dof_ids: NDArray, element_densities: NDArray, K_e: NDArray
) -> NDArray:
    # Make the stiffness matrix
    num_dofs: int = np.max(element_dof_ids) + 1
    K: NDArray = np.zeros((num_dofs, num_dofs))

    # TODO vectorised way currently broken
    # K[
    #     element_dof_ids[:, :, np.newaxis],
    #     element_dof_ids[:, np.newaxis, :],
    # ] += K_e * element_densities.flatten(order="F")[:, np.newaxis, np.newaxis]

    # Old way of doing it

    for element, dof_ids in enumerate(element_dof_ids):
        for i, j in itertools.product(range(8), range(8)):
            K[dof_ids[i], dof_ids[j]] += (
                K_e[i, j] * element_densities.flatten(order="F")[element]
            )

    return K


def solve_displacements(K, F):
    return jnp.linalg.solve(K, F)
