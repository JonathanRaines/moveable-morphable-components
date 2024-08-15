import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import scipy


def element_stiffness_matrix(
    youngs_modulus: float,
    poissons_ratio: float,
    element_size: tuple[float, float],
    element_thickness: float,
) -> NDArray:
    """
    Create the stiffness matrix for a single element

    Parameters:
        youngs_modulus: float - Young's modulus
        poissons_ratio: float - Poisson's ratio
        element_size: tuple[float, float] - The size of the element
        element_thickness: float - The thickness of the element

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

    # Rename to symbols for equations
    E: float = youngs_modulus
    ν: float = poissons_ratio
    t: float = element_thickness

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

    element_stiffness_matrix_triu: NDArray = (
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

    # Mirror the upper triangular matrix to get the full matrix
    element_stiffness_matrix: NDArray = (
        element_stiffness_matrix_triu
        + element_stiffness_matrix_triu.T
        - np.diag(np.diag(element_stiffness_matrix_triu))
    )
    return element_stiffness_matrix


def assemble_stiffness_matrix(
    element_dof_ids: NDArray[np.uint],
    element_densities: NDArray[float],
    element_stiffness_matrix: NDArray[float],
) -> scipy.sparse.csc_matrix:
    """
    Assemble the stiffness matrix from the element stiffness matrices and densities

    Parameters:
        element_dof_ids: NDArray[np.uint] - The global DOF indices for each element (8 x num_elements)
            Numbering is clockwise from the bottom left corner
        element_densities: NDArray[float] - The density of each element
        element_stiffness_matrix: NDArray[float] - The stiffness matrix for a single element
    """
    num_dofs: int = np.max(element_dof_ids) + 1

    row: list[int] = []
    column: list[int] = []
    data: list[float] = []
    element_densities_flat = element_densities.flatten(order="F")
    for element, dof_ids in enumerate(element_dof_ids):
        element_stiffness = element_stiffness_matrix * element_densities_flat[element]
        i, j = np.meshgrid(dof_ids, dof_ids)
        row.extend(i.flatten())
        column.extend(j.flatten())
        data.extend(element_stiffness.flatten())
        pass
    return scipy.sparse.coo_array(
        (data, (row, column)), shape=(num_dofs, num_dofs)
    ).tocsc()
