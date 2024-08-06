"""Derives the element stiffness matrix from the constitutive law and shape functions."""

import numpy as np
from numpy.typing import NDArray
import sympy as sp


def derive_stiffness_matrix() -> (
    tuple[NDArray, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]
):
    """Derives the element stiffness matrix from the constitutive law and shape functions.

    For an explanation of the derivation, see:
    Intro to the Finite Element Method Lecture 6 | Isoparametric Elements and Gaussian Integration by Dr. Clayton Pettit
    https://www.youtube.com/watch?v=gJzqCaOEqsA (Example from 1:42:41)

    Args:


    Returns:
        NDArray: the 8 x 8 element stiffness matrix
        Symbol: The symbol being used for element thickness
        Symbol: The symbol being used for Young's modulus
        Symbol: The symbol being used for Poisson's ratio
        Symbol: The symbol being used for element size in x direction
        Symbol: The symbol being used for element size in y direction
    """

    # For regular quadrilateral element, every is the same.
    # As such, we can calculate the stiffness matrix for the first element and reuse.
    # The element goes from 0,0 to a,b where a and b are the element sizes in x and y directions
    t, E, ν, a, b = sp.symbols("t E ν a b")
    node_coords = sp.Matrix(
        [
            [0, 0],
            [a, 0],
            [a, b],
            [0, b],
        ]
    )

    # Constitutive law - Plain Strain
    # C = (
    #     E
    #     / ((1 + ν) * (1 - 2 * ν))
    #     * np.array(
    #         [
    #             [1 - ν, ν, 0],
    #             [ν, 1 - ν, 0],
    #             [0, 0, (1 - 2 * ν) / 2],
    #         ]
    #     )
    # )
    # Constitutive law - Plain Stress
    C = (
        E
        / (1 + ν)
        * np.array(
            [
                [1 / (1 - ν), ν / (1 - ν), 0],
                [ν / (1 - ν), 1 / (1 - ν), 0],
                [0, 0, 1 / 2],
            ]
        )
    )

    # Shape Functions - Linear Quadrilateral Element
    ξ, η = sp.symbols("ξ η")
    N1 = (1 - η) * (1 - ξ) / 4
    N2 = (1 - η) * (1 + ξ) / 4
    N3 = (1 + η) * (1 + ξ) / 4
    N4 = (1 + η) * (1 - ξ) / 4
    N = sp.Matrix([N1, N2, N3, N4])

    # Coordinates map
    x = sum([N[i] * node_coords[i, 0] for i in range(4)])
    y = sum([N[i] * node_coords[i, 1] for i in range(4)])

    # Jacobian
    J = sp.Matrix(
        [
            [x.diff(ξ), x.diff(η)],
            [y.diff(ξ), y.diff(η)],
        ]
    )

    # Strains  (B matrix)
    # ε (strain) = B * u_e (element displacement) where u_e = [u1, v1, u2, v2, u3, v3, u4, v4].T
    # u1, v1, u2, v2, u3, v3, u4, v4 are the displacements at the 4 nodes in local ξ and η directions

    # B1 is from ε = [du/dx, dv/dy, du/dy + dv/dx].T -> B1 * [du/dx, du/dy, dv/dx, dv/dy].T
    B1 = sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
        ]
    )

    # u and v are in terms of ξ and η rather than x and y so replace
    # [du/dx, dv/dy, du/dy + dv/dx].T with B2 * [du/dξ, dv/dξ, du/dη, dv/dη].T
    # via multivariable chain rule
    J_inv = J.inv()
    B2 = sp.Matrix(
        [
            [J_inv[0, 0], J_inv[0, 1], 0, 0],
            [J_inv[1, 0], J_inv[1, 1], 0, 0],
            [0, 0, J_inv[0, 0], J_inv[0, 1]],
            [0, 0, J_inv[1, 0], J_inv[1, 1]],
        ]
    )
    # Finally replace [du/dξ, dv/dξ, du/dη, dv/dη].T with B3 * [u1, v1, u2, v2, u3, v3, u4, v4].T
    # to get strain in terms of nodal displacements

    # 4 x 8 matrix for 4 node element (linear)
    # 4 x 16 for 8 node element (quadratic)
    B3 = sp.Matrix(
        [
            [N1.diff(ξ), 0, N2.diff(ξ), 0, N3.diff(ξ), 0, N4.diff(ξ), 0],
            [N1.diff(η), 0, N2.diff(η), 0, N3.diff(η), 0, N4.diff(η), 0],
            [0, N1.diff(ξ), 0, N2.diff(ξ), 0, N3.diff(ξ), 0, N4.diff(ξ)],
            [0, N1.diff(η), 0, N2.diff(η), 0, N3.diff(η), 0, N4.diff(η)],
        ]
    )
    # pprint(B3)
    B = B1 * B2 * B3
    # print("B")
    # pprint(B)

    # Stiffness Matrix
    # K_e = ∫(B^T * C * B) * det(J) dξ dη
    integrand = B.T * C * B * sp.det(J)

    # Numerical Integration - very slow.
    # K_e = sp.integrate(integrand, (ξ, -1, 1), (η, -1, 1))
    # print("K_e")
    # # pprint(K_e)
    # pprint(K_e.subs({t: 1, E: 1, ν: 0.2}))

    # Integration by Gaussian Quadrature 3rd order - fast
    gauss_integral_points = [1 / sp.sqrt(3), -1 / sp.sqrt(3)]
    # Both Gauss weights are 1

    K_e = sp.ZeroMatrix(8, 8)
    for ξ_ in gauss_integral_points:
        for η_ in gauss_integral_points:
            K_e += integrand.subs({ξ: ξ_, η: η_})
    K_e = K_e * t
    print("K_e")
    # pprint(sp.simplify(K_e / E / t * (1 - ν**2)))

    # TODO: This produces a matrix of matrices rather than a list that then gets summed
    # K_e: sp.MatrixBase = t * sum(
    #     [
    #         integrand.subs({ξ: ξ_, η: η_})
    #         for ξ_ in gauss_integral_points
    #         for η_ in gauss_integral_points
    #     ]
    # )

    # Substitute values
    # K_e_np: NDArray = np.array(
    #     sp.N(
    #         K_e.subs(
    #             {t: t_val, E: E_val, ν: ν_val, a: element_size[0], b: element_size[1]}
    #         )
    #     )
    # ).astype(np.float64)
    return K_e, t, E, ν, a, b


def print_for_hard_coding(
    K_e: sp.MatrixBase, E: sp.Symbol, ν: sp.Symbol, t: sp.Symbol
) -> list[str]:
    """
    Rather than doing the derivation for each run of Movable Morphable Components,
    it can be advantageous to hard code the stiffness matrix as per the 218 line MMC-218 code.

    This function prints the stiffness matrix in a format that can be easily copied into the code.

    Args:
        K_e (sp.MatrixBase): The stiffness matrix
        E (sp.Symbol): The symbol being used for Young's modulus
        t (sp.Symbol): The symbol being used for element thickness
    """
    K_e_simp = K_e / E / t * (1 - ν**2)  # Same denominator as the 218 line code

    # Want to only hard-code equations once
    # Initialise a register of unique elements
    unique_elements = {}
    # Stiffness matrix is 8 x 8
    assert K_e_simp.shape == (8, 8), "Stiffness matrix is not 8 x 8"
    output: list[str] = []
    for i in range(8):
        for j in range(8):
            # Only consider the upper triangular part
            if i > j:
                continue

            # Simplify and factor the expression
            val = K_e_simp[i, j].simplify().factor()
            unique = True
            # Check if the expression is already in the register
            # See SymPy suggestions for comparing expressions
            for exp, index in unique_elements.items():
                if val - exp == 0:
                    output.append(f"K_e[{i+1}, {j+1}] = {index}")
                    unique = False
                # Negative versions of expressions are also considered
                if val + exp == 0:
                    output.append(f"K_e[{i+1}, {j+1}] = -{index}")
                    unique = False
            # If unique, add to the register
            if unique:
                unique_elements[val] = f"k_{i+1}_{j+1}"
                output.append(f"K_e[{i+1}, {j+1}] = {sp.N(val,3)}")
    return output


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    k_e, t, E, ν, a, b = derive_stiffness_matrix()
    print("\n".join(print_for_hard_coding(k_e, E, ν, t)))
    print(
        np.array(sp.N(k_e.subs({E: 1.0, ν: 0.3, a: 1.0, b: 1.0, t: 1.0}))).astype(
            np.float64
        )
    )
