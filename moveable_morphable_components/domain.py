import itertools
from typing import Generator

import numpy as np
from numpy.typing import NDArray


class Domain2D:
    """
    Represents the 2D domain for the problem. The domain is a rectangle with dimensions [x, y].
    Quadrilateral bi-linear elements are used to discretise the domain.

    Attributes:
        dimensions: tuple[float, float] - The dimensions of the domain [x, y]
        element_shape: tuple[int, int] - The number of elements in each direction [x, y]
        node_shape: tuple[int, int] - The number of nodes in each direction [x, y]
        ν: float - Poisson's ratio - used to calculate the stiffness matrix
        num_dofs: int - The number of degrees of freedom
        element_size: NDArray - The size of each element [x, y]
    """

    def __init__(
        self,
        dimensions: tuple[float, float],
        element_shape: tuple[int, int],
        E: float = 1.0,
        ν: float = 0.3,
        t: float = 1.0,
    ) -> None:
        self.dimensions: NDArray = np.array(dimensions, dtype=np.float32)

        self.element_shape: NDArray = np.array(element_shape, dtype=np.int32)
        self.num_elements: int = np.prod(self.element_shape)
        self.element_size: NDArray = dimensions / self.element_shape

        self.node_shape: NDArray = self.element_shape + 1
        self.num_nodes: int = np.prod(self.node_shape)

        self.num_dofs: int = 2 * np.prod(self.num_nodes)

        # Make a map of element ID to element x, y index
        element_ids: NDArray = np.arange(self.num_elements)
        self.element_multi_index = np.array(
            np.unravel_index(element_ids, self.element_shape, order="F")
        ).T

        # Make a map of node ID to node x, y index
        # num_elements x 4 nodes per element x 2 coordinates per node
        self.element_node_xys = np.array(
            [
                self.element_multi_index,
                self.element_multi_index + [1, 0],
                self.element_multi_index + [1, 1],
                self.element_multi_index + [0, 1],
            ]
        ).swapaxes(0, 1)

        # The global node ids for each element
        # Start in top left, sweep along x direction, then down a row.
        # Nodes are numbered starting in the bottom left corner and moving anti-clockwise
        # 4 nodes per element. Adjacent elements share nodes.
        #
        #    8   9  10  11
        #    *---*---*---*
        #    | 3 | 4 | 5 |
        #  4 *---*---*---* 7
        #    | 0 | 1 | 2 |
        #    *---*---*---*
        #    0   1   2   3

        # Reshape into an ordered list of nodes, convert xy position to node id then reshape back
        # num_elements x 4 nodes ids per element
        self.element_node_ids = np.ravel_multi_index(
            self.element_node_xys.reshape(-1, 2).T,
            self.node_shape,
            order="F",
        ).reshape(-1, 4)

        # For each element, the id of the degrees of freedom
        # bottom left clockwise going clockwise, x, y for each nodes
        self.element_dof_ids = np.moveaxis(
            np.array(
                [
                    2 * self.element_node_ids,
                    2 * self.element_node_ids + 1,
                ]
            ),
            0,
            -1,
        ).reshape(-1, 8)

        self.node_volumes = (
            self.element_value_to_nodes(np.ones(self.element_shape)).flatten(order="F")
            * np.prod(self.element_size)
            / np.prod(self.dimensions)
        )

    # TODO currently (y, x) as the way product generates it.
    @property
    def node_coordinates(self) -> Generator[tuple[float, float], None, None]:
        return itertools.product(
            np.linspace(0, self.dimensions[1], self.node_shape[1]),
            np.linspace(0, self.dimensions[0], self.node_shape[0]),
        )

    def map_node_xy_to_id(self, point: tuple[int, int]) -> int:
        return np.ravel_multi_index(point, self.node_shape, order="F")

    def select_node_with_point(self, point: tuple[float, float]) -> NDArray:
        """
        Selects the node corresponding to the point in the domain

        Parameters:
            point: tuple[float, float] - The point in the domain in dimension units [x, y]

        Returns:
            NDArray - The node corresponding to the point
        """
        # Find the closes x, y node index by dividing by the element size and rounding
        node_xy = np.rint(point / self.element_size).astype(np.int32)
        return self.map_node_xy_to_id(node_xy)

    def select_dofs_with_point(self, point: tuple[float, float]) -> NDArray:
        """
        Selects the degrees of freedom corresponding to the point in the domain

        Parameters:
            point: tuple[float, float] - The point in the domain in dimension units [x, y]

        Returns:
            NDArray - The degrees of freedom corresponding to the point
        """
        assert (
            point[0] <= self.dimensions[0] and point[1] <= self.dimensions[1]
        ), "Point is outside the domain"
        assert point[0] >= 0 and point[1] >= 0, "Point is outside the domain"
        selected_node_id = self.select_node_with_point(point)
        return np.array([selected_node_id * 2, selected_node_id * 2 + 1])

    def select_dofs_on_left_boundary(self):
        node_ids: NDArray = np.arange(self.num_nodes, step=self.node_shape[0])
        return np.concatenate([node_ids * 2, node_ids * 2 + 1])

    def average_node_values_to_element(self, node_property: NDArray) -> NDArray:
        """
        Converts an array of values sampled at each node to an array of values within each element
        The value of each element is the average of the values at each node

        Parameters:
            node_property: NDArray - The property at each node

        Returns:
            NDArray - The property averaged to the element
        """
        # Reshape to the domain
        _node_prop = node_property.reshape(self.node_shape, order="F")

        element_densities: NDArray = np.mean(
            np.array(
                [
                    _node_prop[:-1, :-1],
                    _node_prop[1:, :-1],
                    _node_prop[:-1, 1:],
                    _node_prop[1:, 1:],
                ]
            ),
            axis=0,
        )
        return element_densities

    def element_value_to_nodes(self, element_value: NDArray) -> NDArray:
        """
        Converts an array of values sampled at each element to an array of values within each node
        The value of each node is 1/4 of the value of each element it is part of

        Parameters:
            element_value: NDArray - The property at each element

        Returns:
            NDArray - The property at each node
        """
        # Reshape to the domain
        _element_value = element_value.reshape(self.element_shape)

        node_values: NDArray = np.zeros(self.node_shape)
        node_values[:-1, :-1] += element_value / 4
        node_values[1:, :-1] += element_value / 4
        node_values[1:, 1:] += element_value / 4
        node_values[:-1, 1:] += element_value / 4
        return node_values


if __name__ == "__main__":
    d = Domain2D(dimensions=(2.0, 1.0), element_shape=(20, 10))
    a = d.select_dofs_with_point((2.0, 1.0))
    d = Domain2D(dimensions=(5.0, 1.0), element_shape=(5, 1))
    a = d.select_dofs_with_point((5.0, 0.5))
