import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from domain import Domain2D


@pytest.fixture
def d_5_1():
    return Domain2D(dimensions=(5.0, 1.0), element_shape=(5, 1))


@pytest.fixture
def d_20_10():
    return Domain2D(dimensions=(2.0, 1.0), element_shape=(20, 10))


def test_elements(d_20_10):
    assert all(d_20_10.element_shape == (20, 10))
    assert d_20_10.num_elements == 200
    assert all(d_20_10.element_size == (0.1, 0.1))


def test_element_multi_index(d_20_10):
    assert all(d_20_10.element_multi_index[0] == (0, 0))
    assert all(d_20_10.element_multi_index[20] == (0, 1))
    assert all(d_20_10.element_multi_index[-1] == (19, 9))


def test_nodes(d_20_10):
    assert all(d_20_10.node_shape == (21, 11))
    assert d_20_10.num_nodes == 231
    assert d_20_10.num_dofs == 462  # 21 * 11 * 2D


def test_element_node_xys(d_20_10):
    assert_allclose(
        d_20_10.element_node_xys[0], np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    )


def test_element_node_ids(d_20_10):
    assert_allclose(d_20_10.element_node_ids[0], np.array([0, 1, 22, 21]))


def test_select_dofs_with_point(d_20_10, d_5_1):
    # The dofs at 0.0, 0.0 are 0 (x) and 1 (y)
    assert_equal(d_20_10.select_dofs_with_point((0.0, 0.0)), np.array([0, 1]))
    assert_equal(d_20_10.select_dofs_with_point((2.0, 1.0)), np.array([460, 461]))

    assert_equal(d_5_1.select_dofs_with_point((5.0, 0.5)), np.array([10, 11]))
