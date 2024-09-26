import numpy as np
import pytest
from domain import Domain2D
from numpy.testing import assert_allclose


@pytest.fixture
def d_5_1():
    """Create a 5 by 1 element domain fixture."""
    return Domain2D(dimensions=(5.0, 1.0), element_shape=(5, 1))


@pytest.fixture
def d_20_10():
    """Create a 20 by 10 element domain fixture."""
    return Domain2D(dimensions=(2.0, 1.0), element_shape=(20, 10))


def test_elements(d_20_10: Domain2D):
    """Test elment functions."""
    assert all(d_20_10.element_shape == (20, 10))
    assert d_20_10.num_elements == 200
    assert all(d_20_10.element_size == (0.1, 0.1))


def test_element_multi_index(d_20_10: Domain2D):
    """Test multi-intexing."""
    assert all(d_20_10.element_multi_index[0] == (0, 0))
    assert all(d_20_10.element_multi_index[20] == (0, 1))
    assert all(d_20_10.element_multi_index[-1] == (19, 9))


def test_nodes(d_20_10: Domain2D):
    """Test ndoe related function."""
    assert all(d_20_10.node_shape == (21, 11))
    assert d_20_10.num_nodes == 231
    assert d_20_10.num_dofs == 462  # 21 * 11 * 2D


def test_element_node_xys(d_20_10: Domain2D):
    """Check node coordinates are correct."""
    assert_allclose(
        d_20_10.element_node_xys[0], np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]]),
    )


def test_element_node_ids(d_20_10: Domain2D):
    """Test nodes are ids are in the correct order for FEA.

    The nodes are labelled from teh bottom-left and go clockwise.
    The local ids are always 0, 1, 2, 3. These have to be converted to global ids.
    Global ids start in the bottom-left but go row-by-row.
    """
    assert_allclose(d_20_10.element_node_ids[0], np.array([0, 1, 22, 21]))
