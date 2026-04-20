import porepy as pp
import numpy as np
import pytest
from grid_manipulation import paste_grids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_match_nodes(seed):
    np.random.seed(seed)
    sz = 10
    n1 = np.random.rand(3, sz)
    n2 = n1[:, np.random.permutation(sz)]
    common, mapped_1, mapped_2 = paste_grids.match_nodes(n1, n2)

    np.testing.assert_allclose(n1, common[:, mapped_1])
    np.testing.assert_allclose(n2, common[:, mapped_2])


@pytest.mark.parametrize(
    "surface_coord, const",
    [
        ([1, 0, 0], 0),
        ([1, 0, 0], 1),
        ([0, 1, 0], 0),
        ([0, 1, 0], 1),
        ([0, 0, 1], 0),
        ([0, 0, 1], 1),
    ],
)
def test_find_nodes_on_surface(surface_coord, const):
    surface_coord = np.array(surface_coord)
    nx = np.array([1, 1, 1])
    g = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    g.compute_geometry()
    tol = 1e-8

    active_dim = np.where(surface_coord != 0)[0][0]

    nodes = paste_grids.find_nodes_on_surface(g, surface_coord, const, tol)

    expected_nodes = np.where(g.nodes[active_dim, :] == const)[0]
    np.testing.assert_array_equal(np.sort(nodes), np.sort(expected_nodes))
