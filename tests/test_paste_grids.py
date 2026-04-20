import porepy as pp
import numpy as np
import pytest
from grid_manipulation import paste_grids


@pytest.mark.parametrize("nx", [np.array([1, 1, 1]), np.array([2, 2, 2])])
def test_faces_from_node_set(nx):
    g = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    node_set = np.where(g.nodes[2, :] == 1)[0]
    g.compute_geometry()
    faces = paste_grids.faces_from_node_set(g, node_set)
    np.testing.assert_array_equal(g.face_centers[2, faces], 1)
    np.testing.assert_equal(len(faces), np.prod(nx[:2]) * 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_match_nodes(seed):
    np.random.seed(seed)
    sz = 10
    n1 = np.random.rand(3, sz)
    n2 = n1[:, np.random.permutation(sz)]
    common, mapped_1, mapped_2 = paste_grids.match_nodes(n1, n2)

    np.testing.assert_allclose(n1, common[:, mapped_1])
    np.testing.assert_allclose(n2, common[:, mapped_2])


@pytest.mark.parametrize("nx", np.array([[1, 1, 1], [2, 2, 2]]))
@pytest.mark.parametrize("active_dim", [0, 1, 2])
def test_merge_node_coords(nx, active_dim):
    g1 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    g2 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    offset = 1

    g2.nodes[active_dim, :] += offset

    plane_coefficients = np.zeros(3)
    plane_coefficients[active_dim] = 1
    nodes, _, _, ind_2 = paste_grids.merge_node_coords(
        g1, g2, plane_coefficients, offset
    )

    passive_dim = np.delete(np.arange(3), active_dim)
    num_overlapping_nodes = np.prod(nx[passive_dim] + 1)
    assert nodes.shape[1] == g1.num_nodes + g2.num_nodes - num_overlapping_nodes
    np.testing.assert_allclose(nodes[:, ind_2], g2.nodes)
    np.testing.assert_allclose(nodes[:, : g1.num_nodes], g1.nodes)


@pytest.mark.parametrize("nx", np.array([[1, 1, 1], [2, 2, 2]]))
@pytest.mark.parametrize("active_dim", [0, 1, 2])
def test_merge_face_nodes(nx, active_dim):
    g1 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    g2 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    offset = 1

    g2.nodes[active_dim, :] += offset
    g1.compute_geometry()
    g2.compute_geometry()

    plane_coefficients = np.zeros(3)
    plane_coefficients[active_dim] = 1

    nx_merged = np.copy(nx)
    nx_merged[active_dim] = 2 * nx[active_dim]
    g_merged = pp.StructuredTetrahedralGrid(nx_merged, [1, 1, 1])

    faces_1 = np.where(g1.face_centers[active_dim, :] == offset)[0]
    faces_2 = np.where(g2.face_centers[active_dim, :] == offset)[0]

    node_map = np.arange(g2.num_nodes) + g1.num_nodes
    _mapping_from_coordinates(g1.nodes, g2.nodes, node_map, active_dim, offset)

    fn_merged, face_ind_2 = paste_grids.merge_face_nodes(
        g1, g2, faces_1, faces_2, node_map
    )
    assert fn_merged.shape[1] == g_merged.num_faces
    passive_dim = np.delete(np.arange(3), active_dim)
    num_overlapping_faces = np.prod(nx[passive_dim]) * 2
    assert fn_merged.shape[1] == g1.num_faces + g2.num_faces - num_overlapping_faces
    assert face_ind_2.size == g2.num_faces
    assert np.sum(face_ind_2 < g1.num_faces) == num_overlapping_faces

    for fi in range(g2.num_faces):
        if fi in faces_2:
            np.testing.assert_array_equal(offset, g2.face_centers[active_dim, fi])
        else:
            np.testing.assert_array_less(offset, g2.face_centers[active_dim, fi])


def _mapping_from_coordinates(coord_1, coord_2, mapping, active_dim, offset):
    for ni in range(coord_2.shape[1]):
        if coord_2[active_dim, ni] == offset:
            ind_in_g1 = np.where(
                np.all(coord_2[:, ni][:, None] == coord_1[:, :], axis=0)
            )[0]
            mapping[ni] = ind_in_g1


@pytest.mark.parametrize("nx", np.array([[1, 1, 1], [2, 2, 2]]))
@pytest.mark.parametrize("active_dim", [0, 1, 2])
def test_merge_cell_faces(nx, active_dim):
    g1 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    g2 = pp.StructuredTetrahedralGrid(nx, [1, 1, 1])
    offset = 1

    g2.nodes[active_dim, :] += offset
    g1.compute_geometry()
    g2.compute_geometry()

    plane_coefficients = np.zeros(3)
    plane_coefficients[active_dim] = 1

    nx_merged = np.copy(nx)
    nx_merged[active_dim] = 2 * nx[active_dim]
    g_merged = pp.StructuredTetrahedralGrid(nx_merged, [1, 1, 1])

    faces_1 = np.where(g1.face_centers[active_dim, :] == offset)[0]
    faces_2 = np.where(g2.face_centers[active_dim, :] == offset)[0]

    face_map = np.arange(g2.num_faces) + g1.num_faces
    _mapping_from_coordinates(
        g1.face_centers, g2.face_centers, face_map, active_dim, offset
    )

    cf_merged = paste_grids.merge_cell_faces(g1, g2, faces_1, faces_2, face_map)
    assert cf_merged.shape[1] == g_merged.num_cells

    assert np.unique(cf_merged).size == g_merged.num_faces


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
