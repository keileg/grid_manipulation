import numpy as np
import porepy as pp
import scipy.sparse as sps


def paste_3d_simplex_grids(
    g1: pp.Grid, g2: pp.Grid, plane_coefficients: np.array, offset: float
) -> pp.Grid:
    nodes, nodes_1, nodes_2, node_ind_2 = merge_node_coords(
        g1, g2, plane_coefficients, offset
    )
    faces_1 = faces_from_node_set(g1, nodes_1)
    faces_2 = faces_from_node_set(g2, nodes_2)

    face_node_indices, face_ind_2 = merge_face_nodes(
        g1, g2, faces_1, faces_2, node_ind_2
    )
    num_faces = face_node_indices.shape[1]
    face_nodes = sps.csc_matrix(
        (
            np.ones(face_node_indices.size, dtype=bool),
            face_node_indices.ravel(order="F"),
            np.arange(num_faces * 3 + 1, step=3),
        ),
        shape=(nodes.shape[1], num_faces),
    )

    cell_face_indices = merge_cell_faces(g1, g2, faces_1, faces_2, face_ind_2)

    return pp.Grid(
        dim=g1.dim,
        nodes=nodes,
        face_nodes=face_nodes,
        cell_faces=cell_faces,
        name="glued_grid",
    )


def merge_node_coords(g1, g2, plane_coefficients, offset):
    nodes_1 = find_nodes_on_surface(g1, plane_coefficients, offset)
    nodes_2 = find_nodes_on_surface(g2, plane_coefficients, offset)

    _, mapped_1, mapped_2 = match_nodes(g1.nodes[:, nodes_1], g2.nodes[:, nodes_2])

    assert np.allclose(mapped_1, np.arange(mapped_1.size))

    reduced_nodes_2 = np.delete(g2.nodes, nodes_2, axis=1)

    ind_2 = g1.num_nodes + np.arange(g2.nodes.shape[1])
    reduction = np.cumsum(np.isin(np.arange(g2.num_nodes), nodes_2))
    ind_2 -= reduction
    for i in range(len(mapped_2)):
        # This is a weak spot, I have not tested towards on a case where the ordering
        # of the nodes is different in the two grids. This could give rise to a mapping
        # error.
        ind_2[nodes_2[mapped_2[i]]] = nodes_1[mapped_1[i]]

    assert ind_2.size == np.unique(ind_2).size

    return np.hstack((g1.nodes, reduced_nodes_2)), nodes_1, nodes_2, ind_2


def merge_face_nodes(g1, g2, faces_1, faces_2, node_ind_2):
    face_nodes_1 = g1.face_nodes.tocsc().indices.reshape(
        (g1.dim, g1.num_faces), order="F"
    )
    face_nodes_2 = g2.face_nodes.tocsc().indices.reshape(
        (g2.dim, g2.num_faces), order="F"
    )
    face_nodes_2_mapped = node_ind_2[face_nodes_2]
    faces_to_delete = face_nodes_2_mapped[:, faces_2].copy()
    face_nodes_2_reduced = np.delete(face_nodes_2_mapped, faces_2, axis=1)

    fn_unique, mapping = np.unique(
        np.sort(np.hstack((face_nodes_1[:, faces_1], faces_to_delete)), axis=0),
        return_inverse=True,
        axis=1,
    )
    assert fn_unique.shape[1] == faces_1.size

    mapping_1 = mapping[: faces_1.size]
    mapping_2 = mapping[faces_1.size :]
    faces_1_mapped = faces_1[mapping_1]
    faces_2_mapped = faces_2[mapping_2]

    face_ind_2 = g1.num_faces + np.arange(g2.num_faces)
    reduction = np.cumsum(np.isin(np.arange(g2.num_faces), faces_2))
    face_ind_2 -= reduction
    for i in range(faces_2.size):
        face_ind_2[faces_2_mapped[i]] = faces_1_mapped[i]

    return (
        np.hstack((face_nodes_1, face_nodes_2_reduced)),
        face_ind_2,
    )


def merge_cell_faces(g1, g2, faces_1, faces_2, face_ind_2):
    cf_1 = g1.cell_faces_as_dense()
    cf_2 = g2.cell_faces_as_dense()

    num_faces = cf_1.shape[1] + cf_2.shape[1] - faces_2.size

    for f1, f2 in zip(faces_1, faces_2):
        if cf_2[0, f2] > -1:
            c2 = cf_2[0, f2]
        else:
            c2 = cf_2[1, f2]
        assert c2 >= 0
        if cf_1[0, f1] > -1:
            cf_1[1, f1] = c2 + g1.num_cells
        else:
            cf_1[0, f1] = c2 + g1.num_cells

    cf_2_reduced = np.delete(cf_2, faces_2, axis=1)
    cf_2_adjusted = cf_2_reduced.copy() + g1.num_cells

    mask = np.hstack((cf_1, cf_2_reduced)).ravel(order="F") > -1

    cf_merged = np.hstack((cf_1, cf_2_adjusted))
    row = np.repeat(np.arange(num_faces), 2)
    col = cf_merged.ravel(order="F")
    data = np.vstack((np.ones(num_faces), -np.ones(num_faces))).ravel(order="F")
    cell_faces = sps.coo_matrix(
        (data[mask], (row[mask], col[mask])),
        shape=(num_faces, g1.num_cells + g2.num_cells),
    ).tocsc()
    return cell_faces


def match_nodes(n_1, n_2):
    num_nodes = n_1.shape[1]
    assert n_2.shape[1] == num_nodes

    common, preserved, mapped = pp.array_operations.uniquify_point_set(
        np.hstack((n_1, n_2)), 1e-12
    )
    assert common.shape[1] == num_nodes
    assert preserved.size == num_nodes
    assert np.allclose(np.bincount(mapped), 2)

    return common, mapped[:num_nodes], mapped[num_nodes:]


def faces_from_node_set(g: pp.Grid, node_set: np.ndarray) -> np.ndarray:
    fn = g.face_nodes.tocsc().indices.reshape((g.dim, g.num_faces), order="F")
    return np.where(np.isin(fn, node_set).all(axis=0))[0]


def find_nodes_on_surface(
    g: pp.Grid, surface_coord: np.array, const: float, tol: float = 1e-8
) -> np.ndarray:
    return np.where(
        np.abs(np.sum(g.nodes * surface_coord.reshape((-1, 1)), axis=0) - const) < tol
    )[0]
