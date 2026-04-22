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

    face_nodes, face_ind_2 = merge_face_nodes(g1, g2, faces_1, faces_2, node_ind_2)

    cell_faces = merge_cell_faces(g1, g2, faces_1, faces_2, face_ind_2)

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
    # We can relax this assumption, but go for the simpler implementation for now.
    assert np.all(np.diff(g1.face_nodes.tocsc().indptr) == 3)
    # Stop at the last row, since we will extend this.
    ind_ptr_1 = g1.face_nodes.tocsc().indptr[:-1]

    face_nodes_1 = g1.face_nodes.tocsc().indices.reshape(
        (g1.dim, g1.num_faces), order="F"
    )
    face_inds, mapped_fn = [], []
    faces_to_delete = []
    for loc_faces, loc_nodes in _spit_face_nodes_by_num_nodes(g2.face_nodes.tocsc()):
        loc_fn_mapped = node_ind_2[loc_nodes]
        cols_to_delete = np.isin(loc_faces, faces_2)
        # This is not strictly necessary. Intertwined number of nodes per faces must
        # be handled by looping over individual faces instead of sets of faces with the
        # same number of nodes - see loop on 'outer' below.
        assert np.all(np.diff(loc_faces) == 1)

        if np.any(cols_to_delete):
            # This should only kick in for faces with 3 nodes (since we assume we are
            # merging along simplex-type faces)
            assert len(faces_to_delete) == 0
            assert loc_nodes.shape[0] == 3
            faces_to_delete.append(loc_fn_mapped[:, cols_to_delete].copy())
        face_inds.append(loc_faces[~cols_to_delete])
        mapped_fn.append(loc_fn_mapped[:, ~cols_to_delete])

    order = np.argsort([fi[0] for fi in face_inds])
    ind_ptr_2 = []
    node_inds_2 = []
    counter = ind_ptr_1[-1] + 3
    for outer in order:
        nn = mapped_fn[outer].shape[0]
        ind_ptr_2.append(counter + np.arange(0, nn * mapped_fn[outer].shape[1], nn))
        counter += nn * mapped_fn[outer].shape[1]
        node_inds_2.append(mapped_fn[outer].ravel(order="F"))

    ind_ptr_2.append(counter)

    indptr = np.hstack((ind_ptr_1, np.hstack(ind_ptr_2)))
    node_indices = np.hstack((face_nodes_1.ravel(order="F"), np.hstack(node_inds_2)))
    data = np.ones(node_indices.size, dtype=bool)

    num_faces = indptr.size - 1
    num_nodes = node_indices.max() + 1
    face_nodes = sps.csc_matrix(
        (
            data,
            node_indices.ravel(order="F"),
            indptr.ravel(order="F"),
        ),
        shape=(num_nodes, num_faces),
    )

    # Find mapping from the deleted faces to the equivalent faces in g1 (hence the full
    # grid).
    fn_unique, mapping = np.unique(
        np.sort(np.hstack((face_nodes_1[:, faces_1], faces_to_delete[0])), axis=0),
        return_inverse=True,
        axis=1,
    )
    assert fn_unique.shape[1] == faces_1.size

    face_ind_2 = _map_face_indices(g1, g2, faces_1, faces_2, node_ind_2, mapping)
    return (face_nodes, face_ind_2)


def _map_face_indices(g1, g2, faces_1, faces_2, node_ind_2, mapping):
    mapping_1 = mapping[: faces_1.size]
    mapping_2 = mapping[faces_1.size :]
    faces_1_mapped = faces_1[mapping_1]
    faces_2_mapped = faces_2[mapping_2]

    face_ind_2 = g1.num_faces + np.arange(g2.num_faces)
    reduction = np.cumsum(np.isin(np.arange(g2.num_faces), faces_2))
    face_ind_2 -= reduction
    for i in range(faces_2.size):
        face_ind_2[faces_2_mapped[i]] = faces_1_mapped[i]
    return face_ind_2


def merge_cell_faces(g1, g2, faces_1, faces_2, face_ind_2):
    cf_1 = g1.cell_faces_as_dense()
    cf_2 = g2.cell_faces_as_dense()

    num_faces = cf_1.shape[1] + cf_2.shape[1] - faces_2.size

    for f2 in faces_2:
        f1 = face_ind_2[f2]
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
    fn = g.face_nodes.tocsc()

    num_nodes_per_face = np.diff(fn.indptr)
    if np.allclose(num_nodes_per_face, 3):
        nodes = fn.indices.reshape((g.dim, g.num_faces), order="F")
        return np.where(np.isin(nodes, node_set).all(axis=0))[0]
    else:
        faces = []
        for loc_faces, loc_nodes in _spit_face_nodes_by_num_nodes(fn):
            faces.append(loc_faces[np.isin(loc_nodes, node_set).all(axis=0)])
        return np.hstack(faces)


def _spit_face_nodes_by_num_nodes(fn):
    num_nodes_per_face = np.diff(fn.indptr)
    for nn in np.unique(num_nodes_per_face):
        loc_faces = np.where(num_nodes_per_face == nn)[0]
        loc_nodes = fn[:, loc_faces].indices.reshape((nn, loc_faces.size), order="F")
        yield loc_faces, loc_nodes


def find_nodes_on_surface(
    g: pp.Grid, surface_coord: np.array, const: float, tol: float = 1e-8
) -> np.ndarray:
    return np.where(
        np.abs(np.sum(g.nodes * surface_coord.reshape((-1, 1)), axis=0) - const) < tol
    )[0]
