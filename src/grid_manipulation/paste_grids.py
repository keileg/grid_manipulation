import numpy as np
import porepy as pp


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
