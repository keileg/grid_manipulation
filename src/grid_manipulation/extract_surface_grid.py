import numpy as np
import porepy as pp


def extract(g: pp.Grid, target_faces: np.ndarray) -> pp.Grid:
    sparse_fn = g.face_nodes.tocsc()
    global_fn = sparse_fn.indices.reshape((g.dim, g.num_faces), order="F")
    local_fn = global_fn[:, target_faces]
    unique_nodes, fn_unique = np.unique(local_fn, return_inverse=True)

    unique_coords = g.nodes[:, unique_nodes]

    return pp.TriangleGrid(unique_coords, fn_unique, g.dim - 1)
