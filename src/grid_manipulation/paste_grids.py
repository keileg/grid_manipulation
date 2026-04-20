import numpy as np
import porepy as pp


def find_nodes_on_surface(
    g: pp.Grid, surface_coord: np.array, const: float, tol: float = 1e-8
) -> np.ndarray:
    return np.where(
        np.abs(np.sum(g.nodes * surface_coord.reshape((-1, 1)), axis=0) - const) < tol
    )[0]
