import numpy as np
import porepy as pp

from grid_manipulation import glue_grids, extract_surface_grid


def bottom_domain():
    domain = pp.domains.nd_cube_domain(3, 1)
    fracture_network = pp.create_fracture_network(domain=domain)

    mdg = pp.create_mdg(
        "simplex", {"cell_size": 0.5}, fracture_network=fracture_network
    )
    return mdg


if __name__ == "__main__":
    mdg = bottom_domain()

    g_3d_bottom = mdg.subdomains(dim=3)[0]

    traget_faces_bottom = np.where(g_3d_bottom.face_centers[2, :] == 1)[0]
    g_2d_top = extract_surface_grid.extract(g_3d_bottom, traget_faces_bottom)

    g_2d_top.compute_geometry()

    print(g_2d_top)

    z_layers = g_2d_top.nodes[2, 0] + np.array([0, 0.3, 0.7, 1])

    g_3d_top, *_ = pp.grid_extrusion.extrude_grid(g_2d_top, z_layers)

    plane_coefficients = np.array([0, 0, 1]).reshape((3, 1))

    g = glue_grids.paste_3d_simplex_grids(
        g_3d_bottom, g_3d_top, plane_coefficients=plane_coefficients, offset=1
    )
    g.compute_geometry()

    pp.plot_grid(g)

    debug = []
