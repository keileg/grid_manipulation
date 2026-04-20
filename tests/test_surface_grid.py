import numpy as np
import porepy as pp
import pytest

from grid_manipulation import extract_surface_grid


@pytest.mark.parametrize("nx", [(1, 1, 1), (2, 2, 2)])
def test_extract_surface_grid(nx):
    top_coord = 1
    g = pp.StructuredTetrahedralGrid(nx, [1, 1, top_coord])
    g.compute_geometry()
    top_faces = np.where(g.face_centers[2, :] == 1)[0]

    surface_grid = extract_surface_grid.extract(g, top_faces)

    num_top_faces = np.prod(nx[:2]) * 2
    assert surface_grid.num_cells == num_top_faces

    surface_grid.compute_geometry()
    np.testing.assert_allclose(g.face_centers[:, top_faces], surface_grid.cell_centers)
