import pyvista as pv
import numpy as np
from pathlib import Path
from numpy.typing import NDArray

class MeshProjector:
    def __init__(self, mesh_file: Path|str):
        self.mesh: pv.PolyData = pv.read(mesh_file) # type: ignore
        assert isinstance(self.mesh, pv.PolyData), "Loaded mesh is not a PolyData object."
        self.mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False)

    def project_points(self, points: NDArray) -> NDArray:
        """
        Projects 3D points onto the mesh surface.

        Args:
            points: An array of shape (N, 3) representing N 3D points.

        Returns:
            np.ndarray: An array of shape (N, 3) representing the projected points on the mesh surface.
        """
        _, closest_points = self.mesh.find_closest_cell(points, return_closest_point=True) # type: ignore
        closest_points = np.where(np.isnan(closest_points), points, closest_points)
        return closest_points

    def get_normals(self, points: NDArray) -> NDArray:
        """
        Returns the normal vectors at the closest points on the mesh surface.

        Args:
            points: An array of shape (N, 3) representing N 3D points.

        Returns:
            np.ndarray: An array of shape (N, 3) representing the normal vectors.
        """
        cell_indices = self.mesh.find_closest_cell(points)
        return self.mesh.cell_normals[cell_indices] # type: ignore