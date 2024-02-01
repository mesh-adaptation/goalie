import firedrake
import firedrake.mesh as fmesh
from .mesh_seq import MeshSeq

__all__ = ["PointSeq"]


class PointSeq(MeshSeq):
    """
    A simplified subset of :class:`~.MeshSeq` for ODE problems.

    In this version, a single mesh comprised of a single vertex is shared across all
    subintervals.
    """

    def __init__(self, time_partition, **kwargs):
        # TODO: docstring
        mesh = fmesh.VertexOnlyMesh(firedrake.UnitIntervalMesh(1), [[0.5]])
        super().__init__(time_partition, mesh, **kwargs)

    def set_meshes(self, meshes, copy=False):
        super().set_meshes(meshes, copy=copy)
