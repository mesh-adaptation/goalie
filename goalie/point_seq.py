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
        r"""
        :arg time_partition: the :class:`~.TimePartition` which partitions the temporal
            domain
        :kwarg get_function_spaces: a function, whose only argument is a
            :class:`~.MeshSeq`, which constructs prognostic
            :class:`firedrake.functionspaceimpl.FunctionSpace`\s for each subinterval
        :kwarg get_initial_condition: a function, whose only argument is a
            :class:`~.MeshSeq`, which specifies initial conditions on the first mesh
        :kwarg get_form: a function, whose only argument is a :class:`~.MeshSeq`, which
            returns a function that generates the ODE weak form
        :kwarg get_solver: a function, whose only argument is a :class:`~.MeshSeq`,
            which returns a function that integrates initial data over a subinterval
        :kwarg get_bcs: a function, whose only argument is a :class:`~.MeshSeq`, which
            returns a function that determines any Dirichlet boundary conditions
        """
        mesh = fmesh.VertexOnlyMesh(firedrake.UnitIntervalMesh(1), [[0.5]])
        super().__init__(time_partition, mesh, **kwargs)

    def set_meshes(self, mesh):
        """
        Update the mesh associated with the :class:`~.PointSeq`, as well as the
        associated attributes.

        :arg mesh: the vertex-only mesh
        """
        self.meshes = [mesh for _ in self.subintervals]
        self.dim = mesh.topological_dimension()
        assert self.dim == 0
        self._reset_counts()
