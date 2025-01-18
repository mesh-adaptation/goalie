"""
Sequences of meshes corresponding to a :class:`~.TimePartition`.
"""

from collections.abc import Iterable

import firedrake
import numpy as np
from animate.quality import QualityMeasure
from animate.utility import Mesh
from firedrake.petsc import PETSc
from firedrake.pyplot import triplot

from .log import DEBUG, debug, info, logger, pyrint, warning

__all__ = ["MeshSeq"]


class MeshSeq:
    """
    A sequence of meshes for solving a PDE associated with a particular
    :class:`~.TimePartition` of the temporal domain.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, initial_meshes, **kwargs):
        r"""
        :arg initial_meshes: a list of meshes corresponding to the subinterval of the
            time partition, or a single mesh to use for all subintervals
        :type initial_meshes: :class:`list` or :class:`~.MeshGeometry`
        """
        self.set_meshes(initial_meshes)
        self.sections = [{} for mesh in self]

    def __str__(self):
        return f"{[str(mesh) for mesh in self.meshes]}"

    def __repr__(self):
        name = type(self).__name__
        if len(self) == 1:
            return f"{name}([{repr(self.meshes[0])}])"
        elif len(self) == 2:
            return f"{name}([{repr(self.meshes[0])}, {repr(self.meshes[1])}])"
        else:
            return f"{name}([{repr(self.meshes[0])}, ..., {repr(self.meshes[-1])}])"

    def debug(self, msg):
        """
        Print a ``debug`` message.

        :arg msg: the message to print
        :type msg: :class:`str`
        """
        debug(f"{type(self).__name__}: {msg}")

    def warning(self, msg):
        """
        Print a ``warning`` message.

        :arg msg: the message to print
        :type msg: :class:`str`
        """
        warning(f"{type(self).__name__}: {msg}")

    def info(self, msg):
        """
        Print an ``info`` level message.

        :arg msg: the message to print
        :type msg: :class:`str`
        """
        info(f"{type(self).__name__}: {msg}")

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, subinterval):
        """
        :arg subinterval: a subinterval index
        :type subinterval: :class:`int`
        :returns: the corresponding mesh
        :rtype: :class:`firedrake.MeshGeometry`
        """
        return self.meshes[subinterval]

    def __setitem__(self, subinterval, mesh):
        """
        :arg subinterval: a subinterval index
        :type subinterval: :class:`int`
        :arg mesh: the mesh to use for that subinterval
        :type subinterval: :class:`firedrake.MeshGeometry`
        """
        self.meshes[subinterval] = mesh

    def count_elements(self):
        r"""
        Count the number of elements in each mesh in the sequence.

        :returns: list of element counts
        :rtype: :class:`list` of :class:`int`\s
        """
        comm = firedrake.COMM_WORLD
        return [comm.allreduce(mesh.coordinates.cell_set.size) for mesh in self]

    def count_vertices(self):
        r"""
        Count the number of vertices in each mesh in the sequence.

        :returns: list of vertex counts
        :rtype: :class:`list` of :class:`int`\s
        """
        comm = firedrake.COMM_WORLD
        return [comm.allreduce(mesh.coordinates.node_set.size) for mesh in self]

    def _reset_counts(self):
        """
        Reset the lists of element and vertex counts.
        """
        self.element_counts = [self.count_elements()]
        self.vertex_counts = [self.count_vertices()]

    def set_meshes(self, meshes):
        r"""
        Set all meshes in the sequence and deduce various properties.

        :arg meshes: list of meshes to use in the sequence, or a single mesh to use for
            all subintervals
        :type meshes: :class:`list` of :class:`firedrake.MeshGeometry`\s or
            :class:`firedrake.MeshGeometry`
        """
        # TODO #122: Refactor to use the set method
        if not isinstance(meshes, Iterable):
            meshes = [Mesh(meshes) for subinterval in self.subintervals]  # FIXME
        self.meshes = meshes
        dim = np.array([mesh.topological_dimension() for mesh in meshes])
        if dim.min() != dim.max():
            raise ValueError("Meshes must all have the same topological dimension.")
        self.dim = dim.min()
        self._reset_counts()
        if logger.level == DEBUG:
            for i, mesh in enumerate(meshes):
                nc = self.element_counts[0][i]
                nv = self.vertex_counts[0][i]
                qm = QualityMeasure(mesh)
                ar = qm("aspect_ratio")
                mar = ar.vector().gather().max()
                self.debug(
                    f"{i}: {nc:7d} cells, {nv:7d} vertices,  max aspect ratio {mar:.2f}"
                )
            debug(100 * "-")

    def plot(self, fig=None, axes=None, **kwargs):
        """
        Plot the meshes comprising a 2D :class:`~.MeshSeq`.

        :kwarg fig: matplotlib figure to use
        :type fig: :class:`matplotlib.figure.Figure`
        :kwarg axes: matplotlib axes to use
        :type axes: :class:`matplotlib.axes._axes.Axes`
        :returns: matplotlib figure and axes for the plots
        :rtype1: :class:`matplotlib.figure.Figure`
        :rtype2: :class:`matplotlib.axes._axes.Axes`

        All keyword arguments are passed to :func:`firedrake.pyplot.triplot`.
        """
        from matplotlib.pyplot import subplots

        if self.dim != 2:
            raise ValueError("MeshSeq plotting only supported in 2D.")

        # Process kwargs
        interior_kw = {"edgecolor": "k"}
        interior_kw.update(kwargs.pop("interior_kw", {}))
        boundary_kw = {"edgecolor": "k"}
        boundary_kw.update(kwargs.pop("boundary_kw", {}))
        kwargs["interior_kw"] = interior_kw
        kwargs["boundary_kw"] = boundary_kw
        if fig is None or axes is None:
            n = len(self)
            fig, axes = subplots(ncols=n, nrows=1, figsize=(5 * n, 5))

        # Loop over all axes and plot the meshes
        k = 0
        if not isinstance(axes, Iterable):
            axes = [axes]
        for i, axis in enumerate(axes):
            if not isinstance(axis, Iterable):
                axis = [axis]
            for ax in axis:
                ax.set_title(f"MeshSeq[{k}]")
                triplot(self.meshes[k], axes=ax, **kwargs)
                ax.axis(False)
                k += 1
            if len(axis) == 1:
                axes[i] = axis[0]
        if len(axes) == 1:
            axes = axes[0]
        return fig, axes

    def check_element_count_convergence(self):
        r"""
        Check for convergence of the fixed point iteration due to the relative
        difference in element count being smaller than the specified tolerance.

        :return: an array, whose entries are ``True`` if convergence is detected on the
            corresponding subinterval
        :rtype: :class:`list` of :class:`bool`\s
        """
        if self.params.drop_out_converged:
            converged = self.converged
        else:
            converged = np.array([False] * len(self), dtype=bool)
        if len(self.element_counts) >= max(2, self.params.miniter + 1):
            for i, (ne_, ne) in enumerate(zip(*self.element_counts[-2:])):
                if not self.check_convergence[i]:
                    self.info(
                        f"Skipping element count convergence check on subinterval {i})"
                        f" because check_convergence[{i}] == False."
                    )
                    continue
                if abs(ne - ne_) <= self.params.element_rtol * ne_:
                    converged[i] = True
                    if len(self) == 1:
                        pyrint(
                            f"Element count converged after {self.fp_iteration+1}"
                            " iterations under relative tolerance"
                            f" {self.params.element_rtol}."
                        )
                    else:
                        pyrint(
                            f"Element count converged on subinterval {i} after"
                            f" {self.fp_iteration+1} iterations under relative"
                            f" tolerance {self.params.element_rtol}."
                        )

        # Check only early subintervals are marked as converged
        if self.params.drop_out_converged and not converged.all():
            first_not_converged = converged.argsort()[0]
            converged[first_not_converged:] = False

        return converged
