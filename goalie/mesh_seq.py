"""
Sequences of meshes corresponding to a :class:`~.TimePartition`.
"""

from collections.abc import Iterable

import firedrake
import numpy as np
from animate.interpolation import transfer
from animate.quality import QualityMeasure
from animate.utility import Mesh
from firedrake.adjoint import pyadjoint
from firedrake.petsc import PETSc
from firedrake.pyplot import triplot

from .function_data import ForwardSolutionData
from .log import DEBUG, debug, info, logger, pyrint, warning
from .options import AdaptParameters
from .utility import AttrDict

__all__ = ["MeshSeq"]


class MeshSeq:
    """
    A sequence of meshes for solving a PDE associated with a particular
    :class:`~.TimePartition` of the temporal domain.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, time_partition, initial_meshes, **kwargs):
        r"""
        :arg time_partition: a partition of the temporal domain
        :type time_partition: :class:`~.TimePartition`
        :arg initial_meshes: a list of meshes corresponding to the subinterval of the
            time partition, or a single mesh to use for all subintervals
        :type initial_meshes: :class:`list` or :class:`~.MeshGeometry`
        :kwarg get_function_spaces: a function as described in
            :meth:`~.MeshSeq.get_function_spaces`
        :kwarg get_initial_condition: a function as described in
            :meth:`~.MeshSeq.get_initial_condition`
        :kwarg get_form: a function as described in :meth:`~.MeshSeq.get_form`
        :kwarg get_solver: a function as described in :meth:`~.MeshSeq.get_solver`
        :kwarg transfer_method: the method to use for transferring fields between
            meshes. Options are "project" (default) and "interpolate". See
            :func:`animate.interpolation.transfer` for details
        :type transfer_method: :class:`str`
        :kwarg transfer_kwargs: kwargs to pass to the chosen transfer method
        :type transfer_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg parameters: parameters to apply to the mesh adaptation process
        :type parameters: :class:`~.AdaptParameters`
        """
        self.time_partition = time_partition
        self.fields = {field_name: None for field_name in time_partition.field_names}
        self.field_types = {
            field: field_type
            for field, field_type in zip(self.fields, time_partition.field_types)
        }
        self.subintervals = time_partition.subintervals
        self.num_subintervals = time_partition.num_subintervals
        self.set_meshes(initial_meshes)
        self._fs = None
        self._get_function_spaces = kwargs.get("get_function_spaces")
        self._get_initial_condition = kwargs.get("get_initial_condition")
        self._get_form = kwargs.get("get_form")
        self._get_solver = kwargs.get("get_solver")
        self._transfer_method = kwargs.get("transfer_method", "project")
        self._transfer_kwargs = kwargs.get("transfer_kwargs", {})
        self.params = kwargs.get("parameters")
        self.steady = time_partition.steady
        self.check_convergence = np.array([True] * len(self), dtype=bool)
        self.converged = np.array([False] * len(self), dtype=bool)
        self.fp_iteration = 0
        if self.params is None:
            self.params = AdaptParameters()
        self.sections = [{} for mesh in self]

        self._outputs_consistent()

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
            meshes = [Mesh(meshes) for subinterval in self.subintervals]
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
            raise ValueError("MeshSeq plotting only supported in 2D")

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

    def get_function_spaces(self, mesh):
        """
        Construct the function spaces corresponding to each field, for a given mesh.

        :arg mesh: the mesh to base the function spaces on
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :returns: a dictionary whose keys are field names and whose values are the
            corresponding function spaces
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.functionspaceimpl.FunctionSpace` values
        """
        if self._get_function_spaces is None:
            raise NotImplementedError("'get_function_spaces' needs implementing.")
        return self._get_function_spaces(mesh)

    def get_initial_condition(self):
        r"""
        Get the initial conditions applied on the first mesh in the sequence.

        :returns: the dictionary, whose keys are field names and whose values are the
            corresponding initial conditions applied
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        if self._get_initial_condition is not None:
            return self._get_initial_condition(self)
        return {
            field: firedrake.Function(fs[0])
            for field, fs in self.function_spaces.items()
        }

    def get_form(self):
        """
        Get the function mapping a subinterval index and a solution dictionary to a
        dictionary containing parts of the PDE weak form corresponding to each solution
        component.

        Signature for the function to be returned:
        ```
        :arg index: the subinterval index
        :type index: :class:`int`
        :arg solutions: map from fields to tuples of current and previous solutions
        :type solutions: :class:`dict` with :class:`str` keys and :class:`tuple` values
        :return: map from fields to the corresponding forms
        :rtype: :class:`dict` with :class:`str` keys and :class:`ufl.form.Form` values
        ```

        :returns: the function for obtaining the form
        :rtype: see docstring above
        """
        if self._get_form is None:
            raise NotImplementedError("'get_form' needs implementing.")
        return self._get_form(self)

    def get_solver(self):
        """
        Get the function mapping a subinterval index and an initial condition dictionary
        to a dictionary of solutions for the corresponding solver step.

        Signature for the function to be returned:
        ```
        :arg index: the subinterval index
        :type index: :class:`int`
        :arg ic: map from fields to the corresponding initial condition components
        :type ic: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        :return: map from fields to the corresponding solutions
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        ```

        :returns: the function for obtaining the solver
        :rtype: see docstring above
        """
        if self._get_solver is None:
            raise NotImplementedError("'get_solver' needs implementing.")
        return self._get_solver(self)

    def _transfer(self, source, target_space, **kwargs):
        """
        Transfer a field between meshes using the specified transfer method.

        :arg source: the function to be transferred
        :type source: :class:`firedrake.function.Function` or
            :class:`firedrake.cofunction.Cofunction`
        :arg target_space: the function space which we seek to transfer onto, or the
            function or cofunction to use as the target
        :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`,
            :class:`firedrake.function.Function`
            or :class:`firedrake.cofunction.Cofunction`
        :returns: the transferred function
        :rtype: :class:`firedrake.function.Function` or
            :class:`firedrake.cofunction.Cofunction`

        Extra keyword arguments are passed to :func:`goalie.interpolation.transfer`.
        """
        # Update kwargs with those specified by the user
        transfer_kwargs = kwargs.copy()
        transfer_kwargs.update(self._transfer_kwargs)
        return transfer(source, target_space, self._transfer_method, **transfer_kwargs)

    def _outputs_consistent(self):
        """
        Assert that function spaces, initial conditions, and forms are given in a
        dictionary format with :attr:`MeshSeq.fields` as keys.
        """
        for method in ["function_spaces", "initial_condition", "form", "solver"]:
            if getattr(self, f"_get_{method}") is None:
                continue
            method_map = getattr(self, f"get_{method}")
            if method == "function_spaces":
                method_map = method_map(self.meshes[0])
            elif method == "initial_condition":
                method_map = method_map()
            elif method == "form":
                self._reinitialise_fields(self.get_initial_condition())
                method_map = method_map()(0)
            elif method == "solver":
                self._reinitialise_fields(self.get_initial_condition())
                solver_gen = method_map()(0)
                assert hasattr(solver_gen, "__next__"), "solver should yield"
                if logger.level == DEBUG:
                    next(solver_gen)
                    f, f_ = self.fields[next(iter(self.fields))]
                    if np.array_equal(f.vector().array(), f_.vector().array()):
                        self.debug(
                            "Current and lagged solutions are equal. Does the"
                            " solver yield before updating lagged solutions?"
                        )
                break
            assert isinstance(method_map, dict), f"get_{method} should return a dict"
            mesh_seq_fields = set(self.fields)
            method_fields = set(method_map.keys())
            diff = mesh_seq_fields.difference(method_fields)
            assert len(diff) == 0, f"missing fields {diff} in get_{method}"
            diff = method_fields.difference(mesh_seq_fields)
            assert len(diff) == 0, f"unexpected fields {diff} in get_{method}"

    def _function_spaces_consistent(self):
        """
        Determine whether the mesh sequence's function spaces are consistent with its
        meshes.

        :returns: ``True`` if the meshes and function spaces are consistent, otherwise
            ``False``
        :rtype: `:class:`bool`
        """
        consistent = len(self.time_partition) == len(self)
        consistent &= all(len(self) == len(self._fs[field]) for field in self.fields)
        for field in self.fields:
            consistent &= all(
                mesh == fs.mesh() for mesh, fs in zip(self.meshes, self._fs[field])
            )
            consistent &= all(
                self._fs[field][0].ufl_element() == fs.ufl_element()
                for fs in self._fs[field]
            )
        return consistent

    def _update_function_spaces(self):
        """
        Update the function space dictionary associated with the mesh sequence.
        """
        if self._fs is None or not self._function_spaces_consistent():
            self._fs = AttrDict(
                {
                    field: [self.get_function_spaces(mesh)[field] for mesh in self]
                    for field in self.fields
                }
            )
        assert (
            self._function_spaces_consistent()
        ), "Meshes and function spaces are inconsistent"

    @property
    def function_spaces(self):
        """
        Get the function spaces associated with the mesh sequence.

        :returns: a dictionary whose keys are field names and whose values are the
            corresponding function spaces
        :rtype: :class:`~.AttrDict` with :class:`str` keys and
            :class:`firedrake.functionspaceimpl.FunctionSpace` values
        """
        self._update_function_spaces()
        return self._fs

    @property
    def initial_condition(self):
        """
        Get the initial conditions associated with the first subinterval.

        :returns: a dictionary whose keys are field names and whose values are the
            corresponding initial conditions applied on the first subinterval
        :rtype: :class:`~.AttrDict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        return AttrDict(self.get_initial_condition())

    @property
    def form(self):
        """
        See :meth:`~.MeshSeq.get_form`.
        """
        return self.get_form()

    @property
    def solver(self):
        """
        See :meth:`~.MeshSeq.get_solver`.
        """
        return self.get_solver()

    def _create_solutions(self):
        """
        Create the :class:`~.FunctionData` instance for holding solution data.
        """
        self._solutions = ForwardSolutionData(self.time_partition, self.function_spaces)

    @property
    def solutions(self):
        """
        :returns: the solution data object
        :rtype: :class:`~.FunctionData`
        """
        if not hasattr(self, "_solutions"):
            self._create_solutions()
        return self._solutions

    def _reinitialise_fields(self, initial_conditions):
        """
        Reinitialise fields and assign initial conditions on the given subinterval.

        :arg initial_conditions: the initial conditions to assign to lagged solutions
        :type initial_conditions: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        for field, ic in initial_conditions.items():
            fs = ic.function_space()
            if self.field_types[field] == "steady":
                self.fields[field] = firedrake.Function(fs, name=f"{field}").assign(ic)
            else:
                self.fields[field] = (
                    firedrake.Function(fs, name=field),
                    firedrake.Function(fs, name=f"{field}_old").assign(ic),
                )

    @PETSc.Log.EventDecorator()
    def _solve_forward(self, update_solutions=True, solver_kwargs=None):
        r"""
        Solve a forward problem on a sequence of subintervals. Yields the final solution
        on each subinterval.

        :kwarg update_solutions: if ``True``, updates the solution data
        :type update_solutions: :class:`bool`
        :kwarg solver_kwargs: parameters for the forward solver
        :type solver_kwargs: :class:`dict` whose keys are :class:`str`\s and whose values
            may take various types
        :yields: the solution data of the forward solves
        :ytype: :class:`~.ForwardSolutionData`
        """
        solver_kwargs = solver_kwargs or {}
        num_subintervals = len(self)
        tp = self.time_partition

        if update_solutions:
            # Reinitialise the solution data object
            self._create_solutions()
            solutions = self.solutions.extract(layout="field")

        # Stop annotating
        if pyadjoint.annotate_tape():
            tape = pyadjoint.get_working_tape()
            if tape is not None:
                tape.clear_tape()
            pyadjoint.pause_annotation()

        # Loop over the subintervals
        checkpoint = self.initial_condition
        for i in range(num_subintervals):
            solver_gen = self.solver(i, **solver_kwargs)

            # Reinitialise fields and assign initial conditions
            self._reinitialise_fields(checkpoint)

            if update_solutions:
                # Solve sequentially between each export time
                for j in range(tp.num_exports_per_subinterval[i] - 1):
                    for _ in range(tp.num_timesteps_per_export[i]):
                        next(solver_gen)
                    # Update the solution data
                    for field, sol in self.fields.items():
                        if not self.steady:
                            assert isinstance(sol, tuple)
                            solutions[field].forward[i][j].assign(sol[0])
                            solutions[field].forward_old[i][j].assign(sol[1])
                        else:
                            assert isinstance(sol, firedrake.Function)
                            solutions[field].forward[i][j].assign(sol)
            else:
                # Solve over the entire subinterval in one go
                for _ in range(tp.num_timesteps_per_subinterval[i]):
                    next(solver_gen)

            # Transfer the checkpoint to the next subintervals
            if i < num_subintervals - 1:
                checkpoint = AttrDict(
                    {
                        field: self._transfer(
                            self.fields[field]
                            if self.field_types[field] == "steady"
                            else self.fields[field][0],
                            fs[i + 1],
                        )
                        for field, fs in self._fs.items()
                    }
                )

            yield checkpoint

    @PETSc.Log.EventDecorator()
    def get_checkpoints(self, run_final_subinterval=False, solver_kwargs=None):
        r"""
        Get checkpoints corresponding to the starting fields on each subinterval.

        :kwarg run_final_subinterval: if ``True``, the solver is run on the final
            subinterval
        :type run_final_subinterval: :class:`bool`
        :kwarg solver_kwargs: parameters for the forward solver
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :returns: checkpoints for each subinterval
        :rtype: :class:`list` of :class:`firedrake.function.Function`\s
        """
        solver_kwargs = solver_kwargs or {}
        N = len(self)

        # The first checkpoint is the initial condition
        checkpoints = [self.initial_condition]

        # If there is only one subinterval then we are done
        if N == 1 and not run_final_subinterval:
            return checkpoints

        # Otherwise, solve each subsequent subinterval and append the checkpoint
        solver_gen = self._solve_forward(
            update_solutions=False, solver_kwargs=solver_kwargs
        )
        for _ in range(N if run_final_subinterval else N - 1):
            checkpoints.append(next(solver_gen))

        return checkpoints

    @PETSc.Log.EventDecorator()
    def solve_forward(self, solver_kwargs=None):
        r"""
        Solve a forward problem on a sequence of subintervals.

        A dictionary of solution fields is computed - see :class:`~.ForwardSolutionData`
        for more details.

        :kwarg solver_kwargs: parameters for the forward solver
        :type solver_kwargs: :class:`dict` whose keys are :class:`str`\s and whose values
            may take various types
        :returns: the solution data of the forward solves
        :rtype: :class:`~.ForwardSolutionData`
        """
        solver_kwargs = solver_kwargs or {}
        solver_gen = self._solve_forward(update_solutions=True, **solver_kwargs)
        for _ in range(len(self)):
            next(solver_gen)

        return self.solutions

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
                            f" {self.fp_iteration+1} iterations under relative tolerance"
                            f" {self.params.element_rtol}."
                        )

        # Check only early subintervals are marked as converged
        if self.params.drop_out_converged and not converged.all():
            first_not_converged = converged.argsort()[0]
            converged[first_not_converged:] = False

        return converged

    @PETSc.Log.EventDecorator()
    def fixed_point_iteration(
        self, adaptor, update_params=None, solver_kwargs=None, adaptor_kwargs=None
    ):
        r"""
        Apply mesh adaptation using a fixed point iteration loop approach.

        :arg adaptor: function for adapting the mesh sequence. Its arguments are the mesh
            sequence and the solution data object. It should return ``True`` if the
            convergence criteria checks are to be skipped for this iteration. Otherwise,
            it should return ``False``.
        :kwarg update_params: function for updating :attr:`~.MeshSeq.params` at each
            iteration. Its arguments are the parameter class and the fixed point
            iteration
        :kwarg solver_kwargs: parameters to pass to the solver
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg adaptor_kwargs: parameters to pass to the adaptor
        :type adaptor_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :returns: solution data object
        :rtype: :class:`~.ForwardSolutionData`
        """
        # TODO #124: adaptor no longer needs solution data to be passed explicitly
        solver_kwargs = solver_kwargs or {}
        adaptor_kwargs = adaptor_kwargs or {}

        self._reset_counts()
        self.converged[:] = False
        self.check_convergence[:] = True

        for fp_iteration in range(self.params.maxiter):
            self.fp_iteration = fp_iteration
            if update_params is not None:
                update_params(self.params, self.fp_iteration)

            # Solve the forward problem over all meshes
            self.solve_forward(solver_kwargs=solver_kwargs)

            # Adapt meshes, logging element and vertex counts
            continue_unconditionally = adaptor(self, self.solutions, **adaptor_kwargs)
            if self.params.drop_out_converged:
                self.check_convergence[:] = np.logical_not(
                    np.logical_or(continue_unconditionally, self.converged)
                )
            self.element_counts.append(self.count_elements())
            self.vertex_counts.append(self.count_vertices())

            # Check for element count convergence
            self.converged[:] = self.check_element_count_convergence()
            if self.converged.all():
                break
        else:
            for i, conv in enumerate(self.converged):
                if not conv:
                    pyrint(
                        f"Failed to converge on subinterval {i} in"
                        f" {self.params.maxiter} iterations."
                    )

        return self.solutions
