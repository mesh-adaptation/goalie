import firedrake
import numpy as np
from animate.interpolation import transfer
from firedrake.adjoint import pyadjoint
from firedrake.petsc import PETSc

from .function_data import ForwardSolutionData
from .log import DEBUG, debug, info, logger, pyrint, warning
from .options import AdaptParameters
from .utility import AttrDict

__all__ = ["Solver"]


class Solver:
    r"""
    Base class for all solvers.

    Your solver should inherit from this class and implement the following methods:

    - :meth:`get_function_spaces`
    - :meth:`get_solver`.

    Additionally, if your problem requires non-zero initial conditions (for
    time-dependent problems) or a non-zero initial guess (for time-independent
    problems), you should implement the :meth:`get_initial_condition` method.

    If your solver is solving an adjoint problem, you should implement the
    :meth:`get_qoi` method.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, time_partition, mesh_sequence, **kwargs):
        r"""
        :arg time_partition: a partition of the temporal domain
        :type time_partition: :class:`~.TimePartition`
        :arg mesh_sequence: a sequence of meshes on which to solve the problem
        :type mesh_sequence: :class:`~.MeshSeq`
        :kwarg transfer_method: the method to use for transferring fields between
            meshes. Options are "project" (default) and "interpolate". See
            :func:`animate.interpolation.transfer` for details
        :type transfer_method: :class:`str`
        :kwarg transfer_kwargs: kwargs to pass to the chosen transfer method
        :type transfer_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        """
        self.time_partition = time_partition
        self.meshes = mesh_sequence
        self._transfer_method = kwargs.get("transfer_method", "project")
        self._transfer_kwargs = kwargs.get("transfer_kwargs", {})
        self.fields = {field_name: None for field_name in time_partition.field_names}
        self.field_types = dict(zip(self.fields, time_partition.field_types))
        self.subintervals = time_partition.subintervals
        self.num_subintervals = time_partition.num_subintervals
        self._fs = None
        self.steady = time_partition.steady
        self.fp_iteration = 0
        self.params = None

        self._outputs_consistent()

    def get_function_spaces(self, *args, **kwargs):
        """
        Construct the function spaces corresponding to each field, for a given mesh.

        Should be overridden by all subclasses.

        Signature for the function:
        ```
            :arg mesh: the mesh to base the function spaces on
            :type mesh: :class:`firedrake.mesh.MeshGeometry`
            :returns: a dictionary whose keys are field names and whose values are the
                corresponding function spaces
            :rtype: :class:`dict` with :class:`str` keys and
                :class:`firedrake.functionspaceimpl.FunctionSpace` values
        ```
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_function_spaces'"
            " method."
        )

    def get_initial_condition(self, *args, **kwargs):
        """
        Get the initial conditions applied on the first mesh in the sequence.

        :returns: the dictionary, whose keys are field names and whose values are the
            corresponding initial conditions applied
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_initial_condition'"
            " method."
        )

    def get_solver(self, *args, **kwargs):
        """
        Get the function mapping a subinterval index and an initial condition dictionary
        to a dictionary of solutions for the corresponding solver step.

        Should be overridden by all subclasses.

        Signature for the function:
        ```
            :arg mesh_seq: the sequence of meshes on which to solve the problem
            :type mesh_seq: :class:`~goalie.mesh_seq.MeshSequence`
            :arg index: the subinterval index
            :type index: :class:`int`
            :arg ic: map from fields to the corresponding initial condition components
            :type ic: :class:`dict` with :class:`str` keys and
                :class:`firedrake.function.Function` values
            :return: map from fields to the corresponding solutions
            :rtype: :class:`dict` with :class:`str` keys and
                :class:`firedrake.function.Function` values
        ```
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_solver' method."
        )

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

    def _outputs_consistent(self):
        """
        Assert that function spaces and initial conditions are given in a
        dictionary format with :attr:`Solver.fields` as keys.
        """
        for method in ["function_spaces", "initial_condition", "solver"]:
            if getattr(self, f"get_{method}") is None:
                continue
            method_map = getattr(self, f"get_{method}")
            if method == "function_spaces":
                method_map = method_map(self.meshes[0])
            elif method == "initial_condition":
                method_map = method_map(self.meshes)
            elif method == "solver":
                self._reinitialise_fields(self.get_initial_condition(self.meshes))
                solver_gen = method_map(self.meshes, 0)
                assert hasattr(solver_gen, "__next__"), "solver should yield"
                if logger.level == DEBUG:
                    next(solver_gen)
                    f, f_ = self.fields[next(iter(self.fields))]
                    if np.array_equal(f.vector().array(), f_.vector().array()):
                        self.debug(
                            "Current and lagged solutions are equal. Does the"
                            " solver yield before updating lagged solutions?"
                        )  # noqa
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
        consistent = len(self.time_partition) == len(self.meshes)
        consistent &= all(
            len(self.meshes) == len(self._fs[field]) for field in self.fields
        )
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
                    field: [
                        self.get_function_spaces(mesh)[field] for mesh in self.meshes
                    ]
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
        return AttrDict(self.get_initial_condition(self.meshes))

    @property
    def solver(self):
        """
        See :meth:`~.MeshSeq.get_solver`.
        """
        # return self.get_solver()
        return self.get_solver

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

    @PETSc.Log.EventDecorator()
    def _solve_forward(self, update_solutions=True, solver_kwargs=None):
        r"""
        Solve a forward problem on a sequence of subintervals. Yields the final solution
        on each subinterval.

        :kwarg update_solutions: if ``True``, updates the solution data
        :type update_solutions: :class:`bool`
        :kwarg solver_kwargs: parameters for the forward solver
        :type solver_kwargs: :class:`dict` whose keys are :class:`str`\s and whose
            values may take various types
        :yields: the solution data of the forward solves
        :ytype: :class:`~.ForwardSolutionData`
        """
        solver_kwargs = solver_kwargs or {}
        num_subintervals = self.num_subintervals
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
            solver_gen = self.solver(self.meshes, i, **solver_kwargs)

            # Reinitialise fields and assign initial conditions
            self._reinitialise_fields(checkpoint)

            if update_solutions:
                # Solve sequentially between each export time
                for j in range(tp.num_exports_per_subinterval[i] - 1):
                    for _ in range(tp.num_timesteps_per_export[i]):
                        next(solver_gen)
                    # Update the solution data
                    for field, sol in self.fields.items():
                        if not self.field_types[field] == "steady":
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
        N = self.num_subintervals  # FIXME

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
        :type solver_kwargs: :class:`dict` whose keys are :class:`str`\s and whose
            values may take various types
        :returns: the solution data of the forward solves
        :rtype: :class:`~.ForwardSolutionData`
        """
        solver_kwargs = solver_kwargs or {}
        solver_gen = self._solve_forward(update_solutions=True, **solver_kwargs)
        for _ in range(self.num_subintervals):
            next(solver_gen)

        return self.solutions

    @PETSc.Log.EventDecorator()
    def fixed_point_iteration(
        self,
        adaptor,
        parameters=None,
        update_params=None,
        solver_kwargs=None,
        adaptor_kwargs=None,
    ):
        r"""
        Apply mesh adaptation using a fixed point iteration loop approach.

        :arg adaptor: function for adapting the mesh sequence. Its arguments are the
            mesh sequence and the solution data object. It should return ``True`` if the
            convergence criteria checks are to be skipped for this iteration. Otherwise,
            it should return ``False``.
        :kwarg parameters: parameters to apply to the mesh adaptation process
        :type parameters: :class:`~.AdaptParameters`
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
        self.params = parameters or AdaptParameters()
        solver_kwargs = solver_kwargs or {}
        adaptor_kwargs = adaptor_kwargs or {}

        self.meshes.params = self.params  # FIXME

        self.meshes._reset_counts()
        self.meshes.converged[:] = False
        self.meshes.check_convergence[:] = True

        for fp_iteration in range(self.params.maxiter):
            self.fp_iteration = fp_iteration
            self.meshes.fp_iteration = fp_iteration  # FIXME
            if update_params is not None:
                update_params(self.params, self.fp_iteration)

            # Solve the forward problem over all meshes
            self.solve_forward(solver_kwargs=solver_kwargs)

            # Adapt meshes, logging element and vertex counts
            continue_unconditionally = adaptor(
                self, self.meshes, self.solutions, **adaptor_kwargs
            )
            if self.params.drop_out_converged:
                self.check_convergence[:] = np.logical_not(
                    np.logical_or(continue_unconditionally, self.converged)
                )
            self.meshes.element_counts.append(self.meshes.count_elements())
            self.meshes.vertex_counts.append(self.meshes.count_vertices())

            # Check for element count convergence
            self.meshes.converged[:] = self.meshes.check_element_count_convergence()
            if self.meshes.converged.all():
                break
        else:
            for i, conv in enumerate(self.meshes.converged):
                if not conv:
                    pyrint(
                        f"Failed to converge on subinterval {i} in"
                        f" {self.params.maxiter} iterations."
                    )

        return self.solutions
