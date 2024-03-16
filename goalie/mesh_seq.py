"""
Sequences of meshes corresponding to a :class:`~.TimePartition`.
"""

import firedrake
from firedrake.adjoint import pyadjoint
from firedrake.adjoint_utils.solving import get_solve_blocks
from firedrake.petsc import PETSc
from firedrake.pyplot import triplot
from .function_data import ForwardSolutionData
from .interpolation import transfer
from .log import pyrint, debug, warning, info, logger, DEBUG
from .options import AdaptParameters
from animate.quality import QualityMeasure
from .utility import AttrDict, Mesh
from collections.abc import Iterable
import numpy as np


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
        :kwarg get_bcs: a function as described in :meth:`~.MeshSeq.get_bcs`
        :kwarg parameters: parameters to apply to the mesh adaptation process
        :type parameters: :class:`~.AdaptParameters`
        """
        self.time_partition = time_partition
        self.fields = time_partition.fields
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
        self._get_bcs = kwargs.get("get_bcs")
        self.params = kwargs.get("parameters")
        self.steady = time_partition.steady
        self.check_convergence = np.array([True] * len(self), dtype=bool)
        self.converged = np.array([False] * len(self), dtype=bool)
        self.fp_iteration = 0
        if self.params is None:
            self.params = AdaptParameters()
        self.sections = [{} for mesh in self]

        # Set the method for transferring fields between meshes
        transfer_method = kwargs.get("transfer_method", "interpolate")
        self.transfer = lambda source, target: transfer(source, target, transfer_method)

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
        return [mesh.num_cells() for mesh in self]  # TODO #123: make parallel safe

    def count_vertices(self):
        r"""
        Count the number of vertices in each mesh in the sequence.

        :returns: list of vertex counts
        :rtype: :class:`list` of :class:`int`\s
        """
        return [mesh.num_vertices() for mesh in self]  # TODO #123: make parallel safe

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
                nc = mesh.num_cells()
                nv = mesh.num_vertices()
                qm = QualityMeasure(mesh)
                ar = qm("aspect_ratio")
                mar = ar.vector().gather().max()
                self.debug(
                    f"{i}: {nc:7d} cells, {nv:7d} vertices,   max aspect ratio {mar:.2f}"
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

    def get_bcs(self):
        """
        Get the function mapping a subinterval index to a set of Dirichlet boundary
        conditions.

        Signature for the function to be returned:
        ```
        :arg index: the subinterval index
        :type index: :class:`int`
        :return: boundary conditions
        :rtype: :class:`~.DirichletBC` or :class:`list` thereof
        :rtype: see docstring above
        ```

        :returns: the function for obtaining the boundary conditions
        """
        if self._get_bcs is not None:
            return self._get_bcs(self)

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
        initial_condition_map = self.get_initial_condition()
        assert isinstance(
            initial_condition_map, dict
        ), "`get_initial_condition` should return a dict"
        mesh_seq_fields = set(self.fields)
        initial_condition_fields = set(initial_condition_map.keys())
        assert mesh_seq_fields.issubset(
            initial_condition_fields
        ), "missing fields in initial condition"
        assert initial_condition_fields.issubset(
            mesh_seq_fields
        ), "more initial conditions than fields"
        return AttrDict(initial_condition_map)

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

    @property
    def bcs(self):
        """
        See :meth:`~.MeshSeq.get_bcs`.
        """
        return self.get_bcs()

    @PETSc.Log.EventDecorator()
    def get_checkpoints(self, solver_kwargs={}, run_final_subinterval=False):
        r"""
        Solve forward on the sequence of meshes, extracting checkpoints corresponding
        to the starting fields on each subinterval.

        :kwarg solver_kwargs: parameters for the forward solver
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg run_final_subinterval: if ``True``, the solver is run on the final
            subinterval
        :type run_final_subinterval: :class:`bool`
        :returns: checkpoints for each subinterval
        :rtype: :class:`list` of :class:`firedrake.function.Function`\s
        """
        N = len(self)

        # The first checkpoint is the initial condition
        checkpoints = [self.initial_condition]

        # If there is only one subinterval then we are done
        if N == 1 and not run_final_subinterval:
            return checkpoints

        # Otherwise, solve each subsequent subinterval, in each case making use of the
        # previous checkpoint
        for i in range(N if run_final_subinterval else N - 1):
            sols = self.solver(i, checkpoints[i], **solver_kwargs)
            if not isinstance(sols, dict):
                raise TypeError(
                    f"Solver should return a dictionary, not '{type(sols)}'."
                )

            # Check that the output of the solver is as expected
            fields = set(sols.keys())
            if not set(self.fields).issubset(fields):
                diff = set(self.fields).difference(fields)
                raise ValueError(f"Fields are missing from the solver: {diff}.")
            if not fields.issubset(set(self.fields)):
                diff = fields.difference(set(self.fields))
                raise ValueError(f"Unexpected solver outputs: {diff}.")

            # Transfer between meshes
            if i < N - 1:
                checkpoints.append(
                    AttrDict(
                        {
                            field: self.transfer(sols[field], fs[i + 1])
                            for field, fs in self._fs.items()
                        }
                    )
                )

        return checkpoints

    @PETSc.Log.EventDecorator()
    def get_solve_blocks(self, field, subinterval):
        r"""
        Get all blocks of the tape corresponding to solve steps for prognostic solution
        field on a given subinterval.

        :arg field: name of the prognostic solution field
        :type field: :class:`str`
        :arg subinterval: subinterval index
        :type subinterval: :class:`int`
        :returns: list of solve blocks
        :rtype: :class:`list` of :class:`pyadjoint.block.Block`\s
        """
        tape = pyadjoint.get_working_tape()
        if tape is None:
            self.warning("Tape does not exist!")
            return []

        blocks = tape.get_blocks()
        if len(blocks) == 0:
            self.warning("Tape has no blocks!")
            return blocks

        # Restrict to solve blocks
        solve_blocks = get_solve_blocks()
        if len(solve_blocks) == 0:
            self.warning("Tape has no solve blocks!")
            return solve_blocks

        # Select solve blocks whose tags correspond to the field name
        solve_blocks = [
            block
            for block in solve_blocks
            if isinstance(block.tag, str) and block.tag.startswith(field)
        ]
        N = len(solve_blocks)
        if N == 0:
            self.warning(
                f"No solve blocks associated with field '{field}'."
                " Has ad_block_tag been used correctly?"
            )
            return solve_blocks
        self.debug(
            f"Field '{field}' on subinterval {subinterval} has {N} solve blocks."
        )

        # Check FunctionSpaces are consistent across solve blocks
        element = self.function_spaces[field][subinterval].ufl_element()
        for block in solve_blocks:
            if element != block.function_space.ufl_element():
                raise ValueError(
                    f"Solve block list for field '{field}' contains mismatching elements:"
                    f" {element} vs. {block.function_space.ufl_element()}."
                )

        # Check that the number of timesteps does not exceed the number of solve blocks
        num_timesteps = self.time_partition.num_timesteps_per_subinterval[subinterval]
        if num_timesteps > N:
            raise ValueError(
                f"Number of timesteps exceeds number of solve blocks for field '{field}'"
                f" on subinterval {subinterval}: {num_timesteps} > {N}."
            )

        # Check the number of timesteps is divisible by the number of solve blocks
        ratio = num_timesteps / N
        if not np.isclose(np.round(ratio), ratio):
            raise ValueError(
                "Number of timesteps is not divisible by number of solve blocks for"
                f" field '{field}' on subinterval {subinterval}: {num_timesteps} vs."
                f" {N}."
            )
        return solve_blocks

    def _output(self, field, subinterval, solve_block):
        """
        For a given solve block and solution field, get the block's outputs corresponding
        to the solution from the current timestep.

        :arg field: field of interest
        :type field: :class:`str`
        :arg subinterval: subinterval index
        :type subinterval: :class:`int`
        :arg solve_block: taped solve block
        :type solve_block: :class:`firedrake.adjoint.blocks.GenericSolveBlock`
        :returns: the output
        :rtype: :class:`firedrake.function.Function`
        """
        # TODO #93: Inconsistent return value - can be None
        fs = self.function_spaces[field][subinterval]

        # Loop through the solve block's outputs
        candidates = []
        for out in solve_block._outputs:
            # Look for Functions with matching function spaces
            if not isinstance(out.output, firedrake.Function):
                continue
            if out.output.function_space() != fs:
                continue

            # Look for Functions whose name matches that of the field
            # NOTE: Here we assume that the user has set this correctly in their
            #       get_solver method
            if not out.output.name() == field:
                continue

            # Add to the list of candidates
            candidates.append(out)

        # Check for existence and uniqueness
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise AttributeError(
                "Cannot determine a unique output index for the solution associated"
                f" with field '{field}' out of {len(candidates)} candidates."
            )
        elif not self.steady:
            raise AttributeError(
                f"Solve block for field '{field}' on subinterval {subinterval} has no"
                " outputs."
            )

    def _dependency(self, field, subinterval, solve_block):
        """
        For a given solve block and solution field, get the block's dependency which
        corresponds to the solution from the previous timestep.

        :arg field: field of interest
        :type field: :class:`str`
        :arg subinterval: subinterval index
        :type subinterval: :class:`int`
        :arg solve_block: taped solve block
        :type solve_block: :class:`firedrake.adjoint.blocks.GenericSolveBlock`
        :returns: the dependency
        :rtype: :class:`firedrake.function.Function`
        """
        # TODO #93: Inconsistent return value - can be None
        if self.field_types[field] == "steady":
            return
        fs = self.function_spaces[field][subinterval]

        # Loop through the solve block's dependencies
        candidates = []
        for dep in solve_block._dependencies:
            # Look for Functions with matching function spaces
            if not isinstance(dep.output, firedrake.Function):
                continue
            if dep.output.function_space() != fs:
                continue

            # Look for Functions whose name is the lagged version of the field's
            # NOTE: Here we assume that the user has set this correctly in their
            #       get_solver method
            if not dep.output.name() == f"{field}_old":
                continue

            # Add to the list of candidates
            candidates.append(dep)

        # Check for existence and uniqueness
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise AttributeError(
                "Cannot determine a unique dependency index for the lagged solution"
                f" associated with field '{field}' out of {len(candidates)} candidates."
            )
        elif not self.steady:
            raise AttributeError(
                f"Solve block for field '{field}' on subinterval {subinterval} has no"
                " dependencies."
            )

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

    @PETSc.Log.EventDecorator()
    def solve_forward(self, solver_kwargs={}):
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
        num_subintervals = len(self)
        P = self.time_partition
        solver = self.solver

        # Reinitialise the solution data object
        self._create_solutions()

        # Start annotating
        if pyadjoint.annotate_tape():
            tape = pyadjoint.get_working_tape()
            if tape is not None:
                tape.clear_tape()
        else:
            pyadjoint.continue_annotation()

        # Loop over the subintervals
        checkpoint = self.initial_condition
        for i in range(num_subintervals):
            stride = P.num_timesteps_per_export[i]
            num_exports = P.num_exports_per_subinterval[i]

            # Annotate tape on current subinterval
            checkpoint = solver(i, checkpoint, **solver_kwargs)

            # Loop over prognostic variables
            for field, fs in self.function_spaces.items():
                # Get solve blocks
                solve_blocks = self.get_solve_blocks(field, i)
                num_solve_blocks = len(solve_blocks)
                if num_solve_blocks == 0:
                    raise ValueError(
                        "Looks like no solves were written to tape!"
                        " Does the solution depend on the initial condition?"
                    )
                if fs[0].ufl_element() != solve_blocks[0].function_space.ufl_element():
                    raise ValueError(
                        f"Solve block list for field '{field}' contains mismatching"
                        f" finite elements: ({fs[0].ufl_element()} vs. "
                        f" {solve_blocks[0].function_space.ufl_element()})"
                    )

                # Extract solution data
                if len(solve_blocks[::stride]) >= num_exports:
                    raise ValueError(
                        f"More solve blocks than expected"
                        f" ({len(solve_blocks[::stride])} > {num_exports-1})"
                    )

                # Update solution data based on block dependencies and outputs
                sols = self.solutions[field]
                for j, block in enumerate(reversed(solve_blocks[::-stride])):
                    # Current solution is determined from outputs
                    out = self._output(field, i, block)
                    if out is not None:
                        sols.forward[i][j].assign(out.saved_output)

                    # Lagged solution comes from dependencies
                    dep = self._dependency(field, i, block)
                    if not self.steady and dep is not None:
                        sols.forward_old[i][j].assign(dep.saved_output)

            # Transfer the checkpoint between subintervals
            if i < num_subintervals - 1:
                checkpoint = AttrDict(
                    {
                        field: self.transfer(checkpoint[field], fs[i + 1])
                        for field, fs in self._fs.items()
                    }
                )

            # Clear the tape to reduce the memory footprint
            pyadjoint.get_working_tape().clear_tape()

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
        self, adaptor, update_params=None, solver_kwargs={}, adaptor_kwargs={}
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
        self.element_counts = [self.count_elements()]
        self.vertex_counts = [self.count_vertices()]
        self.converged[:] = False
        self.check_convergence[:] = True

        for self.fp_iteration in range(self.params.maxiter):
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
