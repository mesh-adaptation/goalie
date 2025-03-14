"""
Drivers for solving adjoint problems on sequences of meshes.
"""

from functools import wraps

import firedrake
import numpy as np
from animate.utility import norm
from firedrake.adjoint import pyadjoint
from firedrake.adjoint_utils.solving import get_solve_blocks
from firedrake.petsc import PETSc

from .function_data import AdjointSolutionData
from .log import pyrint
from .mesh_seq import MeshSeq
from .utility import AttrDict

__all__ = ["AdjointMeshSeq", "annotate_qoi"]


def annotate_qoi(get_qoi):
    """
    Decorator that ensures QoIs are annotated properly.

    To be applied to the :meth:`~.AdjointMeshSeq.get_qoi` method.

    :arg get_qoi: a function mapping a dictionary of solution data and an integer index
        to a QoI function
    """

    @wraps(get_qoi)
    def wrap_get_qoi(mesh_seq, i):
        qoi = get_qoi(mesh_seq, i)

        # Count number of arguments
        num_kwargs = 0 if qoi.__defaults__ is None else len(qoi.__defaults__)
        num_args = qoi.__code__.co_argcount - num_kwargs
        if num_args == 0:
            if mesh_seq.qoi_type not in ["end_time", "steady"]:
                raise ValueError(
                    "Expected qoi_type to be 'end_time' or 'steady',"
                    f" not '{mesh_seq.qoi_type}'."
                )
        elif num_args == 1:
            if mesh_seq.qoi_type != "time_integrated":
                raise ValueError(
                    "Expected qoi_type to be 'time_integrated',"
                    f" not '{mesh_seq.qoi_type}'."
                )
        else:
            raise ValueError(f"QoI should have 0 or 1 args, not {num_args}.")

        @PETSc.Log.EventDecorator("goalie.AdjointMeshSeq.evaluate_qoi")
        @wraps(qoi)
        def wrap_qoi(*args, **kwargs):
            j = firedrake.assemble(qoi(*args, **kwargs))
            if pyadjoint.tape.annotate_tape():
                j.block_variable.adj_value = 1.0
            return j

        mesh_seq.qoi = wrap_qoi
        return wrap_qoi

    return wrap_get_qoi


class AdjointMeshSeq(MeshSeq):
    """
    An extension of :class:`~.MeshSeq` to account for solving adjoint problems on a
    sequence of meshes.

    For time-dependent quantities of interest, the solver should access and modify
    :attr:`~AdjointMeshSeq.J`, which holds the QoI value.
    """

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
        :kwarg get_solver: a function as described in :meth:`~.MeshSeq.get_solver`
        :kwarg get_qoi: a function as described in :meth:`~.AdjointMeshSeq.get_qoi`
        """
        self.qoi_type = kwargs.pop("qoi_type")
        if self.qoi_type not in ["end_time", "time_integrated", "steady"]:
            raise ValueError(
                f"QoI type '{self.qoi_type}' not recognised."
                " Choose from 'end_time', 'time_integrated', or 'steady'."
            )
        self._get_qoi = kwargs.get("get_qoi")
        self.J = 0
        super().__init__(time_partition, initial_meshes, **kwargs)
        if self.qoi_type == "steady" and not self.steady:
            raise ValueError(
                "QoI type is set to 'steady' but the time partition is not steady."
            )
        elif self.qoi_type != "steady" and self.steady:
            raise ValueError(
                "Time partition is steady but the QoI type is set to"
                f" '{self.qoi_type}'."
            )
        self._controls = None
        self._gradient = None
        self.qoi_values = []

    @property
    @pyadjoint.no_annotations
    def initial_condition(self):
        return super().initial_condition

    @property
    def gradient(self):
        if self._gradient is None:
            raise AttributeError(
                "To compute the gradient, pass compute_gradient=True to the"
                " solve_adjoint method."
            )
        return self._gradient

    @annotate_qoi
    def get_qoi(self, subinterval):
        """
        Get the function for evaluating the QoI, which has either zero or one arguments,
        corresponding to either an end time or time integrated quantity of interest,
        respectively. If the QoI has an argument then it is for the current time.

        Signature for the function to be returned:
        ```
        :arg t: the current time (for time-integrated QoIs)
        :type t: :class:`float`
        :return: the QoI as a 0-form
        :rtype: :class:`ufl.form.Form`
        ```

        :arg solution_map: a dictionary whose keys are the solution field names and
            whose values are the corresponding solutions
        :type solution_map: :class:`dict` with :class:`str` keys and values and
            :class:`firedrake.function.Function` values
        :arg subinterval: the subinterval index
        :type subinterval: :class:`int`
        :returns: the function for obtaining the QoI
        :rtype: see docstring above
        """
        if self._get_qoi is None:
            raise NotImplementedError("'get_qoi' is not implemented.")
        return self._get_qoi(self, subinterval)

    @pyadjoint.no_annotations
    @PETSc.Log.EventDecorator()
    def get_checkpoints(self, solver_kwargs=None, run_final_subinterval=False):
        r"""
        Solve forward on the sequence of meshes, extracting checkpoints corresponding
        to the starting fields on each subinterval.

        The QoI is also evaluated.

        :kwarg solver_kwargs: additional keyword arguments to be passed to the solver
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg run_final_subinterval: if ``True``, the solver is run on the final
            subinterval
        :type run_final_subinterval: :class:`bool`
        :returns: checkpoints for each subinterval
        :rtype: :class:`list` of :class:`firedrake.function.Function`\s
        """
        solver_kwargs = solver_kwargs or {}

        # In some cases we run over all subintervals to check the QoI that is computed
        if run_final_subinterval:
            self.J = 0

        # Generate the checkpoints as in MeshSeq
        checkpoints = super().get_checkpoints(
            solver_kwargs=solver_kwargs, run_final_subinterval=run_final_subinterval
        )

        # Account for end time QoI
        if self.qoi_type in ["end_time", "steady"] and run_final_subinterval:
            self._reinitialise_fields(checkpoints[-1])
            qoi = self.get_qoi(len(self) - 1)
            self.J = qoi(**solver_kwargs.get("qoi_kwargs", {}))
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
        blocks = pyadjoint.get_working_tape().get_blocks()
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
                    f"Solve block list for field '{field}' contains mismatching"
                    f" elements: {element} vs. {block.function_space.ufl_element()}."
                )

        # Check that the number of timesteps does not exceed the number of solve blocks
        num_timesteps = self.time_partition.num_timesteps_per_subinterval[subinterval]
        if num_timesteps > N:
            raise ValueError(
                f"Number of timesteps exceeds number of solve blocks for field"
                f" '{field}' on subinterval {subinterval}: {num_timesteps} > {N}."
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
        For a given solve block and solution field, get the block's outputs
        corresponding to the solution from the current timestep.

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
        for out in solve_block.get_outputs():
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
        if not self.tmp_fields[field].unsteady:
            return
        fs = self.function_spaces[field][subinterval]

        # Loop through the solve block's dependencies
        candidates = []
        for dep in solve_block.get_dependencies():
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
        self._solutions = AdjointSolutionData(self.time_partition, self.function_spaces)

    @PETSc.Log.EventDecorator()
    def _solve_adjoint(
        self,
        solver_kwargs=None,
        adj_solver_kwargs=None,
        get_adj_values=False,
        test_checkpoint_qoi=False,
        track_coefficients=False,
        compute_gradient=False,
    ):
        """
        A generator for solving an adjoint problem on a sequence of subintervals.

        As well as the quantity of interest value, solution fields are computed - see
        :class:`~.AdjointSolutionData` for more information. The solution data are
        yielded at the end of each subinterval, before clearing the tape.

        :kwarg solver_kwargs: parameters for the forward solver, as well as any
            parameters for the QoI, which should be included as a sub-dictionary with
            key 'qoi_kwargs'
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg adj_solver_kwargs: parameters for the adjoint solver
        :type adj_solver_kwargs: :class:`dict` with :class:`str` keys and values which
            may take various types
        :kwarg get_adj_values: if ``True``, adjoint actions are also returned at
            exported timesteps
        :type get_adj_values: :class:`bool`
        :kwarg test_checkpoint_qoi: solve over the final subinterval when checkpointing
            so that the QoI value can be checked across runs
        :kwarg track_coefficients: if ``True``, coefficients in the variational form
            will be stored whenever they change between export times. Only relevant for
            goal-oriented error estimation on unsteady problems.
        :type track_coefficients: :class:`bool`
        :kwarg compute_gradient: if ``True``, the gradient of the QoI with respect to
            the initial conditions is computed and is available via the `gradient`
            attribute
        :type compute_gradient: :class:`bool`
        :yields: the solution data of the forward and adjoint solves
        :ytype: :class:`~.AdjointSolutionData`
        """
        # TODO #125: Support get_adj_values in AdjointSolutionData
        # TODO #126: Separate out qoi_kwargs
        solver_kwargs = solver_kwargs or {}
        adj_solver_kwargs = adj_solver_kwargs or {}
        tp = self.time_partition
        num_subintervals = len(self)
        solver = self.solver
        qoi_kwargs = solver_kwargs.get("qoi_kwargs", {})

        # Reinitialise the solution data object
        self._create_solutions()

        # Solve forward to get checkpoints and evaluate QoI
        checkpoints = self.get_checkpoints(
            solver_kwargs=solver_kwargs,
            run_final_subinterval=test_checkpoint_qoi,
        )
        J_chk = float(self.J)
        if test_checkpoint_qoi and np.isclose(J_chk, 0.0):
            self.warning("Zero QoI. Is it implemented as intended?")

        # Reset the QoI to zero
        self.J = 0

        if get_adj_values:
            for field in self.field_data:
                self.solutions.extract(layout="field")[field]["adj_value"] = []
                for i, fs in enumerate(self.function_spaces[field]):
                    self.solutions.extract(layout="field")[field]["adj_value"].append(
                        [
                            firedrake.Cofunction(fs.dual(), name=f"{field}_adj_value")
                            for j in range(tp.num_exports_per_subinterval[i] - 1)
                        ]
                    )

        @PETSc.Log.EventDecorator("goalie.AdjointMeshSeq.solve_adjoint.evaluate_fwd")
        @wraps(solver)
        def wrapped_solver(subinterval, initial_condition_map, **kwargs):
            """
            Decorator to allow the solver to stash its initial conditions as controls.

            :arg subinterval: the subinterval index
            :type subinterval: :class:`int`
            :arg initial_condition_map: a dictionary of initial conditions, keyed by
                field name
            :type initial_condition_map: :class:`dict` with :class:`str` keys and
                :class:`firedrake.function.Function` values

            All keyword arguments are passed to the solver.
            """
            copy_map = AttrDict(
                {
                    field: initial_condition.copy(deepcopy=True)
                    for field, initial_condition in initial_condition_map.items()
                }
            )
            self._controls = list(map(pyadjoint.Control, copy_map.values()))

            # Reinitialise fields and assign initial conditions
            self._reinitialise_fields(copy_map)

            return solver(subinterval, **kwargs)

        # Loop over subintervals in reverse
        seeds = {}
        for i in reversed(range(num_subintervals)):
            stride = tp.num_timesteps_per_export[i]
            num_exports = tp.num_exports_per_subinterval[i]

            # Clear tape and start annotation
            if not pyadjoint.annotate_tape():
                pyadjoint.continue_annotation()
            tape = pyadjoint.get_working_tape()
            if tape is not None:
                tape.clear_tape()

            # Initialise the solver generator
            solver_gen = wrapped_solver(i, checkpoints[i], **solver_kwargs)

            # Annotate tape on current subinterval.
            # If we are using a goal-oriented approach on an unsteady problem, we need
            # to keep track of the coefficients in the variational form to detect their
            # changes between export times. In that case, we solve the forward problem
            # sequentially between each export time and save changing coefficients.
            # Otherwise, solve over the entire subinterval in one go.
            if track_coefficients:
                for j in range(tp.num_exports_per_subinterval[i] - 1):
                    for _ in range(tp.num_timesteps_per_export[i]):
                        next(solver_gen)
                    self._detect_changing_coefficients(j)
            else:
                for _ in range(tp.num_timesteps_per_subinterval[i]):
                    next(solver_gen)
            pyadjoint.pause_annotation()

            # Final solution is used as the initial condition for the next subinterval
            checkpoint = {
                field: sol[0] if self.tmp_fields[field].unsteady else sol
                for field, sol in self.field_data.items()
            }

            # Get seed vector for reverse propagation
            if i == num_subintervals - 1:
                if self.qoi_type in ["end_time", "steady"]:
                    pyadjoint.continue_annotation()
                    qoi = self.get_qoi(i)
                    self.J = qoi(**qoi_kwargs)
                    if np.isclose(float(self.J), 0.0):
                        self.warning("Zero QoI. Is it implemented as intended?")
                    pyadjoint.pause_annotation()
            else:
                for field, fs in self.function_spaces.items():
                    checkpoint[field].block_variable.adj_value = self._transfer(
                        seeds[field], fs[i]
                    )

            # Update adjoint solver kwargs
            for field in self.field_data:
                for block in self.get_solve_blocks(field, i):
                    block.adj_kwargs.update(adj_solver_kwargs)

            # Solve adjoint problem
            tape = pyadjoint.get_working_tape()
            with PETSc.Log.Event("goalie.AdjointMeshSeq.solve_adjoint.evaluate_adj"):
                controls = pyadjoint.enlisting.Enlist(self._controls)
                with pyadjoint.stop_annotating():
                    with tape.marked_nodes(controls):
                        tape.evaluate_adj(markings=True)

                # Compute the gradient on the first subinterval
                if i == 0 and compute_gradient:
                    self._gradient = controls.delist(
                        [control.get_derivative() for control in controls]
                    )

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

                # Detect whether we have a steady problem
                steady = self.steady or num_subintervals == num_solve_blocks == 1
                if steady and "adjoint_next" in checkpoint:
                    checkpoint.pop("adjoint_next")

                # Check that there are as many solve blocks as expected
                if len(solve_blocks[::stride]) >= num_exports:
                    self.warning(
                        "More solve blocks than expected:"
                        f" ({len(solve_blocks[::stride])} > {num_exports-1})."
                    )

                # Update forward and adjoint solution data based on block dependencies
                # and outputs
                solutions = self.solutions.extract(layout="field")[field]
                for j, block in enumerate(reversed(solve_blocks[::-stride])):
                    # Current forward solution is determined from outputs
                    out = self._output(field, i, block)
                    if out is not None:
                        solutions.forward[i][j].assign(out.saved_output)

                    # Current adjoint solution is determined from the adj_sol attribute
                    solutions.adjoint[i][j].assign(block.adj_sol)

                    # Lagged forward solution comes from dependencies
                    dep = self._dependency(field, i, block)
                    if not self.steady and dep is not None:
                        solutions.forward_old[i][j].assign(dep.saved_output)

                    # Adjoint action also comes from dependencies
                    if get_adj_values and dep is not None:
                        solutions.adj_value[i][j].assign(dep.adj_value)

                    # The adjoint solution at the 'next' timestep is determined from the
                    # adj_sol attribute of the next solve block
                    if not steady:
                        if (j + 1) * stride < num_solve_blocks:
                            solutions.adjoint_next[i][j].assign(
                                solve_blocks[(j + 1) * stride].adj_sol
                            )
                        elif (j + 1) * stride > num_solve_blocks:
                            raise IndexError(
                                "Cannot extract solve block"
                                f" {(j + 1) * stride} > {num_solve_blocks}."
                            )

                # The initial timestep of the current subinterval is the 'next' timestep
                # after the final timestep of the previous subinterval
                if i > 0:
                    self._transfer(
                        solve_blocks[0].adj_sol, solutions.adjoint_next[i - 1][-1]
                    )

                # Check non-zero adjoint solution/value
                if np.isclose(norm(solutions.adjoint[i][0]), 0.0):
                    self.warning(
                        f"Adjoint solution for field '{field}' on {self.th(i)}"
                        " subinterval is zero."
                    )
                if get_adj_values and np.isclose(norm(solutions.adj_value[i][0]), 0.0):
                    self.warning(
                        f"Adjoint action for field '{field}' on {self.th(i)}"
                        " subinterval is zero."
                    )

            # Get adjoint action on each subinterval
            with pyadjoint.stop_annotating():
                for field, control in zip(self.field_data, self._controls):
                    seeds[field] = firedrake.Cofunction(
                        self.function_spaces[field][i].dual()
                    )
                    if control.block_variable.adj_value is not None:
                        seeds[field].assign(control.block_variable.adj_value)
                    if not self.steady and np.isclose(norm(seeds[field]), 0.0):
                        self.warning(
                            f"Adjoint action for field '{field}' on {self.th(i)}"
                            " subinterval is zero."
                        )

            yield self.solutions

            # Clear the tape to reduce the memory footprint
            tape.clear_tape()

        # Check the QoI value agrees with that due to the checkpointing run
        if self.qoi_type == "time_integrated" and test_checkpoint_qoi:
            if not np.isclose(J_chk, self.J):
                raise ValueError(
                    "QoI values computed during checkpointing and annotated"
                    f" run do not match ({J_chk} vs. {self.J})"
                )

        tape.clear_tape()
        if not compute_gradient:
            self._gradient = None

    def solve_adjoint(
        self,
        solver_kwargs=None,
        adj_solver_kwargs=None,
        get_adj_values=False,
        test_checkpoint_qoi=False,
        compute_gradient=False,
    ):
        """
        Solve an adjoint problem on a sequence of subintervals.

        As well as the quantity of interest value, solution fields are computed - see
        :class:`~.AdjointSolutionData` for more information.

        :kwarg solver_kwargs: parameters for the forward solver, as well as any
            parameters for the QoI, which should be included as a sub-dictionary with
            key 'qoi_kwargs'
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg adj_solver_kwargs: parameters for the adjoint solver
        :type adj_solver_kwargs: :class:`dict` with :class:`str` keys and values which
            may take various types
        :kwarg get_adj_values: if ``True``, adjoint actions are also returned at
            exported timesteps
        :type get_adj_values: :class:`bool`
        :kwarg test_checkpoint_qoi: solve over the final subinterval when checkpointing
            so that the QoI value can be checked across runs
        :kwarg compute_gradient: if ``True``, the gradient of the QoI with respect to
            the initial conditions is computed and is available via the `gradient`
            attribute
        :type compute_gradient: :class:`bool`
        :returns: the solution data of the forward and adjoint solves
        :rtype: :class:`~.AdjointSolutionData`
        """

        # Initialise the adjoint solver generator
        adjoint_solver_gen = self._solve_adjoint(
            solver_kwargs=solver_kwargs,
            adj_solver_kwargs=adj_solver_kwargs,
            get_adj_values=get_adj_values,
            test_checkpoint_qoi=test_checkpoint_qoi,
            compute_gradient=compute_gradient,
        )
        # Solve the adjoint problem over each subinterval
        for _ in range(len(self)):
            next(adjoint_solver_gen)

        return self.solutions

    @staticmethod
    def th(num):
        """
        Convert from cardinal to ordinal.

        :arg num: the cardinal number to convert
        :type num: :class:`int`
        :returns: the corresponding ordinal number
        :rtype: :class:`str`
        """
        end = int(str(num)[-1])
        try:
            c = {1: "st", 2: "nd", 3: "rd"}[end]
        except KeyError:
            c = "th"
        return f"{num}{c}"

    def _subintervals_not_checked(self):
        num_not_checked = len(self.check_convergence[not self.check_convergence])
        return self.check_convergence.argsort()[num_not_checked]

    def check_qoi_convergence(self):
        """
        Check for convergence of the fixed point iteration due to the relative
        difference in QoI value being smaller than the specified tolerance.

        :return: ``True`` if QoI convergence is detected, else ``False``
        :rtype: :class:`bool`
        """
        if not self.check_convergence.any():
            self.info(
                "Skipping QoI convergence check because check_convergence contains"
                f" False values for indices {self._subintervals_not_checked}."
            )
            return False
        if len(self.qoi_values) >= max(2, self.params.miniter + 1):
            qoi_, qoi = self.qoi_values[-2:]
            if abs(qoi - qoi_) < self.params.qoi_rtol * abs(qoi_):
                pyrint(
                    f"QoI converged after {self.fp_iteration+1} iterations"
                    f" under relative tolerance {self.params.qoi_rtol}."
                )
                return True
        return False
