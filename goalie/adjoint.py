"""
Drivers for solving adjoint problems on sequences of meshes.
"""

from functools import wraps

import firedrake
import numpy as np
from animate.utility import norm
from firedrake.adjoint import pyadjoint
from firedrake.petsc import PETSc

from .function_data import AdjointSolutionData
from .log import pyrint
from .mesh_seq import MeshSeq
from .options import GoalOrientedParameters
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
    def wrap_get_qoi(mesh_seq, solution_map, i):
        qoi = get_qoi(mesh_seq, solution_map, i)

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
        :kwarg get_form: a function as described in :meth:`~.MeshSeq.get_form`
        :kwarg get_solver: a function as described in :meth:`~.MeshSeq.get_solver`
        :kwarg get_bcs: a function as described in :meth:`~.MeshSeq.get_bcs`
        :kwarg parameters: parameters to apply to the mesh adaptation process
        :type parameters: :class:`~.AdaptParameters`
        :kwarg get_qoi: a function as described in :meth:`~.AdjointMeshSeq.get_qoi`
        """
        if kwargs.get("parameters") is None:
            kwargs["parameters"] = GoalOrientedParameters()
        self.qoi_type = kwargs.pop("qoi_type")
        if self.qoi_type not in ["end_time", "time_integrated", "steady"]:
            raise ValueError(
                f"QoI type '{self.qoi_type}' not recognised."
                " Choose from 'end_time', 'time_integrated', or 'steady'."
            )
        super().__init__(time_partition, initial_meshes, **kwargs)
        if self.qoi_type == "steady" and not self.steady:
            raise ValueError(
                "QoI type is set to 'steady' but the time partition is not steady."
            )
        elif self.qoi_type != "steady" and self.steady:
            raise ValueError(
                f"Time partition is steady but the QoI type is set to '{self.qoi_type}'."
            )
        self._get_qoi = kwargs.get("get_qoi")
        self.J = 0
        self._controls = None
        self.qoi_values = []

    @property
    @pyadjoint.no_annotations
    def initial_condition(self):
        return super().initial_condition

    @annotate_qoi
    def get_qoi(self, solution_map, subinterval):
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

        :arg solution_map: a dictionary whose keys are the solution field names and whose
            values are the corresponding solutions
        :type solution_map: :class:`dict` with :class:`str` keys and values and
            :class:`firedrake.function.Function` values
        :arg subinterval: the subinterval index
        :type subinterval: :class:`int`
        :returns: the function for obtaining the QoI
        :rtype: see docstring above
        """
        if self._get_qoi is None:
            raise NotImplementedError("'get_qoi' is not implemented.")
        return self._get_qoi(self, solution_map, subinterval)

    @pyadjoint.no_annotations
    @PETSc.Log.EventDecorator()
    def get_checkpoints(self, solver_kwargs={}, run_final_subinterval=False):
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

        # In some cases we run over all subintervals to check the QoI that is computed
        if run_final_subinterval:
            self.J = 0

        # Generate the checkpoints as in MeshSeq
        checkpoints = super().get_checkpoints(
            solver_kwargs=solver_kwargs, run_final_subinterval=run_final_subinterval
        )

        # Account for end time QoI
        if self.qoi_type in ["end_time", "steady"] and run_final_subinterval:
            qoi = self.get_qoi(checkpoints[-1], len(self) - 1)
            self.J = qoi(**solver_kwargs.get("qoi_kwargs", {}))
        return checkpoints

    @PETSc.Log.EventDecorator()
    def get_solve_blocks(self, field, subinterval, has_adj_sol=True):
        r"""
        Get all blocks of the tape corresponding to solve steps for prognostic solution
        field on a given subinterval.

        :arg field: name of the prognostic solution field
        :type field: :class:`str`
        :arg subinterval: subinterval index
        :type subinterval: :class:`int`
        :kwarg has_adj_sol: if ``True``, only blocks with ``adj_sol`` attributes will be
            considered
        :type has_adj_sol: :class:`bool`
        :returns: list of solve blocks
        :rtype: :class:`list` of :class:`pyadjoint.block.Block`\s
        """
        solve_blocks = super().get_solve_blocks(field, subinterval)
        if not has_adj_sol:
            return solve_blocks

        # Check that adjoint solutions exist
        if all(block.adj_sol is None for block in solve_blocks):
            self.warning(
                "No block has an adjoint solution. Has the adjoint equation been solved?"
            )

        # Default adjoint solution to zero, rather than None
        for block in solve_blocks:
            if block.adj_sol is None:
                block.adj_sol = firedrake.Function(
                    self.function_spaces[field][subinterval], name=field
                )
        return solve_blocks

    def _create_solutions(self):
        """
        Create the :class:`~.FunctionData` instance for holding solution data.
        """
        self._solutions = AdjointSolutionData(self.time_partition, self.function_spaces)

    def _extract_adjoint_solutions(self, field, i, solve_blocks, get_adj_values=False):
        """
        Extract adjoint solutions from the tape for a given subinterval.

        :arg i: subinterval index
        :type i: :class:`int`
        :kwarg get_adj_values: if ``True``, adjoint actions are also returned at exported
            timesteps
        :type get_adj_values: :class:`bool`
        """
        # TODO #125: Support get_adj_values in AdjointSolutionData
        tp = self.time_partition
        solutions = self.solutions.extract(layout="field")[field]
        if get_adj_values and "adj_value" not in solutions:
            fs = self.function_spaces[field]
            self.solutions.extract(layout="field")[field]["adj_value"] = []
            for i, fs in enumerate(self.function_spaces[field]):
                self.solutions.extract(layout="field")[field]["adj_value"].append(
                    [
                        firedrake.Cofunction(fs.dual(), name=f"{field}_adj_value")
                        for j in range(tp.num_exports_per_subinterval[i] - 1)
                    ]
                )

        # Update adjoint solution data based on block dependencies and outputs
        stride = tp.num_timesteps_per_export[i]
        for j, block in enumerate(reversed(solve_blocks[::-stride])):
            # Current adjoint solution is determined from the adj_sol attribute
            if block.adj_sol is not None:
                solutions.adjoint[i][j].assign(block.adj_sol)

            # Adjoint action comes from dependencies
            dep = self._dependency(field, i, block)
            if get_adj_values and dep is not None:
                solutions.adj_value[i][j].assign(dep.adj_value)

            # The adjoint solution at the 'next' timestep is determined from the
            # adj_sol attribute of the next solve block
            num_solve_blocks = len(solve_blocks)
            steady = self.steady or len(self) == num_solve_blocks == 1
            if not steady:
                if (j + 1) * stride < num_solve_blocks:
                    if solve_blocks[(j + 1) * stride].adj_sol is not None:
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
        if i > 0 and solve_blocks[0].adj_sol is not None:
            self._transfer(solve_blocks[0].adj_sol, solutions.adjoint_next[i - 1][-1])

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

    @PETSc.Log.EventDecorator()
    def solve_adjoint(
        self,
        solver_kwargs={},
        adj_solver_kwargs={},
        get_adj_values=False,
        test_checkpoint_qoi=False,
    ):
        """
        Solve an adjoint problem on a sequence of subintervals.

        As well as the quantity of interest value, solution fields are computed - see
        :class:`~.AdjointSolutionData` for more information.

        :kwarg solver_kwargs: parameters for the forward solver, as well as any
            parameters for the QoI, which should be included as a sub-dictionary with key
            'qoi_kwargs'
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg adj_solver_kwargs: parameters for the adjoint solver
        :type adj_solver_kwargs: :class:`dict` with :class:`str` keys and values which
            may take various types
        :kwarg get_adj_values: if ``True``, adjoint actions are also returned at exported
            timesteps
        :type get_adj_values: :class:`bool`
        :kwarg test_checkpoint_qoi: solve over the final subinterval when checkpointing
            so that the QoI value can be checked across runs
        :returns: the solution data of the forward and adjoint solves
        :rtype: :class:`~.AdjointSolutionData`
        """
        # TODO #126: Separate out qoi_kwargs
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
            return solver(subinterval, copy_map, **kwargs)

        # Loop over subintervals in reverse
        seeds = {}
        for i in reversed(range(num_subintervals)):
            # Clear tape and start annotation
            if not pyadjoint.annotate_tape():
                pyadjoint.continue_annotation()
            tape = pyadjoint.get_working_tape()
            if tape is not None:
                tape.clear_tape()

            # Annotate tape on current subinterval
            checkpoint = wrapped_solver(i, checkpoints[i], **solver_kwargs)
            pyadjoint.pause_annotation()

            # Get seed vector for reverse propagation
            if i == num_subintervals - 1:
                if self.qoi_type in ["end_time", "steady"]:
                    pyadjoint.continue_annotation()
                    qoi = self.get_qoi(checkpoint, i)
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
            for field in self.fields:
                for block in self.get_solve_blocks(field, i, has_adj_sol=False):
                    block.adj_kwargs.update(adj_solver_kwargs)

            # Solve adjoint problem
            tape = pyadjoint.get_working_tape()
            with PETSc.Log.Event("goalie.AdjointMeshSeq.solve_adjoint.evaluate_adj"):
                m = pyadjoint.enlisting.Enlist(self._controls)
                with pyadjoint.stop_annotating():
                    with tape.marked_nodes(m):
                        tape.evaluate_adj(markings=True)

            # Extract forward and adjoint solutions from tape
            for field in self.fields:
                solve_blocks = self._extract_forward_solutions(
                    field, i, return_blocks=True
                )
                self._extract_adjoint_solutions(
                    field, i, solve_blocks, get_adj_values=get_adj_values
                )

            # Get adjoint action on each subinterval
            with pyadjoint.stop_annotating():
                for field, control in zip(self.fields, self._controls):
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
