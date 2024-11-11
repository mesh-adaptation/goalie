"""
Drivers for goal-oriented error estimation on sequences of meshes.
"""

from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import ufl
from animate.interpolation import interpolate
from animate.utility import Mesh
from firedrake import Function, FunctionSpace, MeshHierarchy, TransferManager
from firedrake.petsc import PETSc

from .adjoint import AdjointMeshSeq
from .error_estimation import get_dwr_indicator
from .function_data import IndicatorData
from .log import pyrint
from .options import GoalOrientedAdaptParameters

__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`~.AdjointMeshSeq` to account for goal-oriented problems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_values = []

    @PETSc.Log.EventDecorator()
    def get_enriched_mesh_seq(self, enrichment_method="p", num_enrichments=1):
        """
        Construct a sequence of globally enriched spaces.

        The following global enrichment methods are supported:
        * h-refinement (``enrichment_method='h'``) - refine each mesh element
          uniformly in each direction;
        * p-refinement (``enrichment_method='p'``) - increase the function space
          polynomial order by one globally.

        :kwarg enrichment_method: the method for enriching the mesh sequence
        :type enrichment_method: :class:`str`
        :kwarg num_enrichments: the number of enrichments to apply
        :type num_enrichments: :class:`int`
        :returns: the enriched mesh sequence
        :type: the type is inherited from the parent mesh sequence
        """
        if enrichment_method not in ("h", "p"):
            raise ValueError(f"Enrichment method '{enrichment_method}' not supported.")
        if num_enrichments <= 0:
            raise ValueError("A positive number of enrichments is required.")

        # Apply h-refinement
        if enrichment_method == "h":
            if any(mesh == self.meshes[0] for mesh in self.meshes[1:]):
                raise ValueError(
                    "h-enrichment is not supported for shallow-copied meshes."
                )
            meshes = [MeshHierarchy(mesh, num_enrichments)[-1] for mesh in self.meshes]
        else:
            meshes = [Mesh(mesh) for mesh in self.meshes]

        # Create copy of time_partition
        time_partition = deepcopy(self.time_partition)

        # Construct object to hold enriched spaces
        enriched_mesh_seq = type(self)(
            time_partition,
            meshes,
            get_function_spaces=self._get_function_spaces,
            get_initial_condition=self._get_initial_condition,
            get_form=self._get_form,
            get_solver=self._get_solver,
            get_qoi=self._get_qoi,
            qoi_type=self.qoi_type,
        )
        enriched_mesh_seq._update_function_spaces()

        # Apply p-refinement
        if enrichment_method == "p":
            for label, fs in enriched_mesh_seq.function_spaces.items():
                for n, _space in enumerate(fs):
                    element = _space.ufl_element()
                    element = element.reconstruct(
                        degree=element.degree() + num_enrichments
                    )
                    enriched_mesh_seq._fs[label][n] = FunctionSpace(
                        enriched_mesh_seq.meshes[n], element
                    )

        return enriched_mesh_seq

    @staticmethod
    def _get_transfer_function(enrichment_method):
        """
        Get the function for transferring function data between a mesh sequence and its
        enriched counterpart.

        :arg enrichment_method: the enrichment method used to generate the counterpart
            - see :meth:`~.GoalOrientedMeshSeq.get_enriched_mesh_seq` for the supported
            enrichment methods
        :type enrichment_method: :class:`str`
        :returns: the function for mapping function data between mesh sequences
        """
        if enrichment_method == "h":
            return TransferManager().prolong
        else:
            return interpolate

    def _create_indicators(self):
        """
        Create the :class:`~.FunctionData` instance for holding error indicator data.
        """
        self._indicators = IndicatorData(self.time_partition, self.meshes)

    @property
    def indicators(self):
        """
        :returns: the error indicator data object
        :rtype: :class:`~.IndicatorData`
        """
        if not hasattr(self, "_indicators"):
            self._create_indicators()
        return self._indicators

    @PETSc.Log.EventDecorator()
    def indicate_errors(
        self, enrichment_kwargs=None, solver_kwargs=None, indicator_fn=get_dwr_indicator
    ):
        """
        Compute goal-oriented error indicators for each subinterval based on solving the
        adjoint problem in a globally enriched space.

        :kwarg enrichment_kwargs: keyword arguments to pass to the global enrichment
            method - see :meth:`~.GoalOrientedMeshSeq.get_enriched_mesh_seq` for the
            supported enrichment methods and options
        :type enrichment_kwargs: :class:`dict` with :class:`str` keys and values which
            may take various types
        :kwarg solver_kwargs: parameters for the forward solver, as well as any
            parameters for the QoI, which should be included as a sub-dictionary with key
            'qoi_kwargs'
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg indicator_fn: function which maps the form, adjoint error and enriched
            space(s) as arguments to the error indicator
            :class:`firedrake.function.Function`
        :returns: solution and indicator data objects
        :rtype1: :class:`~.AdjointSolutionData
        :rtype2: :class:`~.IndicatorData
        """
        solver_kwargs = solver_kwargs or {}
        default_enrichment_kwargs = {"enrichment_method": "p", "num_enrichments": 1}
        enrichment_kwargs = dict(default_enrichment_kwargs, **(enrichment_kwargs or {}))
        enriched_mesh_seq = self.get_enriched_mesh_seq(**enrichment_kwargs)
        transfer = self._get_transfer_function(enrichment_kwargs["enrichment_method"])

        # Reinitialise the error indicator data object
        self._create_indicators()

        # Initialise adjoint solver generators on the MeshSeq and its enriched version
        adj_sol_gen = self._solve_adjoint(**solver_kwargs)
        adj_sol_gen_enriched = enriched_mesh_seq._solve_adjoint(**solver_kwargs)

        FWD, ADJ = "forward", "adjoint"
        FWD_OLD = "forward" if self.steady else "forward_old"
        ADJ_NEXT = "adjoint" if self.steady else "adjoint_next"
        P0_spaces = [FunctionSpace(mesh, "DG", 0) for mesh in self]

        # Loop over each subinterval in reverse
        for i in reversed(range(len(self))):
            # Solve the adjoint problem on the current subinterval
            next(adj_sol_gen)
            next(adj_sol_gen_enriched)

            # Get Functions
            u, u_, u_star, u_star_next, u_star_e = {}, {}, {}, {}, {}
            enriched_spaces = {
                f: enriched_mesh_seq.function_spaces[f][i] for f in self.fields
            }
            mapping = {}
            for f, fs_e in enriched_spaces.items():
                u[f] = Function(fs_e)
                u_[f] = Function(fs_e)
                mapping[f] = (
                    (u[f], u_[f])
                    if enriched_mesh_seq.field_types[f] == "unsteady"
                    else u[f]
                )
                u_star[f] = Function(fs_e)
                u_star_next[f] = Function(fs_e)
                u_star_e[f] = Function(fs_e)

            # Get forms for each equation in enriched space
            enriched_mesh_seq.fields = mapping
            forms = enriched_mesh_seq.form(i)

            # Loop over each timestep
            for j in range(self.time_partition.num_exports_per_subinterval[i] - 1):
                # In case of having multiple solution fields that are solved for one
                # after another, the field that is solved for first uses the values of
                # latter fields from the previous timestep. Therefore, we must transfer
                # the lagged solution of latter fields as if they were the current
                # timestep solutions. This assumes that the order of fields being solved
                # for in get_solver is the same as their order in self.fields
                for f_next in self.time_partition.field_names[1:]:
                    transfer(self.solutions[f_next][FWD_OLD][i][j], u[f_next])
                # Loop over each strongly coupled field
                for f in self.fields:
                    # Transfer solutions associated with the current field f
                    transfer(self.solutions[f][FWD][i][j], u[f])
                    transfer(self.solutions[f][FWD_OLD][i][j], u_[f])
                    transfer(self.solutions[f][ADJ][i][j], u_star[f])
                    transfer(self.solutions[f][ADJ_NEXT][i][j], u_star_next[f])

                    # Combine adjoint solutions as appropriate
                    u_star[f].assign(0.5 * (u_star[f] + u_star_next[f]))
                    u_star_e[f].assign(
                        0.5
                        * (
                            enriched_mesh_seq.solutions[f][ADJ][i][j]
                            + enriched_mesh_seq.solutions[f][ADJ_NEXT][i][j]
                        )
                    )
                    u_star_e[f] -= u_star[f]

                    # Evaluate error indicator
                    indi_e = indicator_fn(forms[f], u_star_e[f])

                    # Transfer back to the base space
                    indi = self._transfer(indi_e, P0_spaces[i])
                    indi.interpolate(abs(indi))
                    self.indicators[f][i][j].interpolate(ufl.max_value(indi, 1.0e-16))

            # discard current subinterval duplicate solution fields
            if not self.steady:
                for f in self.fields:
                    self.solutions[f][FWD_OLD].pop(-1)
                    self.solutions[f][ADJ_NEXT].pop(-1)
                    enriched_mesh_seq.solutions[f][FWD_OLD].pop(-1)
                    enriched_mesh_seq.solutions[f][ADJ_NEXT].pop(-1)

            # delete current subinterval enriched mesh to reduce the memory footprint
            if len(enriched_mesh_seq.meshes) > 1:
                for f in self.fields:
                    enriched_mesh_seq._fs[f].pop(-1)
                enriched_mesh_seq.meshes.pop(-1)
                enriched_mesh_seq.time_partition.drop_last_subinterval()

        # clear empty labels
        for f in self.fields:
            if self.steady:
                self.solutions.labels = ("forward",)
                self.solutions[f].pop("forward_old", None)
            else:
                self.solutions.labels = ("forward", "adjoint")
                self.solutions[f].pop("forward_old", None)
                self.solutions[f].pop("adjoint_next", None)

        return self.solutions, self.indicators

    @PETSc.Log.EventDecorator()
    def error_estimate(self, absolute_value=False):
        r"""
        Deduce the error estimator value associated with error indicator fields defined
        over the mesh sequence.

        :kwarg absolute_value: if ``True``, the modulus is taken on each element
        :type absolute_value: :class:`bool`
        :returns: the error estimator value
        :rtype: :class:`float`
        """
        assert isinstance(self.indicators, IndicatorData)
        if not isinstance(absolute_value, bool):
            raise TypeError(
                f"Expected 'absolute_value' to be a bool, not '{type(absolute_value)}'."
            )
        estimator = 0
        for field, by_field in self.indicators.items():
            if field not in self.time_partition.field_names:
                raise ValueError(
                    f"Key '{field}' does not exist in the TimePartition provided."
                )
            assert not isinstance(by_field, Function) and isinstance(by_field, Iterable)
            for by_mesh, dt in zip(by_field, self.time_partition.timesteps):
                assert not isinstance(by_mesh, Function) and isinstance(
                    by_mesh, Iterable
                )
                for indicator in by_mesh:
                    if absolute_value:
                        indicator.interpolate(abs(indicator))
                    estimator += dt * indicator.vector().gather().sum()
        return estimator

    def check_estimator_convergence(self):
        """
        Check for convergence of the fixed point iteration due to the relative
        difference in error estimator value being smaller than the specified tolerance.

        :return: ``True`` if estimator convergence is detected, else ``False``
        :rtype: :class:`bool`
        """
        if not self.check_convergence.any():
            self.info(
                "Skipping estimator convergence check because check_convergence"
                f" contains False values for indices {self._subintervals_not_checked}."
            )
            return False
        if len(self.estimator_values) >= max(2, self.params.miniter + 1):
            ee_, ee = self.estimator_values[-2:]
            if abs(ee - ee_) < self.params.estimator_rtol * abs(ee_):
                pyrint(
                    f"Error estimator converged after {self.fp_iteration+1} iterations"
                    f" under relative tolerance {self.params.estimator_rtol}."
                )
                return True
        return False

    @PETSc.Log.EventDecorator()
    def fixed_point_iteration(
        self,
        adaptor,
        parameters=None,
        update_params=None,
        enrichment_kwargs=None,
        adaptor_kwargs=None,
        solver_kwargs=None,
        indicator_fn=get_dwr_indicator,
    ):
        r"""
        Apply goal-oriented mesh adaptation using a fixed point iteration loop approach.

        :arg adaptor: function for adapting the mesh sequence. Its arguments are the mesh
            sequence and the solution and indicator data objects. It should return
            ``True`` if the convergence criteria checks are to be skipped for this
            iteration. Otherwise, it should return ``False``.
        :kwarg parameters: parameters to apply to the mesh adaptation process
        :type parameters: :class:`~.GoalOrientedAdaptParameters`
        :kwarg update_params: function for updating :attr:`~.MeshSeq.params` at each
            iteration. Its arguments are the parameter class and the fixed point
            iteration
        :kwarg enrichment_kwargs: keyword arguments to pass to the global enrichment
            method
        :type enrichment_kwargs: :class:`dict` with :class:`str` keys and values which
            may take various types
        :kwarg solver_kwargs: parameters to pass to the solver
        :type solver_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg adaptor_kwargs: parameters to pass to the adaptor
        :type adaptor_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        :kwarg indicator_fn: function which maps the form, adjoint error and enriched
            space(s) as arguments to the error indicator
            :class:`firedrake.function.Function`
        :returns: solution and indicator data objects
        :rtype1: :class:`~.AdjointSolutionData
        :rtype2: :class:`~.IndicatorData
        """
        # TODO #124: adaptor no longer needs solution and indicator data to be passed
        #            explicitly
        self.params = parameters or GoalOrientedAdaptParameters()
        enrichment_kwargs = enrichment_kwargs or {}
        adaptor_kwargs = adaptor_kwargs or {}
        solver_kwargs = solver_kwargs or {}
        self._reset_counts()
        self.qoi_values = []
        self.estimator_values = []
        self.converged[:] = False
        self.check_convergence[:] = True

        for fp_iteration in range(self.params.maxiter):
            self.fp_iteration = fp_iteration
            if update_params is not None:
                update_params(self.params, self.fp_iteration)

            # Indicate errors over all meshes
            self.indicate_errors(
                enrichment_kwargs=enrichment_kwargs,
                solver_kwargs=solver_kwargs,
                indicator_fn=indicator_fn,
            )

            # Check for QoI convergence
            # TODO #23: Put this check inside the adjoint solve as an optional return
            #           condition so that we can avoid unnecessary extra solves
            self.qoi_values.append(self.J)
            qoi_converged = self.check_qoi_convergence()
            if self.params.convergence_criteria == "any" and qoi_converged:
                self.converged[:] = True
                break

            # Check for error estimator convergence
            self.estimator_values.append(self.error_estimate())
            ee_converged = self.check_estimator_convergence()
            if self.params.convergence_criteria == "any" and ee_converged:
                self.converged[:] = True
                break

            # Adapt meshes and log element counts
            continue_unconditionally = adaptor(
                self, self.solutions, self.indicators, **adaptor_kwargs
            )
            if self.params.drop_out_converged:
                self.check_convergence[:] = np.logical_not(
                    np.logical_or(continue_unconditionally, self.converged)
                )
            self.element_counts.append(self.count_elements())
            self.vertex_counts.append(self.count_vertices())

            # Check for element count convergence
            self.converged[:] = self.check_element_count_convergence()
            elem_converged = self.converged.all()
            if self.params.convergence_criteria == "any" and elem_converged:
                break

            # Convergence check for 'all' mode
            if qoi_converged and ee_converged and elem_converged:
                break
        else:
            if self.params.convergence_criteria == "all":
                pyrint(f"Failed to converge in {self.params.maxiter} iterations.")
                self.converged[:] = False
            else:
                for i, conv in enumerate(self.converged):
                    if not conv:
                        pyrint(
                            f"Failed to converge on subinterval {i} in"
                            f" {self.params.maxiter} iterations."
                        )

        return self.solutions, self.indicators
