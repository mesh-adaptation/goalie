"""
Drivers for goal-oriented error estimation on sequences of meshes.
"""

from .adjoint import AdjointMeshSeq
from .error_estimation import get_dwr_indicator
from .log import pyrint
from .utility import AttrDict
from firedrake import Function, FunctionSpace, MeshHierarchy, TransferManager, project
from firedrake.petsc import PETSc
from collections.abc import Callable, Iterable
import numpy as np
from typing import Tuple
import ufl


__all__ = ["GoalOrientedMeshSeq"]


class GoalOrientedMeshSeq(AdjointMeshSeq):
    """
    An extension of :class:`~.AdjointMeshSeq` to account for goal-oriented problems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_values = []

    @PETSc.Log.EventDecorator()
    def get_enriched_mesh_seq(
        self, enrichment_method: str = "p", num_enrichments: int = 1
    ) -> AdjointMeshSeq:
        """
        Construct a sequence of globally enriched spaces.

        Currently, global enrichment may be achieved using one of:
        * h-refinement (``enrichment_method = 'h'``);
        * p-refinement (``enrichment_method = 'p'``).

        The number of refinements may be controlled by the keyword argument
        ``num_enrichments``.
        """
        if enrichment_method not in ("h", "p"):
            raise ValueError(f"Enrichment method '{enrichment_method}' not supported.")
        if num_enrichments <= 0:
            raise ValueError("A positive number of enrichments is required.")

        # Apply h-refinement
        if enrichment_method == "h":
            meshes = [MeshHierarchy(mesh, num_enrichments)[-1] for mesh in self.meshes]
        else:
            meshes = self.meshes

        # Construct object to hold enriched spaces
        mesh_seq_e = self.__class__(
            self.time_partition,
            meshes,
            get_function_spaces=self._get_function_spaces,
            get_initial_condition=self._get_initial_condition,
            get_form=self._get_form,
            get_solver=self._get_solver,
            get_qoi=self._get_qoi,
            get_bcs=self._get_bcs,
            qoi_type=self.qoi_type,
            parameters=self.params,
        )

        # Update function spaces for enriched mesh
        mesh_seq_e.update_function_spaces

        # Apply p-refinement
        if enrichment_method == "p":
            for label, fs in mesh_seq_e.function_spaces.items():
                for n, _space in enumerate(fs):
                    element = _space.ufl_element()
                    element = element.reconstruct(
                        degree=element.degree() + num_enrichments
                    )
                    mesh_seq_e._fs[label][n] = FunctionSpace(
                        mesh_seq_e.meshes[n], element
                    )

        return mesh_seq_e

    @staticmethod
    def _get_transfer_function(enrichment_method):
        if enrichment_method == "h":
            return TransferManager().prolong
        else:
            return lambda source, target: target.interpolate(source)

    def _create_indicators(self):
        P0_spaces = [FunctionSpace(mesh, "DG", 0) for mesh in self]
        self._indicators = AttrDict(
            {
                field: [
                    [
                        Function(fs, name=f"{field}_error_indicator")
                        for _ in range(
                            self.time_partition.num_exports_per_subinterval[i] - 1
                        )
                    ]
                    for i, fs in enumerate(P0_spaces)
                ]
                for field in self.fields
            }
        )

    @property
    def indicators(self):
        """
        Arrays holding exported error indicators.
        """
        if not hasattr(self, "_indicators"):
            self._create_indicators()
        return self._indicators

    @PETSc.Log.EventDecorator()
    def indicate_errors(
        self,
        enrichment_kwargs: dict = {},
        adj_kwargs: dict = {},
        indicator_fn: Callable = get_dwr_indicator,
    ) -> Tuple[dict, AttrDict]:
        """
        Compute goal-oriented error indicators for each subinterval based on solving the
        adjoint problem in a globally enriched space.

        :kwarg enrichment_kwargs: keyword arguments to pass to the global enrichment
            method
        :kwarg adj_kwargs: keyword arguments to pass to the adjoint solver
        :kwarg indicator_fn: function for error indication, which takes the form,
            adjoint error and enriched space(s) as arguments
        """
        enrichment_kwargs.setdefault("enrichment_method", "p")
        enrichment_kwargs.setdefault("num_enrichments", 1)
        mesh_seq_e = self.get_enriched_mesh_seq(**enrichment_kwargs)
        transfer = self._get_transfer_function(enrichment_kwargs["enrichment_method"])

        # Solve the forward and adjoint problems on the MeshSeq and its enriched version
        self.solve_adjoint(**adj_kwargs)
        mesh_seq_e.solve_adjoint(**adj_kwargs)

        FWD, ADJ = "forward", "adjoint"
        FWD_OLD = "forward" if self.steady else "forward_old"
        ADJ_NEXT = "adjoint" if self.steady else "adjoint_next"
        P0_spaces = [FunctionSpace(mesh, "DG", 0) for mesh in self]
        for i, mesh in enumerate(self):
            # Get Functions
            u, u_, u_star, u_star_next, u_star_e = {}, {}, {}, {}, {}
            enriched_spaces = {f: mesh_seq_e.function_spaces[f][i] for f in self.fields}
            mapping = {}
            for f, fs_e in enriched_spaces.items():
                u[f] = Function(fs_e)
                u_[f] = Function(fs_e)
                mapping[f] = (u[f], u_[f])
                u_star[f] = Function(fs_e)
                u_star_next[f] = Function(fs_e)
                u_star_e[f] = Function(fs_e)

            # Get forms for each equation in enriched space
            forms = mesh_seq_e.form(i, mapping)
            if not isinstance(forms, dict):
                raise TypeError(
                    "The function defined by get_form should return a dictionary"
                    f", not type '{type(forms)}'."
                )

            # Loop over each strongly coupled field
            for f in self.fields:
                # Loop over each timestep
                for j in range(len(self.solutions[f]["forward"][i])):
                    # Update fields
                    transfer(self.solutions[f][FWD][i][j], u[f])
                    transfer(self.solutions[f][FWD_OLD][i][j], u_[f])
                    transfer(self.solutions[f][ADJ][i][j], u_star[f])
                    transfer(self.solutions[f][ADJ_NEXT][i][j], u_star_next[f])

                    # Combine adjoint solutions as appropriate
                    u_star[f].assign(0.5 * (u_star[f] + u_star_next[f]))
                    u_star_e[f].assign(
                        0.5
                        * (
                            mesh_seq_e.solutions[f][ADJ][i][j]
                            + mesh_seq_e.solutions[f][ADJ_NEXT][i][j]
                        )
                    )
                    u_star_e[f] -= u_star[f]

                    # Evaluate error indicator
                    indi_e = indicator_fn(forms[f], u_star_e[f])

                    # Project back to the base space
                    indi = project(indi_e, P0_spaces[i])
                    indi.interpolate(abs(indi))
                    self.indicators[f][i][j].interpolate(ufl.max_value(indi, 1.0e-16))

        return self.solutions, self.indicators

    @PETSc.Log.EventDecorator()
    def error_estimate(self, absolute_value: bool = False) -> float:
        r"""
        Deduce the error estimator value associated with error indicator fields defined over
        a :class:`~.MeshSeq`.

        :kwarg absolute_value: toggle whether to take the modulus on each element
        """
        assert isinstance(self.indicators, dict)
        if not isinstance(absolute_value, bool):
            raise TypeError(
                f"Expected 'absolute_value' to be a bool, not '{type(absolute_value)}'."
            )
        estimator = 0
        for field, by_field in self.indicators.items():
            if field not in self.time_partition.fields:
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
        adaptor: Callable,
        enrichment_kwargs: dict = {},
        adaptor_kwargs: dict = {},
        adj_kwargs: dict = {},
        indicator_fn: Callable = get_dwr_indicator,
        **kwargs,
    ):
        r"""
        Apply goal-oriented mesh adaptation using a fixed point iteration loop.

        :arg adaptor: function for adapting the mesh sequence. Its arguments are the
            :class:`~.MeshSeq` instance, the dictionary of solution
            :class:`firedrake.function.Function`\s and the list of error indicators.
            It should return ``True`` if the convergence criteria checks are to be
            skipped for this iteration. Otherwise, it should return ``False``.
        :kwarg update_params: function for updating :attr:`~.GoalOrientedMeshSeq.params`
            at each iteration. Its arguments are the parameter class and the fixed point
            iteration number
        :kwarg enrichment_kwargs: keyword arguments to pass to the global enrichment
            method
        :kwarg adaptor_kwargs: a dictionary providing parameters to the adaptor
        :kwarg adj_kwargs: keyword arguments to pass to the adjoint solver
        :kwarg indicator_fn: function for error indication, which takes the form, adjoint
            error and enriched space(s) as arguments
        """
        update_params = kwargs.get("update_params")
        self.element_counts = [self.count_elements()]
        self.vertex_counts = [self.count_vertices()]
        self.qoi_values = []
        self.estimator_values = []
        self.converged[:] = False
        self.check_convergence[:] = True

        for self.fp_iteration in range(self.params.maxiter):
            if update_params is not None:
                update_params(self.params, self.fp_iteration)

            # Indicate errors over all meshes
            self._create_solutions()
            self._create_indicators()
            self.indicate_errors(
                enrichment_kwargs=enrichment_kwargs,
                adj_kwargs=adj_kwargs,
                indicator_fn=indicator_fn,
            )

            # Check for QoI convergence
            # TODO: Put this check inside the adjoint solve as
            #       an optional return condition so that we
            #       can avoid unnecessary extra solves
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
