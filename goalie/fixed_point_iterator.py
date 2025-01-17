import numpy as np
from firedrake.petsc import PETSc

from .log import pyrint
from .options import AdaptParameters


class FixedPointIterator:
    def __init__(self, *args, **kwargs):
        self.check_convergence = np.array([True] * len(self), dtype=bool)
        self.converged = np.array([False] * len(self), dtype=bool)
        self.params = None

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
