"""
Module for handling PDE-constrained optimisation.
"""

import abc

import numpy as np
import pyadjoint
import ufl
from firedrake.assemble import assemble
from firedrake.exceptions import ConvergenceError
from firedrake.function import Function

from .go_mesh_seq import GoalOrientedMeshSeq
from .log import pyrint, warning
from .utility import AttrDict

__all__ = ["OptimisationProgress", "QoIOptimiser"]


class OptimisationProgress(AttrDict):
    """
    Class for stashing progress of an optimisation routine.

    The class is implemented as an :class:`goalie.utility.AttrDict` so that attributes
    may be accessed either as class attributes or with dictionary keys.

    The progress of three quantities are tracked as lists:
    * `qoi`: quantity of interest (QoI), i.e., cost function
    * `control`: control variable(s)
    * `gradient`: gradient(s) of QoI with respect to the control variable(s)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the progress tracks to their initial state as empty lists.
        """
        self["count"] = []
        self["qoi"] = []
        self["control"] = []
        self["gradient"] = []

    def convert_to_float(self):
        r"""
        Convert all progress tracks to be lists of :class:`float`\s.

        This is required because the optimisation algorithms will track progress of
        :class:`firedrake.function.Function`\s and :class:`~.AdjFloat`\s, whereas
        post-processing typically expects real values.
        """
        for key in self.keys():
            self[key] = [float(x) for x in self[key]]


class QoIOptimiser_Base(abc.ABC):
    """
    Base class for handling PDE-constrained optimisation.
    """

    # TODO: Use Goalie Solver rather than MeshSeq (#239)
    def __init__(self, mesh_seq, control, params, adaptor=None, adaptor_kwargs=None):
        """
        :arg mesh_seq: a mesh sequence that implements the forward model and
            computes the objective functional
        :type mesh_seq: :class:`~.AdjointMeshSeq`
        :arg control: name of the field to use as the control
        :type control: :class:`str`
        :kwarg params: Class holding parameters for optimisation routine
        :type params: :class:`~.OptimisationParameters`
        :kwarg adaptor: function for adapting the mesh sequence. Its arguments are the
            mesh sequence and the solution and indicator data objects. It should return
            ``True`` if the convergence criteria checks are to be skipped for this
            iteration. Otherwise, it should return ``False``.
        :kwarg adaptor: :class:`function`
        :kwarg adaptor_kwargs: parameters to pass to the adaptor
        :type adaptor_kwargs: :class:`dict` with :class:`str` keys and values which may
            take various types
        """
        self.mesh_seq = mesh_seq
        self.control = control
        if control not in mesh_seq.field_metadata:
            raise ValueError(f"Invalid control '{control}'.")
        if mesh_seq.field_metadata[control].family != "Real":
            raise NotImplementedError(
                "Only controls in R-space are currently implemented."
            )
        self.params = params
        self.progress = OptimisationProgress()
        self.adaptor = adaptor
        self.adaptive = adaptor is not None
        self.adaptor_kwargs = adaptor_kwargs
        self.goal_oriented = isinstance(self.mesh_seq, GoalOrientedMeshSeq)

    @abc.abstractmethod
    def step(self):
        """
        Take a step with the chosen optimisation approach.

        This method should be implemented in the subclass.
        """
        pass

    @property
    def converged(self):
        """
        Check for convergence of the optimiser.

        :return: `True` if the gradients have converged according to the gradient
            convergence tolerance, else `False`
        :rtype: :class:`bool`
        """
        dJ = float(self.mesh_seq.gradient[self.control])
        if abs(dJ / float(self.progress["gradient"][0])) < self.params.gtol:
            pyrint("Gradient convergence detected.")
            return True
        return False

    def check_divergence(self):
        """
        Check for divergence of the optimiser.

        :raises: :class:`~.ConvergenceError` if the QoI values have diverged according
            to the divergence tolerance
        """
        if np.abs(self.mesh_seq.J / np.min(self.progress["qoi"])) > self.params.dtol:
            raise ConvergenceError("QoI divergence detected.")

    def minimise(self):
        """
        Custom minimisation routine, where the tape is re-annotated each iteration to
        support mesh adaptation.

        Progress of the optimisation is tracked by the
        :class:`goalie.optimisation.OptimisationParameters` instance stashed on the
        :class:`goalie.optimisation.QoIOptimiser`
        as :meth:`goalie.optimisation.QoIOptimiser.progress`.

        :raises: :class:`~.ConvergenceError` if the maximum number of iterations are
            reached.
        """
        mesh_seq = self.mesh_seq
        for it in range(1, self.params.maxiter + 1):
            tape = pyadjoint.get_working_tape()
            tape.clear_tape()

            # Solve the forward and adjoint problems for the current control values
            pyadjoint.continue_annotation()
            if self.adaptive and self.goal_oriented:
                mesh_seq.indicate_errors(compute_gradient=True)
            else:
                mesh_seq.solve_adjoint(compute_gradient=True)
            pyadjoint.pause_annotation()
            J = mesh_seq.J
            u = mesh_seq.controls[self.control].tape_value()
            dJ = mesh_seq.gradient[self.control]

            # Take a step with the specified optimisation method and track progress
            self.step(u, J, dJ)
            pyrint(
                f"it={it:2d}, "
                f"control={float(u):11.4e}, "
                f"J={J:11.4e}, "
                f"dJ={float(dJ):11.4e}, "
                f"lr={self.params.lr:10.4e}"
            )
            self.progress["count"].append(it)
            self.progress["control"].append(u)
            self.progress["qoi"].append(J)
            self.progress["gradient"].append(dJ)

            if self.adaptive:
                # Check for QoI convergence
                # TODO #23: Put this check inside the adjoint solve as an optional
                #           return condition so that we can avoid unnecessary extra
                #           solves
                mesh_seq.qoi_values.append(J)
                qoi_converged = mesh_seq.check_qoi_convergence()
                if qoi_converged:
                    pyrint("QoI convergence detected.")
                    if mesh_seq.params.convergence_criteria == "any":
                        mesh_seq.converged[:] = True
                        self.adaptive = False

                # Check for error estimator convergence
                if self.goal_oriented:
                    mesh_seq.estimator_values.append(mesh_seq.error_estimate())
                    ee_converged = mesh_seq.check_estimator_convergence()
                    if ee_converged:
                        pyrint("Error estimator convergence detected.")
                        if mesh_seq.params.convergence_criteria == "any":
                            mesh_seq.converged[:] = True
                            self.adaptive = False
                else:
                    ee_converged = True

                # Adapt meshes and log element counts
                continue_unconditionally = self.adaptor(
                    mesh_seq,
                    mesh_seq.solutions,
                    mesh_seq.indicators,
                    **self.adaptor_kwargs,
                )
                if mesh_seq.params.drop_out_converged:
                    mesh_seq.check_convergence[:] = np.logical_not(
                        np.logical_or(continue_unconditionally, mesh_seq.converged)
                    )
                mesh_seq.element_counts.append(mesh_seq.count_elements())
                mesh_seq.vertex_counts.append(mesh_seq.count_vertices())

                # Check for element count convergence
                mesh_seq.converged[:] = mesh_seq.check_element_count_convergence()
                elem_converged = mesh_seq.converged.all()
                if elem_converged:
                    pyrint("Element count convergence detected.")
                    if mesh_seq.params.convergence_criteria == "any":
                        self.adaptive = False

                # Convergence check for 'all' mode
                if qoi_converged and ee_converged and elem_converged:
                    pyrint("Convergence of all quantities detected.")
                    self.adaptive = False

            # Update initial condition getter for the next iteration
            ics = mesh_seq.get_initial_condition()
            ics[self.control] = Function(
                mesh_seq.function_spaces[self.control][0], val=float(u)
            )
            if mesh_seq._get_initial_condition is None:
                # NOTE:
                # * mesh_seq.get_initial_condition may have been defined directly in the
                #   'object-oriented' approach (for example, see
                #   https://mesh-adaptation.github.io/docs/demos/burgers_oo.py.html)
                # * Ruff raises 'B023 Function definition does not bind loop variable
                #   `ics`' but this is intentional so we suppress the error.
                mesh_seq.get_initial_condition = lambda: ics  # noqa: B023
            else:
                mesh_seq._get_initial_condition = lambda mesh_seq: ics  # noqa: B023

            # Check for convergence and divergence
            if it == 1:
                continue
            if self.converged:
                self.progress.convert_to_float()
                return
            self.check_divergence()
        raise ConvergenceError("Reached maximum number of iterations.")


class QoIOptimiser_GradientDescent(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the gradient descent approach.
    """

    order = 1
    method_type = "gradient-based"

    def step(self, u, J, dJ):
        """
        Take one gradient descent step.

        Note that this method modifies
        :meth:`goalie.mesh_seq.MeshSeq._get_initial_condition` so that the control
        variable value corresponds to the updated value from this iteration.

        :arg u: control value at the current iteration
        :type u: :class:`firedrake.function.Function`, and
        :arg J: QoI value at the current iteration
        :type J: :class:`tuple` of :class:`~.AdjFloat`,
        :arg dJ: gradient at the current iteration
        :type dJ: :class:`firedrake.function.Function`
        """

        # Compute descent direction
        P = dJ.copy(deepcopy=True)
        P *= -1

        # Barzilai-Borwein formula
        if len(self.progress["control"]) > 1:
            u_prev = self.progress["control"][-2]
            dJ_prev = self.progress["gradient"][-1]
            dJ_diff = assemble(ufl.inner(dJ_prev - dJ, dJ_prev - dJ) * ufl.dx)
            if dJ_diff < 1e-12:
                warning(
                    "Near-zero difference between successive gradient values."
                    " Skipping Barzilai-Borwein step length update."
                )
            else:
                product = assemble(abs(ufl.inner(u_prev - u, dJ_prev - dJ)) * ufl.dx)
                self.params.lr = max(product / dJ_diff, self.params.lr_min)

        # Take a step downhill
        u.dat.data[:] += self.params.lr * P.dat.data


def QoIOptimiser(mesh_seq, control, params, method="gradient_descent"):
    """
    Factory method for constructing handlers for PDE-constrained optimisation.

    :return: Instance of the subclass corresponding to the requested implementation
    :rtype: Subclass of :class:`goalie.optimisation.QoIOptimiser_Base`
    """
    try:
        return {
            "gradient_descent": QoIOptimiser_GradientDescent,
        }[method](mesh_seq, control, params)
    except KeyError as ke:
        raise ValueError(f"Method '{method}' not supported.") from ke
