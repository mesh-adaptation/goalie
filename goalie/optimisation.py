"""
Module for handling PDE-constrained optimisation.
"""

import abc
from time import perf_counter

import numpy as np
import pyadjoint
import ufl
from firedrake.assemble import assemble
from firedrake.exceptions import ConvergenceError
from firedrake.function import Function
from firedrake.petsc import PETSc

from .go_mesh_seq import GoalOrientedMeshSeq
from .log import pyrint, warning
from .options import GoalOrientedAdaptParameters
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
        self["cputime"] = []
        self["qoi"] = []
        self["control"] = []
        self["gradient"] = []

    def convert_for_output(self):
        r"""
        Convert all progress tracks to be lists of :class:`float`\s and convert the
        cputimes to seconds elapsed.

        This is required because the optimisation algorithms will track progress of
        :class:`firedrake.function.Function`\s and :class:`~.AdjFloat`\s, whereas
        post-processing typically expects real values.
        """
        for key in self.keys():
            if key == "cputime":
                self[key] = [t - self[key][0] for t in self[key][1:]]
            else:
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

    def check_gradient_convergence(self):
        """
        Check for convergence of the optimisation routine due to the relative
        difference in gradient norm value being smaller than the specified tolerance.

        :return: ``True`` if gradient convergence is detected, else ``False``
        :rtype: :class:`bool`
        """
        dJ = float(self.mesh_seq.gradient[self.control])
        if abs(dJ / float(self.progress["gradient"][0])) < self.params.gtol:
            pyrint("Gradient convergence detected.")
            return True
        return False

    def check_qoi_divergence(self):
        """
        Check for divergence of the optimisation routine due to diverging QoI values.

        :raises: :class:`~.ConvergenceError` if the QoI values have diverged according
            to the divergence tolerance
        """
        if np.abs(self.mesh_seq.J / np.min(self.progress["qoi"])) > self.params.dtol:
            raise ConvergenceError("QoI divergence detected.")

    @PETSc.Log.EventDecorator()
    def minimise(self, adaptation_parameters=None, dropout=True):
        """
        Custom minimisation routine, where the tape is re-annotated each iteration to
        support mesh adaptation.

        Progress of the optimisation is tracked by the
        :class:`goalie.optimisation.OptimisationParameters` instance stashed on the
        :class:`goalie.optimisation.QoIOptimiser`
        as :meth:`goalie.optimisation.QoIOptimiser.progress`.

        :kwarg adaptation_parameters: parameters to apply to the mesh adaptation process
        :type adaptation_parameters: :class:`~.GoalOrientedAdaptParameters`
        :kwarg dropout: whether to stop adapting once the mesh has converged
        :type dropout: :class:`bool`
        :return: solution data from the last iteration
        :rtype: :class:`goalie.function.FunctionData`

        :raises: :class:`~.ConvergenceError` if the maximum number of iterations are
            reached.
        """
        mesh_seq = self.mesh_seq
        mesh_seq.params = adaptation_parameters or GoalOrientedAdaptParameters()
        for mesh_seq.fp_iteration in range(1, self.params.maxiter + 1):
            self.progress["cputime"].append(perf_counter())
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
            if mesh_seq.fp_iteration == 1:
                self.progress["control"].append(float(u))

            # Take a step with the specified optimisation method and track progress
            self.step(u, J, dJ)
            pyrint(
                f"it={mesh_seq.fp_iteration:2d}, "
                f"control={float(u):11.4e}, "
                f"J={J:11.4e}, "
                f"dJ={float(dJ):11.4e}, "
                f"lr={self.params.lr:10.4e}"
            )
            self.progress["count"].append(mesh_seq.fp_iteration)
            self.progress["control"].append(u)
            self.progress["qoi"].append(J)
            self.progress["gradient"].append(dJ)

            # Apply mesh adaptation, if enabled
            if self.adaptive:
                mesh_converged = mesh_seq._adapt_and_check(
                    self.adaptor, adaptor_kwargs=self.adaptor_kwargs
                )
                if dropout and mesh_converged:
                    self.adaptive = False

            # Update the value of the control in the MeshSeq
            mesh_seq.controls[self.control].assign(float(u))

            # Check for convergence and divergence
            if mesh_seq.fp_iteration == 1:
                continue
            if self.check_gradient_convergence():
                self.progress["control"].pop()
                self.progress["cputime"].append(perf_counter())
                self.progress.convert_for_output()
                return mesh_seq.solve_forward()
            self.check_qoi_divergence()
        raise ConvergenceError("Reached maximum number of iterations.")


class QoIOptimiser_GradientDescent(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the gradient descent approach.
    """

    order = 1
    method_type = "gradient-based"

    @PETSc.Log.EventDecorator()
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
            R = u.function_space()

            # Get all the quantities in the same space
            assert R == dJ.function_space()
            u_prev = Function(R, val=float(self.progress["control"][-2]))
            dJ_prev = Function(R, val=float(self.progress["gradient"][-1]))

            # Evaluate the formula
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


@PETSc.Log.EventDecorator()
def QoIOptimiser(mesh_seq, control, params, method="gradient_descent", **kwargs):
    """
    Factory method for constructing handlers for PDE-constrained optimisation.

    :arg mesh_seq: a mesh sequence that implements the forward model and computes the
        objective functional
    :type mesh_seq: :class:`~.AdjointMeshSeq`
    :arg control: name of the field to use as the control
    :type control: :class:`str`
    :kwarg params: Class holding parameters for optimisation routine
    :type params: :class:`~.OptimisationParameters`
    :kwarg adaptor: function for adapting the mesh sequence. Its arguments are the mesh
        sequence and the solution and indicator data objects. It should return ``True``
        if the convergence criteria checks are to be skipped for this iteration.
        Otherwise, it should return ``False``.
    :kwarg adaptor: :class:`function`
    :kwarg adaptor_kwargs: parameters to pass to the adaptor
    :type adaptor_kwargs: :class:`dict` with :class:`str` keys and values which may take
        various types
    :return: Instance of the subclass corresponding to the requested implementation
    :rtype: Subclass of :class:`goalie.optimisation.QoIOptimiser_Base`
    """
    try:
        return {
            "gradient_descent": QoIOptimiser_GradientDescent,
        }[method](mesh_seq, control, params, **kwargs)
    except KeyError as ke:
        raise ValueError(f"Method '{method}' not supported.") from ke
