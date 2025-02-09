"""
Module for handling PDE-constrained optimisation.
"""

import abc

import firedrake.function as ffunc
import numpy as np
import pyadjoint
from firedrake import ConvergenceError

from .log import log
from .utility import AttrDict

__all__ = ["OptimisationProgress", "QoIOptimiser"]


class OptimisationProgress(AttrDict):
    """
    Class for stashing progress of an optimisation routine.
    """

    def __init__(self):
        self["qoi"] = []
        self["control"] = []
        self["gradient"] = []
        self["hessian"] = []


class QoIOptimiser_Base(abc.ABC):
    """
    Base class for handling PDE-constrained optimisation.
    """

    # TODO: Use Goalie Solver rather than MeshSeq
    def __init__(self, mesh_seq, params):
        """
        :arg mesh_seq: a mesh sequence that implements the forward model and
            computes the objective functional
        :type mesh_seq: :class:`~.AdjointMeshSeq`
        :kwarg params: Class holding parameters for optimisation routine
        :type params: :class:`~.OptimisationParameters`
        """
        self.mesh_seq = mesh_seq
        control = mesh_seq._control
        if (not isinstance(control, ffunc.Function)) or (
            control.ufl_element().family() != "Real"
        ):
            raise NotImplementedError(
                "Only controls in R-space are currently implemented."
            )
        self.params = params
        self.progress = OptimisationProgress()

    def line_search(self, P, J, dJ):
        """
        Apply a backtracking line search method to update the step length (i.e.,
        learning rate).

        :arg P: the current descent direction
        :type P: :class:`firedrake.function.Function`
        :arg J: the current value of objective function
        :type J: :class:`~.AdjFloat`
        :arg dJ: the current gradient value
        :type dJ: :class:`firedrake.function.Function`
        """

        lr = self.params.lr
        if not self.params.line_search:
            return lr
        alpha = self.params.ls_rtol
        tau = self.params.ls_frac
        maxiter = self.params.ls_maxiter

        # Compute initial slope
        initial_slope = np.dot(dJ.dat.data, P.dat.data)
        if np.isclose(initial_slope, 0.0):
            return self.params.lr

        # Perform line search
        log(f"  Applying line search with alpha = {alpha} and tau = {tau}")
        ext = ""
        for i in range(maxiter):
            log(f"  {i:3d}:      lr = {lr:.4e}{ext}")
            u_plus = self.mesh_seq._control + lr * P
            # TODO: Use Solver rather than MeshSeq; better implementation of controls
            self.mesh_seq._control = u_plus
            self.mesh_seq.get_checkpoints(run_final_subinterval=True)
            J_plus = self.mesh_seq.J
            ext = f"  diff {J_plus - J:.4e}"

            # Check Armijo rule:
            if J_plus - J <= alpha * lr * initial_slope:
                break
            lr *= tau
            if lr < self.params.lr_min:
                lr = self.params.lr_min
                break
        else:
            raise Exception("Line search did not converge")
        log(f"  converged lr = {lr:.4e}")
        self.lr = lr

    @abc.abstractmethod
    def step(self):
        """
        Take a step with the chosen optimisation approach.

        This method should be implemented in the subclass.
        """
        pass

    def minimise(self):
        # TODO: Docstring
        # TODO: Upstream implementation from opt_adapt
        params = self.params
        for it in range(self.params.maxiter):
            J, dJ, u = (float(x) for x in self.step())
            print(
                f"it={it+1:2d}, "
                f"u={u:.4e}, "
                f"J={J:.4e}, "
                f"dJ={dJ:.4e}, "
                f"lr={self.params.lr:.4e}"
            )
            self.progress["control"].append(u)
            self.progress["qoi"].append(J)
            self.progress["gradient"].append(dJ)
            if it == 0:
                continue

            # Check for QoI divergence
            if np.abs(J / np.min(self.progress["qoi"])) > params.dtol:
                raise ConvergenceError("QoI divergence detected")

            # Check for gradient convergence
            if dJ / self.progress["gradient"][0] < params.gtol:
                print("Gradient convergence detected")
                return
        raise ConvergenceError("Reached maximum number of iterations")


class QoIOptimiser_GradientDescent(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the gradient descent approach.
    """

    order = 1
    method_type = "gradient-based"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        tape = pyadjoint.get_working_tape()
        tape.clear_tape()
        pyadjoint.continue_annotation()
        self.mesh_seq.solve_adjoint()
        pyadjoint.pause_annotation()
        J = self.mesh_seq.J
        u = self.mesh_seq._control
        control = pyadjoint.Control(u)
        # TODO: Compute gradient in Goalie as part of solve_adjoint above
        dJ = pyadjoint.compute_gradient(J, control)
        u = u.copy(deepcopy=True)

        P = dJ.copy(deepcopy=True)
        P *= -1

        # FIXME:
        # # Barzilai-Borwein formula
        # if len(self.progress["control"]) > 0:
        #     u_prev = self.progress["control"][-1]
        #     dJ_prev = self.progress["gradient"][-1]
        #     dJ_diff = assemble(ufl.inner(dJ_prev - dJ, dJ_prev - dJ) * ufl.dx)
        #     lr = abs(assemble(ufl.inner(u_prev - u, dJ_prev - dJ) * ufl.dx) / dJ_diff)
        #     print(u_prev, dJ_prev, float(u), float(dJ), lr)
        #     self.params.lr = max(lr, self.params.lr_min)

        # Take a step downhill
        self.mesh_seq._control.dat.data[:] += self.params.lr * P.dat.data
        return J, dJ, u


class QoIOptimiser_Adam(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the Adam approach.
    """

    order = 1
    method_type = "gradient-based"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO: Upstream Adam implementation


class QoIOptimiser_Newton(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the Newton approach.
    """

    order = 2
    method_type = "newton"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO: Upstream Newton implementation


class QoIOptimiser_BFGS(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the BFGS approach.
    """

    order = 2
    method_type = "quasi-newton"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO: Upstream BFGS implementation


class QoIOptimiser_LBFGS(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the L-BFGS approach.
    """

    order = 2
    method_type = "quasi-newton"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO: Upstream L-BFGS implementation


def QoIOptimiser(mesh_seq, params, method="gradient_descent"):
    """
    Factory method for constructing handlers for PDE-constrained optimisation.
    """
    try:
        return {
            "gradient_descent": QoIOptimiser_GradientDescent,
            "adam": QoIOptimiser_Adam,
            "newton": QoIOptimiser_Newton,
            "bfgs": QoIOptimiser_BFGS,
            "lbfgs": QoIOptimiser_LBFGS,
        }[method](mesh_seq, params)
    except KeyError as ke:
        raise ValueError(f"Method {method} not supported.") from ke
