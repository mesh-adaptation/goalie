"""
Module for handling PDE-constrained optimisation.
"""

import abc

import firedrake.function as ffunc
import numpy as np

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

    def __init__(self, forward_run, mesh, control, params):
        """
        :arg forward_run: a Python function that implements the forward model and
            computes the objective functional
        :type forward_run: :class:`~.Callable`
        :arg mesh: the initial mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg control: the initial control value
        :type control: :class:`~.Control`
        :kwarg params: Class holding parameters for optimisation routine
        :type params: :class:`~.OptimisationParameters`
        """
        # TODO: Use Goalie Solver rather than forward_run
        self.forward_run = forward_run
        self.mesh = mesh
        if (not isinstance(control, ffunc.Function)) or (
            control.ufl_element().family() != "R"
        ):
            raise NotImplementedError(
                "Only controls in R-space are currently implemented."
            )
        self.control = control
        self.params = params

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
            u_plus = self.control + lr * P
            # TODO: Use Goalie Solver rather than forward_run
            J_plus, u_plus = self.forward_run(self.mesh, u_plus)
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
        raise NotImplementedError  # TODO: Upstream implementation from opt_adapt


class QoIOptimiser_GradientDescent(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the gradient descent approach.
    """

    order = 1
    method_type = "gradient-based"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO: Upstream gradient descent implementation


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


def QoIOptimiser(method="gradient_descent"):
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
        }[method]
    except KeyError as ke:
        raise ValueError(f"Method {method} not supported.") from ke
