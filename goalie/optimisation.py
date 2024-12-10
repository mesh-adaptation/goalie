"""
Module for handling PDE-constrained optimisation.
"""

import abc

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


def line_search(forward_run, m, u, P, J, dJ, params):
    """
    Apply a backtracking line search method to compute the step length / learning rate
    (lr).

    :arg forward_run: a Python function that implements the forward model and computes
        the objective functional
    :type forward_run: :class:`~.Callable`
    :arg m: the current mesh
    :type m: :class:`firedrake.mesh.MeshGeometry`
    :arg u: the current control value
    :type u: :class:`~.Control`
    :arg P: the current descent direction
    :type P: :class:`firedrake.function.Function`
    :arg J: the current value of objective function
    :type J: :class:`~.AdjFloat`
    :arg dJ: the current gradient value
    :type dJ: :class:`firedrake.function.Function`
    :kwarg params: Class holding parameters for optimisation routine
    :type params: :class:`~.OptimisationParameters`
    """

    lr = params.lr
    if not params.line_search:
        return lr
    alpha = params.ls_rtol
    tau = params.ls_frac
    maxiter = params.ls_maxiter

    # Compute initial slope
    initial_slope = np.dot(dJ.dat.data, P.dat.data)
    if np.isclose(initial_slope, 0.0):
        return params.lr

    # Perform line search
    log(f"  Applying line search with alpha = {alpha} and tau = {tau}")
    ext = ""
    for i in range(maxiter):
        log(f"  {i:3d}:      lr = {lr:.4e}{ext}")
        u_plus = u + lr * P
        J_plus, u_plus = forward_run(m, u_plus)
        ext = f"  diff {J_plus - J:.4e}"

        # Check Armijo rule:
        if J_plus - J <= alpha * lr * initial_slope:
            break
        lr *= tau
        if lr < params.lr_min:
            lr = params.lr_min
            break
    else:
        raise Exception("Line search did not converge")
    log(f"  converged lr = {lr:.4e}")
    return lr


class QoIOptimiser_Base(abc.ABC):
    """
    Base class for handling PDE-constrained optimisation.
    """

    @abc.abstractmethod
    def __init__(self):
        pass  # TODO


class QoIOptimiser_GradientDescent(QoIOptimiser_Base):
    """
    Class for handling PDE-constrained optimisation using the gradient descent approach.
    """

    order = 1
    method_type = "gradient-based"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError  # TODO


def QoIOptimiser(method="gradient_descent"):
    """
    Factory method for constructing handlers for PDE-constrained optimisation.
    """
    try:
        return {"gradient_descent": QoIOptimiser_GradientDescent}[method]
    except KeyError as ke:
        raise ValueError(f"Method {method} not supported.") from ke
