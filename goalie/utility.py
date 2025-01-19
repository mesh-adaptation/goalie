"""
Utility functions and classes for mesh adaptation.
"""

import os

import firedrake
import numpy as np

__all__ = ["AttrDict", "create_directory", "effectivity_index"]


class AttrDict(dict):
    """
    Dictionary that provides both ``self[key]`` and ``self.key`` access to members.

    **Disclaimer**: Copied from `stackoverflow
    <http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python>`__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def effectivity_index(error_indicator, Je):
    r"""
    Compute the overestimation factor of some error estimator for the QoI error.

    Note that this is only typically used for simple steady-state problems with
    analytical solutions.

    :arg error_indicator: a :math:`\mathbb P0` error indicator which localises
        contributions to an error estimator to individual elements
    :type error_indicator: :class:`firedrake.function.Function`
    :arg Je: the error in the quantity of interest
    :type Je: :class:`float`
    :returns: the effectivity index
    :rtype: :class:`float`
    """
    if not isinstance(error_indicator, firedrake.Function):
        raise ValueError("Error indicator must return a Function.")
    el = error_indicator.ufl_element()
    if not (el.family() == "Discontinuous Lagrange" and el.degree() == 0):
        raise ValueError("Error indicator must be P0.")
    eta = error_indicator.vector().gather().sum()
    return np.abs(eta / Je)


def create_directory(path, comm=firedrake.COMM_WORLD):
    """
    Create a directory on disk.

    **Disclaimer**: Code copied from `Thetis <https://thetisproject.org>`__.

    :arg path: path to the directory
    :type path: :class:`str`
    :kwarg comm: MPI communicator
    :type comm: :class:`mpi4py.MPI.Intracomm`
    :returns: the path in absolute form
    :rtype path: :class:`str`
    """
    if comm.rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    comm.barrier()
    return path
