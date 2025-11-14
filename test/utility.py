"""
Functions used frequently for testing.
"""

import firedrake
import ufl

from goalie.metric import RiemannianMetric

__all__ = ["uniform_mesh", "uniform_metric", "mesh_for_sensors"]


def uniform_mesh(dim, n, length=1, **kwargs):
    args = [n] * dim + [length]
    return (firedrake.SquareMesh if dim == 2 else firedrake.CubeMesh)(*args, **kwargs)


def uniform_metric(function_space, scaling):
    dim = function_space.mesh().topological_dimension
    metric = RiemannianMetric(function_space)
    metric.interpolate(scaling * ufl.Identity(dim))
    return metric


def mesh_for_sensors(dim, n):
    mesh = uniform_mesh(dim, n, length=2)
    coords = firedrake.Function(mesh.coordinates)
    coords.interpolate(coords - ufl.as_vector([1] * dim))
    return firedrake.Mesh(coords)
