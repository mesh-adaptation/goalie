"""
Testing for gradient computation.
"""

import unittest

import numpy as np
import ufl
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitIntervalMesh
from parameterized import parameterized

from goalie.adjoint import AdjointMeshSeq, annotate_qoi
from goalie.time_partition import TimeInterval, TimePartition


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised during gradient computation.
    """

    def test_attribute_error(self):
        mesh_seq = AdjointMeshSeq(
            TimeInterval(1.0, 1.0, "field"),
            UnitIntervalMesh(1),
            qoi_type="steady",
        )
        with self.assertRaises(AttributeError) as cm:
            _ = mesh_seq.gradient
        msg = (
            "To compute the gradient, pass compute_gradient=True to the solve_adjoint"
            " method."
        )
        self.assertEqual(str(cm.exception), msg)


class GradientTestMeshSeq(AdjointMeshSeq):
    """
    Custom MeshSeq that accepts options to define the test problem.
    """

    def __init__(self, options_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_value = options_dict.get("initial_value", 1.0)
        self.family = options_dict.get("family", "CG")
        self.degree = options_dict.get("degree", 1)

    def get_function_spaces(self, mesh):
        return {"field": FunctionSpace(mesh, self.family, self.degree)}

    def get_initial_condition(self):
        u = Function(self.function_spaces["field"][0])
        u.assign(self.initial_value)
        return {"field": u}

    def get_solver(self):
        def solver(index):
            """
            Artificial solve that just assigns the initial condition.
            """
            fs = self.function_spaces["field"][index]
            u = self.fields["field"]
            u0 = Function(fs).assign(u)
            v = TestFunction(fs)
            F = u * v * ufl.dx - u0 * v * ufl.dx
            solve(F == 0, u, ad_block_tag="field")
            yield

        return solver

    @annotate_qoi
    def get_qoi(self, index):
        # TODO: Test some more interesting QoIs with parameterized
        def steady_qoi():
            """
            QoI that squares the solution field.
            """
            u = self.fields["field"]
            return ufl.inner(u, u) * ufl.dx

        return steady_qoi


class TestSingleSubinterval(unittest.TestCase):
    """
    Unit tests for gradient computation on mesh sequences with a single subinterval.
    """

    def time_partition(self, dt):
        return TimeInterval(1.0, dt, "field")

    @parameterized.expand([("R", 0), ("CG", 1)])
    def test_single_timestep(self, family, degree):
        options_dict = {"family": family, "degree": degree}
        mesh_seq = GradientTestMeshSeq(
            options_dict,
            self.time_partition(1.0),
            UnitIntervalMesh(1),
            qoi_type="steady",
        )
        mesh_seq.solve_adjoint(compute_gradient=True)
        self.assertTrue(np.allclose(mesh_seq.gradient[0].dat.data, 2.0))

    def test_two_timesteps(self):
        raise NotImplementedError("TODO")  # TODO


class TestTwoSubintervals(unittest.TestCase):
    """
    Unit tests for gradient computation on mesh sequences with two subintervals.
    """

    def setUp(self):
        end_time = 1.0
        num_subintervals = 2
        dt = end_time / num_subintervals
        self.time_partition = TimePartition(end_time, num_subintervals, dt, "field")
        self.meshes = [UnitIntervalMesh(1) for _ in range(num_subintervals)]

    def test_single_timestep(self):
        raise NotImplementedError("TODO")  # TODO

    def test_two_timesteps(self):
        raise NotImplementedError("TODO")  # TODO
