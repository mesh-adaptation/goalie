"""
Testing for gradient computation.
"""

import unittest

import numpy as np
import ufl
from firedrake.constant import Constant
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
        self.qoi_expr = options_dict.get("qoi_expr", "quadratic")
        self.scalar = options_dict.get("scalar", 1.2)

    def get_function_spaces(self, mesh):
        return {"field": FunctionSpace(mesh, "R", 0)}

    def get_initial_condition(self):
        u = Function(self.function_spaces["field"][0])
        u.assign(self.initial_value)
        return {"field": u}

    def get_solver(self):
        def solver(index):
            """
            Artificial solve that just scales the initial condition.
            """
            fs = self.function_spaces["field"][index]
            u = self.fields["field"]
            u0 = Function(fs).assign(u)
            v = TestFunction(fs)
            F = u * v * ufl.dx - Constant(self.scalar) * u0 * v * ufl.dx

            tp = self.time_partition
            for _ in range(tp.num_subintervals):
                solve(F == 0, u, ad_block_tag="field")
                u0.assign(u)
            yield

        return solver

    def integrand(self, u):
        if self.qoi_expr == "linear":
            return 3 * u
        elif self.qoi_expr == "quadratic":
            return 0.5 * u**2
        elif self.qoi_expr == "cubic":
            return ufl.pi * u**3
        elif self.qoi_expr == "sqrt":
            return u**0.5
        else:
            raise NotImplementedError

    @annotate_qoi
    def get_qoi(self, index):
        """
        Various QoIs as determined by the `qoi_expr` option.
        """

        def steady_qoi():
            """
            QoI that squares the solution field.
            """
            u = self.fields["field"]
            return self.integrand(u) * ufl.dx

        return steady_qoi

    def expected_gradient(self):
        """
        Method for determining the expected value of the gradient.
        """
        if self.qoi_expr == "linear":
            return self.integrand(self.scalar)
        elif self.qoi_expr == "quadratic":
            return self.integrand(self.scalar) * 2 * self.initial_value
        elif self.qoi_expr == "cubic":
            return self.integrand(self.scalar) * 3 * self.initial_value**2
        elif self.qoi_expr == "sqrt":
            return self.integrand(self.scalar) * 0.5 * self.initial_value ** (-0.5)
        else:
            raise NotImplementedError


class TestSingleSubinterval(unittest.TestCase):
    """
    Unit tests for gradient computation on mesh sequences with a single subinterval.
    """

    def time_partition(self, dt):
        return TimeInterval(1.0, dt, "field")

    @parameterized.expand(
        [
            ("linear", 2.3),
            ("linear", 0.004),
            ("quadratic", 7.8),
            ("quadratic", -3),
            ("cubic", 0.0),
            ("cubic", np.exp(1)),
            ("sqrt", 1.0),
            ("sqrt", 4.2),
        ]
    )
    def test_single_timestep(self, qoi_expr, initial_value):
        options_dict = {
            "qoi_expr": qoi_expr,
            "initial_value": initial_value,
        }
        mesh_seq = GradientTestMeshSeq(
            options_dict,
            self.time_partition(1.0),
            UnitIntervalMesh(1),
            qoi_type="steady",
        )
        mesh_seq.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                mesh_seq.gradient[0].dat.data,
                mesh_seq.expected_gradient(),
            )
        )

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
