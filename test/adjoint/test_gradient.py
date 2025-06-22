"""
Testing for gradient computation.
"""

import unittest

import numpy as np
import ufl
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitIntervalMesh
from parameterized import parameterized

from goalie.adjoint import AdjointSolver, annotate_qoi
from goalie.field import Field
from goalie.mesh_seq import MeshSeq
from goalie.time_partition import TimeInterval, TimePartition


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised during gradient computation.
    """

    def test_attribute_error(self):
        field = Field("field", family="Real", degree=0, unsteady=False)
        solver = AdjointSolver(
            TimeInterval(1.0, 1.0, field),
            MeshSeq(UnitIntervalMesh(1)),
            qoi_type="steady",
        )
        with self.assertRaises(AttributeError) as cm:
            _ = solver.gradient
        msg = (
            "To compute the gradient, pass compute_gradient=True to the solve_adjoint"
            " method."
        )
        self.assertEqual(str(cm.exception), msg)


class GradientTestSolver(AdjointSolver):
    """
    Custom AdjointSolver that accepts options to define the test problem.
    """

    def __init__(self, options_dict, *args, **kwargs):
        self.initial_value = options_dict.get("initial_value", 1.0)
        self.qoi_degree = options_dict.get("qoi_degree", 2)
        self.scalar = options_dict.get("scalar", 1.2)
        super().__init__(*args, **kwargs)

    # @staticmethod
    # def get_function_spaces(mesh):
    #     return {"field": FunctionSpace(mesh, "R", 0)}

    def get_initial_condition(self):
        u = Function(self.function_spaces["field"][0])
        u.assign(self.initial_value)
        return {"field": u}

    def get_solver(self, index):
        """
        Artificial solve that just scales the initial condition.
        """
        fs = self.function_spaces["field"][index]
        tp = self.time_partition
        if tp.steady:
            u = self.field_functions["field"]
            u_ = Function(fs, name="field_old").assign(u)
        else:
            u, u_ = self.field_functions["field"]
        v = TestFunction(fs)
        F = u * v * ufl.dx - Constant(self.scalar) * u_ * v * ufl.dx

        # Scale the initial condition at each timestep
        t_start, t_end = self.subintervals[index]
        dt = tp.timesteps[index]
        t = t_start
        qoi = self.get_qoi(index)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="field")
            if self.qoi_type == "time_integrated":
                self.J += qoi(t)
            yield

            u_.assign(u)
            t += dt

    def integrand(self, u):
        """
        Expression for the integrand of the QoI in terms of the solution field.
        """
        return ufl.pi * u**self.qoi_degree

    @annotate_qoi
    def get_qoi(self, index):
        """
        Various QoIs as determined by the `qoi_degree` option.
        """
        tp = self.time_partition

        def steady_qoi():
            return self.integrand(self.field_functions["field"]) * ufl.dx

        def end_time_qoi():
            return self.integrand(self.field_functions["field"][0]) * ufl.dx

        def time_integrated_qoi(t):
            dt = tp.timesteps[index]
            return dt * self.integrand(self.field_functions["field"][0]) * ufl.dx

        if self.qoi_type == "steady":
            return steady_qoi
        elif self.qoi_type == "end_time":
            return end_time_qoi
        else:
            return time_integrated_qoi

    def expected_gradient(self):
        """
        Method for determining the expected value of the gradient.
        """
        tp = self.time_partition
        if self.qoi_type in ("steady", "end_time"):
            # In the steady and end-time cases, the gradient accumulates the scale
            # factor as many times as there are timesteps
            scaling = self.integrand(self.scalar**tp.num_timesteps)
        else:
            # In the time-integrated case, the gradient becomes a sum, where each term
            # accumulates an additional scale factor in each timestep. Each contribution
            # is multiplied by the timestep on the corresponding subinterval
            scaling = 0
            p = 0
            for subinterval in range(tp.num_subintervals):
                dt = tp.timesteps[subinterval]
                for _ in range(tp.num_timesteps_per_subinterval[subinterval]):
                    p += 1
                    scaling += dt * self.integrand(self.scalar**p)
        return scaling * self.qoi_degree * self.initial_value ** (self.qoi_degree - 1)


class TestGradientComputation(unittest.TestCase):
    """
    Unit tests that check gradient values can be computed correctly.
    """

    def time_partition(self, num_subintervals, dt, unsteady=True):
        field = Field("field", family="Real", degree=0, unsteady=unsteady)
        return TimePartition(1.0, num_subintervals, dt, field)

    def mesh_seq(self, num_subintervals):
        return MeshSeq([UnitIntervalMesh(1) for _ in range(num_subintervals)])

    @parameterized.expand(
        [
            (1, 2.3),
            (1, 0.004),
            (2, 7.8),
            (2, -3),
            (3, 0.0),
            (3, np.exp(1)),
            (0.5, 1.0),
            (0.5, 4.2),
        ]
    )
    def test_single_timestep_steady_qoi(self, qoi_degree, initial_value):
        solver = GradientTestSolver(
            {"qoi_degree": qoi_degree, "initial_value": initial_value},
            self.time_partition(1, 1.0, unsteady=False),
            MeshSeq([UnitIntervalMesh(1)]),
            qoi_type="steady",
        )
        solver.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                solver.gradient["field"].dat.data,
                solver.expected_gradient(),
            )
        )

    @parameterized.expand(
        [
            (1, 2.3),
            (1, 0.004),
            (2, 7.8),
            (2, -3),
            (3, 0.0),
            (3, np.exp(1)),
            (0.5, 1.0),
            (0.5, 4.2),
        ]
    )
    def test_two_timesteps_end_time_qoi(self, qoi_degree, initial_value):
        solver = GradientTestSolver(
            {"qoi_degree": qoi_degree, "initial_value": initial_value},
            self.time_partition(1, 0.5),
            self.mesh_seq(1),
            qoi_type="end_time",
        )
        solver.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                solver.gradient["field"].dat.data,
                solver.expected_gradient(),
            )
        )

    @parameterized.expand(
        [
            (1, 2.3),
            (1, 0.004),
            (2, 7.8),
            (2, -3),
            (3, 0.0),
            (3, np.exp(1)),
            (0.5, 1.0),
            (0.5, 4.2),
        ]
    )
    def test_two_subintervals_end_time_qoi(self, qoi_degree, initial_value):
        solver = GradientTestSolver(
            {"qoi_degree": qoi_degree, "initial_value": initial_value},
            self.time_partition(2, [0.25, 0.125]),
            self.mesh_seq(2),
            qoi_type="end_time",
        )
        solver.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                solver.gradient["field"].dat.data,
                solver.expected_gradient(),
            )
        )

    @parameterized.expand(
        [
            (1, 2.3),
            (1, 0.004),
            (2, 7.8),
            (2, -3),
            (3, 0.0),
            (3, np.exp(1)),
            (0.5, 1.0),
            (0.5, 4.2),
        ]
    )
    def test_two_subintervals_time_integrated_qoi(self, qoi_degree, initial_value):
        solver = GradientTestSolver(
            {"qoi_degree": qoi_degree, "initial_value": initial_value},
            self.time_partition(2, [0.25, 0.125]),
            self.mesh_seq(2),
            qoi_type="time_integrated",
        )
        solver.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                solver.gradient["field"].dat.data,
                solver.expected_gradient(),
            )
        )
