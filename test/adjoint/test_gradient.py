"""
Testing for gradient computation.
"""

import unittest

import numpy as np
import ufl
from firedrake.function import Function
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitIntervalMesh
from parameterized import parameterized

from goalie.adjoint import AdjointMeshSeq, annotate_qoi
from goalie.field import Field
from goalie.time_partition import TimeInterval, TimePartition


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised during gradient computation.
    """

    def test_attribute_error(self):
        mesh_seq = AdjointMeshSeq(
            TimeInterval(1.0, 1.0, Field("field", unsteady=False)),
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


class BaseTestMeshSeq(AdjointMeshSeq):
    """
    Base class for MeshSeqs for testing gradient computations.
    """

    def __init__(self, options_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_value = options_dict.get("initial_value", 1.0)
        self.qoi_degree = options_dict.get("qoi_degree", 2)
        self.scaling = options_dict.get("alpha", 1.2)
        self.power = options_dict.get("power", 1)

    def get_initial_condition(self):
        R = self.function_spaces["field"][0]
        return {
            "field": Function(R).assign(self.initial_value),
            "alpha": Function(R).assign(self.scaling),
        }

    def integrand(self):
        raise NotImplementedError("Need to implement integrand")

    @annotate_qoi
    def get_qoi(self, index):
        """
        Various QoIs as determined by the `integrand` method.
        """

        def steady_qoi():
            return self.integrand(self.field_functions["field"]) * ufl.dx

        def end_time_qoi():
            return self.integrand(self.field_functions["field"][0]) * ufl.dx

        def time_integrated_qoi(t):
            dt = self.time_partition.timesteps[index]
            return dt * self.integrand(self.field_functions["field"][0]) * ufl.dx

        if self.qoi_type == "steady":
            return steady_qoi
        elif self.qoi_type == "end_time":
            return end_time_qoi
        else:
            return time_integrated_qoi

    def expected_gradient(self):
        raise NotImplementedError("Need to implement expected_gradient")


class ScalingTestMeshSeq(BaseTestMeshSeq):
    """
    MeshSeq defining a simple scaling test problem.
    """

    def get_solver(self):
        def solver(index):
            """
            Artificial solve that just scales the initial condition.
            """
            fs = self.function_spaces["field"][index]
            tp = self.time_partition
            if self.field_metadata["field"].unsteady:
                u, u_ = self.field_functions["field"]
            else:
                u = self.field_functions["field"]
                u_ = Function(fs, name="field_old").assign(u)
            alpha = self.field_functions["alpha"]
            v = TestFunction(fs)
            F = u * v * ufl.dx - alpha**self.power * u_ * v * ufl.dx

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

        return solver

    def integrand(self, u):
        """
        Expression for the integrand of the QoI in terms of the solution field.
        """
        return ufl.pi * u**self.qoi_degree

    def expected_gradient(self, field):
        """
        Method for determining the expected value of the gradient.
        """
        assert field in ("field", "alpha")
        tp = self.time_partition
        N = tp.num_timesteps
        q = self.qoi_degree
        alpha = self.scaling
        p = self.power
        if field == "field":
            if self.qoi_type in ("steady", "end_time"):
                # In the steady and end-time cases, the gradient accumulates the scale
                # factor as many times as there are timesteps
                integrand = self.integrand(alpha ** (p * N))
            else:
                # In the time-integrated case, the gradient becomes a sum, where each
                # term accumulates an additional scale factor in each timestep. Each
                # contribution is multiplied by the timestep on the corresponding
                # subinterval
                integrand = 0
                k = 0
                for subinterval in range(tp.num_subintervals):
                    dt = tp.timesteps[subinterval]
                    for _ in range(tp.num_timesteps_per_subinterval[subinterval]):
                        k += 1
                        integrand += dt * self.integrand(alpha ** (p * k))
            return integrand * q * self.initial_value ** (q - 1)
        else:
            if self.qoi_type in ("steady", "end_time"):
                multiplier = p * N * q / alpha
                return multiplier * self.integrand(
                    alpha ** (p * N) * self.initial_value
                )
            else:
                integrand = 0
                k = 0
                for subinterval in range(tp.num_subintervals):
                    dt = tp.timesteps[subinterval]
                    for _ in range(tp.num_timesteps_per_subinterval[subinterval]):
                        k += 1
                        multiplier = dt * p * k * q / alpha
                        integrand += multiplier * self.integrand(
                            alpha ** (p * k) * self.initial_value
                        )
                return integrand


# First entry: power of field in QoI
# Second entry: initial value for field
fixture_pairs = [
    (1, 2.3),
    (1, 0.004),
    (2, 7.8),
    (2, -3),
    (3, 0.0),
    (3, np.exp(1)),
    (0.5, 1.0),
    (0.5, 4.2),
]

# First entry: power of field in QoI
# Second entry: initial value for field
# Third entry: power of scaling in form
fixture_triples = [tup + (i,) for tup in fixture_pairs for i in range(4)]


class BaseTestGradient(unittest.TestCase):
    """
    Base class for unit tests that check gradients can be computed correctly.
    """

    fieldname = None

    def setUp(self):
        assert self.fieldname is not None
        self.mesh = UnitIntervalMesh(1)

    def time_partition(self, num_subintervals, dt):
        unsteady = num_subintervals > 1 or not np.allclose(dt, 1.0)
        fields = [
            Field("field", unsteady=unsteady),
            Field("alpha", unsteady=False, solved_for=False),
        ]
        return TimePartition(1.0, num_subintervals, dt, fields)

    def check_gradient(self, mesh_seq):
        mesh_seq.solve_adjoint(compute_gradient=True)
        self.assertTrue(
            np.allclose(
                mesh_seq.gradient[self.fieldname].dat.data,
                mesh_seq.expected_gradient(self.fieldname),
            )
        )


class TestGradientFieldInitialCondition(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the initial condition of the field
    can be computed correctly.
    """

    fieldname = "field"

    @parameterized.expand(fixture_pairs)
    def test_single_timestep_steady_qoi(self, qoi_degree, initial_value):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": initial_value},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_pairs)
    def test_two_timesteps_end_time_qoi(self, qoi_degree, initial_value):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": initial_value},
                self.time_partition(1, 0.5),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs)
    def test_two_subintervals_end_time_qoi(self, qoi_degree, initial_value):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": initial_value},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs)
    def test_two_subintervals_time_integrated_qoi(self, qoi_degree, initial_value):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": initial_value},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )


class TestGradientScaling(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the scaling can be computed
    correctly.
    """

    fieldname = "alpha"

    @parameterized.expand(fixture_triples)
    def test_single_timestep_steady_qoi(self, qoi_degree, init_value, power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": init_value, "power": power},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_triples)
    def test_two_subintervals_end_time_qoi(self, qoi_degree, init_value, power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": init_value, "power": power},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_triples)
    def test_two_subintervals_time_integrated_qoi(self, qoi_degree, init_value, power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_degree": qoi_degree, "initial_value": init_value, "power": power},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )
