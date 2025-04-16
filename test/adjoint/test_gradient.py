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
        field = Field("field", family="Real", degree=0, unsteady=False)
        mesh_seq = AdjointMeshSeq(
            TimeInterval(1.0, 1.0, field),
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
        self.u0 = options_dict.get("u0", 1.0)
        self.qoi_power = options_dict.get("qoi_power", 2)
        self.scaling = options_dict.get("theta", 1.2)
        self.scaling_power = options_dict.get("scaling_power", 1)

    def get_initial_condition(self):
        R = self.function_spaces["field"][0]
        return {
            "field": Function(R).assign(self.u0),
            "theta": Function(R).assign(self.scaling),
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
            theta = self.field_functions["theta"]
            v = TestFunction(fs)
            F = u * v * ufl.dx - theta**self.scaling_power * u_ * v * ufl.dx

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
        return ufl.pi * u**self.qoi_power

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

    def expected_gradient(self, field):
        """
        Method for determining the expected value of the gradient.
        """
        assert field in ("field", "theta")
        tp = self.time_partition
        N = tp.num_timesteps
        q = self.qoi_power
        theta = self.scaling
        p = self.scaling_power
        if field == "field":
            if self.qoi_type in ("steady", "end_time"):
                # In the steady and end-time cases, the gradient accumulates the scale
                # factor as many times as there are timesteps
                integrand = self.integrand(theta ** (p * N))
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
                        integrand += dt * self.integrand(theta ** (p * k))
            return integrand * q * self.u0 ** (q - 1)
        else:
            if self.qoi_type in ("steady", "end_time"):
                multiplier = p * N * q / theta
                integrand = multiplier * self.integrand(theta ** (p * N) * self.u0)
            else:
                integrand = 0
                k = 0
                for subinterval in range(tp.num_subintervals):
                    dt = tp.timesteps[subinterval]
                    for _ in range(tp.num_timesteps_per_subinterval[subinterval]):
                        k += 1
                        multiplier = dt * p * k * q / theta
                        integrand += multiplier * self.integrand(
                            theta ** (p * k) * self.u0
                        )
            return integrand


class ThetaMethodTestMeshSeq(BaseTestMeshSeq):
    """
    MeshSeq defining a theta-timestepper test problem applied to :math:`y = e^t`.
    """

    def get_solver(self):
        def solver(index):
            fs = self.function_spaces["field"][index]
            tp = self.time_partition
            if self.field_metadata["field"].unsteady:
                u, u_ = self.field_functions["field"]
            else:
                u = self.field_functions["field"]
                u_ = Function(fs, name="field_old").assign(u)
            theta = self.field_functions["theta"]
            v = TestFunction(fs)
            F = (u - u_) * v * ufl.dx - (theta * u + (1 - theta) * u_) * v * ufl.dx

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
        # TODO: Use consistent theta method
        return u**self.qoi_power

    def expected_gradient(self, field):
        """
        Method for determining the expected value of the gradient.
        """
        assert field in ("field", "theta")
        tp = self.time_partition
        N = tp.num_timesteps
        q = self.qoi_power
        theta = self.scaling
        if field == "field":
            if self.qoi_type in ("steady", "end_time"):
                # In the steady and end-time cases, the gradient accumulates the scale
                # factor as many times as there are timesteps
                dt = tp.timesteps[-1]
                print(dt)
                S = (1 + dt * (1 - theta)) / (1 - dt * theta)
                print(S)
                integrand = self.integrand(S**N)
                # FIXME: multiple timesteps gives incorrect answers
            else:
                # In the time-integrated case, the gradient becomes a sum, where each
                # term accumulates an additional scale factor in each timestep. Each
                # contribution is multiplied by the timestep on the corresponding
                # subinterval
                integrand = 0
                k = 0
                for subinterval in range(tp.num_subintervals):
                    dt = tp.timesteps[subinterval]
                    S = (1 + dt * (1 - theta)) / (1 - dt * theta)
                    for _ in range(tp.num_timesteps_per_subinterval[subinterval]):
                        k += 1
                        integrand += dt * self.integrand(S**k)
                # FIXME: incorrect answers
            return integrand * q * self.u0 ** (q - 1)
        else:
            if self.qoi_type in ("steady", "end_time"):
                dt = tp.timesteps[-1]
                S = (1 + dt * (1 - theta)) / (1 - dt * theta)
                u = S**N * self.u0
                dudtheta = (
                    (dt * (dt + 1) * theta - 2 * dt - 1)
                    / (1 - dt * theta) ** 2
                    * self.u0
                )
                integrand = q * self.integrand(u) / self.u0 * dudtheta
                # FIXME: incorrect answers
            else:
                raise NotImplementedError  # TODO: Figure out the expected values
            return integrand


# First entry: scaling_power of field in QoI
# Second entry: initial value for field
fixture_pairs_scaling = [
    (1, 2.3),
    (1, 0.004),
    (2, 7.8),
    (2, -3),
    (3, 0.0),
    (3, np.exp(1)),
    (0.5, 1.0),
    (0.5, 4.2),
]

# First entry: scaling_power of field in QoI
# Second entry: initial value for field
# Third entry: scaling_power of scaling in form
fixture_triples_scaling = [
    tup + (i,) for tup in fixture_pairs_scaling for i in range(4)
]

# First entry: scaling_power of field in QoI
# Second entry: initial value for field
# Third entry: initial value for theta parameter
fixture_pairs_theta = [
    (1, 2.3, 0.01),
    (1, -37, 0.5),
    (1, 0.004, 0.99),
    (2, 7.8, 0.01),
    (2, 0.223, 0.5),
    (2, -3, 0.99),
]


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
            Field("field", family="Real", degree=0, unsteady=unsteady),
            Field("theta", family="Real", degree=0, unsteady=False, solved_for=False),
        ]
        return TimePartition(1.0, num_subintervals, dt, fields)

    def check_gradient(self, mesh_seq):
        mesh_seq.solve_adjoint(compute_gradient=True)
        got = mesh_seq.gradient[self.fieldname].dat.data
        expected = mesh_seq.expected_gradient(self.fieldname)
        test_pass = np.allclose(got, expected)
        if not test_pass:
            print(f"Got {got} but expected {expected}.")
        self.assertTrue(test_pass)


class TestGradient_Scaling_FieldInitialCondition(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the initial condition of the field
    can be computed correctly for the scaling problem.
    """

    fieldname = "field"

    @parameterized.expand(fixture_pairs_scaling)
    def test_single_timestep_steady_qoi(self, qoi_power, u0):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_pairs_scaling)
    def test_two_timesteps_end_time_qoi(self, qoi_power, u0):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0},
                self.time_partition(1, 0.5),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs_scaling)
    def test_two_subintervals_end_time_qoi(self, qoi_power, u0):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs_scaling)
    def test_two_subintervals_time_integrated_qoi(self, qoi_power, u0):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )


class TestGradient_Scaling_Scaling(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the scaling can be computed
    correctly for the scaling problem.
    """

    fieldname = "theta"

    @parameterized.expand(fixture_triples_scaling)
    def test_single_timestep_steady_qoi(self, qoi_power, u0, scaling_power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "scaling_power": scaling_power},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_triples_scaling)
    def test_two_subintervals_end_time_qoi(self, qoi_power, u0, scaling_power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "scaling_power": scaling_power},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_triples_scaling)
    def test_two_subintervals_time_integrated_qoi(self, qoi_power, u0, scaling_power):
        self.check_gradient(
            ScalingTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "scaling_power": scaling_power},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )


class TestGradient_Theta_FieldInitialCondition(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the initial condition of the field
    can be computed correctly for the theta-method problem.
    """

    fieldname = "field"

    @parameterized.expand(fixture_pairs_theta)
    def test_single_timestep_steady_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_pairs_theta)
    def test_two_timesteps_end_time_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(1, 0.5),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs_theta)
    def test_two_subintervals_end_time_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs_theta)
    def test_two_subintervals_time_integrated_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )


class TestGradient_Theta_Theta(BaseTestGradient):
    """
    Unit tests that check gradients with respect to the scaling can be computed
    correctly for the theta problem.
    """

    fieldname = "theta"

    @parameterized.expand(fixture_pairs_theta)
    def test_single_timestep_steady_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(1, 1.0),
                self.mesh,
                qoi_type="steady",
            )
        )

    @parameterized.expand(fixture_pairs_theta)
    def test_two_subintervals_end_time_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="end_time",
            )
        )

    @parameterized.expand(fixture_pairs_theta)
    def test_two_subintervals_time_integrated_qoi(self, qoi_power, u0, theta0):
        self.check_gradient(
            ThetaMethodTestMeshSeq(
                {"qoi_power": qoi_power, "u0": u0, "theta": theta0},
                self.time_partition(2, [0.25, 0.125]),
                self.mesh,
                qoi_type="time_integrated",
            )
        )
