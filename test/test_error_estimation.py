from firedrake import *
from goalie.error_estimation import (
    form2indicator,
    get_dwr_indicator,
)
from goalie.function_data import IndicatorData
from goalie.go_mesh_seq import GoalOrientedMeshSeq
from goalie.time_partition import TimeInstant, TimePartition
from parameterized import parameterized
import unittest


class ErrorEstimationTestCase(unittest.TestCase):
    """
    Base class for error estimation testing.
    """

    def setUp(self):
        self.mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(self.mesh, "CG", 1)
        self.trial = TrialFunction(self.fs)
        self.test = TestFunction(self.fs)
        self.one = Function(self.fs, name="Uno")
        self.one.assign(1)


class TestForm2Indicator(ErrorEstimationTestCase):
    """
    Unit tests for :func:`form2indicator`.
    """

    def test_form_type_error(self):
        with self.assertRaises(TypeError) as cm:
            form2indicator(1)
        msg = "Expected 'F' to be a Form, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_exterior_facet_integral(self):
        F = self.one * ds(1) - self.one * ds(2)
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], -1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_interior_facet_integral(self):
        F = avg(self.one) * dS
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], sqrt(2))
        self.assertAlmostEqual(indicator.dat.data[1], sqrt(2))

    def test_cell_integral(self):
        x, y = SpatialCoordinate(self.mesh)
        F = conditional(x + y < 1, 1, 0) * dx
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], 0)
        self.assertAlmostEqual(indicator.dat.data[1], 0.5)


class TestIndicators2Estimator(ErrorEstimationTestCase):
    """
    Unit tests for :meth:`error_estimate`.
    """

    def mesh_seq(self, time_partition=None):
        num_timesteps = 1 if time_partition is None else time_partition.num_timesteps
        return GoalOrientedMeshSeq(
            time_partition or TimeInstant("field"),
            self.mesh,
            qoi_type="steady" if num_timesteps == 1 else "end_time",
        )

    def test_time_partition_wrong_field_error(self):
        mesh_seq = self.mesh_seq(TimeInstant("field"))
        time_partition = TimeInstant("f")
        mesh_seq._indicators = IndicatorData(time_partition, mesh_seq.meshes)
        with self.assertRaises(ValueError) as cm:
            mesh_seq.error_estimate()
        msg = "Key 'f' does not exist in the TimePartition provided."
        self.assertEqual(str(cm.exception), msg)

    def test_absolute_value_type_error(self):
        mesh_seq = self.mesh_seq()
        with self.assertRaises(TypeError) as cm:
            mesh_seq.error_estimate(absolute_value=0)
        msg = "Expected 'absolute_value' to be a bool, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_unit_time_instant(self):
        mesh_seq = self.mesh_seq(time_partition=TimeInstant("field", time=1.0))
        mesh_seq.indicators["field"][0][0].assign(form2indicator(self.one * dx))
        estimator = mesh_seq.error_estimate()
        self.assertAlmostEqual(estimator, 1)  # 1 * (0.5 + 0.5)

    @parameterized.expand([[False], [True]])
    def test_unit_time_instant_abs(self, absolute_value):
        mesh_seq = self.mesh_seq(time_partition=TimeInstant("field", time=1.0))
        mesh_seq.indicators["field"][0][0].assign(form2indicator(-self.one * dx))
        estimator = mesh_seq.error_estimate(absolute_value=absolute_value)
        self.assertAlmostEqual(
            estimator, 1 if absolute_value else -1
        )  # (-)1 * (0.5 + 0.5)

    def test_half_time_instant(self):
        mesh_seq = self.mesh_seq(time_partition=TimeInstant("field", time=0.5))
        mesh_seq.indicators["field"][0][0].assign(form2indicator(self.one * dx))
        estimator = mesh_seq.error_estimate()
        self.assertAlmostEqual(estimator, 0.5)  # 0.5 * (0.5 + 0.5)

    def test_time_partition_same_timestep(self):
        mesh_seq = self.mesh_seq(
            time_partition=TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        )
        mesh_seq.indicators["field"][0][0].assign(form2indicator(2 * self.one * dx))
        estimator = mesh_seq.error_estimate()
        self.assertAlmostEqual(estimator, 1)  # 2 * 0.5 * (0.5 + 0.5)

    def test_time_partition_different_timesteps(self):
        mesh_seq = self.mesh_seq(
            time_partition=TimePartition(1.0, 2, [0.5, 0.25], ["field"])
        )
        indicator = form2indicator(self.one * dx)
        mesh_seq.indicators["field"][0][0].assign(indicator)
        mesh_seq.indicators["field"][1][0].assign(indicator)
        mesh_seq.indicators["field"][1][1].assign(indicator)
        estimator = mesh_seq.error_estimate()
        self.assertAlmostEqual(
            estimator, 1
        )  # 0.5 * (0.5 + 0.5) + 0.25 * 2 * (0.5 + 0.5)

    def test_time_instant_multiple_fields(self):
        mesh_seq = self.mesh_seq(
            time_partition=TimeInstant(["field1", "field2"], time=1.0)
        )
        indicator = form2indicator(self.one * dx)
        mesh_seq.indicators["field1"][0][0].assign(indicator)
        mesh_seq.indicators["field2"][0][0].assign(indicator)
        estimator = mesh_seq.error_estimate()
        self.assertAlmostEqual(estimator, 2)  # 2 * (1 * (0.5 + 0.5))


class TestGetDWRIndicator(ErrorEstimationTestCase):
    """
    Unit tests for :func:`get_dwr_indicator`.
    """

    def setUp(self):
        super().setUp()
        self.two = Function(self.fs, name="Dos")
        self.two.assign(2)
        self.F = self.one * self.test * dx

    def test_form_type_error(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.one, self.one)
        msg = "Expected 'F' to be a Form, not '<class 'firedrake.function.Function'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_adjoint_error_type_error(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, 1)
        msg = "Expected 'adjoint_error' to be a Function or dict, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_test_space_type_error1(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, self.one, test_space=self.one)
        msg = (
            "Expected 'test_space' to be a FunctionSpace or dict,"
            " not '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_input_error(self):
        adjoint_error = {"field1": self.one, "field2": self.one}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=self.fs)
        msg = "Inconsistent input for 'adjoint_error' and 'test_space'."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_test_space_error(self):
        adjoint_error = {"field": self.one}
        test_space = {"f": self.fs}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        msg = "Key 'field' does not exist in the test space provided."
        self.assertEqual(str(cm.exception), msg)

    def test_test_space_type_error2(self):
        adjoint_error = {"field": self.one}
        test_space = {"field": self.one}
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        msg = (
            "Expected 'test_space['field']' to be a FunctionSpace,"
            " not '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_mesh_error1(self):
        adjoint_error = Function(FunctionSpace(UnitTriangleMesh(), "CG", 1))
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error)
        msg = "Meshes underlying the form and adjoint error do not match."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_mesh_error2(self):
        test_space = FunctionSpace(UnitTriangleMesh(), "CG", 1)
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, self.one, test_space=test_space)
        msg = "Meshes underlying the form and test space do not match."
        self.assertEqual(str(cm.exception), msg)

    def test_convert_neither(self):
        adjoint_error = {"field": self.two}
        test_space = {"field": self.one.function_space()}
        indicator = get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_both(self):
        test_space = self.one.function_space()
        indicator = get_dwr_indicator(self.F, self.two, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_test_space(self):
        adjoint_error = {"field": self.two}
        test_space = self.one.function_space()
        indicator = get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error(self):
        test_space = {"Dos": self.one.function_space()}
        indicator = get_dwr_indicator(self.F, self.two, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error_no_test_space(self):
        indicator = get_dwr_indicator(self.F, self.two)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error_mismatch(self):
        test_space = {"field": self.one.function_space()}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, self.two, test_space=test_space)
        msg = "Key 'Dos' does not exist in the test space provided."
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
